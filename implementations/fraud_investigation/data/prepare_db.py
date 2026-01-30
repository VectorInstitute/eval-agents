"""Prepare and populate the fraud investigation database."""

import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Literal

import kagglehub
import numpy as np
import pandas as pd
from aieng.agent_evals.configs import Configs
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


def _compute_fan_metrics(transactions: pd.DataFrame, window: str, direction: Literal["in", "out"]) -> pd.Series:
    """Compute fan-in or fan-out degree over a rolling time window.

    Fan-in means many unique senders to one receiver.
    Fan-out means one sender to many unique receivers.
    """
    acct_col = "from_account" if direction == "out" else "to_account"
    partner_col = "to_account" if direction == "out" else "from_account"
    time_window = pd.Timedelta(window)

    # Sort once upfront
    transactions_sorted = transactions.sort_values([acct_col, "timestamp"]).copy()

    # Vectorized: identify new counterparties
    prev_ts = transactions_sorted.groupby([acct_col, partner_col], sort=False)["timestamp"].shift()
    is_new = (prev_ts.isna() | ((transactions_sorted["timestamp"] - prev_ts) > time_window)).astype(int)
    transactions_sorted["is_new"] = is_new

    # Vectorized rolling sum on sorted data
    rolled = (
        transactions_sorted.groupby(acct_col, sort=False)
        .rolling(time_window, on="timestamp", closed="left")["is_new"]
        .sum()
        .fillna(0)
        .reset_index(level=0, drop=True)  # Drop the account level from multi-index
    )

    # Map back to original index
    result = pd.Series(0, index=transactions.index)
    result.loc[transactions_sorted.index] = rolled.values + is_new.values

    return result.astype(int)


def _detect_cycles(transactions: pd.DataFrame, window: str = "10D") -> pd.Series:
    """Detect 2-hop cycles within a time window.

    A cycle is defined as A->B followed by B->A within the specified time window.
    """
    time_window = pd.Timedelta(window)

    # Create bidirectional edge lookup
    edges = transactions[["from_account", "to_account", "timestamp"]].copy()
    edges["pair_key"] = edges["from_account"] + "|" + edges["to_account"]
    edges["reverse_key"] = edges["to_account"] + "|" + edges["from_account"]

    # Self-join on reverse pairs within time window
    merged = pd.merge(
        edges,
        edges[["pair_key", "timestamp"]].rename(columns={"timestamp": "reverse_ts"}),
        left_on="reverse_key",
        right_on="pair_key",
        how="left",
        suffixes=("", "_reverse"),
    )

    # Vectorized time window check
    has_cycle = (
        merged["reverse_ts"].notna()
        & (merged["reverse_ts"] < merged["timestamp"])
        & (merged["reverse_ts"] > merged["timestamp"] - time_window)
    )

    # Group by original index and check if any reverse exists
    result = has_cycle.groupby(merged.index).any()
    return result.reindex(transactions.index, fill_value=False)


def _detect_stack(transactions: pd.DataFrame, window: str = "3D", similarity_threshold: float = 0.75) -> pd.Series:
    """Detect stack/layering patterns.

    A stack pattern is defined as a transaction from A->B that is closely preceded
    by an inbound transaction to A from another party within a time window.
    In other words, A receives funds and quickly passes them on to B.
    """
    inbound = transactions[["to_account", "timestamp", "amount_received", "receiving_currency"]].copy()
    inbound = inbound.rename(columns={"to_account": "account"})

    transactions_sorted = transactions.sort_values("timestamp").copy()
    transactions_sorted["account"] = transactions_sorted["from_account"]

    merged = pd.merge_asof(
        transactions_sorted,
        inbound.sort_values("timestamp"),
        left_on="timestamp",
        right_on="timestamp",
        by="account",
        suffixes=("", "_in"),
        direction="backward",
        tolerance=pd.Timedelta(window),
    )

    # More forgiving similarity threshold to match ground truth
    amount_diff = (merged["amount_paid"] - merged["amount_received"]).abs()
    denom = merged["amount_received"].clip(lower=1)
    similarity = (1 - amount_diff / denom).clip(0, 1)

    has_prior_inbound = merged["amount_received"].notna()

    result = pd.Series(False, index=transactions.index)
    result.loc[merged.index] = (similarity > similarity_threshold) & has_prior_inbound

    return result


def _score_fan_pattern(fan_degree: pd.Series, prefix: str) -> tuple[pd.Series, pd.Series]:
    """Score fan patterns using tiered thresholds.

    Lower degrees get some points, higher degrees get more points.
    """
    tiers = [
        (10, 45, f"{prefix}: Max degree = 10+"),
        (5, 30, f"{prefix}: Max degree = 5-9"),
        (2, 15, f"{prefix}: Max degree = 2-4"),
    ]

    scores = pd.Series(0, index=fan_degree.index)
    evidence_strs = pd.Series("", index=fan_degree.index)

    for threshold, points, label in tiers:
        mask = (fan_degree >= threshold) & (evidence_strs == "")  # Only assign once
        scores[mask] = points
        evidence_strs[mask] = label

    return scores, evidence_strs


def _compute_risk_score(sorted_transactions: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute risk scores using pattern detection."""
    # Ensure timestamps are real datetimes so rolling windows transactions correctly.
    if not np.issubdtype(sorted_transactions["timestamp"].dtype, np.datetime64):
        sorted_transactions["timestamp"] = pd.to_datetime(sorted_transactions["timestamp"])

    score = pd.Series(0.0, index=sorted_transactions.index)
    evidence = pd.Series([[] for _ in range(len(sorted_transactions))], index=sorted_transactions.index)

    # Compute all pattern detections
    fan_out = _compute_fan_metrics(sorted_transactions, "3D", "out")
    fan_in = _compute_fan_metrics(sorted_transactions, "3D", "in")
    has_cycle = _detect_cycles(sorted_transactions, "10D")
    has_stack = _detect_stack(sorted_transactions, "3D", similarity_threshold=0.8)

    # Fan-out scoring (ground truth shows 2-16 degree, most common 10-16)
    fan_out_score, fan_out_evidence = _score_fan_pattern(fan_out, "FAN-OUT")
    score += fan_out_score
    for idx in fan_out_evidence[fan_out_evidence != ""].index:
        evidence.at[idx] = evidence.at[idx] + [fan_out_evidence.at[idx]]

    # Fan-in scoring (ground truth shows 2-16 degree, most common 10-16)
    fan_in_score, fan_in_evidence = _score_fan_pattern(fan_in, "FAN-IN")
    score += fan_in_score
    for idx in fan_in_evidence[fan_in_evidence != ""].index:
        evidence.at[idx] = evidence.at[idx] + [fan_in_evidence.at[idx]]

    # Cycles (ground truth: 2-10 hops within 10 days)
    score[has_cycle] += 25
    evidence[has_cycle] = evidence[has_cycle].apply(lambda x: x + ["CYCLE: Max hops >=2"])

    # Stack/Layering (ground truth: A→B→C pass-through)
    score[has_stack] += 20
    evidence[has_stack] = evidence[has_stack].apply(lambda x: x + ["STACK/LAYERING"])

    # Gather-scatter (ground truth: hub behavior with both fan-in and fan-out)
    gather_scatter = (fan_in >= 3) & (fan_out >= 3)
    score[gather_scatter] += 15
    evidence[gather_scatter] = evidence[gather_scatter].apply(lambda x: x + ["GATHER-SCATTER"])

    # Pattern combinations (multiple signals = higher confidence)
    high_signal_count = (
        has_cycle.astype(int)
        + has_stack.astype(int)
        + (fan_in >= 5).astype(int)
        + (fan_out >= 5).astype(int)
        + gather_scatter.astype(int)
    )

    mask = high_signal_count == 2
    score[mask] += 20
    evidence[mask] = evidence[mask].apply(lambda x: x + ["MULTIPLE PATTERNS: 2 high-risk signals"])

    mask = high_signal_count >= 3
    score[mask] += 35
    evidence[mask] = evidence[mask].apply(lambda x: x + ["MULTIPLE PATTERNS: 3+ high-risk signals"])

    triggered_rules: pd.Series = evidence.apply(lambda x: ",".join(x) if len(x) > 0 else "")
    return score.clip(0, 100), triggered_rules


def _process_transaction_data(file_path: str) -> pd.DataFrame:
    """Process transaction data and populate the database."""
    transc_df = pd.read_csv(file_path, dtype_backend="pyarrow")
    transc_df.rename(
        columns={
            "Timestamp": "timestamp",
            "From Bank": "from_bank",
            "Account": "from_account",
            "To Bank": "to_bank",
            "Account.1": "to_account",
            "Amount Received": "amount_received",
            "Receiving Currency": "receiving_currency",
            "Amount Paid": "amount_paid",
            "Payment Currency": "payment_currency",
            "Payment Format": "payment_format",
            "Is Laundering": "is_laundering",
        },
        inplace=True,
    )
    transc_df.drop_duplicates(inplace=True)

    # Convert Timestamp column to ISO 8601 format
    transc_df["timestamp"] = pd.to_datetime(transc_df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Add new columns: date, day_of_week, time_of_day
    transc_df["date"] = pd.to_datetime(transc_df["timestamp"]).dt.date
    transc_df["day_of_week"] = pd.to_datetime(transc_df["timestamp"]).dt.day_name()
    transc_df["time_of_day"] = pd.to_datetime(transc_df["timestamp"]).dt.time

    # Create Transaction ID by hashing row content, excluding label
    # Concatenate values of desired columns into a string
    cols = [
        "timestamp",
        "time_of_day",
        "from_bank",
        "from_account",
        "to_bank",
        "to_account",
        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
    ]

    txn_str = transc_df[cols].astype("string").agg("|".join, axis=1)
    transc_df["transaction_id"] = txn_str.map(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])

    transc_df_sorted = transc_df.copy()

    transc_df_sorted["timestamp"] = pd.to_datetime(transc_df_sorted["timestamp"])
    transc_df_sorted = transc_df_sorted.sort_values("timestamp").reset_index(drop=True)

    pattern_scores, pattern_evidence = _compute_risk_score(transc_df_sorted)
    transc_df_sorted.loc[: len(pattern_scores) - 1, "risk_score"] = pattern_scores
    transc_df_sorted.loc[: len(pattern_scores) - 1, "triggered_rules"] = pattern_evidence

    # Set Transaction ID as index
    transc_df_sorted.set_index("transaction_id", drop=True, inplace=True)

    return transc_df_sorted


def _curate_test_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Curate a test set with diverse fraud and legitimate cases."""
    test_cases = []

    fraud = df[df["is_laundering"] == 1]
    legit = df[df["is_laundering"] == 0]

    # All distinct fraud patterns (at least 2 examples each)
    for evidence_pattern in fraud["triggered_rules"].value_counts().index:
        cases = fraud[fraud["triggered_rules"] == evidence_pattern].sample(
            min(2, len(fraud[fraud["triggered_rules"] == evidence_pattern])), random_state=42
        )
        test_cases.append(cases)

    # Use percentiles to define thresholds
    high_risk_threshold = fraud["risk_score"].quantile(0.75)
    low_risk_threshold = legit["risk_score"].quantile(0.25)
    normal_threshold = legit["risk_score"].quantile(0.5)

    # High-score false positives
    false_positives = df[(df["is_laundering"] == 0) & (df["risk_score"] >= high_risk_threshold)]
    test_cases.append(false_positives.sample(min(50, len(false_positives)), random_state=42))

    # Low-score true positives
    false_negatives = df[(df["is_laundering"] == 1) & (df["risk_score"] < low_risk_threshold)]
    if len(false_negatives) > 0:
        test_cases.append(false_negatives)  # Include all

    # 4. Representative sample of normal transactions
    normal = df[(df["is_laundering"] == 0) & (df["risk_score"] < normal_threshold)]
    test_cases.append(normal.sample(min(100, len(normal)), random_state=42))

    # 5. Edge cases: single-transaction fraud attempts
    single_txn_fraud = df[df["is_laundering"] == 1].groupby("from_account").filter(lambda x: len(x) == 1)
    if len(single_txn_fraud) > 0:
        test_cases.append(single_txn_fraud.sample(min(10, len(single_txn_fraud)), random_state=42))

    return pd.concat(test_cases).drop_duplicates().sort_values("timestamp")


def main() -> None:
    """Prepare and populate the fraud investigation database."""
    configs = Configs()
    if configs.aml_db is None:
        raise ValueError("AML database path is not configured.")

    db_path = Path(configs.aml_db.database or "./aml_prod.db").resolve()
    ddl_file_path = Path("implementations/fraud_investigation/data/schema.ddl")

    # Download datasets from Kaggle
    path_to_transc_csv = kagglehub.dataset_download(
        handle="ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
        path="HI-Small_Trans.csv",
    )
    path_to_accts_csv = kagglehub.dataset_download(
        handle="ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
        path="HI-Small_accounts.csv",
    )
    logger.info(f"Downloaded transaction data to {path_to_transc_csv}")
    logger.info(f"Downloaded account data to {path_to_accts_csv}")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")

    with open(ddl_file_path, "r") as file:
        conn.executescript(file.read())
    conn.commit()

    accts_df = pd.read_csv(path_to_accts_csv, dtype_backend="pyarrow")

    # Rename duplicate 'Account' columns to distinguish sender and receiver
    accts_df.rename(
        columns={
            "Bank Name": "bank_name",
            "Bank ID": "bank_id",
            "Account Number": "account_number",
            "Entity ID": "entity_id",
            "Entity Name": "entity_name",
        },
        inplace=True,
    )

    accts_df.to_sql("accounts", conn, if_exists="append", index=False)
    logger.info("✅ Populated accounts table.")

    # Create the transactions table
    logger.info("Processing transaction data...")
    transc_df = _process_transaction_data(path_to_transc_csv)

    # Drop the "is_laundering" column and insert into the database
    transc_df.drop(columns=["is_laundering", "triggered_rules"]).to_sql("transactions", conn, if_exists="append")
    logger.info("✅ Populated transactions table.")
    logger.info(f"✅ Populated database at {db_path}")

    # Split out a test set with the last 3 days of transactions
    test_df = _curate_test_cases(transc_df)
    test_df.to_csv(ddl_file_path.parent / "test_transactions.csv")
    logger.info(f"✅ Created curated test set with {len(test_df)} cases.")

    conn.close()


if __name__ == "__main__":
    main()

"""Case generation and parsing utilities for AML evaluation data.

This module parses laundering patterns, builds case records, and provides
Pydantic models used by downstream evaluation and agent inputs.
"""

import csv
import json
import logging
import random
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from .utils import (
    _canonicalize_numeric,
    _canonicalize_text,
    _canonicalize_timestamp,
    _create_id,
    _parse_timestamp,
    apply_lookback_window,
)


logger = logging.getLogger(__name__)

__all__ = [
    "LaunderingPattern",
    "GroundTruth",
    "CaseFile",
    "CaseRecord",
    "parse_patterns_file",
    "build_cases",
]

_TRANSACTION_ID_COLUMNS = [
    "timestamp",
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
_REQUIRED_TRANSACTION_COLUMNS = {"timestamp", "transaction_id", "is_laundering", "from_account"}


class LaunderingPattern(str, Enum):
    """Enumeration of laundering pattern types."""

    FAN_IN = "FAN-IN"
    FAN_OUT = "FAN-OUT"
    CYCLE = "CYCLE"
    GATHER_SCATTER = "GATHER-SCATTER"
    SCATTER_GATHER = "SCATTER-GATHER"
    STACK = "STACK"
    RANDOM = "RANDOM"
    BIPARTITE = "BIPARTITE"
    NONE = "NONE"


_RANDOM_PATTERN_TYPES = [pattern.value for pattern in LaunderingPattern if pattern not in {LaunderingPattern.NONE}]


class CaseFile(BaseModel):
    """Metadata for a laundering case file."""

    case_id: str
    """Unique identifier for the case."""
    seed_transaction_id: str
    """The transaction ID that seeded the laundering attempt. This is typically the last
    transaction in the pattern."""
    seed_timestamp: str
    """The timestamp of the seed transaction."""
    window_start: str
    """The start timestamp of the case window."""
    suspected_pattern_type: str
    """The type of laundering pattern suspected in the case."""


class GroundTruth(BaseModel):
    """Ground truth information for a laundering case."""

    is_laundering: bool
    """Whether the case involves money laundering."""
    pattern_type: LaunderingPattern
    """The type of laundering pattern in the case."""
    pattern_description: str
    """A short description of the laundering pattern."""
    attempt_transaction_ids: str
    """Comma-separated list of transaction IDs involved in the laundering attempt."""


class AnalystOutput(BaseModel):
    """Analyst notes for a laundering case."""

    summary_narrative: str
    """Analyst's reasoning/evidence summary."""
    is_laundering: bool
    """Whether the case involves money laundering."""
    pattern_type: LaunderingPattern
    """The type of laundering pattern in the case."""
    pattern_description: str
    """A short description of the laundering pattern."""
    flagged_transaction_ids: list[str]
    """A comma-separeted list of transaction IDs flagged by the analyst as suspicious.
    """


class CaseRecord(BaseModel):
    """Combined case file and ground truth record."""

    case: CaseFile
    """Metadata for the laundering case."""
    groundtruth: GroundTruth
    """Ground truth information for the laundering case."""
    analysis: AnalystOutput | None = None
    """Optional analyst output for the laundering case. Typically populated after \
    investigation."""


def parse_patterns_file(path: str | Path, lookback_days: int = 0, min_timestamp: str | None = None) -> list[CaseRecord]:
    """Parse laundering pattern attempts from a patterns file.

    Parameters
    ----------
    path : str | Path
        Path to the patterns file.
    lookback_days : int, default=0
        Number of days to look back from the first transaction timestamp to
        compute the case window start.
    min_timestamp : str | None, default=None
        Minimum timestamp boundary (ISO 8601 or dataset format). If provided,
        the window start will not precede this timestamp.

    Returns
    -------
    list[CaseRecord]
        Parsed laundering cases with seed metadata and ground truth.

    Raises
    ------
    FileNotFoundError
        If the patterns file does not exist.
    ValueError
        If the file contains malformed transaction rows or an unterminated block.
    """
    if lookback_days < 0:
        raise ValueError("lookback_days must be >= 0")

    patterns_path = Path(path)
    if not patterns_path.exists():
        raise FileNotFoundError(f"Patterns file not found: {patterns_path}")
    if min_timestamp:
        _parse_timestamp(min_timestamp)

    cases: list[CaseRecord] = []
    current: dict[str, Any] | None = None
    begin_prefix = "BEGIN LAUNDERING ATTEMPT - "
    end_prefix = "END LAUNDERING ATTEMPT"

    with patterns_path.open(newline="") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(begin_prefix):
                header = line[len(begin_prefix) :]
                pattern_type, pattern_description = _parse_pattern_header(header)
                current = {
                    "pattern_type": pattern_type,
                    "pattern_description": pattern_description,
                    "transactions": [],
                }
                continue

            if line.startswith(end_prefix):
                if current and current["transactions"]:
                    txns: list[dict[str, str]] = current["transactions"]
                    txns_sorted = sorted(txns, key=lambda t: _parse_timestamp(t["timestamp"]))
                    seed_txn = txns_sorted[-1]
                    window_start = apply_lookback_window(
                        txns_sorted[0]["timestamp"], lookback_days, min_timestamp=min_timestamp
                    )
                    attempt_ids = _serialize_attempt_ids([txn["transaction_id"] for txn in txns_sorted])
                    case_file = CaseFile(
                        case_id=_create_id(json.dumps(txns_sorted)),
                        seed_transaction_id=seed_txn["transaction_id"],
                        seed_timestamp=seed_txn["timestamp"],
                        window_start=window_start,
                        suspected_pattern_type=current["pattern_type"],
                    )
                    groundtruth = GroundTruth(
                        is_laundering=True,
                        pattern_type=current["pattern_type"],
                        pattern_description=current["pattern_description"],
                        attempt_transaction_ids=attempt_ids,
                    )
                    cases.append(CaseRecord(case=case_file, groundtruth=groundtruth))
                current = None
                continue

            if current:
                row = next(csv.reader([line]))
                if len(row) < 11:
                    raise ValueError(f"Malformed transaction row at line {line_number}: {line}")

                txn = {
                    "timestamp": _canonicalize_timestamp(row[0]),
                    "from_bank": _canonicalize_numeric(row[1]),
                    "from_account": _canonicalize_text(row[2]),
                    "to_bank": _canonicalize_numeric(row[3]),
                    "to_account": _canonicalize_text(row[4]),
                    "amount_received": _canonicalize_numeric(row[5]),
                    "receiving_currency": _canonicalize_text(row[6]),
                    "amount_paid": _canonicalize_numeric(row[7]),
                    "payment_currency": _canonicalize_text(row[8]),
                    "payment_format": _canonicalize_text(row[9]),
                }
                txn_str = "|".join(txn[col] for col in _TRANSACTION_ID_COLUMNS)
                txn["transaction_id"] = _create_id(txn_str)
                current["transactions"].append(txn)

    if current is not None:
        raise ValueError("Unterminated laundering attempt block in patterns file.")

    return cases


def build_cases(
    patterns_filepath: str | Path,
    transactions: pd.DataFrame,
    num_laundering_cases: int,
    num_false_positive_cases: int,
    num_false_negative_cases: int,
    num_normal_cases: int,
    lookback_days: int,
) -> list[CaseRecord]:
    """Build case files from laundering patterns and a transactions dataset.

    Parameters
    ----------
    patterns_filepath : str | Path
        Path to the patterns file.
    transactions : pandas.DataFrame
        Normalized transactions dataframe. Must include the following columns:
        ``timestamp``, ``transaction_id``, ``is_laundering``, and ``from_account``.
    num_laundering_cases : int
        Number of laundering cases to sample from known pattern attempts.
    num_false_positive_cases : int
        Number of false positive cases to generate from benign activity.
    num_false_negative_cases : int
        Number of false negative cases to generate from laundering activity.
    num_normal_cases : int
        Number of normal (benign) cases to sample.
    lookback_days : int
        Number of days to look back for case window start.

    Returns
    -------
    list[CaseRecord]
        Combined list of case records.

    Raises
    ------
    ValueError
        If required columns are missing or numeric arguments are negative.
    TypeError
        If ``transactions`` is not a pandas DataFrame.
    FileNotFoundError
        If ``patterns_filepath`` does not exist.
    """
    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("transactions must be a pandas DataFrame")
    if lookback_days < 0:
        raise ValueError("lookback_days must be >= 0")
    for name, value in [
        ("num_laundering_cases", num_laundering_cases),
        ("num_false_positive_cases", num_false_positive_cases),
        ("num_false_negative_cases", num_false_negative_cases),
        ("num_normal_cases", num_normal_cases),
    ]:
        if value < 0:
            raise ValueError(f"{name} must be >= 0")

    missing_columns = _REQUIRED_TRANSACTION_COLUMNS - set(transactions.columns)
    if missing_columns:
        raise ValueError(f"transactions is missing required columns: {sorted(missing_columns)}")

    min_timestamp = transactions["timestamp"].min() if not transactions.empty else None
    rng = random.Random(42)

    all_attempts = parse_patterns_file(patterns_filepath, lookback_days=lookback_days, min_timestamp=min_timestamp)
    rng.shuffle(all_attempts)
    laundering_cases = all_attempts[:num_laundering_cases]
    remaining_attempts = all_attempts[num_laundering_cases:]

    laundering_attempt_txn_ids: set[str] = set()
    for case in laundering_cases:
        laundering_attempt_txn_ids.update(_deserialize_attempt_ids(case.groundtruth.attempt_transaction_ids))

    false_negative_cases = _build_false_negative_cases(
        remaining_attempts, num_false_negative_cases, laundering_attempt_txn_ids
    )

    false_positive_cases = _build_false_positive_cases(transactions, num_false_positive_cases)

    fp_seed_ids = {case.case.seed_transaction_id for case in false_positive_cases}
    normal_cases = _build_normal_cases(transactions, num_normal_cases, fp_seed_ids, lookback_days, min_timestamp)

    return laundering_cases + false_negative_cases + false_positive_cases + normal_cases


def _build_false_negative_cases(
    remaining_attempts: list[CaseRecord], num_false_negative_cases: int, laundering_attempt_txn_ids: set[str]
) -> list[CaseRecord]:
    """Create false negative cases from remaining attempts or sampled transactions."""
    false_negative_cases: list[CaseRecord] = []
    fn_attempts = remaining_attempts[:num_false_negative_cases]
    for case in fn_attempts:
        if case.case.seed_transaction_id in laundering_attempt_txn_ids:
            continue
        case_file = CaseFile(
            case_id=_create_id(f"fn:{case.case.case_id}"),
            seed_transaction_id=case.case.seed_transaction_id,
            seed_timestamp=case.case.seed_timestamp,
            window_start=case.case.window_start,
            suspected_pattern_type="UNKNOWN",
        )
        groundtruth = GroundTruth(
            is_laundering=case.groundtruth.is_laundering,
            pattern_type=case.groundtruth.pattern_type,
            pattern_description=case.groundtruth.pattern_description,
            attempt_transaction_ids=case.groundtruth.attempt_transaction_ids,
        )
        false_negative_cases.append(CaseRecord(case=case_file, groundtruth=groundtruth))

    if len(false_negative_cases) < num_false_negative_cases:
        logger.warning(
            "Only %d false negative cases could be built; requested %d.",
            len(false_negative_cases),
            num_false_negative_cases,
        )

    return false_negative_cases


def _build_false_positive_cases(transc_df: pd.DataFrame, num_false_positive_cases: int) -> list[CaseRecord]:
    """Create false positive cases from benign transaction bursts."""
    false_positive_cases: list[CaseRecord] = []
    if num_false_positive_cases <= 0:
        return false_positive_cases

    benign_df = transc_df[transc_df["is_laundering"] == 0].copy()
    if benign_df.empty:
        return false_positive_cases

    benign_df["date"] = pd.to_datetime(benign_df["timestamp"]).dt.date
    window_stats = (
        benign_df.groupby(["from_account", "date"], as_index=False)
        .size()
        .rename(columns={"size": "txn_count"})
        .sort_values(["txn_count", "date"], ascending=[False, True])
    )
    top_windows = window_stats.head(num_false_positive_cases)
    for _, window in top_windows.iterrows():
        window_txns = benign_df[
            (benign_df["from_account"] == window["from_account"]) & (benign_df["date"] == window["date"])
        ].sort_values("timestamp")
        if window_txns.empty:
            continue
        seed_row = window_txns.iloc[-1]
        case_file = CaseFile(
            case_id=_create_id(f"{seed_row['transaction_id']}|false_positive"),
            seed_transaction_id=seed_row["transaction_id"],
            seed_timestamp=seed_row["timestamp"],
            window_start=_date_window_start(window["date"]),
            suspected_pattern_type=random.choice(_RANDOM_PATTERN_TYPES),
        )
        groundtruth = GroundTruth(
            is_laundering=False,
            pattern_type=LaunderingPattern.NONE,
            pattern_description="Normal transaction",
            attempt_transaction_ids="",
        )
        false_positive_cases.append(CaseRecord(case=case_file, groundtruth=groundtruth))

    return false_positive_cases


def _build_normal_cases(
    transc_df: pd.DataFrame,
    num_normal_cases: int,
    fp_seed_ids: set[str],
    lookback_days: int,
    min_timestamp: str | None,
) -> list[CaseRecord]:
    """Create normal (benign) cases from the transaction pool."""
    normal_cases: list[CaseRecord] = []
    if num_normal_cases <= 0:
        return normal_cases

    normal_pool = transc_df[(transc_df["is_laundering"] == 0) & (~transc_df["transaction_id"].isin(fp_seed_ids))]
    if normal_pool.empty:
        return normal_cases

    sample_count = min(num_normal_cases, len(normal_pool))
    normal_txns = normal_pool.sample(n=sample_count, random_state=42)
    for _, row in normal_txns.iterrows():
        window_start = apply_lookback_window(row["timestamp"], lookback_days, min_timestamp=min_timestamp)
        case_file = CaseFile(
            case_id=_create_id(row.drop(labels=["transaction_id", "is_laundering"]).to_json()),
            seed_transaction_id=row["transaction_id"],
            seed_timestamp=row["timestamp"],
            window_start=window_start,
            suspected_pattern_type=LaunderingPattern.NONE.value,
        )
        groundtruth = GroundTruth(
            is_laundering=False,
            pattern_type=LaunderingPattern.NONE,
            pattern_description="Normal transaction",
            attempt_transaction_ids="",
        )
        normal_cases.append(CaseRecord(case=case_file, groundtruth=groundtruth))

    return normal_cases


def _parse_pattern_header(header: str) -> tuple[str, str]:
    """Parse a pattern header line into type and description."""
    cleaned = header.strip()
    if ":" in cleaned:
        pattern_type, description = cleaned.split(":", 1)
        pattern_type = pattern_type.strip()
        description = description.strip() or "N/A"
    else:
        pattern_type = cleaned
        description = "N/A"
    return pattern_type, description


def _date_window_start(date_value: Any) -> str:
    """Return an ISO 8601 window start string for a given date value."""
    if isinstance(date_value, datetime):
        date_value = date_value.date()
    if hasattr(date_value, "isoformat"):
        return f"{date_value.isoformat()}T00:00:00"
    parsed = _parse_timestamp(str(date_value))
    return parsed.strftime("%Y-%m-%dT00:00:00")


def _serialize_attempt_ids(ids: list[str]) -> str:
    """Serialize attempt transaction IDs as a comma-separated string."""
    return ",".join(ids)


def _deserialize_attempt_ids(value: str) -> list[str]:
    """Parse a comma-separated attempt transaction ID string."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]

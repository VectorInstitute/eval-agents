"""Shared utilities for AML dataset preparation and normalization."""

import hashlib
import math
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, get_args

import kagglehub
import pandas as pd


__all__ = ["download_dataset_file", "normalize_transactions_data", "apply_lookback_window"]

_TRANSACTION_ID_TEXT_COLUMNS = [
    "from_account",
    "to_account",
    "receiving_currency",
    "payment_currency",
    "payment_format",
]

IllicitRatios = Literal["HI", "LI"]
TransactionsSizes = Literal["Small", "Medium", "Large"]
Filenames = Literal["Trans.csv", "accounts.csv", "Patterns.txt"]


def download_dataset_file(
    illicit_ratio: IllicitRatios = "HI",
    transactions_size: TransactionsSizes = "Small",
    filename: Filenames = "Trans.csv",
) -> str:
    """Download a specific file from the Kaggle AML dataset.

    Parameters
    ----------
    illicit_ratio : {"HI", "LI"}, default="HI"
        The illicit transaction ratio.
    transactions_size : {"Small", "Medium", "Large"}, default="Small"
        The size of the transactions dataset.
    filename : {"Trans.csv", "accounts.csv", "Patterns.txt"}, default="Trans.csv"
        The name of the file to download.

    Returns
    -------
    str
        Local path to the downloaded file (from Kaggle cache).

    Raises
    ------
    ValueError
        If any option is not a supported value.
    """
    if illicit_ratio not in get_args(IllicitRatios):
        raise ValueError(f"illicit_ratio must be one of {sorted(get_args(IllicitRatios))}")
    if transactions_size not in get_args(TransactionsSizes):
        raise ValueError(f"transactions_size must be one of {sorted(get_args(TransactionsSizes))}")
    if filename not in get_args(Filenames):
        raise ValueError(f"filename must be one of {sorted(get_args(Filenames))}")

    dataset_path = f"{illicit_ratio}-{transactions_size}_{filename}"
    return kagglehub.dataset_download(
        handle="ealtman2019/ibm-transactions-for-anti-money-laundering-aml", path=dataset_path
    )


def normalize_transactions_data(transc_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and enrich the raw transactions dataframe.

    This function mutates the input dataframe in place.

    Parameters
    ----------
    transc_df : pandas.DataFrame
        Raw transactions dataframe with the original Kaggle column names.

    Returns
    -------
    pandas.DataFrame
        Normalized dataframe with canonical column names, a generated
        ``transaction_id``, and sorted by timestamp.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required_columns = {
        "Timestamp",
        "From Bank",
        "Account",
        "To Bank",
        "Account.1",
        "Amount Received",
        "Receiving Currency",
        "Amount Paid",
        "Payment Currency",
        "Payment Format",
        "Is Laundering",
    }
    missing = required_columns - set(transc_df.columns)
    if missing:
        raise ValueError(f"transc_df is missing required columns: {sorted(missing)}")

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
    # Canonicalize fields used for transaction_id hashing so all sources match
    transc_df["timestamp"] = pd.to_datetime(transc_df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    for col in _TRANSACTION_ID_TEXT_COLUMNS:
        transc_df[col] = transc_df[col].map(_canonicalize_text)

    txn_str = (
        transc_df["timestamp"]
        + "|"
        + transc_df["from_bank"].map(_canonicalize_numeric)
        + "|"
        + transc_df["from_account"]
        + "|"
        + transc_df["to_bank"].map(_canonicalize_numeric)
        + "|"
        + transc_df["to_account"]
        + "|"
        + transc_df["amount_received"].map(_canonicalize_numeric)
        + "|"
        + transc_df["receiving_currency"]
        + "|"
        + transc_df["amount_paid"].map(_canonicalize_numeric)
        + "|"
        + transc_df["payment_currency"]
        + "|"
        + transc_df["payment_format"]
    )
    transc_df["transaction_id"] = txn_str.map(_create_id)
    transc_df.drop_duplicates(subset=["transaction_id"], inplace=True)

    # Sort by timestamp
    return transc_df.sort_values("timestamp").reset_index(drop=True)


def apply_lookback_window(base_timestamp: str, lookback_days: int, min_timestamp: str | None = None) -> str:
    """Compute the case window start timestamp.

    Parameters
    ----------
    base_timestamp : str
        Seed transaction timestamp (ISO 8601 or dataset format).
    lookback_days : int
        Number of days to look back from the base timestamp.
    min_timestamp : str | None, default=None
        Minimum timestamp boundary (ISO 8601 or dataset format).

    Returns
    -------
    str
        Window start timestamp in ISO 8601 format.

    Raises
    ------
    ValueError
        If lookback_days is negative or timestamps are invalid.
    """
    if lookback_days < 0:
        raise ValueError("lookback_days must be >= 0")
    if not base_timestamp:
        raise ValueError("base_timestamp must be a non-empty string")

    base_dt = _parse_timestamp(base_timestamp)
    window_start = base_dt - timedelta(days=lookback_days) if lookback_days > 0 else base_dt

    if min_timestamp:
        min_dt = _parse_timestamp(min_timestamp)
        window_start = max(window_start, min_dt)

    return window_start.strftime("%Y-%m-%dT%H:%M:%S")


def _create_id(serialized_data: str) -> str:
    """Create a unique ID by hashing the serialized data."""
    return hashlib.sha256(serialized_data.encode()).hexdigest()[:16]


def _canonicalize_text(value: Any) -> str:
    """Normalize text fields for stable hashing."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _canonicalize_numeric(value: Any) -> str:
    """Normalize numeric fields into a consistent string form."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return ""

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return ""
        try:
            dec = Decimal(cleaned)
        except InvalidOperation:
            return cleaned
    else:
        try:
            dec = Decimal(str(value))
        except InvalidOperation:
            return str(value)

    if dec.is_zero():
        return "0"
    normalized = dec.normalize()
    formatted = format(normalized, "f")
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _canonicalize_timestamp(value: Any) -> str:
    """Normalize timestamps into ISO 8601 strings."""
    if pd.isna(value):
        return ""
    parsed = value if isinstance(value, datetime) else _parse_timestamp(str(value).strip())
    return parsed.strftime("%Y-%m-%dT%H:%M:%S")


def _parse_timestamp(value: str) -> datetime:
    """Parse a timestamp from known dataset formats."""
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(value)

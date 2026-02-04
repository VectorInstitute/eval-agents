"""CLI for building the AML database and case files.

This module provides Click commands to:
- download Kaggle AML datasets,
- create and populate the SQLite database, and
- generate case files for downstream evaluation.

Examples
--------
Create the database:
    python implementations/aml_investigation/data/cli.py create-db \
        --illicit-ratio HI --transactions-size Small

Generate case files:
    python implementations/aml_investigation/data/cli.py create-cases \
        --illicit-ratio HI --transactions-size Small --output-dir ./data
"""

import functools
import logging
import sqlite3
from pathlib import Path
from typing import Any, Callable, get_args

import click
import pandas as pd
from aieng.agent_evals.aml_investigation.data import (
    CaseRecord,
    IllicitRatios,
    TransactionsSizes,
    build_cases,
    download_dataset_file,
    normalize_transactions_data,
)


logger = logging.getLogger(__name__)


def _create_accounts_table(csv_filepath: str, conn: sqlite3.Connection) -> None:
    """Create and populate the accounts table from a CSV file."""
    accts_df = pd.read_csv(csv_filepath, dtype_backend="pyarrow")

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


def _create_transactions_table(csv_filepath: str, conn: sqlite3.Connection) -> None:
    """Create and populate the transactions table from a CSV file."""
    transc_df = pd.read_csv(csv_filepath, dtype_backend="pyarrow")
    transc_df = normalize_transactions_data(transc_df)

    # Add new columns: date, day_of_week, time_of_day
    transc_df["date"] = pd.to_datetime(transc_df["timestamp"]).dt.date
    transc_df["day_of_week"] = pd.to_datetime(transc_df["timestamp"]).dt.day_name()
    transc_df["time_of_day"] = pd.to_datetime(transc_df["timestamp"]).dt.time

    # Set Transaction ID as index
    transc_df.set_index("transaction_id", drop=True, inplace=True)

    transc_df.drop(columns=["is_laundering"]).to_sql("transactions", conn, if_exists="append")


def _write_jsonl(path: Path, records: list[CaseRecord]) -> None:
    """Write CaseRecord entries to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(record.model_dump_json() + "\n")


def _dataset_options(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Attach shared dataset options to a Click command."""

    @click.option(
        "--illicit-ratio",
        type=click.Choice(get_args(IllicitRatios), case_sensitive=False),
        default="HI",
        show_default=True,
        help="Illicit transaction ratio.",
    )
    @click.option(
        "--transactions-size",
        type=click.Choice(get_args(TransactionsSizes), case_sensitive=False),
        default="Small",
        show_default=True,
        help="Size of the transactions dataset.",
    )
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def _validate_illicit_ratio(value: str) -> IllicitRatios:
    """Validate and narrow illicit_ratio to its Literal type.

    Parameters
    ----------
    value : str
        The illicit ratio value to validate.

    Returns
    -------
    IllicitRatios
        The validated and type-narrowed illicit ratio.

    Raises
    ------
    ValueError
        If value is not a valid illicit ratio.
    """
    if value not in get_args(IllicitRatios):
        raise ValueError(f"illicit_ratio must be one of {sorted(get_args(IllicitRatios))}")
    return value  # type: ignore[return-value]


def _validate_transactions_size(value: str) -> TransactionsSizes:
    """Validate and narrow transactions_size to its Literal type.

    Parameters
    ----------
    value : str
        The transactions size value to validate.

    Returns
    -------
    TransactionsSizes
        The validated and type-narrowed transactions size.

    Raises
    ------
    ValueError
        If value is not a valid transactions size.
    """
    if value not in get_args(TransactionsSizes):
        raise ValueError(f"transactions_size must be one of {sorted(get_args(TransactionsSizes))}")
    return value  # type: ignore[return-value]


@click.group()
def cli() -> None:
    """Entry point for CLI commands."""
    pass


@cli.command()
@_dataset_options
@click.option(
    "--ddl-file-path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    default="implementations/aml_investigation/data/schema.ddl",
    show_default=True,
    help="Path to the SQL DDL file for creating database schema.",
)
@click.option(
    "--db-path",
    type=click.Path(dir_okay=False, writable=True, resolve_path=True, path_type=Path),
    envvar="AML_DB__DATABASE",
    default=None,
    show_default=True,
    help="Optional path to the SQLite database file to create. If not provided, "
    "the AML_DB__DATABASE environment variable will be checked. If neither is provided, "
    "the database will be created in the same directory as the DDL file.",
)
def create_db(illicit_ratio: str, transactions_size: str, ddl_file_path: Path, db_path: Path | None = None) -> None:
    """Create and populate the AML Fraud Investigation SQLite database.

    Parameters
    ----------
    illicit_ratio : {"HI", "LI"}
        Illicit transaction ratio to select the dataset variant.
    transactions_size : {"Small", "Medium", "Large"}
        Size of the transactions dataset.
    ddl_file_path : pathlib.Path
        Path to the SQL DDL file for creating database schema.
    db_path : pathlib.Path | None, default=None
        Optional path to the SQLite database file to create. If ``None``, the
        database will be created in the same directory as the DDL file.

    Raises
    ------
    ValueError
        If any option value is invalid.
    FileNotFoundError
        If the DDL file does not exist.
    """
    # Validate and narrow types
    ratio = _validate_illicit_ratio(illicit_ratio)
    size = _validate_transactions_size(transactions_size)

    if not ddl_file_path.exists():
        raise FileNotFoundError(f"DDL file not found: {ddl_file_path}")

    if db_path is None:
        db_path = ddl_file_path.parent / "aml_transactions.db"

    # Download datasets from Kaggle
    click.echo("Downloading dataset files...")
    path_to_transc_csv = download_dataset_file(ratio, size, "Trans.csv")
    path_to_accts_csv = download_dataset_file(ratio, size, "accounts.csv")
    click.echo("✅ Download complete.")

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        with open(ddl_file_path, "r") as file:
            conn.executescript(file.read())
        conn.commit()
        click.echo("✅ Schema applied.")

        _create_accounts_table(path_to_accts_csv, conn)
        click.echo("✅ Accounts loaded.")
        _create_transactions_table(path_to_transc_csv, conn)
        click.echo("✅ Transactions loaded.")


@cli.command()
@_dataset_options
@click.option(
    "--num-laundering-cases", type=int, default=10, show_default=True, help="Number of laundering cases to create."
)
@click.option("--num-normal-cases", type=int, default=10, show_default=True, help="Number of normal cases to create.")
@click.option(
    "--num-false-negative-cases",
    type=int,
    default=10,
    show_default=True,
    help="Number of false negative cases to create.",
)
@click.option(
    "--num-false-positive-cases",
    type=int,
    default=10,
    show_default=True,
    help="Number of false positive cases to create.",
)
@click.option(
    "--lookback-days",
    type=int,
    default=30,
    show_default=True,
    help="Number of days to look back for the case window start (clamped to available data).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True, resolve_path=True, path_type=Path),
    default=Path("implementations/aml_investigation/data"),
    show_default=True,
    help="Directory to write case JSONL files.",
)
def create_cases(
    illicit_ratio: str,
    transactions_size: str,
    num_laundering_cases: int,
    num_normal_cases: int,
    num_false_negative_cases: int,
    num_false_positive_cases: int,
    lookback_days: int,
    output_dir: Path,
) -> None:
    """Create AML case files as JSONL.

    Parameters
    ----------
    illicit_ratio : {"HI", "LI"}
        Illicit transaction ratio to select the dataset variant.
    transactions_size : {"Small", "Medium", "Large"}
        Size of the transactions dataset.
    num_laundering_cases : int
        Number of laundering cases to sample.
    num_normal_cases : int
        Number of normal cases to sample.
    num_false_negative_cases : int
        Number of false negative cases to generate.
    num_false_positive_cases : int
        Number of false positive cases to generate.
    lookback_days : int
        Number of days to look back for the case window start.
    output_dir : pathlib.Path
        Directory to write case JSONL files.

    Raises
    ------
    ValueError
        If any numeric argument is negative or option values are invalid.
    """
    # Validate and narrow types
    ratio = _validate_illicit_ratio(illicit_ratio)
    size = _validate_transactions_size(transactions_size)

    for name, value in [
        ("num_laundering_cases", num_laundering_cases),
        ("num_normal_cases", num_normal_cases),
        ("num_false_negative_cases", num_false_negative_cases),
        ("num_false_positive_cases", num_false_positive_cases),
        ("lookback_days", lookback_days),
    ]:
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
    if lookback_days == 0:
        logger.warning("lookback_days=0 creates very narrow windows (can be seed timestamp only); consider >= 1.")

    path_to_transc_csv = download_dataset_file(ratio, size, "Trans.csv")
    path_to_patterns_txt = download_dataset_file(ratio, size, "Patterns.txt")
    click.echo("✅ Downloaded dataset files.")

    transc_df = pd.read_csv(path_to_transc_csv, dtype_backend="pyarrow")
    transc_df = normalize_transactions_data(transc_df)
    click.echo("✅ Transactions normalized.")

    cases = build_cases(
        path_to_patterns_txt,
        transc_df,
        num_laundering_cases=num_laundering_cases,
        num_false_negative_cases=num_false_negative_cases,
        num_false_positive_cases=num_false_positive_cases,
        num_normal_cases=num_normal_cases,
        lookback_days=lookback_days,
    )
    click.echo(f"✅ Built {len(cases)} cases.")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "aml_cases.jsonl", cases)
    click.echo(f"✅ Wrote JSONL to {output_dir / 'aml_cases.jsonl'}")


if __name__ == "__main__":
    cli()

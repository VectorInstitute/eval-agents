"""Import the Online Retail dataset to a SQLite database."""

import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()

# Will use this as default if no path is provided in the
# REPORT_GENERATION_DB_PATH env var
DEFAULT_DB_PATH = Path("aieng-eval-agents/aieng/agent_evals/impl/report_generation/data/OnlineRetail.db")
DDL_FILE_PATH = Path("aieng-eval-agents/aieng/agent_evals/impl/report_generation/data/OnlineRetail.ddl")


@click.command()
@click.option("--dataset-path", required=True, help="OnlieRetail dataset CSV path.")
def main(dataset_path: str):
    """Import the Online Retail dataset to the database.

    Parameters
    ----------
    dataset_path : str
        The path to the CSV file containing the dataset.
    """
    db_path = os.getenv("REPORT_GENERATION_DB_PATH", DEFAULT_DB_PATH)

    assert Path(dataset_path).exists(), f"Dataset path {dataset_path} does not exist"
    assert Path(db_path).parent.exists(), f"Database path {db_path} does not exist"

    conn = sqlite3.connect(db_path)
    logger.info("Creating tables according to the OnlineRetail.ddl file")

    with open(DDL_FILE_PATH, "r") as file:
        conn.executescript(file.read())
    conn.commit()

    logger.info(f"Importing dataset from {dataset_path} to database at {db_path}")

    df = pd.read_csv(dataset_path)
    df["InvoiceDate"] = df["InvoiceDate"].apply(convert_date)
    df.to_sql("sales", conn, if_exists="append", index=False)

    conn.close()
    logger.info(f"Dataset imported successfully to database at {db_path}")


def convert_date(date_str: str) -> str | None:
    """Convert date from 'MM/DD/YY HH:MM' to 'YYYY-MM-DD HH:MM'.

    Parameters
    ----------
    date_str : str
        Date string in format 'MM/DD/YY HH:MM' or 'MM/DD/YY H:MM'.
        Example: "12/19/10 16:26" -> "2010-12-19 16:26".

    Returns
    -------
    str | None
        Converted date string in format 'YYYY-MM-DD HH:MM' or None if parsing fails.
    """
    if not date_str or date_str.strip() == "":
        return None

    try:
        # Parse the date - format is DD/MM/YY (day/month/year)
        # Format: "12/1/10 8:26" or "12/1/10 16:26"
        # Split date and time parts
        parts = date_str.strip().split(" ")
        if len(parts) != 2:
            logger.warning(f"Invalid date format (expected 'DD/MM/YY HH:MM'): {date_str}")
            return None

        date_part, time_part = parts

        # Normalize time part to have 2-digit hour
        time_parts = time_part.split(":")
        if len(time_parts) != 2:
            logger.warning(f"Invalid time format: {time_part}")
            return None

        hour, minute = time_parts
        if len(hour) == 1:
            hour = f"0{hour}"
        time_part = f"{hour}:{minute}"

        # Parse as DD/MM/YY (day/month/year)
        dt = datetime.strptime(f"{date_part} {time_part}", "%m/%d/%y %H:%M")
        # Convert to YYYY-MM-DD HH:MM format
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError as e:
        logger.warning(f"Could not parse date: {date_str} - {e}")
        return None


if __name__ == "__main__":
    main()

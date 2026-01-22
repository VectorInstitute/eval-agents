"""Convert InvoiceDate format in OnlineRetail.db.

Convert from 'MM/DD/YY HH:MM' to 'YYYY-MM-DD HH:MM' for better searching abilities.
"""

import logging
import sqlite3
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "aieng-eval-agents/aieng/agent_evals/report_generation/data/OnlineRetail.db"


def convert_date(date_str: str) -> str | None:
    """Convert date from 'MM/DD/YY HH:MM' to 'YYYY-MM-DD HH:MM'.

    Args:
        date_str: Date string in format 'MM/DD/YY HH:MM' or 'MM/DD/YY H:MM'
                  Example: "12/19/10 16:26" -> "2010-12-19 16:26"

    Returns
    -------
        Converted date string in format 'YYYY-MM-DD HH:MM' or None if parsing fails
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


def main():
    """Convert all InvoiceDate values in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all rows with InvoiceDate
    cursor.execute("SELECT rowid, InvoiceDate FROM sales WHERE InvoiceDate IS NOT NULL AND InvoiceDate != ''")
    rows = cursor.fetchall()

    logger.info(f"Found {len(rows)} rows with InvoiceDate to convert")

    updated_count = 0
    error_count = 0

    for rowid, old_date in rows:
        new_date = convert_date(old_date)
        if new_date:
            try:
                cursor.execute("UPDATE sales SET InvoiceDate = ? WHERE rowid = ?", (new_date, rowid))
                updated_count += 1
                if updated_count % 100 == 0:
                    logger.info(f"Updated {updated_count} rows...")
            except Exception as e:
                logger.error(f"Error updating rowid {rowid}: {e}")
                error_count += 1
        else:
            logger.warning(f"Could not convert date for rowid {rowid}: {old_date}")
            error_count += 1

    conn.commit()
    conn.close()

    logger.info("Conversion complete!")
    logger.info(f"  Updated: {updated_count} rows")
    logger.info(f"  Errors: {error_count} rows")


if __name__ == "__main__":
    main()

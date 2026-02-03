"""Environment variables and their defaults for the report generation agent."""

import os
from pathlib import Path


DEFAULT_SQLITE_DB_PATH = "implementations/report_generation/data/OnlineRetail.db"
DEFAULT_REPORTS_OUTPUT_PATH = "implementations/report_generation/reports/"
DEFAULT_LANGFUSE_PROJECT_NAME = "Report Generation"


def get_reports_output_path() -> Path:
    """Get the reports output path.

    If no path is provided in the REPORTS_OUTPUT_PATH env var, will use the
    default path in DEFAULT_REPORTS_OUTPUT_PATH.

    Returns
    -------
    Path
        The reports output path.
    """
    return Path(os.getenv("REPORT_GENERATION_OUTPUT_PATH", DEFAULT_REPORTS_OUTPUT_PATH))


def get_sqlite_db_path() -> Path:
    """Get the SQLite database path for report generation.

    If no path is provided in the REPORT_GENERATION_DB_PATH env var, will use the
    default path in DEFAULT_SQLITE_DB_PATH.

    Returns
    -------
    Path
        The default SQLite database path for report generation.
    """
    return Path(os.getenv("REPORT_GENERATION_DB_PATH", DEFAULT_SQLITE_DB_PATH))


def get_langfuse_project_name() -> str:
    """Get the Langfuse project name for report generation.

    If no project name is provided in the REPORT_GENERATION_LANGFUSE_PROJECT_NAME
    env var, will use the default project name in DEFAULT_LANGFUSE_PROJECT_NAME.

    Returns
    -------
    str
        The default Langfuse project name for report generation.
    """
    return os.getenv("REPORT_GENERATION_LANGFUSE_PROJECT_NAME", DEFAULT_LANGFUSE_PROJECT_NAME)

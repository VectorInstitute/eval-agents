"""Report file writer functions."""

import os
import urllib.parse
from pathlib import Path
from typing import Any

import pandas as pd


# Will use this as default if no path is provided in the REPORTS_OUTPUT_PATH env var
DEFAULT_REPORTS_OUTPUT_PATH = Path("implementations/report_generation/reports/")


def write_report_to_file(
    report_data: list[Any],
    report_columns: list[str],
    filename: str = "report.xlsx",
    gradio_link: bool = True,
) -> str:
    """Write a report to a XLSX file.

    Parameters
    ----------
    report_data : list[Any]
        The data of the report.
    report_columns : list[str]
        The columns of the report.
    filename : str, optional
        The name of the file to create. Default is "report.xlsx".
    gradio_link : bool, optional
        Whether to return a file link that works with Gradio UI.
        Default is True.

    Returns
    -------
    str
        The path to the report file. If `gradio_link` is True, will return
        a URL link that allows Gradio UI to download the file.
    """
    # Create reports directory if it doesn't exist
    reports_output_path = get_reports_output_path()
    reports_output_path.mkdir(exist_ok=True)
    filepath = reports_output_path / filename

    report_df = pd.DataFrame(report_data, columns=report_columns)
    report_df.to_excel(filepath, index=False)

    file_uri = str(filepath)
    if gradio_link:
        file_uri = f"gradio_api/file={urllib.parse.quote(str(file_uri), safe='')}"

    return file_uri


def get_reports_output_path() -> Path:
    """Get the reports output path.

    If no path is provided in the REPORTS_OUTPUT_PATH env var, will use the
    default path in DEFAULT_REPORTS_OUTPUT_PATH.

    Returns
    -------
    Path
        The reports output path.
    """
    return Path(os.getenv("REPORTS_OUTPUT_PATH", DEFAULT_REPORTS_OUTPUT_PATH))

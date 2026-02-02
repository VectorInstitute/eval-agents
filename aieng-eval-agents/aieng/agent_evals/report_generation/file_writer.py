"""
Report file writer class.

Example
-------
>>> from aieng.agent_evals.report_generation.file_writer import ReportFileWriter
>>> report_file_writer = ReportFileWriter(reports_output_path=Path("reports/"))
>>> report_file_writer.write(
...     report_data=[["2026-01-01", 100], ["2026-01-02", 200]],
...     report_columns=["Date", "Sales"],
... )
"""

import urllib.parse
from pathlib import Path
from typing import Any

import pandas as pd


class ReportFileWriter:
    """Write reports to an XLSX file."""

    def __init__(self, reports_output_path: Path):
        """Initialize the report writer.

        Parameters
        ----------
        reports_output_path : Path
            The path to the reports output directory.
        """
        self.reports_output_path = reports_output_path

    def write(
        self,
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
        self.reports_output_path.mkdir(exist_ok=True)
        filepath = self.reports_output_path / filename

        report_df = pd.DataFrame(report_data, columns=report_columns)
        report_df.to_excel(filepath, index=False)

        file_uri = str(filepath)
        if gradio_link:
            file_uri = f"gradio_api/file={urllib.parse.quote(str(file_uri), safe='')}"

        return file_uri

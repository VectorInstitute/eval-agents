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

import logging
import urllib.parse
from pathlib import Path
from typing import Any

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class ReportFileWriter:
    """Write reports to a file."""

    def __init__(self, reports_output_path: Path):
        """Initialize the report writer.

        Parameters
        ----------
        reports_output_path : Path
            The path to the reports output directory.
        """
        self.reports_output_path = reports_output_path

    def write_xlsx(
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
            Returns a string with an error message if the report fails to write.
        """
        try:
            # Create reports directory if it doesn't exist
            self.reports_output_path.mkdir(exist_ok=True)
            filepath = self.reports_output_path / filename

            report_df = pd.DataFrame(report_data, columns=report_columns)
            report_df.to_excel(filepath, index=False)

            file_uri = str(filepath)
            if gradio_link:
                file_uri = f"gradio_api/file={urllib.parse.quote(str(file_uri), safe='')}"

            return file_uri

        except Exception as e:
            logger.exception(f"Error writing report: {e}")
            return str(e)

"""Tests for the report file writer."""

import tempfile
from pathlib import Path

import pandas as pd
from aieng.agent_evals.report_generation.file_writer import ReportFileWriter


def test_write_xlsx():
    """Test the write_xlsx method."""
    test_report_data = [["2026-01-01", 100], ["2026-01-02", 200]]
    test_report_columns = ["Date", "Sales"]
    test_filename = "test_report.xlsx"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        report_file_writer = ReportFileWriter(reports_output_path=temp_dir_path)
        file_path = report_file_writer.write_xlsx(
            report_data=test_report_data,
            report_columns=test_report_columns,
            filename=test_filename,
        )

        expected_file_path = temp_dir_path / test_filename
        expected_file_path_escaped = str(expected_file_path).replace("/", "%2F")
        assert file_path == f"gradio_api/file={expected_file_path_escaped}"

        df = pd.read_excel(expected_file_path)
        assert df.equals(pd.DataFrame(test_report_data, columns=test_report_columns))


def test_write_xlsx_without_gradio_link():
    """Test the write_xlsx method without gradio link."""
    test_report_data = [["2026-01-01", 100], ["2026-01-02", 200]]
    test_report_columns = ["Date", "Sales"]
    test_filename = "test_report.xlsx"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        report_file_writer = ReportFileWriter(reports_output_path=temp_dir_path)
        file_path = report_file_writer.write_xlsx(
            report_data=test_report_data,
            report_columns=test_report_columns,
            filename=test_filename,
            gradio_link=False,
        )

        expected_file_path = temp_dir_path / test_filename
        assert file_path == str(expected_file_path)

        df = pd.read_excel(expected_file_path)
        assert df.equals(pd.DataFrame(test_report_data, columns=test_report_columns))

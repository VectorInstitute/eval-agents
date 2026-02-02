"""Tests for the file tools module.

Tests fetch_file, grep_file, and read_file which handle structured data files
(CSV, XLSX, JSON, text) that need to be downloaded and searched.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from aieng.agent_evals.tools.file import (
    _is_excel_file,
    _read_csv_as_text,
    _read_excel_as_text,
    _read_file_lines,
    create_fetch_file_tool,
    create_grep_file_tool,
    create_read_file_tool,
    fetch_file,
    grep_file,
    read_file,
)


class TestIsExcelFile:
    """Tests for the _is_excel_file helper function."""

    def test_xlsx_extension(self):
        """Test that .xlsx files are detected as Excel."""
        assert _is_excel_file("/path/to/file.xlsx") is True
        assert _is_excel_file("/path/to/file.XLSX") is True

    def test_xls_extension(self):
        """Test that .xls files are detected as Excel."""
        assert _is_excel_file("/path/to/file.xls") is True
        assert _is_excel_file("/path/to/file.XLS") is True

    def test_non_excel_extensions(self):
        """Test that non-Excel files are not detected."""
        assert _is_excel_file("/path/to/file.csv") is False
        assert _is_excel_file("/path/to/file.txt") is False
        assert _is_excel_file("/path/to/file.json") is False


class TestReadExcelAsText:
    """Tests for the _read_excel_as_text helper function."""

    def test_single_sheet_excel(self):
        """Test reading a single-sheet Excel file."""
        # Create a temp Excel file
        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Value": [100, 200]})
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name
        df.to_excel(temp_path, index=False, sheet_name="Data")

        try:
            lines = _read_excel_as_text(temp_path)

            assert len(lines) > 0
            # Should have sheet header
            assert any("Sheet: Data" in line for line in lines)
            # Should have data
            combined = "\n".join(lines)
            assert "Alice" in combined
            assert "Bob" in combined
            assert "100" in combined
        finally:
            os.remove(temp_path)

    def test_multi_sheet_excel(self):
        """Test reading a multi-sheet Excel file."""
        df1 = pd.DataFrame({"A": [1, 2]})
        df2 = pd.DataFrame({"B": [3, 4]})

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name

        with pd.ExcelWriter(temp_path) as writer:
            df1.to_excel(writer, sheet_name="Sheet1", index=False)
            df2.to_excel(writer, sheet_name="Sheet2", index=False)

        try:
            lines = _read_excel_as_text(temp_path)
            combined = "\n".join(lines)

            # Should have both sheet headers
            assert "Sheet: Sheet1" in combined
            assert "Sheet: Sheet2" in combined
        finally:
            os.remove(temp_path)

    def test_excel_with_nan_values(self):
        """Test that NaN values are handled gracefully."""
        df = pd.DataFrame({"A": [1, None, 3], "B": [None, "test", None]})
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name
        df.to_excel(temp_path, index=False)

        try:
            lines = _read_excel_as_text(temp_path)
            # Should not raise an error and should have content
            assert len(lines) > 0
        finally:
            os.remove(temp_path)


class TestReadCsvAsText:
    """Tests for the _read_csv_as_text helper function."""

    def test_basic_csv(self):
        """Test reading a basic CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,value\n")
            f.write("Alice,100\n")
            f.write("Bob,200\n")
            temp_path = f.name

        try:
            lines = _read_csv_as_text(temp_path)

            assert len(lines) == 3
            assert "name" in lines[0]
            assert "Alice" in lines[1]
        finally:
            os.remove(temp_path)

    def test_csv_with_special_characters(self):
        """Test CSV with quoted fields and commas."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,description\n")
            f.write('"Smith, John","A description, with commas"\n')
            temp_path = f.name

        try:
            lines = _read_csv_as_text(temp_path)
            assert len(lines) >= 1
        finally:
            os.remove(temp_path)


class TestReadFileLines:
    """Tests for the _read_file_lines helper function."""

    def test_reads_text_file(self):
        """Test reading a plain text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Line 1\n")
            f.write("Line 2\n")
            temp_path = f.name

        try:
            lines = _read_file_lines(temp_path)
            assert len(lines) == 2
            assert "Line 1" in lines[0]
        finally:
            os.remove(temp_path)

    def test_reads_excel_file(self):
        """Test that Excel files are read via pandas."""
        df = pd.DataFrame({"Col1": ["data1", "data2"]})
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name
        df.to_excel(temp_path, index=False)

        try:
            lines = _read_file_lines(temp_path)
            combined = "\n".join(lines)
            assert "data1" in combined
            assert "data2" in combined
        finally:
            os.remove(temp_path)

    def test_reads_csv_file(self):
        """Test that CSV files are read via pandas."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n")
            f.write("val1,val2\n")
            temp_path = f.name

        try:
            lines = _read_file_lines(temp_path)
            assert len(lines) >= 1
        finally:
            os.remove(temp_path)

    def test_handles_encoding_fallback(self):
        """Test that latin-1 encoding is used as fallback."""
        # Create a file with latin-1 specific character
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
            f.write("Café résumé\n".encode("latin-1"))
            temp_path = f.name

        try:
            lines = _read_file_lines(temp_path)
            assert len(lines) >= 1
        finally:
            os.remove(temp_path)


class TestFetchFile:
    """Tests for the fetch_file function."""

    @patch("aieng.agent_evals.tools.file.httpx.Client")
    def test_fetch_csv_success(self, mock_client_class):
        """Test successful CSV file download."""
        csv_content = "name,value\nfoo,100\nbar,200\n"
        mock_response = MagicMock()
        mock_response.text = csv_content
        mock_response.headers = {"content-type": "text/csv"}
        mock_response.url = "https://example.com/data.csv"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = fetch_file("https://example.com/data.csv")

        assert result["status"] == "success"
        assert "file_path" in result
        assert result["file_path"].endswith(".csv")
        assert os.path.exists(result["file_path"])
        assert "preview" in result
        assert "foo" in result["preview"]

        # Cleanup
        os.remove(result["file_path"])

    @patch("aieng.agent_evals.tools.file.httpx.Client")
    def test_fetch_pdf_returns_error(self, mock_client_class):
        """Test that PDF URLs redirect to web_fetch."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/pdf"}

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = fetch_file("https://example.com/doc.pdf")

        assert result["status"] == "error"
        assert "PDF" in result["error"]

    def test_fetch_invalid_url(self):
        """Test that invalid URLs return error."""
        result = fetch_file("not-a-url")
        assert result["status"] == "error"
        assert "Invalid URL" in result["error"]


class TestGrepFile:
    """Tests for the grep_file function."""

    def test_search_finds_matches(self):
        """Test that search finds matching lines."""
        # Create a temp file with test content - spread out so matches don't overlap
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for i in range(50):
                if i == 10:
                    f.write(f"Line {i}: operating expenses were $100\n")
                elif i == 40:
                    f.write(f"Line {i}: total costs increased by 10%\n")
                else:
                    f.write(f"Line {i}: Regular content\n")
            temp_path = f.name

        try:
            result = grep_file(temp_path, "operating expenses, total costs", context_lines=5)

            assert result["status"] == "success"
            assert result["total_matches"] == 2
            assert len(result["matches"]) == 2
            assert "operating expenses" in result["patterns"]
        finally:
            os.remove(temp_path)

    def test_search_no_matches(self):
        """Test search with no matching terms."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Line 1: Hello world\n")
            f.write("Line 2: Goodbye world\n")
            temp_path = f.name

        try:
            result = grep_file(temp_path, "foobar, nonexistent")

            assert result["status"] == "success"
            assert result["total_matches"] == 0
            assert "No matches found" in result["message"]
        finally:
            os.remove(temp_path)

    def test_search_file_not_found(self):
        """Test search with non-existent file."""
        result = grep_file("/nonexistent/path.txt", "test")

        assert result["status"] == "error"
        assert "File not found" in result["error"]

    def test_search_returns_context(self):
        """Test that search returns context around matches."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for i in range(20):
                if i == 10:
                    f.write(f"Line {i}: operating expenses data here\n")
                else:
                    f.write(f"Line {i}: Regular content\n")
            temp_path = f.name

        try:
            result = grep_file(temp_path, "operating expenses", context_lines=3)

            assert result["status"] == "success"
            assert len(result["matches"]) == 1
            # Context should include surrounding lines
            context = result["matches"][0]["context"]
            assert "operating expenses" in context
        finally:
            os.remove(temp_path)

    def test_search_excel_file(self):
        """Test grep_file works with Excel files."""
        df = pd.DataFrame(
            {
                "Category": ["Revenue", "Expenses", "Profit"],
                "Q1": [1000, 500, 500],
                "Q2": [1200, 600, 600],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name
        df.to_excel(temp_path, index=False)

        try:
            result = grep_file(temp_path, "revenue, profit")

            assert result["status"] == "success"
            assert result["total_matches"] >= 1
            # Check that we found the content
            combined_context = " ".join(m["context"] for m in result["matches"])
            assert "revenue" in combined_context.lower() or "profit" in combined_context.lower()
        finally:
            os.remove(temp_path)

    def test_search_csv_file(self):
        """Test grep_file works with CSV files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Category,Value\n")
            f.write("Revenue,1000\n")
            f.write("Expenses,500\n")
            f.write("Profit,500\n")
            temp_path = f.name

        try:
            result = grep_file(temp_path, "revenue, profit")

            assert result["status"] == "success"
            assert result["total_matches"] >= 1
        finally:
            os.remove(temp_path)


class TestReadFile:
    """Tests for the read_file function."""

    def test_read_section(self):
        """Test reading a section of a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for i in range(100):
                f.write(f"Line {i + 1}: Content\n")
            temp_path = f.name

        try:
            result = read_file(temp_path, start_line=10, num_lines=5)

            assert result["status"] == "success"
            assert result["start_line"] == 10
            assert result["end_line"] == 14  # 5 lines from 10 = lines 10-14 (indices 9-13)
            assert "Line 10" in result["content"]
            assert "Line 14" in result["content"]
        finally:
            os.remove(temp_path)

    def test_read_section_file_not_found(self):
        """Test reading from non-existent file."""
        result = read_file("/nonexistent/path.txt")

        assert result["status"] == "error"
        assert "File not found" in result["error"]

    def test_read_excel_file(self):
        """Test read_file works with Excel files."""
        df = pd.DataFrame(
            {
                "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "Score": [85, 90, 78, 92, 88],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name
        df.to_excel(temp_path, index=False)

        try:
            result = read_file(temp_path, start_line=1, num_lines=10)

            assert result["status"] == "success"
            assert "content" in result
            # Should contain data from the Excel file
            assert "Alice" in result["content"] or "Bob" in result["content"]
        finally:
            os.remove(temp_path)

    def test_read_csv_file(self):
        """Test read_file works with CSV files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Name,Score\n")
            f.write("Alice,85\n")
            f.write("Bob,90\n")
            temp_path = f.name

        try:
            result = read_file(temp_path, start_line=1, num_lines=10)

            assert result["status"] == "success"
            assert "Alice" in result["content"] or "Name" in result["content"]
        finally:
            os.remove(temp_path)


class TestCreateFetchFileTool:
    """Tests for the create_fetch_file_tool function."""

    def test_creates_tool_with_correct_function(self):
        """Test that fetch file tool is created with the correct function."""
        tool = create_fetch_file_tool()
        assert tool is not None
        assert tool.func == fetch_file


class TestCreateGrepFileTool:
    """Tests for the create_grep_file_tool function."""

    def test_creates_tool_with_correct_function(self):
        """Test that grep file tool is created with the correct function."""
        tool = create_grep_file_tool()
        assert tool is not None
        assert tool.func == grep_file


class TestCreateReadFileTool:
    """Tests for the create_read_file_tool function."""

    def test_creates_tool_with_correct_function(self):
        """Test that read file tool is created with the correct function."""
        tool = create_read_file_tool()
        assert tool is not None
        assert tool.func == read_file


@pytest.mark.integration_test
class TestFileToolsIntegration:
    """Integration tests for file tools in the agent workflow.

    The agent's typical workflow for data files is:
    1. google_search -> find data file URLs
    2. fetch_file -> download file, get file_path
    3. grep_file -> find relevant sections by keyword
    4. read_file -> read context around matches

    These tests verify this workflow works end-to-end.
    """

    def test_fetch_file_and_search_workflow(self):
        """Test the fetch_file -> grep_file -> read_file workflow.

        This tests the workflow for CSV/text data files where the agent
        needs to search for specific information.
        """
        # Create a mock CSV-like file to test
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("date,category,value,notes\n")
            for i in range(100):
                if i == 42:
                    f.write(f"2024-01-{i:02d},revenue,1000000,quarterly report\n")
                elif i == 75:
                    f.write(f"2024-01-{i:02d},expenses,500000,operating costs\n")
                else:
                    f.write(f"2024-01-{i:02d},other,{i * 100},regular entry\n")
            temp_path = f.name

        try:
            # Step 1: Search for relevant content
            grep_result = grep_file(temp_path, "revenue, expenses", context_lines=3)
            assert grep_result["status"] == "success"
            assert grep_result["total_matches"] == 2

            # Get line number from first match
            first_match = grep_result["matches"][0]
            line_number = first_match["line_number"]

            # Step 2: Read context around the match
            read_result = read_file(temp_path, start_line=max(1, line_number - 2), num_lines=10)
            assert read_result["status"] == "success"

            # Verify the read content contains the financial data
            content = read_result["content"]
            assert "revenue" in content.lower() or "expenses" in content.lower()

        finally:
            os.remove(temp_path)

    def test_grep_file_or_matching(self):
        """Test that grep_file supports OR matching with comma-separated patterns.

        The agent uses this to search for multiple related terms at once,
        e.g., "revenue, income, profit" to find financial metrics.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Section 1: Introduction\n")
            for i in range(20):
                f.write(f"Content line {i}\n")
            f.write("Section 2: Revenue Analysis\n")
            for i in range(20):
                f.write(f"More content {i}\n")
            f.write("Section 3: Profit Margins\n")
            temp_path = f.name

        try:
            # Search for multiple financial terms
            result = grep_file(temp_path, "revenue, profit, income", context_lines=2)

            assert result["status"] == "success"
            # Should find at least 2 matches (Revenue and Profit sections)
            assert result["total_matches"] >= 2

            # Verify patterns are tracked
            assert "revenue" in result["patterns"]
            assert "profit" in result["patterns"]
            assert "income" in result["patterns"]

        finally:
            os.remove(temp_path)

    def test_read_file_for_specific_section(self):
        """Test that read_file can extract specific sections by line number.

        After grep_file finds relevant lines, the agent uses read_file
        to get enough context to understand the data.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            # Create a document with structured sections
            for i in range(1, 101):
                if i == 50:
                    f.write("IMPORTANT: Key Financial Data\n")
                elif 51 <= i <= 55:
                    f.write(f"  Q{i - 50} Revenue: ${(i - 50) * 1000000}\n")
                else:
                    f.write(f"Line {i}: Regular content\n")
            temp_path = f.name

        try:
            # Simulate agent finding the important section
            grep_result = grep_file(temp_path, "key financial data")
            assert grep_result["status"] == "success"

            line_num = grep_result["matches"][0]["line_number"]

            # Read the section with financial data
            read_result = read_file(temp_path, start_line=line_num, num_lines=10)
            assert read_result["status"] == "success"

            # Verify we got the financial data
            content = read_result["content"]
            assert "Key Financial Data" in content
            assert "Revenue" in content

        finally:
            os.remove(temp_path)

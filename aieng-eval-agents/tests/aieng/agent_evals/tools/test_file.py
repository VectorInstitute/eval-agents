"""Tests for the file tools module.

Tests fetch_file, grep_file, and read_file which handle structured data files
(CSV, XLSX, JSON, text) that need to be downloaded and searched.
"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from aieng.agent_evals.knowledge_qa.agent import KnowledgeGroundedAgent
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

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.tools.file.httpx.AsyncClient")
    async def test_fetch_csv_success(self, mock_client_class):
        """Test successful CSV file download."""
        csv_content = "name,value\nfoo,100\nbar,200\n"
        mock_response = MagicMock()
        mock_response.text = csv_content
        mock_response.headers = {"content-type": "text/csv"}
        mock_response.url = "https://example.com/data.csv"

        async def mock_get(*_args, **_kwargs):
            return mock_response

        mock_client = MagicMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await fetch_file("https://example.com/data.csv")

        assert result["status"] == "success"
        assert "file_path" in result
        assert result["file_path"].endswith(".csv")
        assert os.path.exists(result["file_path"])
        assert "preview" in result
        assert "foo" in result["preview"]

        # Cleanup
        os.remove(result["file_path"])

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.tools.file.httpx.AsyncClient")
    async def test_fetch_pdf_returns_error(self, mock_client_class):
        """Test that PDF URLs redirect to web_fetch."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/pdf"}

        async def mock_get(*_args, **_kwargs):
            return mock_response

        mock_client = MagicMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await fetch_file("https://example.com/doc.pdf")

        assert result["status"] == "error"
        assert "PDF" in result["error"]

    @pytest.mark.asyncio
    async def test_fetch_invalid_url(self):
        """Test that invalid URLs return error."""
        result = await fetch_file("not-a-url")
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
    """Integration tests for file tools using the actual agent.

    Tests the agent's ability to:
    1. Download a real file from the web using fetch_file
    2. Search within the file using grep_file
    3. Read specific sections using read_file

    Uses stable public datasets (CSV and Excel) as test files.
    """

    # Test file configurations
    CSV_TEST_URL = "https://raw.githubusercontent.com/plotly/datasets/master/iris.csv"
    CSV_EXPECTED_COLUMNS = ["sepallength", "sepalwidth", "petallength", "petalwidth", "name"]
    CSV_SEARCH_TERM = "virginica"

    EXCEL_TEST_URL = "https://raw.githubusercontent.com/frictionlessdata/datasets/main/files/excel/sample-1-sheet.xlsx"
    EXCEL_EXPECTED_COLUMNS = ["number", "string", "boolean"]
    EXCEL_SEARCH_TERM = "four"

    def _create_test_agent(self):
        """Create a test agent with consistent settings."""
        return KnowledgeGroundedAgent(
            enable_planning=False,
            enable_caching=False,
            enable_compaction=False,
        )

    def _verify_fetch_call(self, tool_calls: list[dict], expected_url: str) -> str:
        """Verify fetch_file was called correctly and return the file path.

        Parameters
        ----------
        tool_calls : list[dict]
            All tool calls from the agent response.
        expected_url : str
            The expected URL that should have been fetched.

        Returns
        -------
        str
            The file path from the fetch call (for further verification).
        """
        fetch_calls = [tc for tc in tool_calls if tc.get("name") == "fetch_file"]
        assert len(fetch_calls) >= 1, "Agent should have called fetch_file"

        fetch_args = fetch_calls[0].get("args", {})
        assert fetch_args.get("url") == expected_url, (
            f"Agent should fetch exact URL. Expected: {expected_url}, Got: {fetch_args.get('url')}"
        )
        return fetch_args.get("url", "")

    def _verify_tool_used_cached_file(self, tool_calls: list[dict], tool_name: str) -> str:
        """Verify a tool was called on a cached file that exists.

        Parameters
        ----------
        tool_calls : list[dict]
            All tool calls from the agent response.
        tool_name : str
            Name of the tool to verify (e.g., 'grep_file', 'read_file').

        Returns
        -------
        str
            The file path that was used.
        """
        tool_uses = [tc for tc in tool_calls if tc.get("name") == tool_name]
        assert len(tool_uses) >= 1, f"Agent should have used {tool_name}"

        tool_args = tool_uses[0].get("args", {})
        file_path = tool_args.get("file_path", "")
        assert "agent_file_cache" in file_path, f"{tool_name} should use a cached file. Got: {file_path}"
        assert os.path.exists(file_path), f"File should exist at: {file_path}"

        return file_path

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "file_url,file_type,search_term,expected_columns",
        [
            (CSV_TEST_URL, "CSV", CSV_SEARCH_TERM, CSV_EXPECTED_COLUMNS),
            (EXCEL_TEST_URL, "Excel", EXCEL_SEARCH_TERM, EXCEL_EXPECTED_COLUMNS),
        ],
        ids=["CSV", "Excel"],
    )
    async def test_agent_downloads_and_searches_file(
        self, file_url: str, file_type: str, search_term: str, expected_columns: list[str]
    ):
        """Test that agent can download and search a data file (CSV or Excel).

        Verifies the agent can:
        - Successfully download a data file
        - Use grep_file to find specific patterns
        - Report findings from the file
        """
        agent = self._create_test_agent()

        try:
            question = f"Please fetch the {file_type} file at {file_url} and search for the word '{search_term}'. Tell me if you find it."

            response = await agent.answer_async(question)

            # Verify we got a response
            assert response.text, "Agent should return a non-empty response"

            # Verify fetch_file was called correctly
            self._verify_fetch_call(response.tool_calls, file_url)

            # Verify grep was used to search the downloaded file
            self._verify_tool_used_cached_file(response.tool_calls, "grep_file")

            # Verify the response mentions the search term
            assert search_term.lower() in response.text.lower(), f"Response should mention {search_term}"

        finally:
            agent.reset()

    @pytest.mark.asyncio
    async def test_agent_file_workflow_full_integration_csv(self):
        """Test the full CSV workflow: fetch -> grep -> read with complex question.

        This test verifies the agent can handle a complex question requiring:
        1. Download a CSV file from a URL
        2. Search for specific content
        3. Read and analyze data to answer a numerical question
        """
        agent = self._create_test_agent()

        try:
            question = (
                f"Download the Iris dataset from {self.CSV_TEST_URL}. "
                f"Search for rows containing 'setosa' species. "
                f"Then tell me approximately how many setosa samples are in the file."
            )

            response = await agent.answer_async(question)

            # Verify we got a response
            assert response.text, "Agent should return a non-empty response"

            # Verify the agent used the file tools
            tool_names = [tc.get("name", "") for tc in response.tool_calls]
            assert "fetch_file" in tool_names, "Agent should have used fetch_file tool"
            assert "grep_file" in tool_names or "read_file" in tool_names, (
                "Agent should have used grep_file or read_file tool"
            )

            # Verify the response mentions expected content
            response_lower = response.text.lower()
            assert "setosa" in response_lower, "Response should mention setosa"

        finally:
            agent.reset()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "file_url,file_type,expected_columns",
        [
            (CSV_TEST_URL, "CSV", CSV_EXPECTED_COLUMNS),
            (EXCEL_TEST_URL, "Excel", EXCEL_EXPECTED_COLUMNS),
        ],
        ids=["CSV", "Excel"],
    )
    async def test_agent_reads_file_sections_and_extracts_columns(
        self, file_url: str, file_type: str, expected_columns: list[str]
    ):
        """Test that agent can read file sections and extract column information.

        Verifies the agent can:
        - Download a data file (CSV or Excel)
        - Use read_file to extract specific sections
        - Correctly identify and report column names
        """
        agent = self._create_test_agent()

        try:
            question = (
                f"Download the {file_type} file from {file_url} "
                f"and read the first 10 lines. List all the column names you find in the header."
            )

            response = await agent.answer_async(question)

            # Verify we got a response
            assert response.text, "Agent should return a non-empty response"

            # Verify fetch_file and read_file were called correctly
            self._verify_fetch_call(response.tool_calls, file_url)
            self._verify_tool_used_cached_file(response.tool_calls, "read_file")

            # Response should mention ALL expected columns (strict check)
            response_lower = response.text.lower()
            missing_columns = [col for col in expected_columns if col not in response_lower]
            assert len(missing_columns) == 0, (
                f"Response should mention ALL expected columns. Missing: {missing_columns}. "
                f"Expected: {expected_columns}. Response: {response.text}"
            )

        finally:
            agent.reset()

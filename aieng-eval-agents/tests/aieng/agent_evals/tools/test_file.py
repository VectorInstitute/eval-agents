"""Tests for the file tools module."""

import os
import tempfile

import pytest
from aieng.agent_evals.tools.file import create_grep_file_tool, create_read_file_tool, grep_file, read_file
from aieng.agent_evals.tools.web import fetch_url


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
    """Integration tests for file tools combined with web tools."""

    def test_fetch_and_search_real(self):
        """Test fetch then search workflow."""
        # Fetch
        fetch_result = fetch_url("https://www.iana.org/help/example-domains")
        assert fetch_result["status"] == "success"

        # Search
        search_result = grep_file(fetch_result["file_path"], "example, domain")
        assert search_result["status"] == "success"
        assert search_result["total_matches"] >= 1

        # Cleanup
        os.remove(fetch_result["file_path"])

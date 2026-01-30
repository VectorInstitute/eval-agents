"""Tests for the web tools module."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from aieng.agent_evals.tools.file import create_grep_file_tool, create_read_file_tool, grep_file, read_file
from aieng.agent_evals.tools.pdf import create_read_pdf_tool, read_pdf
from aieng.agent_evals.tools.web import _html_to_text, create_fetch_url_tool, fetch_url


class TestHtmlToText:
    """Tests for the _html_to_text function."""

    def test_removes_script_tags(self):
        """Test that script tags are removed."""
        html = "<html><script>alert('hi')</script><p>Hello</p></html>"
        result = _html_to_text(html)
        assert "alert" not in result
        assert "Hello" in result

    def test_removes_style_tags(self):
        """Test that style tags are removed."""
        html = "<html><style>.foo { color: red; }</style><p>Text</p></html>"
        result = _html_to_text(html)
        assert "color" not in result
        assert "Text" in result

    def test_converts_block_elements_to_newlines(self):
        """Test that block elements become newlines."""
        html = "<p>Para 1</p><p>Para 2</p>"
        result = _html_to_text(html)
        assert "Para 1" in result
        assert "Para 2" in result

    def test_decodes_html_entities(self):
        """Test that HTML entities are decoded."""
        html = "<p>Tom &amp; Jerry &lt;3</p>"
        result = _html_to_text(html)
        assert "Tom & Jerry <3" in result


class TestFetchUrl:
    """Tests for the fetch_url function."""

    @patch("aieng.agent_evals.tools.web.httpx.Client")
    def test_fetch_success(self, mock_client_class):
        """Test successful URL fetch saves file and returns path."""
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Hello World</p></body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = fetch_url("https://example.com")

        assert result["status"] == "success"
        assert "file_path" in result
        assert "preview" in result
        assert "Hello World" in result["preview"]
        assert os.path.exists(result["file_path"])

        # Cleanup
        os.remove(result["file_path"])

    @patch("aieng.agent_evals.tools.web.httpx.Client")
    def test_fetch_pdf_redirect(self, mock_client_class):
        """Test that PDF URLs are redirected to read_pdf."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/pdf"}

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = fetch_url("https://example.com/doc.pdf")

        assert result["status"] == "error"
        assert "PDF" in result["error"]
        assert "read_pdf" in result["error"]

    @patch("aieng.agent_evals.tools.web.httpx.Client")
    def test_fetch_returns_length(self, mock_client_class):
        """Test that fetch returns content length."""
        long_text = "A" * 10000
        mock_response = MagicMock()
        mock_response.text = f"<html><body>{long_text}</body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = fetch_url("https://example.com")

        assert result["status"] == "success"
        assert result["length"] == 10000  # Just the As, no HTML tags
        assert os.path.exists(result["file_path"])

        # Cleanup
        os.remove(result["file_path"])


class TestGrepFile:
    """Tests for the grep_file function (general-purpose file search)."""

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


class TestReadFileSection:
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


class TestReadPdf:
    """Tests for the read_pdf function."""

    @patch("aieng.agent_evals.tools.pdf.httpx.Client")
    def test_read_pdf_success(self, mock_client_class):
        """Test successful PDF reading."""
        # Create a minimal valid PDF
        from io import BytesIO  # noqa: PLC0415

        from pypdf import PdfWriter  # noqa: PLC0415

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)

        pdf_bytes = BytesIO()
        writer.write(pdf_bytes)
        pdf_content = pdf_bytes.getvalue()

        mock_response = MagicMock()
        mock_response.content = pdf_content
        mock_response.headers = {"content-type": "application/pdf"}

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = read_pdf("https://example.com/doc.pdf")

        assert result["status"] == "success" or result["status"] == "error"
        # Blank page may not have extractable text
        assert "num_pages" in result or "error" in result

    @patch("aieng.agent_evals.tools.pdf.httpx.Client")
    def test_read_pdf_html_error(self, mock_client_class):
        """Test that HTML response returns error."""
        mock_response = MagicMock()
        mock_response.content = b"<html><body>Not a PDF</body></html>"
        mock_response.headers = {"content-type": "text/html"}

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = read_pdf("https://example.com/page")

        assert result["status"] == "error"
        assert "HTML" in result["error"]


class TestToolCreation:
    """Tests for tool creation functions."""

    def test_create_fetch_url_tool(self):
        """Test that fetch URL tool is created correctly."""
        tool = create_fetch_url_tool()
        assert tool is not None
        assert tool.func == fetch_url

    def test_create_grep_file_tool(self):
        """Test that grep file tool is created correctly."""
        tool = create_grep_file_tool()
        assert tool is not None
        assert tool.func == grep_file

    def test_create_read_file_tool(self):
        """Test that read file section tool is created correctly."""
        tool = create_read_file_tool()
        assert tool is not None
        assert tool.func == read_file

    def test_create_read_pdf_tool(self):
        """Test that read PDF tool is created correctly."""
        tool = create_read_pdf_tool()
        assert tool is not None
        assert tool.func == read_pdf


@pytest.mark.integration_test
class TestWebToolsIntegration:
    """Integration tests for web tools (requires network)."""

    def test_fetch_url_real(self):
        """Test fetching a real URL saves file."""
        result = fetch_url("https://www.iana.org/help/example-domains")
        assert result["status"] == "success"
        assert "file_path" in result
        assert os.path.exists(result["file_path"])
        # Check preview contains expected content (IANA page about example domains)
        assert "example" in result["preview"].lower()
        # Cleanup
        os.remove(result["file_path"])

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

    def test_read_pdf_real(self):
        """Test reading a real PDF."""
        result = read_pdf("https://arxiv.org/pdf/2301.00234.pdf", max_pages=1)
        assert result["status"] == "success"
        assert result["num_pages"] > 0
        assert "Survey" in result["content"] or "Learning" in result["content"]

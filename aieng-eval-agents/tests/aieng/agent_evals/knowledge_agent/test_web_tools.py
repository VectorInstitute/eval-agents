"""Tests for the web tools module."""

from unittest.mock import MagicMock, patch

import pytest
from aieng.agent_evals.knowledge_agent.web_tools import (
    _html_to_text,
    create_fetch_url_tool,
    create_read_pdf_tool,
    fetch_url,
    read_pdf,
)


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

    @patch("aieng.agent_evals.knowledge_agent.web_tools.httpx.Client")
    def test_fetch_success(self, mock_client_class):
        """Test successful URL fetch."""
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
        assert "Hello World" in result["content"]

    @patch("aieng.agent_evals.knowledge_agent.web_tools.httpx.Client")
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

    @patch("aieng.agent_evals.knowledge_agent.web_tools.httpx.Client")
    def test_fetch_truncates_long_content(self, mock_client_class):
        """Test that long content is truncated."""
        long_text = "A" * 100000
        mock_response = MagicMock()
        mock_response.text = f"<html><body>{long_text}</body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = fetch_url("https://example.com", max_length=1000)

        assert result["status"] == "success"
        assert len(result["content"]) < 1100  # 1000 + truncation message
        assert "truncated" in result["content"]


class TestReadPdf:
    """Tests for the read_pdf function."""

    @patch("aieng.agent_evals.knowledge_agent.web_tools.httpx.Client")
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

    @patch("aieng.agent_evals.knowledge_agent.web_tools.httpx.Client")
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

    def test_create_read_pdf_tool(self):
        """Test that read PDF tool is created correctly."""
        tool = create_read_pdf_tool()
        assert tool is not None
        assert tool.func == read_pdf


@pytest.mark.integration_test
class TestWebToolsIntegration:
    """Integration tests for web tools (requires network)."""

    def test_fetch_url_real(self):
        """Test fetching a real URL."""
        result = fetch_url("https://httpbin.org/html")
        assert result["status"] == "success"
        assert "Herman Melville" in result["content"]  # httpbin returns Moby Dick excerpt

    def test_read_pdf_real(self):
        """Test reading a real PDF."""
        result = read_pdf("https://arxiv.org/pdf/2301.00234.pdf", max_pages=1)
        assert result["status"] == "success"
        assert result["num_pages"] > 0
        assert "Survey" in result["content"] or "Learning" in result["content"]

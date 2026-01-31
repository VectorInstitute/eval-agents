"""Tests for the web tools module.

Tests web_fetch which handles both HTML pages and PDF documents.
"""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from aieng.agent_evals.tools.web import _html_to_markdown, create_web_fetch_tool, web_fetch
from pypdf import PdfWriter


class TestHtmlToMarkdown:
    """Tests for the _html_to_markdown function."""

    def test_removes_script_tags(self):
        """Test that script tags are removed."""
        html = "<html><script>alert('hi')</script><p>Hello</p></html>"
        result = _html_to_markdown(html)
        assert "alert" not in result
        assert "Hello" in result

    def test_removes_style_tags(self):
        """Test that style tags are removed."""
        html = "<html><style>.foo { color: red; }</style><p>Text</p></html>"
        result = _html_to_markdown(html)
        assert "color" not in result
        assert "Text" in result

    def test_converts_paragraphs(self):
        """Test that paragraphs are preserved."""
        html = "<p>Para 1</p><p>Para 2</p>"
        result = _html_to_markdown(html)
        assert "Para 1" in result
        assert "Para 2" in result

    def test_decodes_html_entities(self):
        """Test that HTML entities are decoded."""
        html = "<p>Tom &amp; Jerry</p>"
        result = _html_to_markdown(html)
        assert "Tom & Jerry" in result

    def test_preserves_links(self):
        """Test that links are preserved in markdown format."""
        html = '<a href="https://example.com">Example Link</a>'
        result = _html_to_markdown(html)
        assert "[Example Link]" in result
        assert "https://example.com" in result

    def test_preserves_links_with_base_url(self):
        """Test that relative links are converted to absolute."""
        html = '<a href="/page">Link</a>'
        result = _html_to_markdown(html, base_url="https://example.com")
        assert "https://example.com/page" in result

    def test_preserves_headings(self):
        """Test that headings are converted to markdown."""
        html = "<h1>Title</h1><h2>Subtitle</h2>"
        result = _html_to_markdown(html)
        assert "Title" in result
        assert "Subtitle" in result


class TestWebFetch:
    """Tests for the web_fetch function."""

    @patch("aieng.agent_evals.tools.web.httpx.Client")
    def test_fetch_html_success(self, mock_client_class):
        """Test successful HTML fetch returns content."""
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Hello World</p></body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = web_fetch("https://example.com")

        assert result["status"] == "success"
        assert "content" in result
        assert "Hello World" in result["content"]
        assert result["content_type"] == "text/html"

    @patch("aieng.agent_evals.tools.web.httpx.Client")
    def test_fetch_pdf_success(self, mock_client_class):
        """Test that PDF content is extracted successfully."""
        # Create a PDF with text
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        pdf_bytes = BytesIO()
        writer.write(pdf_bytes)
        pdf_content = pdf_bytes.getvalue()

        mock_response = MagicMock()
        mock_response.content = pdf_content
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.url = "https://example.com/doc.pdf"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = web_fetch("https://example.com/doc.pdf")

        assert result["status"] == "success"
        assert result["content_type"] == "application/pdf"
        assert "num_pages" in result
        assert result["num_pages"] >= 1

    @patch("aieng.agent_evals.tools.web.httpx.Client")
    def test_fetch_returns_content_length(self, mock_client_class):
        """Test that fetch returns content length."""
        long_text = "A" * 10000
        mock_response = MagicMock()
        mock_response.text = f"<html><body><p>{long_text}</p></body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = web_fetch("https://example.com")

        assert result["status"] == "success"
        # Content length should include the 10000 As (may have some markdown formatting)
        assert result["content_length"] >= 10000
        assert not result["truncated"]

    @patch("aieng.agent_evals.tools.web.httpx.Client")
    def test_fetch_truncates_large_content(self, mock_client_class):
        """Test that very large content is truncated."""
        # Create content larger than MAX_CONTENT_CHARS (100KB)
        large_text = "A" * 150_000
        mock_response = MagicMock()
        mock_response.text = f"<html><body>{large_text}</body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        result = web_fetch("https://example.com")

        assert result["status"] == "success"
        assert result["truncated"] is True
        assert "[Content truncated" in result["content"]

    def test_fetch_invalid_url(self):
        """Test that invalid URLs return error."""
        result = web_fetch("not-a-url")
        assert result["status"] == "error"
        assert "Invalid URL" in result["error"]


class TestCreateWebFetchTool:
    """Tests for the create_web_fetch_tool function."""

    def test_creates_tool_with_correct_function(self):
        """Test that web fetch tool is created with the correct function."""
        tool = create_web_fetch_tool()
        assert tool is not None
        assert tool.func == web_fetch


@pytest.mark.integration_test
class TestWebFetchIntegration:
    """Integration tests for web_fetch (requires network).

    These tests verify that web_fetch works correctly for both HTML pages
    and PDF documents, returning content suitable for the agent to analyze.
    """

    def test_fetch_html_page_returns_readable_content(self):
        """Test that HTML pages are converted to readable markdown."""
        result = web_fetch("https://www.iana.org/help/example-domains")
        assert result["status"] == "success"
        assert result["content_type"] == "text/html" or "html" in result["content_type"].lower()

        # Verify content is markdown (no raw HTML tags)
        content = result["content"]
        assert "<html>" not in content.lower()
        assert "<body>" not in content.lower()

        # Verify content has meaningful text
        assert len(content) > 100
        assert "example" in content.lower()

        # Verify links are preserved in markdown format (if any exist)
        # The page should have links that are converted to [text](url) format
        if "http" in content:
            # Links should be in markdown format, not raw <a> tags
            assert "<a " not in content.lower()

    def test_fetch_pdf_extracts_text(self):
        """Test that PDF content is extracted as searchable text."""
        result = web_fetch("https://arxiv.org/pdf/2301.00234.pdf", max_pages=2)
        assert result["status"] == "success"
        assert result["content_type"] == "application/pdf"
        assert result["num_pages"] > 0

        # Verify extracted text is substantial
        content = result["content"]
        assert len(content) > 500

        # Verify page markers are present
        assert "--- Page" in content

    def test_fetch_pdf_pagination(self):
        """Test that PDF max_pages parameter limits extraction."""
        result = web_fetch("https://arxiv.org/pdf/2301.00234.pdf", max_pages=1)
        assert result["status"] == "success"
        assert result["pages_extracted"] == 1
        assert result["num_pages"] >= 1

"""Tests for the web tools module."""

import os
from unittest.mock import MagicMock, patch

import pytest
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


class TestCreateFetchUrlTool:
    """Tests for the create_fetch_url_tool function."""

    def test_creates_tool_with_correct_function(self):
        """Test that fetch URL tool is created with the correct function."""
        tool = create_fetch_url_tool()
        assert tool is not None
        assert tool.func == fetch_url


@pytest.mark.integration_test
class TestFetchUrlIntegration:
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

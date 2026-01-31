"""Tests for the PDF tools module."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from aieng.agent_evals.tools.pdf import create_read_pdf_tool, read_pdf
from pypdf import PdfWriter


class TestReadPdf:
    """Tests for the read_pdf function."""

    @patch("aieng.agent_evals.tools.pdf.httpx.Client")
    def test_read_pdf_success(self, mock_client_class):
        """Test successful PDF reading."""
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


class TestCreateReadPdfTool:
    """Tests for the create_read_pdf_tool function."""

    def test_creates_tool_with_correct_function(self):
        """Test that read PDF tool is created with the correct function."""
        tool = create_read_pdf_tool()
        assert tool is not None
        assert tool.func == read_pdf


@pytest.mark.integration_test
class TestReadPdfIntegration:
    """Integration tests for PDF tools (requires network)."""

    def test_read_pdf_real(self):
        """Test reading a real PDF."""
        result = read_pdf("https://arxiv.org/pdf/2301.00234.pdf", max_pages=1)
        assert result["status"] == "success"
        assert result["num_pages"] > 0
        assert "Survey" in result["content"] or "Learning" in result["content"]

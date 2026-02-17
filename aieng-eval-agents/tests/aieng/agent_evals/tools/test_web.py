"""Tests for the web tools module.

Tests web_fetch which handles both HTML pages and PDF documents.
"""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from aieng.agent_evals.tools._redirect import (
    _redirect_cache,
    resolve_redirect_url_async,
    resolve_redirect_urls_async,
)
from aieng.agent_evals.tools.web import (
    _html_to_markdown,
    create_web_fetch_tool,
    web_fetch,
)
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

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.tools.web.AsyncClientManager")
    @patch("aieng.agent_evals.tools.web.httpx.AsyncClient")
    async def test_fetch_html_success(self, mock_http_client_class, mock_client_manager_class):
        """Test successful HTML fetch with LLM extraction."""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Hello World</p></body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com"

        async def mock_get(*_args, **_kwargs):
            return mock_response

        mock_http_client = MagicMock()
        mock_http_client.get = mock_get
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client_class.return_value = mock_http_client

        # Mock LLM extraction
        mock_llm_response = MagicMock()
        mock_llm_response.choices = [MagicMock()]
        mock_llm_response.choices[0].message.content = "Extracted: Hello World"
        mock_llm_response.usage = MagicMock()
        mock_llm_response.usage.prompt_tokens = 100
        mock_llm_response.usage.completion_tokens = 20

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_llm_response)

        mock_configs = MagicMock()
        mock_configs.default_worker_model = "gemini-2.5-flash"

        mock_client_manager = MagicMock()
        mock_client_manager.openai_client = mock_openai_client
        mock_client_manager.configs = mock_configs
        mock_client_manager_class.get_instance.return_value = mock_client_manager

        result = await web_fetch("https://example.com", query="get the main message")

        assert result["status"] == "success"
        assert "extracted_info" in result
        assert "Hello World" in result["extracted_info"]
        assert result["content_type"] == "text/html"
        assert result["query"] == "get the main message"
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 20

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.tools.web.AsyncClientManager")
    @patch("aieng.agent_evals.tools.web.httpx.AsyncClient")
    async def test_fetch_pdf_success(self, mock_http_client_class, mock_client_manager_class):
        """Test that PDF content is extracted and processed with LLM."""
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

        async def mock_get(*_args, **_kwargs):
            return mock_response

        mock_http_client = MagicMock()
        mock_http_client.get = mock_get
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client_class.return_value = mock_http_client

        # Mock LLM extraction
        mock_llm_response = MagicMock()
        mock_llm_response.choices = [MagicMock()]
        mock_llm_response.choices[0].message.content = "PDF summary information"
        mock_llm_response.usage = MagicMock()
        mock_llm_response.usage.prompt_tokens = 150
        mock_llm_response.usage.completion_tokens = 30

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_llm_response)

        mock_configs = MagicMock()
        mock_configs.default_worker_model = "gemini-2.5-flash"

        mock_client_manager = MagicMock()
        mock_client_manager.openai_client = mock_openai_client
        mock_client_manager.configs = mock_configs
        mock_client_manager_class.get_instance.return_value = mock_client_manager

        result = await web_fetch("https://example.com/doc.pdf", query="summarize the document")

        assert result["status"] == "success"
        assert result["content_type"] == "application/pdf"
        assert "num_pages" in result
        assert result["num_pages"] >= 1
        assert "extracted_info" in result
        assert "PDF summary" in result["extracted_info"]

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.tools.web.AsyncClientManager")
    @patch("aieng.agent_evals.tools.web.httpx.AsyncClient")
    async def test_fetch_returns_token_counts(self, mock_http_client_class, mock_client_manager_class):
        """Test that fetch returns token usage from LLM extraction."""
        long_text = "A" * 10000
        mock_response = MagicMock()
        mock_response.text = f"<html><body><p>{long_text}</p></body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com"

        async def mock_get(*_args, **_kwargs):
            return mock_response

        mock_http_client = MagicMock()
        mock_http_client.get = mock_get
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client_class.return_value = mock_http_client

        # Mock LLM extraction with realistic token counts for large input
        mock_llm_response = MagicMock()
        mock_llm_response.choices = [MagicMock()]
        mock_llm_response.choices[0].message.content = "This page contains repeated A characters"
        mock_llm_response.usage = MagicMock()
        mock_llm_response.usage.prompt_tokens = 2500  # Large input
        mock_llm_response.usage.completion_tokens = 25

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_llm_response)

        mock_configs = MagicMock()
        mock_configs.default_worker_model = "gemini-2.5-flash"

        mock_client_manager = MagicMock()
        mock_client_manager.openai_client = mock_openai_client
        mock_client_manager.configs = mock_configs
        mock_client_manager_class.get_instance.return_value = mock_client_manager

        result = await web_fetch("https://example.com", query="what is on this page")

        assert result["status"] == "success"
        assert result["input_tokens"] == 2500
        assert result["output_tokens"] == 25
        assert result["source_was_truncated"] is False

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.tools.web.AsyncClientManager")
    @patch("aieng.agent_evals.tools.web.httpx.AsyncClient")
    async def test_fetch_truncates_large_content(self, mock_http_client_class, mock_client_manager_class):
        """Test that very large content is truncated before LLM extraction."""
        # Create content larger than MAX_FETCH_CHARS (200KB)
        large_text = "A" * 250_000
        mock_response = MagicMock()
        mock_response.text = f"<html><body>{large_text}</body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com"

        async def mock_get(*_args, **_kwargs):
            return mock_response

        mock_http_client = MagicMock()
        mock_http_client.get = mock_get
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client_class.return_value = mock_http_client

        # Mock LLM extraction
        mock_llm_response = MagicMock()
        mock_llm_response.choices = [MagicMock()]
        mock_llm_response.choices[0].message.content = "Content was truncated but extracted"
        mock_llm_response.usage = MagicMock()
        mock_llm_response.usage.prompt_tokens = 25000
        mock_llm_response.usage.completion_tokens = 50

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_llm_response)

        mock_configs = MagicMock()
        mock_configs.default_worker_model = "gemini-2.5-flash"

        mock_client_manager = MagicMock()
        mock_client_manager.openai_client = mock_openai_client
        mock_client_manager.configs = mock_configs
        mock_client_manager_class.get_instance.return_value = mock_client_manager

        result = await web_fetch("https://example.com", query="extract info")

        assert result["status"] == "success"
        assert result["source_was_truncated"] is True

    @pytest.mark.asyncio
    async def test_fetch_invalid_url(self):
        """Test that invalid URLs return error."""
        result = await web_fetch("not-a-url", query="get info")
        assert result["status"] == "error"
        assert "Invalid URL" in result["error"]
        assert result["query"] == "get info"


class TestCreateWebFetchTool:
    """Tests for the create_web_fetch_tool function."""

    def test_creates_tool_with_correct_function(self):
        """Test that web fetch tool is created with the correct function."""
        tool = create_web_fetch_tool()
        assert tool is not None
        assert tool.func == web_fetch


class TestResolveRedirectUrlAsync:
    """Tests for async redirect URL resolution."""

    @pytest.mark.asyncio
    async def test_non_redirect_url_returns_unchanged(self):
        """Test that non-redirect URLs are returned unchanged."""
        url = "https://example.com/page"
        result = await resolve_redirect_url_async(url)
        assert result == url

    @pytest.mark.asyncio
    async def test_resolves_redirect_url_async(self):
        """Test async resolution of redirect URLs."""
        redirect_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/async123"
        final_url = "https://example.com/actual-page-async"

        mock_response = MagicMock()
        mock_response.url = final_url

        # Create async mock for head method
        async def mock_head(*_args, **_kwargs):
            return mock_response

        # Clear the cache
        _redirect_cache.clear()

        with patch("aieng.agent_evals.tools.web.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.head = mock_head
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await resolve_redirect_url_async(redirect_url)
            assert result == final_url

    @pytest.mark.asyncio
    async def test_resolve_multiple_urls_in_parallel(self):
        """Test that multiple URLs can be resolved in parallel."""
        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3",
        ]
        results = await resolve_redirect_urls_async(urls)

        # Non-redirect URLs should be returned as-is
        assert results == urls

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        """Test that empty list input returns empty list."""
        results = await resolve_redirect_urls_async([])
        assert results == []

    @pytest.mark.asyncio
    async def test_caches_resolved_urls(self):
        """Test that resolved URLs are cached to avoid repeated HTTP calls."""
        redirect_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/cache-test"
        final_url = "https://example.com/cached-page"

        call_count = 0

        async def mock_head(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.url = final_url
            return mock_response

        # Clear the cache
        _redirect_cache.clear()

        with patch("aieng.agent_evals.tools.web.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.head = mock_head
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            # First call should make HTTP request
            result1 = await resolve_redirect_url_async(redirect_url)
            assert result1 == final_url
            assert call_count == 1

            # Second call should use cache (no HTTP request)
            result2 = await resolve_redirect_url_async(redirect_url)
            assert result2 == final_url
            assert call_count == 1  # Still 1, used cache

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self):
        """Test that resolution retries on transient failures."""
        redirect_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/retry-test"
        final_url = "https://example.com/retried-page"

        head_call_count = 0
        stream_call_count = 0

        # HEAD will always fail (return None triggers GET fallback)
        async def mock_head(*_args, **_kwargs):
            nonlocal head_call_count
            head_call_count += 1
            # Return None to simulate HEAD not supported, triggers GET fallback
            raise httpx.HTTPStatusError(
                "Method Not Allowed",
                request=MagicMock(),
                response=MagicMock(status_code=405),
            )

        # Stream (GET) will fail first two times, then succeed
        class MockStreamContext:
            def __init__(self, fail_count):
                self.fail_count = fail_count

            async def __aenter__(self):
                nonlocal stream_call_count
                stream_call_count += 1
                if stream_call_count <= self.fail_count:
                    raise httpx.TimeoutException("Connection timed out")
                mock_response = MagicMock()
                mock_response.url = final_url
                return mock_response

            async def __aexit__(self, *_args):
                pass

        # Clear the cache
        _redirect_cache.clear()

        with patch("aieng.agent_evals.tools.web.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.head = mock_head
            mock_client.stream.return_value = MockStreamContext(fail_count=2)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await resolve_redirect_url_async(redirect_url)
            assert result == final_url
            # HEAD called once per retry attempt (3 times)
            # Stream called 3 times (2 failures + 1 success)
            assert stream_call_count == 3


@pytest.mark.integration_test
class TestWebFetchIntegration:
    """Integration tests for web_fetch (requires network and API keys).

    These tests verify that web_fetch works correctly for both HTML pages
    and PDF documents, using LLM extraction to return only relevant information
    instead of full page content.
    """

    @pytest.mark.asyncio
    async def test_fetch_html_page_with_extraction(self):
        """Test that HTML pages are fetched and information is extracted via LLM."""
        result = await web_fetch(
            "https://www.iana.org/help/example-domains", query="What is the purpose of example.com domain?"
        )
        assert result["status"] == "success"
        assert result["content_type"] == "text/html" or "html" in result["content_type"].lower()

        # Verify extracted info is returned (not raw content)
        assert "extracted_info" in result
        extracted = result["extracted_info"]

        # Extracted info should be concise (not full page)
        assert len(extracted) < 5000  # Should be much smaller than full page
        assert len(extracted) > 50  # But should have meaningful content

        # Should contain relevant information about example domains
        assert "example" in extracted.lower()

        # Verify token counts are present
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert result["input_tokens"] > 0
        assert result["output_tokens"] > 0

        # Verify query is echoed back
        assert result["query"] == "What is the purpose of example.com domain?"

    @pytest.mark.asyncio
    async def test_fetch_pdf_with_extraction(self):
        """Test that PDF content is extracted and processed via LLM."""
        result = await web_fetch(
            "https://arxiv.org/pdf/2301.00234.pdf", query="What is the main research contribution?", max_pages=2
        )
        assert result["status"] == "success"
        assert result["content_type"] == "application/pdf"
        assert result["num_pages"] > 0

        # Verify extracted info instead of raw text
        assert "extracted_info" in result
        extracted = result["extracted_info"]

        # Extracted info should be concise
        assert len(extracted) < 5000
        assert len(extracted) > 50

        # Verify PDF metadata
        assert result["pages_extracted"] <= 2

        # Verify token usage
        assert result["input_tokens"] > 0
        assert result["output_tokens"] > 0

    @pytest.mark.asyncio
    async def test_fetch_pdf_pagination(self):
        """Test that PDF max_pages parameter limits extraction before LLM processing."""
        result = await web_fetch("https://arxiv.org/pdf/2301.00234.pdf", query="summarize the abstract", max_pages=1)
        assert result["status"] == "success"
        assert result["pages_extracted"] == 1
        assert result["num_pages"] >= 1

        # Even with 1 page, should get meaningful extraction
        assert "extracted_info" in result
        assert len(result["extracted_info"]) > 20

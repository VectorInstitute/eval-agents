"""Tests for Google Search tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aieng.agent_evals.tools import (
    GroundedResponse,
    GroundingChunk,
    create_google_search_tool,
    format_response_with_citations,
    google_search,
)
from aieng.agent_evals.tools.search import _extract_grounding_sources, _extract_summary_from_response
from google.adk.tools.function_tool import FunctionTool


class TestGroundingChunk:
    """Tests for the GroundingChunk model."""

    def test_grounding_chunk_creation(self):
        """Test creating a grounding chunk."""
        chunk = GroundingChunk(title="Test Title", uri="https://example.com")
        assert chunk.title == "Test Title"
        assert chunk.uri == "https://example.com"

    def test_grounding_chunk_defaults(self):
        """Test default values for grounding chunk."""
        chunk = GroundingChunk()
        assert chunk.title == ""
        assert chunk.uri == ""


class TestGroundedResponse:
    """Tests for the GroundedResponse model."""

    def test_grounded_response_creation(self):
        """Test creating a grounded response."""
        response = GroundedResponse(
            text="Test response",
            search_queries=["query1", "query2"],
            sources=[
                GroundingChunk(title="Source 1", uri="https://source1.com"),
            ],
            tool_calls=[{"name": "google_search", "args": {"query": "test"}}],
        )
        assert response.text == "Test response"
        assert len(response.search_queries) == 2
        assert len(response.sources) == 1
        assert len(response.tool_calls) == 1

    def test_grounded_response_defaults(self):
        """Test default values for grounded response."""
        response = GroundedResponse(text="Just text")
        assert response.text == "Just text"
        assert response.search_queries == []
        assert response.sources == []
        assert response.tool_calls == []

    def test_format_with_citations(self):
        """Test format_with_citations method."""
        response = GroundedResponse(
            text="The answer is 42.",
            sources=[
                GroundingChunk(title="Wikipedia", uri="https://en.wikipedia.org/wiki/42"),
            ],
        )

        formatted = response.format_with_citations()

        assert "The answer is 42." in formatted
        assert "**Sources:**" in formatted
        assert "[Wikipedia](https://en.wikipedia.org/wiki/42)" in formatted

    def test_format_with_citations_no_sources(self):
        """Test format_with_citations method without sources."""
        response = GroundedResponse(text="Simple answer.")

        formatted = response.format_with_citations()

        assert formatted == "Simple answer."
        assert "Sources" not in formatted


class TestCreateGoogleSearchTool:
    """Tests for the create_google_search_tool function."""

    def test_creates_function_tool(self):
        """Test that the tool is created as a FunctionTool wrapping google_search."""
        # Create a mock config with the required attribute
        mock_config = MagicMock()
        mock_config.default_worker_model = "gemini-2.5-flash"

        result = create_google_search_tool(config=mock_config)

        assert isinstance(result, FunctionTool)
        # The function tool should wrap the google_search function
        assert result.func.__name__ == "google_search"


class TestFormatResponseWithCitations:
    """Tests for the format_response_with_citations function."""

    def test_format_response_with_citations(self):
        """Test formatting response with citations."""
        response = GroundedResponse(
            text="The answer is 42.",
            search_queries=["meaning of life"],
            sources=[
                GroundingChunk(title="Wikipedia", uri="https://en.wikipedia.org/wiki/42"),
                GroundingChunk(title="Guide", uri="https://example.com/guide"),
            ],
        )

        formatted = format_response_with_citations(response)

        assert "The answer is 42." in formatted
        assert "**Sources:**" in formatted
        assert "[Wikipedia](https://en.wikipedia.org/wiki/42)" in formatted
        assert "[Guide](https://example.com/guide)" in formatted

    def test_format_response_without_sources(self):
        """Test formatting response without sources."""
        response = GroundedResponse(text="Simple answer.")

        formatted = format_response_with_citations(response)

        assert formatted == "Simple answer."
        assert "Sources" not in formatted

    def test_format_response_with_empty_title(self):
        """Test formatting response with source that has empty title."""
        response = GroundedResponse(
            text="Answer here.",
            sources=[
                GroundingChunk(title="", uri="https://example.com/page"),
            ],
        )

        formatted = format_response_with_citations(response)

        assert "[Source](https://example.com/page)" in formatted

    def test_format_response_skips_sources_without_uri(self):
        """Test that sources without URI are skipped."""
        response = GroundedResponse(
            text="Answer here.",
            sources=[
                GroundingChunk(title="No URI", uri=""),
                GroundingChunk(title="Has URI", uri="https://example.com"),
            ],
        )

        formatted = format_response_with_citations(response)

        assert "No URI" not in formatted
        assert "[Has URI](https://example.com)" in formatted


@pytest.mark.integration_test
class TestGoogleSearchToolIntegration:
    """Integration tests for the Google Search tool.

    These tests require a valid GOOGLE_API_KEY environment variable.
    """

    def test_create_google_search_tool_real(self):
        """Test creating a real FunctionTool instance wrapping google_search."""
        tool = create_google_search_tool()
        # The tool should be a FunctionTool wrapping google_search
        assert isinstance(tool, FunctionTool)

    @pytest.mark.asyncio
    async def test_google_search_returns_urls(self):
        """Test that google_search returns actual URLs, not redirect URLs."""
        result = await google_search("capital of France")

        # Should have success status
        assert result["status"] == "success"

        # Should have a summary
        assert result["summary"], "Expected non-empty summary"

        # Should have sources with URLs
        assert result["source_count"] > 0, "Expected at least one source"
        assert len(result["sources"]) == result["source_count"]

        # Each source should have title and url
        for source in result["sources"]:
            assert "title" in source
            assert "url" in source
            # URL should be a real URL, not a redirect URL
            assert source["url"].startswith("http"), f"Expected URL, got: {source['url']}"
            assert "vertexaisearch" not in source["url"], "URL should not be a redirect URL"

    @pytest.mark.asyncio
    async def test_google_search_response_structure(self):
        """Test the complete response structure from google_search."""
        result = await google_search("Python programming language")

        # Check all expected keys exist
        assert "status" in result
        assert "summary" in result
        assert "sources" in result
        assert "source_count" in result

        # Sources should be a list
        assert isinstance(result["sources"], list)

        # If we have sources, verify their structure
        if result["sources"]:
            source = result["sources"][0]
            assert isinstance(source, dict)
            assert "title" in source
            assert "url" in source


class TestExtractSummaryFromResponse:
    """Tests for _extract_summary_from_response."""

    def test_no_candidates_returns_empty(self):
        """Test that an empty candidates list yields an empty summary."""
        response = MagicMock()
        response.candidates = []
        assert _extract_summary_from_response(response) == ""

    def test_candidate_with_no_content_returns_empty(self):
        """Test that a candidate whose content is None yields an empty summary."""
        candidate = MagicMock()
        candidate.content = None
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_summary_from_response(response) == ""

    def test_candidate_with_no_parts_returns_empty(self):
        """Test that content with no parts yields an empty summary."""
        candidate = MagicMock()
        candidate.content.parts = None
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_summary_from_response(response) == ""

    def test_single_text_part_returned(self):
        """Test that a single text part is returned as the summary."""
        part = MagicMock()
        part.text = "Paris is the capital of France."
        candidate = MagicMock()
        candidate.content.parts = [part]
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_summary_from_response(response) == "Paris is the capital of France."

    def test_multiple_text_parts_are_concatenated(self):
        """Test that multiple text parts are joined without a separator."""
        part1, part2 = MagicMock(), MagicMock()
        part1.text = "First part. "
        part2.text = "Second part."
        candidate = MagicMock()
        candidate.content.parts = [part1, part2]
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_summary_from_response(response) == "First part. Second part."

    def test_part_without_text_attribute_is_skipped(self):
        """Test that parts lacking a text attribute are skipped."""
        part_no_text = MagicMock(spec=[])  # hasattr(part, "text") → False
        part_with_text = MagicMock()
        part_with_text.text = "Only this."
        candidate = MagicMock()
        candidate.content.parts = [part_no_text, part_with_text]
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_summary_from_response(response) == "Only this."

    def test_part_with_empty_text_is_skipped(self):
        """Test that parts with an empty string text value are skipped."""
        part_empty = MagicMock()
        part_empty.text = ""
        part_valid = MagicMock()
        part_valid.text = "Non-empty."
        candidate = MagicMock()
        candidate.content.parts = [part_empty, part_valid]
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_summary_from_response(response) == "Non-empty."

    def test_only_first_candidate_is_used(self):
        """Test that only the first candidate contributes to the summary."""
        part1, part2 = MagicMock(), MagicMock()
        part1.text = "First candidate text."
        part2.text = "Second candidate text."
        candidate1, candidate2 = MagicMock(), MagicMock()
        candidate1.content.parts = [part1]
        candidate2.content.parts = [part2]
        response = MagicMock()
        response.candidates = [candidate1, candidate2]
        assert _extract_summary_from_response(response) == "First candidate text."


class TestExtractGroundingSources:
    """Tests for _extract_grounding_sources."""

    @pytest.mark.asyncio
    async def test_no_candidates_returns_empty(self):
        """Test that an empty candidates list yields no sources."""
        response = MagicMock()
        response.candidates = []
        assert await _extract_grounding_sources(response) == []

    @pytest.mark.asyncio
    async def test_no_grounding_metadata_returns_empty(self):
        """Test that a candidate with no grounding_metadata yields no sources."""
        candidate = MagicMock()
        candidate.grounding_metadata = None
        response = MagicMock()
        response.candidates = [candidate]
        assert await _extract_grounding_sources(response) == []

    @pytest.mark.asyncio
    async def test_grounding_chunks_attribute_missing_returns_empty(self):
        """Test that grounding_metadata lacking grounding_chunks yields no sources."""
        # spec=[] makes hasattr(gm, "grounding_chunks") return False
        gm = MagicMock(spec=[])
        candidate = MagicMock()
        candidate.grounding_metadata = gm
        response = MagicMock()
        response.candidates = [candidate]
        assert await _extract_grounding_sources(response) == []

    @pytest.mark.asyncio
    async def test_empty_grounding_chunks_returns_empty(self):
        """Test that an empty grounding_chunks list yields no sources."""
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = []
        response = MagicMock()
        response.candidates = [candidate]
        assert await _extract_grounding_sources(response) == []

    @pytest.mark.asyncio
    async def test_single_valid_source(self):
        """Test that a single web chunk with a valid URL is returned."""
        chunk = MagicMock()
        chunk.web.uri = "https://example.com/article"
        chunk.web.title = "Example Article"
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk]
        response = MagicMock()
        response.candidates = [candidate]

        with patch(
            "aieng.agent_evals.tools.search.resolve_redirect_urls_async",
            new=AsyncMock(return_value=["https://example.com/article"]),
        ):
            result = await _extract_grounding_sources(response)

        assert result == [{"title": "Example Article", "url": "https://example.com/article"}]

    @pytest.mark.asyncio
    async def test_multiple_sources_preserved_in_order(self):
        """Test that multiple sources are returned in the same order as the chunks."""
        chunk1, chunk2 = MagicMock(), MagicMock()
        chunk1.web.uri = "https://site1.com"
        chunk1.web.title = "Site 1"
        chunk2.web.uri = "https://site2.com"
        chunk2.web.title = "Site 2"
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk1, chunk2]
        response = MagicMock()
        response.candidates = [candidate]

        with patch(
            "aieng.agent_evals.tools.search.resolve_redirect_urls_async",
            new=AsyncMock(return_value=["https://site1.com", "https://site2.com"]),
        ):
            result = await _extract_grounding_sources(response)

        assert result == [
            {"title": "Site 1", "url": "https://site1.com"},
            {"title": "Site 2", "url": "https://site2.com"},
        ]

    @pytest.mark.asyncio
    async def test_all_chunks_without_web_skips_url_resolution(self):
        """Test that URL resolution is not called when no chunks have a web source."""
        chunk1, chunk2 = MagicMock(), MagicMock()
        chunk1.web = None
        chunk2.web = None
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk1, chunk2]
        response = MagicMock()
        response.candidates = [candidate]

        with patch("aieng.agent_evals.tools.search.resolve_redirect_urls_async") as mock_resolve:
            result = await _extract_grounding_sources(response)

        mock_resolve.assert_not_called()
        assert result == []

    @pytest.mark.asyncio
    async def test_chunk_without_web_is_skipped(self):
        """Test that chunks with a falsy web attribute are ignored."""
        chunk_no_web = MagicMock()
        chunk_no_web.web = None
        chunk_valid = MagicMock()
        chunk_valid.web.uri = "https://example.com"
        chunk_valid.web.title = "Example"
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk_no_web, chunk_valid]
        response = MagicMock()
        response.candidates = [candidate]

        with patch(
            "aieng.agent_evals.tools.search.resolve_redirect_urls_async",
            new=AsyncMock(return_value=["https://example.com"]),
        ):
            result = await _extract_grounding_sources(response)

        assert result == [{"title": "Example", "url": "https://example.com"}]

    @pytest.mark.asyncio
    async def test_vertexaisearch_url_is_filtered_out(self):
        """Test that resolved URLs beginning with vertexaisearch are excluded."""
        chunk = MagicMock()
        chunk.web.uri = "https://vertexaisearch.cloud.google.com/redirect/abc"
        chunk.web.title = "Redirect"
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk]
        response = MagicMock()
        response.candidates = [candidate]

        with patch(
            "aieng.agent_evals.tools.search.resolve_redirect_urls_async",
            new=AsyncMock(return_value=["https://vertexaisearch.cloud.google.com/redirect/abc"]),
        ):
            result = await _extract_grounding_sources(response)

        assert result == []

    @pytest.mark.asyncio
    async def test_empty_resolved_url_is_filtered_out(self):
        """Test that sources whose resolved URL is an empty string are excluded."""
        chunk = MagicMock()
        chunk.web.uri = "https://example.com"
        chunk.web.title = "Example"
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk]
        response = MagicMock()
        response.candidates = [candidate]

        with patch(
            "aieng.agent_evals.tools.search.resolve_redirect_urls_async",
            new=AsyncMock(return_value=[""]),
        ):
            result = await _extract_grounding_sources(response)

        assert result == []

    @pytest.mark.asyncio
    async def test_valid_and_filtered_sources_mixed(self):
        """Test that vertexaisearch sources are filtered when mixed with valid ones."""
        chunk_valid = MagicMock()
        chunk_valid.web.uri = "https://valid.com/page"
        chunk_valid.web.title = "Valid"
        chunk_vertex = MagicMock()
        chunk_vertex.web.uri = "https://vertexaisearch.cloud.google.com/redirect/xyz"
        chunk_vertex.web.title = "Vertex"
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk_valid, chunk_vertex]
        response = MagicMock()
        response.candidates = [candidate]

        with patch(
            "aieng.agent_evals.tools.search.resolve_redirect_urls_async",
            new=AsyncMock(
                return_value=["https://valid.com/page", "https://vertexaisearch.cloud.google.com/redirect/xyz"]
            ),
        ):
            result = await _extract_grounding_sources(response)

        assert result == [{"title": "Valid", "url": "https://valid.com/page"}]

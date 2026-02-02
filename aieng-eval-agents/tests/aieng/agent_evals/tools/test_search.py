"""Tests for Google Search tool."""

import pytest
from aieng.agent_evals.tools import (
    GroundedResponse,
    GroundingChunk,
    create_google_search_tool,
    format_response_with_citations,
    google_search,
)
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
        result = create_google_search_tool()

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

    def test_google_search_returns_urls(self):
        """Test that google_search returns actual URLs, not redirect URLs."""
        result = google_search("capital of France")

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

    def test_google_search_response_structure(self):
        """Test the complete response structure from google_search."""
        result = google_search("Python programming language")

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

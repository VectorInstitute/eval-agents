"""Tests for Google Search grounding tool."""

from unittest.mock import MagicMock, patch

import pytest
from aieng.agent_evals.knowledge_agent.grounding_tool import (
    GroundedResponse,
    GroundingChunk,
    create_google_search_tool,
    format_response_with_citations,
)


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


class TestCreateGoogleSearchTool:
    """Tests for the create_google_search_tool function."""

    @patch("aieng.agent_evals.knowledge_agent.grounding_tool.GoogleSearchTool")
    def test_creates_tool_with_bypass_flag(self, mock_tool_class):
        """Test that the tool is created with bypass_multi_tools_limit=True."""
        mock_tool = MagicMock()
        mock_tool_class.return_value = mock_tool

        result = create_google_search_tool()

        mock_tool_class.assert_called_once_with(bypass_multi_tools_limit=True)
        assert result is mock_tool


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
        """Test creating a real GoogleSearchTool instance."""
        tool = create_google_search_tool()
        # The tool should be a GoogleSearchTool instance with bypass flag
        assert tool is not None

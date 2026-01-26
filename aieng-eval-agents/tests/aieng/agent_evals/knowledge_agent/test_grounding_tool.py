"""Tests for Gemini grounding tool."""

from unittest.mock import MagicMock, patch

import pytest
from aieng.agent_evals.knowledge_agent.grounding_tool import (
    GeminiGroundingTool,
    GroundedResponse,
    GroundingChunk,
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
        )
        assert response.text == "Test response"
        assert len(response.search_queries) == 2
        assert len(response.sources) == 1

    def test_grounded_response_defaults(self):
        """Test default values for grounded response."""
        response = GroundedResponse(text="Just text")
        assert response.text == "Just text"
        assert response.search_queries == []
        assert response.sources == []


class TestGeminiGroundingTool:
    """Tests for the GeminiGroundingTool class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = MagicMock()
        config.openai_api_key = "test-api-key"
        config.default_worker_model = "gemini-2.5-flash"
        return config

    @patch("aieng.agent_evals.knowledge_agent.grounding_tool.genai.Client")
    def test_tool_initialization(self, mock_client_class, mock_config):
        """Test initializing the grounding tool."""
        tool = GeminiGroundingTool(config=mock_config)

        assert tool.config is mock_config
        assert tool.model == "gemini-2.5-flash"
        mock_client_class.assert_called_once_with(api_key="test-api-key")

    @patch("aieng.agent_evals.knowledge_agent.grounding_tool.genai.Client")
    def test_tool_with_custom_model(self, mock_client_class, mock_config):
        """Test initializing with a custom model."""
        tool = GeminiGroundingTool(config=mock_config, model="gemini-2.5-pro")
        assert tool.model == "gemini-2.5-pro"

    def test_format_response_with_citations(self, mock_config):
        """Test formatting response with citations."""
        with patch("aieng.agent_evals.knowledge_agent.grounding_tool.genai.Client"):
            tool = GeminiGroundingTool(config=mock_config)

        response = GroundedResponse(
            text="The answer is 42.",
            search_queries=["meaning of life"],
            sources=[
                GroundingChunk(
                    title="Wikipedia", uri="https://en.wikipedia.org/wiki/42"
                ),
                GroundingChunk(title="Guide", uri="https://example.com/guide"),
            ],
        )

        formatted = tool.format_response_with_citations(response)

        assert "The answer is 42." in formatted
        assert "**Sources:**" in formatted
        assert "[Wikipedia](https://en.wikipedia.org/wiki/42)" in formatted
        assert "[Guide](https://example.com/guide)" in formatted

    def test_format_response_without_sources(self, mock_config):
        """Test formatting response without sources."""
        with patch("aieng.agent_evals.knowledge_agent.grounding_tool.genai.Client"):
            tool = GeminiGroundingTool(config=mock_config)

        response = GroundedResponse(text="Simple answer.")

        formatted = tool.format_response_with_citations(response)

        assert formatted == "Simple answer."
        assert "Sources" not in formatted


@pytest.mark.integration_test
class TestGeminiGroundingToolIntegration:
    """Integration tests for the GeminiGroundingTool.

    These tests require a valid GOOGLE_API_KEY environment variable.
    """

    def test_search_real_query(self):
        """Test a real search query."""
        from aieng.agent_evals.knowledge_agent import (  # noqa: PLC0415
            GeminiGroundingTool,
        )

        tool = GeminiGroundingTool()
        response = tool.search("What is the capital of France?")

        assert response.text
        assert "Paris" in response.text
        # May or may not have search queries depending on model decision

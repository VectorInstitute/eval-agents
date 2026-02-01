"""Tests for the Knowledge-Grounded QA Agent."""

from unittest.mock import MagicMock, patch

import pytest
from aieng.agent_evals.knowledge_agent.agent import (
    SYSTEM_INSTRUCTIONS,
    EnhancedGroundedResponse,
    KnowledgeAgentManager,
    KnowledgeGroundedAgent,
)
from aieng.agent_evals.knowledge_agent.planner import ResearchPlan
from aieng.agent_evals.tools import GroundingChunk


class TestKnowledgeGroundedAgent:
    """Tests for the KnowledgeGroundedAgent class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = MagicMock()
        config.openai_api_key = "test-api-key"
        config.default_worker_model = "gemini-2.5-flash"
        config.default_planner_model = "gemini-2.5-pro"
        return config

    @patch("aieng.agent_evals.knowledge_agent.agent.ResearchPlanner")
    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_read_file_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_grep_file_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_fetch_file_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_web_fetch_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    def test_agent_initialization(
        self,
        mock_create_search_tool,
        mock_create_web_fetch_tool,
        mock_create_fetch_file_tool,
        mock_create_grep_file_tool,
        mock_create_read_file_tool,
        mock_agent_class,
        mock_session_service,
        mock_runner_class,
        mock_planner_class,
        mock_config,
    ):
        """Test initializing the agent with all tools."""
        mock_search_tool = MagicMock()
        mock_web_fetch_tool = MagicMock()
        mock_create_search_tool.return_value = mock_search_tool
        mock_create_web_fetch_tool.return_value = mock_web_fetch_tool

        agent = KnowledgeGroundedAgent(config=mock_config, enable_caching=False, enable_compaction=False)

        # Verify all tools were created
        mock_create_search_tool.assert_called_once()
        mock_create_web_fetch_tool.assert_called_once()
        mock_create_fetch_file_tool.assert_called_once()
        mock_create_grep_file_tool.assert_called_once()
        mock_create_read_file_tool.assert_called_once()

        # Verify ADK Agent was created with correct params
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args[1]
        assert call_kwargs["name"] == "enhanced_knowledge_agent"
        assert call_kwargs["instruction"] == SYSTEM_INSTRUCTIONS
        assert mock_search_tool in call_kwargs["tools"]
        assert mock_web_fetch_tool in call_kwargs["tools"]

        # Verify planner was created (planning enabled by default)
        mock_planner_class.assert_called_once()
        assert agent.enable_planning is True

    @patch("aieng.agent_evals.knowledge_agent.agent.ResearchPlanner")
    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_read_file_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_grep_file_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_fetch_file_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_web_fetch_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    def test_agent_without_planning(
        self,
        mock_create_search_tool,
        mock_create_web_fetch_tool,
        mock_create_fetch_file_tool,
        mock_create_grep_file_tool,
        mock_create_read_file_tool,
        mock_agent_class,
        mock_session_service,
        mock_runner_class,
        mock_planner_class,
        mock_config,
    ):
        """Test initializing the agent without planning."""
        agent = KnowledgeGroundedAgent(
            config=mock_config, enable_planning=False, enable_caching=False, enable_compaction=False
        )

        # Planner should not be created
        mock_planner_class.assert_not_called()
        assert agent.enable_planning is False
        assert agent._planner is None

    @patch("aieng.agent_evals.knowledge_agent.agent.ResearchPlanner")
    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_read_file_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_grep_file_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_fetch_file_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_web_fetch_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    def test_agent_with_custom_model(
        self,
        mock_create_search_tool,
        mock_create_web_fetch_tool,
        mock_create_fetch_file_tool,
        mock_create_grep_file_tool,
        mock_create_read_file_tool,
        mock_agent_class,
        mock_session_service,
        mock_runner_class,
        mock_planner_class,
        mock_config,
    ):
        """Test initializing with a custom model."""
        agent = KnowledgeGroundedAgent(
            config=mock_config, model="gemini-2.5-pro", enable_caching=False, enable_compaction=False
        )

        call_kwargs = mock_agent_class.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-pro"
        assert agent.model == "gemini-2.5-pro"


class TestKnowledgeAgentManager:
    """Tests for the KnowledgeAgentManager class."""

    @patch("aieng.agent_evals.knowledge_agent.agent.ResearchPlanner")
    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_read_file_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_grep_file_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_fetch_file_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_web_fetch_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    def test_lazy_initialization(self, *mocks):
        """Test that agent is lazily initialized."""
        with patch("aieng.agent_evals.knowledge_agent.agent.Configs") as mock_config_class:
            mock_config = MagicMock()
            mock_config.default_worker_model = "gemini-2.5-flash"
            mock_config.default_planner_model = "gemini-2.5-pro"
            mock_config_class.return_value = mock_config

            manager = KnowledgeAgentManager(enable_caching=False, enable_compaction=False)

            # Should not be initialized yet
            assert not manager.is_initialized()

            # Access agent to trigger initialization
            _ = manager.agent

            # Now should be initialized
            assert manager.is_initialized()

    @patch("aieng.agent_evals.knowledge_agent.agent.ResearchPlanner")
    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_read_file_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_grep_file_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_fetch_file_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_web_fetch_tool")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    def test_close(self, *mocks):
        """Test closing the client manager."""
        with patch("aieng.agent_evals.knowledge_agent.agent.Configs") as mock_config_class:
            mock_config = MagicMock()
            mock_config.default_worker_model = "gemini-2.5-flash"
            mock_config.default_planner_model = "gemini-2.5-pro"
            mock_config_class.return_value = mock_config

            manager = KnowledgeAgentManager(enable_caching=False, enable_compaction=False)
            _ = manager.agent
            assert manager.is_initialized()

            manager.close()
            assert not manager.is_initialized()


class TestEnhancedGroundedResponse:
    """Tests for the EnhancedGroundedResponse model."""

    def test_response_creation(self):
        """Test creating an enhanced response."""
        plan = ResearchPlan(
            original_question="Test question",
            complexity_assessment="simple",
            steps=[],
            reasoning="Test reasoning",
        )

        response = EnhancedGroundedResponse(
            text="Test answer.",
            plan=plan,
            sources=[GroundingChunk(title="Source", uri="https://example.com")],
            search_queries=["test query"],
            reasoning_chain=["Step 1"],
            tool_calls=[{"name": "google_search", "args": {"query": "test"}}],
            total_duration_ms=1000,
        )

        assert response.text == "Test answer."
        assert response.plan.complexity_assessment == "simple"
        assert len(response.sources) == 1
        assert response.sources[0].uri == "https://example.com"
        assert response.search_queries == ["test query"]
        assert response.total_duration_ms == 1000


@pytest.mark.integration_test
class TestKnowledgeGroundedAgentIntegration:
    """Integration tests for the KnowledgeGroundedAgent.

    These tests require a valid GOOGLE_API_KEY environment variable.
    """

    def test_agent_creation_real(self):
        """Test creating a real agent instance."""
        agent = KnowledgeGroundedAgent()
        assert agent is not None
        assert agent.model == "gemini-2.5-flash"
        assert agent.enable_planning is True

    @pytest.mark.asyncio
    async def test_answer_real_question(self):
        """Test answering a real question."""
        agent = KnowledgeGroundedAgent()
        response = await agent.answer_async("What is the capital of France?")

        assert response.text
        assert "Paris" in response.text
        assert isinstance(response, EnhancedGroundedResponse)

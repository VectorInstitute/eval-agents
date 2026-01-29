"""Tests for the Knowledge-Grounded QA Agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aieng.agent_evals.knowledge_agent.agent import (
    SYSTEM_INSTRUCTIONS,
    KnowledgeAgentManager,
    KnowledgeGroundedAgent,
)
from aieng.agent_evals.knowledge_agent.grounding_tool import GroundedResponse


class TestKnowledgeGroundedAgent:
    """Tests for the KnowledgeGroundedAgent class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = MagicMock()
        config.openai_api_key = "test-api-key"
        config.default_worker_model = "gemini-2.5-flash"
        return config

    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    def test_agent_initialization(
        self,
        mock_create_tool,
        mock_agent_class,
        mock_session_service,
        mock_runner_class,
        mock_config,
    ):
        """Test initializing the agent."""
        mock_tool = MagicMock()
        mock_create_tool.return_value = mock_tool

        KnowledgeGroundedAgent(config=mock_config)

        # Verify tool was created
        mock_create_tool.assert_called_once()

        # Verify ADK Agent was created with correct params
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args[1]
        assert call_kwargs["name"] == "knowledge_qa_agent"
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["instruction"] == SYSTEM_INSTRUCTIONS
        assert mock_tool in call_kwargs["tools"]

        # Verify session service and runner were created
        mock_session_service.assert_called_once()
        mock_runner_class.assert_called_once()

    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    def test_agent_with_custom_model(
        self,
        mock_create_tool,
        mock_agent_class,
        mock_session_service,
        mock_runner_class,
        mock_config,
    ):
        """Test initializing with a custom model."""
        KnowledgeGroundedAgent(config=mock_config, model="gemini-2.5-pro")

        call_kwargs = mock_agent_class.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-pro"

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    async def test_get_or_create_session(
        self,
        mock_create_tool,
        mock_agent_class,
        mock_session_service_class,
        mock_runner_class,
        mock_config,
    ):
        """Test session creation and retrieval."""
        # Mock the session service's create_session method
        mock_session = MagicMock()
        mock_session.id = "mock-session-id-1"
        mock_session_service = MagicMock()
        mock_session_service.create_session = AsyncMock(return_value=mock_session)
        mock_session_service_class.return_value = mock_session_service

        agent = KnowledgeGroundedAgent(config=mock_config)

        # Create a new session
        session1 = await agent._get_or_create_session_async("test-session-1")
        assert session1 is not None

        # Same session ID should return same ADK session (cached)
        session2 = await agent._get_or_create_session_async("test-session-1")
        assert session1 == session2

        # Different session ID should create new session
        mock_session.id = "mock-session-id-2"
        session3 = await agent._get_or_create_session_async("test-session-2")
        assert session3 != session1

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    async def test_get_or_create_session_generates_id(
        self,
        mock_create_tool,
        mock_agent_class,
        mock_session_service_class,
        mock_runner_class,
        mock_config,
    ):
        """Test that session ID is generated if not provided."""
        # Mock the session service's create_session method
        mock_session = MagicMock()
        mock_session.id = "mock-session-id"
        mock_session_service = MagicMock()
        mock_session_service.create_session = AsyncMock(return_value=mock_session)
        mock_session_service_class.return_value = mock_session_service

        agent = KnowledgeGroundedAgent(config=mock_config)

        session = await agent._get_or_create_session_async(None)
        assert session is not None

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    async def test_answer_async(
        self,
        mock_create_tool,
        mock_agent_class,
        mock_session_service_class,
        mock_runner_class,
        mock_config,
    ):
        """Test async answer method."""
        # Mock the session service's create_session method
        mock_session = MagicMock()
        mock_session.id = "mock-session-id"
        mock_session_service = MagicMock()
        mock_session_service.create_session = AsyncMock(return_value=mock_session)
        mock_session_service_class.return_value = mock_session_service

        # Create mock event with final response
        mock_event = MagicMock()
        mock_event.is_final_response.return_value = True
        mock_event.content.parts = [MagicMock(text="Paris is the capital of France.")]

        # Make runner.run_async return an async generator
        async def mock_run_async(*args, **kwargs):
            yield mock_event

        mock_runner = MagicMock()
        mock_runner.run_async = mock_run_async
        mock_runner_class.return_value = mock_runner

        agent = KnowledgeGroundedAgent(config=mock_config)
        response = await agent.answer_async("What is the capital of France?")

        assert isinstance(response, GroundedResponse)
        assert response.text == "Paris is the capital of France."

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    async def test_answer_async_extracts_function_calls(
        self,
        mock_create_tool,
        mock_agent_class,
        mock_session_service_class,
        mock_runner_class,
        mock_config,
    ):
        """Test that function calls are extracted from events."""
        # Mock session service
        mock_session = MagicMock()
        mock_session.id = "mock-session-id"
        mock_session_service = MagicMock()
        mock_session_service.create_session = AsyncMock(return_value=mock_session)
        mock_session_service_class.return_value = mock_session_service

        # Create mock function call
        mock_function_call = MagicMock()
        mock_function_call.name = "google_search"
        mock_function_call.args = {"query": "capital of France"}

        # Create mock event with function call
        mock_tool_event = MagicMock()
        mock_tool_event.is_final_response.return_value = False
        mock_tool_event.get_function_calls.return_value = [mock_function_call]
        mock_tool_event.get_function_responses.return_value = None
        mock_tool_event.grounding_metadata = None
        mock_tool_event.content = None

        # Create mock final event
        mock_final_event = MagicMock()
        mock_final_event.is_final_response.return_value = True
        mock_final_event.get_function_calls.return_value = None
        mock_final_event.get_function_responses.return_value = None
        mock_final_event.grounding_metadata = None
        mock_final_event.content.parts = [MagicMock(text="Paris is the capital.")]

        async def mock_run_async(*args, **kwargs):
            yield mock_tool_event
            yield mock_final_event

        mock_runner = MagicMock()
        mock_runner.run_async = mock_run_async
        mock_runner_class.return_value = mock_runner

        agent = KnowledgeGroundedAgent(config=mock_config)
        response = await agent.answer_async("What is the capital of France?")

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "google_search"
        assert response.tool_calls[0]["args"] == {"query": "capital of France"}
        assert "capital of France" in response.search_queries

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    async def test_answer_async_extracts_sources_from_function_responses(
        self,
        mock_create_tool,
        mock_agent_class,
        mock_session_service_class,
        mock_runner_class,
        mock_config,
    ):
        """Test that sources are extracted from function responses."""
        # Mock session service
        mock_session = MagicMock()
        mock_session.id = "mock-session-id"
        mock_session_service = MagicMock()
        mock_session_service.create_session = AsyncMock(return_value=mock_session)
        mock_session_service_class.return_value = mock_session_service

        # Create mock function response with sources
        mock_function_response = MagicMock()
        mock_function_response.response = {
            "sources": [
                {"title": "Wikipedia - Paris", "uri": "https://en.wikipedia.org/wiki/Paris"},
                {"title": "Travel Guide", "url": "https://example.com/paris"},
            ]
        }

        # Create mock event with function response
        mock_response_event = MagicMock()
        mock_response_event.is_final_response.return_value = False
        mock_response_event.get_function_calls.return_value = None
        mock_response_event.get_function_responses.return_value = [mock_function_response]
        mock_response_event.grounding_metadata = None
        mock_response_event.content = None

        # Create mock final event
        mock_final_event = MagicMock()
        mock_final_event.is_final_response.return_value = True
        mock_final_event.get_function_calls.return_value = None
        mock_final_event.get_function_responses.return_value = None
        mock_final_event.grounding_metadata = None
        mock_final_event.content.parts = [MagicMock(text="Paris is the capital.")]

        async def mock_run_async(*args, **kwargs):
            yield mock_response_event
            yield mock_final_event

        mock_runner = MagicMock()
        mock_runner.run_async = mock_run_async
        mock_runner_class.return_value = mock_runner

        agent = KnowledgeGroundedAgent(config=mock_config)
        response = await agent.answer_async("What is the capital of France?")

        assert len(response.sources) == 2
        assert response.sources[0].title == "Wikipedia - Paris"
        assert response.sources[0].uri == "https://en.wikipedia.org/wiki/Paris"
        assert response.sources[1].title == "Travel Guide"
        assert response.sources[1].uri == "https://example.com/paris"

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    async def test_answer_async_extracts_grounding_chunks_from_responses(
        self,
        mock_create_tool,
        mock_agent_class,
        mock_session_service_class,
        mock_runner_class,
        mock_config,
    ):
        """Test that grounding_chunks are extracted from function responses."""
        # Mock session service
        mock_session = MagicMock()
        mock_session.id = "mock-session-id"
        mock_session_service = MagicMock()
        mock_session_service.create_session = AsyncMock(return_value=mock_session)
        mock_session_service_class.return_value = mock_session_service

        # Create mock function response with grounding_chunks
        mock_function_response = MagicMock()
        mock_function_response.response = {
            "grounding_chunks": [
                {"web": {"title": "Official Site", "uri": "https://official.com"}},
                {"web": {"title": "News Article", "uri": "https://news.com/article"}},
            ]
        }

        # Create mock event with function response
        mock_response_event = MagicMock()
        mock_response_event.is_final_response.return_value = False
        mock_response_event.get_function_calls.return_value = None
        mock_response_event.get_function_responses.return_value = [mock_function_response]
        mock_response_event.grounding_metadata = None
        mock_response_event.content = None

        # Create mock final event
        mock_final_event = MagicMock()
        mock_final_event.is_final_response.return_value = True
        mock_final_event.get_function_calls.return_value = None
        mock_final_event.get_function_responses.return_value = None
        mock_final_event.grounding_metadata = None
        mock_final_event.content.parts = [MagicMock(text="Answer.")]

        async def mock_run_async(*args, **kwargs):
            yield mock_response_event
            yield mock_final_event

        mock_runner = MagicMock()
        mock_runner.run_async = mock_run_async
        mock_runner_class.return_value = mock_runner

        agent = KnowledgeGroundedAgent(config=mock_config)
        response = await agent.answer_async("Test question")

        assert len(response.sources) == 2
        assert response.sources[0].title == "Official Site"
        assert response.sources[0].uri == "https://official.com"
        assert response.sources[1].title == "News Article"
        assert response.sources[1].uri == "https://news.com/article"

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    async def test_answer_async_extracts_grounding_metadata(
        self,
        mock_create_tool,
        mock_agent_class,
        mock_session_service_class,
        mock_runner_class,
        mock_config,
    ):
        """Test that grounding metadata is extracted from events."""
        # Mock session service
        mock_session = MagicMock()
        mock_session.id = "mock-session-id"
        mock_session_service = MagicMock()
        mock_session_service.create_session = AsyncMock(return_value=mock_session)
        mock_session_service_class.return_value = mock_session_service

        # Create mock grounding chunk
        mock_web_chunk = MagicMock()
        mock_web_chunk.title = "Grounded Source"
        mock_web_chunk.uri = "https://grounded.com"

        mock_grounding_chunk = MagicMock()
        mock_grounding_chunk.web = mock_web_chunk

        # Create mock grounding metadata
        mock_grounding_metadata = MagicMock()
        mock_grounding_metadata.grounding_chunks = [mock_grounding_chunk]
        mock_grounding_metadata.web_search_queries = ["grounded query"]

        # Create mock event with grounding metadata
        mock_grounding_event = MagicMock()
        mock_grounding_event.is_final_response.return_value = False
        mock_grounding_event.get_function_calls.return_value = None
        mock_grounding_event.get_function_responses.return_value = None
        mock_grounding_event.grounding_metadata = mock_grounding_metadata
        mock_grounding_event.content = None

        # Create mock final event
        mock_final_event = MagicMock()
        mock_final_event.is_final_response.return_value = True
        mock_final_event.get_function_calls.return_value = None
        mock_final_event.get_function_responses.return_value = None
        mock_final_event.grounding_metadata = None
        mock_final_event.content.parts = [MagicMock(text="Final answer.")]

        async def mock_run_async(*args, **kwargs):
            yield mock_grounding_event
            yield mock_final_event

        mock_runner = MagicMock()
        mock_runner.run_async = mock_run_async
        mock_runner_class.return_value = mock_runner

        agent = KnowledgeGroundedAgent(config=mock_config)
        response = await agent.answer_async("Test question")

        assert len(response.sources) == 1
        assert response.sources[0].title == "Grounded Source"
        assert response.sources[0].uri == "https://grounded.com"
        assert "grounded query" in response.search_queries

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    async def test_answer_async_extracts_grounding_metadata_from_content(
        self,
        mock_create_tool,
        mock_agent_class,
        mock_session_service_class,
        mock_runner_class,
        mock_config,
    ):
        """Test grounding metadata extraction from event.content."""
        # Mock session service
        mock_session = MagicMock()
        mock_session.id = "mock-session-id"
        mock_session_service = MagicMock()
        mock_session_service.create_session = AsyncMock(return_value=mock_session)
        mock_session_service_class.return_value = mock_session_service

        # Create mock grounding chunk on content
        mock_web_chunk = MagicMock()
        mock_web_chunk.title = "Content Source"
        mock_web_chunk.uri = "https://content-source.com"

        mock_grounding_chunk = MagicMock()
        mock_grounding_chunk.web = mock_web_chunk

        mock_grounding_metadata = MagicMock()
        mock_grounding_metadata.grounding_chunks = [mock_grounding_chunk]
        mock_grounding_metadata.web_search_queries = ["content query"]

        # Create mock event with grounding metadata on content (not event directly)
        mock_content = MagicMock()
        mock_content.grounding_metadata = mock_grounding_metadata

        mock_event = MagicMock()
        mock_event.is_final_response.return_value = False
        mock_event.get_function_calls.return_value = None
        mock_event.get_function_responses.return_value = None
        mock_event.grounding_metadata = None  # Not on event directly
        mock_event.content = mock_content

        # Create mock final event
        mock_final_event = MagicMock()
        mock_final_event.is_final_response.return_value = True
        mock_final_event.get_function_calls.return_value = None
        mock_final_event.get_function_responses.return_value = None
        mock_final_event.grounding_metadata = None
        mock_final_event.content.parts = [MagicMock(text="Answer.")]
        mock_final_event.content.grounding_metadata = None

        async def mock_run_async(*args, **kwargs):
            yield mock_event
            yield mock_final_event

        mock_runner = MagicMock()
        mock_runner.run_async = mock_run_async
        mock_runner_class.return_value = mock_runner

        agent = KnowledgeGroundedAgent(config=mock_config)
        response = await agent.answer_async("Test question")

        assert len(response.sources) == 1
        assert response.sources[0].title == "Content Source"
        assert response.sources[0].uri == "https://content-source.com"
        assert "content query" in response.search_queries

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    async def test_answer_async_handles_multiple_search_tool_names(
        self,
        mock_create_tool,
        mock_agent_class,
        mock_session_service_class,
        mock_runner_class,
        mock_config,
    ):
        """Test that search queries are extracted from various search tool names."""
        # Mock session service
        mock_session = MagicMock()
        mock_session.id = "mock-session-id"
        mock_session_service = MagicMock()
        mock_session_service.create_session = AsyncMock(return_value=mock_session)
        mock_session_service_class.return_value = mock_session_service

        # Create mock function calls with different search tool names
        mock_fc1 = MagicMock()
        mock_fc1.name = "google_search"
        mock_fc1.args = {"query": "query one"}

        mock_fc2 = MagicMock()
        mock_fc2.name = "web_search"
        mock_fc2.args = {"query": "query two"}

        mock_fc3 = MagicMock()
        mock_fc3.name = "SearchTool"
        mock_fc3.args = {"query": "query three"}

        # Event with all function calls
        mock_tool_event = MagicMock()
        mock_tool_event.is_final_response.return_value = False
        mock_tool_event.get_function_calls.return_value = [mock_fc1, mock_fc2, mock_fc3]
        mock_tool_event.get_function_responses.return_value = None
        mock_tool_event.grounding_metadata = None
        mock_tool_event.content = None

        # Final event
        mock_final_event = MagicMock()
        mock_final_event.is_final_response.return_value = True
        mock_final_event.get_function_calls.return_value = None
        mock_final_event.get_function_responses.return_value = None
        mock_final_event.grounding_metadata = None
        mock_final_event.content.parts = [MagicMock(text="Done.")]

        async def mock_run_async(*args, **kwargs):
            yield mock_tool_event
            yield mock_final_event

        mock_runner = MagicMock()
        mock_runner.run_async = mock_run_async
        mock_runner_class.return_value = mock_runner

        agent = KnowledgeGroundedAgent(config=mock_config)
        response = await agent.answer_async("Test")

        assert len(response.tool_calls) == 3
        assert "query one" in response.search_queries
        assert "query two" in response.search_queries
        assert "query three" in response.search_queries

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    async def test_answer_async_handles_empty_events(
        self,
        mock_create_tool,
        mock_agent_class,
        mock_session_service_class,
        mock_runner_class,
        mock_config,
    ):
        """Test that empty events are handled gracefully."""
        # Mock session service
        mock_session = MagicMock()
        mock_session.id = "mock-session-id"
        mock_session_service = MagicMock()
        mock_session_service.create_session = AsyncMock(return_value=mock_session)
        mock_session_service_class.return_value = mock_session_service

        # Create events with no data
        mock_empty_event = MagicMock()
        mock_empty_event.is_final_response.return_value = False
        mock_empty_event.get_function_calls.return_value = []
        mock_empty_event.get_function_responses.return_value = []
        mock_empty_event.grounding_metadata = None
        mock_empty_event.content = None

        # Final event
        mock_final_event = MagicMock()
        mock_final_event.is_final_response.return_value = True
        mock_final_event.get_function_calls.return_value = None
        mock_final_event.get_function_responses.return_value = None
        mock_final_event.grounding_metadata = None
        mock_final_event.content.parts = [MagicMock(text="Final.")]

        async def mock_run_async(*args, **kwargs):
            yield mock_empty_event
            yield mock_final_event

        mock_runner = MagicMock()
        mock_runner.run_async = mock_run_async
        mock_runner_class.return_value = mock_runner

        agent = KnowledgeGroundedAgent(config=mock_config)
        response = await agent.answer_async("Test")

        assert isinstance(response, GroundedResponse)
        assert response.text == "Final."
        assert response.tool_calls == []
        assert response.search_queries == []
        assert response.sources == []

    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    def test_format_answer(
        self,
        mock_create_tool,
        mock_agent_class,
        mock_session_service,
        mock_runner_class,
        mock_config,
    ):
        """Test format_answer method."""
        agent = KnowledgeGroundedAgent(config=mock_config)

        response = GroundedResponse(
            text="Test answer.",
            sources=[],
        )

        formatted = agent.format_answer(response)
        assert "Test answer." in formatted


class TestKnowledgeAgentManager:
    """Tests for the KnowledgeAgentManager class."""

    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    def test_lazy_initialization(
        self,
        mock_create_tool,
        mock_agent_class,
        mock_session_service,
        mock_runner_class,
    ):
        """Test that clients are lazily initialized."""
        with patch("aieng.agent_evals.knowledge_agent.agent.KnowledgeAgentConfig") as mock_config_class:
            mock_config_class.return_value = MagicMock()

            manager = KnowledgeAgentManager()

            # Should not be initialized yet
            assert not manager.is_initialized()

            # Access agent to trigger initialization
            _ = manager.agent

            # Now should be initialized
            assert manager.is_initialized()

    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    def test_close(
        self,
        mock_create_tool,
        mock_agent_class,
        mock_session_service,
        mock_runner_class,
    ):
        """Test closing the client manager."""
        with patch("aieng.agent_evals.knowledge_agent.agent.KnowledgeAgentConfig") as mock_config_class:
            mock_config_class.return_value = MagicMock()

            manager = KnowledgeAgentManager()
            _ = manager.agent
            assert manager.is_initialized()

            manager.close()
            assert not manager.is_initialized()

    @patch("aieng.agent_evals.knowledge_agent.agent.Runner")
    @patch("aieng.agent_evals.knowledge_agent.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_agent.agent.Agent")
    @patch("aieng.agent_evals.knowledge_agent.agent.create_google_search_tool")
    def test_agent_reuse(
        self,
        mock_create_tool,
        mock_agent_class,
        mock_session_service,
        mock_runner_class,
    ):
        """Test that agent is reused on multiple accesses."""
        with patch("aieng.agent_evals.knowledge_agent.agent.KnowledgeAgentConfig") as mock_config_class:
            mock_config_class.return_value = MagicMock()

            manager = KnowledgeAgentManager()

            agent1 = manager.agent
            agent2 = manager.agent

            assert agent1 is agent2


@pytest.mark.integration_test
class TestKnowledgeGroundedAgentIntegration:
    """Integration tests for the KnowledgeGroundedAgent.

    These tests require a valid GOOGLE_API_KEY environment variable.
    """

    def test_agent_creation_real(self):
        """Test creating a real agent instance."""
        from aieng.agent_evals.knowledge_agent import (  # noqa: PLC0415
            KnowledgeGroundedAgent,
        )

        agent = KnowledgeGroundedAgent()
        assert agent is not None
        assert agent.model == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_answer_real_question(self):
        """Test answering a real question."""
        from aieng.agent_evals.knowledge_agent import (  # noqa: PLC0415
            KnowledgeGroundedAgent,
        )

        agent = KnowledgeGroundedAgent()
        response = await agent.answer_async("What is the capital of France?")

        assert response.text
        assert "Paris" in response.text

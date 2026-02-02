"""Knowledge-grounded QA agent using Google ADK with Google Search.

This module provides a proper ReAct agent that explicitly calls
Google Search and shows the reasoning process through observable tool calls.
"""

import asyncio
import logging
import uuid
from typing import Any

from aieng.agent_evals.configs import Configs
from aieng.agent_evals.tools import (
    GroundedResponse,
    GroundingChunk,
    create_google_search_tool,
)
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


logger = logging.getLogger(__name__)


def _extract_tool_calls(event: Any) -> list[dict[str, Any]]:
    """Extract tool calls from event function calls.

    Parameters
    ----------
    event : Any
        An event from the ADK runner.

    Returns
    -------
    list[dict[str, Any]]
        List of tool call dictionaries with 'name' and 'args' keys.
    """
    if not hasattr(event, "get_function_calls"):
        return []
    function_calls = event.get_function_calls()
    if not function_calls:
        return []

    tool_calls = []
    for fc in function_calls:
        tool_call_info = {
            "name": getattr(fc, "name", "unknown"),
            "args": getattr(fc, "args", {}),
        }
        tool_calls.append(tool_call_info)
        logger.info(f"Tool call: {tool_call_info['name']}({tool_call_info['args']})")
    return tool_calls


def _extract_search_queries_from_tool_calls(tool_calls: list[dict[str, Any]]) -> list[str]:
    """Extract search queries from tool calls.

    Parameters
    ----------
    tool_calls : list[dict[str, Any]]
        List of tool call dictionaries.

    Returns
    -------
    list[str]
        Search queries found in the tool calls.
    """
    queries = []
    for tool_call in tool_calls:
        tool_name = str(tool_call.get("name", ""))
        tool_args = tool_call.get("args", {})
        if "search" in tool_name.lower() and isinstance(tool_args, dict):
            query = tool_args.get("query", "")
            if query:
                queries.append(query)
    return queries


def _extract_sources_from_responses(event: Any) -> list[GroundingChunk]:
    """Extract sources from event function responses.

    Parameters
    ----------
    event : Any
        An event from the ADK runner.

    Returns
    -------
    list[GroundingChunk]
        Sources extracted from the function responses.
    """
    if not hasattr(event, "get_function_responses"):
        return []
    function_responses = event.get_function_responses()
    if not function_responses:
        return []

    sources = []
    for fr in function_responses:
        response_data = getattr(fr, "response", {})
        if not isinstance(response_data, dict):
            continue
        # Extract sources from search tool response
        for src in response_data.get("sources", []):
            if isinstance(src, dict):
                sources.append(
                    GroundingChunk(
                        title=src.get("title", ""),
                        uri=src.get("uri") or src.get("url") or "",
                    )
                )
        # Extract grounding_chunks if present
        for chunk in response_data.get("grounding_chunks", []):
            if isinstance(chunk, dict) and "web" in chunk:
                sources.append(
                    GroundingChunk(
                        title=chunk["web"].get("title", ""),
                        uri=chunk["web"].get("uri", ""),
                    )
                )
    return sources


def _extract_grounding_sources(event: Any) -> list[GroundingChunk]:
    """Extract sources from grounding metadata.

    Parameters
    ----------
    event : Any
        An event from the ADK runner.

    Returns
    -------
    list[GroundingChunk]
        Sources extracted from the grounding metadata.
    """
    gm = getattr(event, "grounding_metadata", None)
    if not gm and hasattr(event, "content") and event.content:
        gm = getattr(event.content, "grounding_metadata", None)
    if not gm:
        return []

    sources = []
    if hasattr(gm, "grounding_chunks") and gm.grounding_chunks:
        for chunk in gm.grounding_chunks:
            if hasattr(chunk, "web") and chunk.web:
                sources.append(
                    GroundingChunk(
                        title=getattr(chunk.web, "title", "") or "",
                        uri=getattr(chunk.web, "uri", "") or "",
                    )
                )
    return sources


def _extract_grounding_queries(event: Any) -> list[str]:
    """Extract search queries from grounding metadata.

    Parameters
    ----------
    event : Any
        An event from the ADK runner.

    Returns
    -------
    list[str]
        Search queries from the grounding metadata.
    """
    gm = getattr(event, "grounding_metadata", None)
    if not gm and hasattr(event, "content") and event.content:
        gm = getattr(event.content, "grounding_metadata", None)
    if not gm:
        return []

    queries = []
    if hasattr(gm, "web_search_queries") and gm.web_search_queries:
        for q in gm.web_search_queries:
            if q:
                queries.append(q)
    return queries


def _extract_final_response(event: Any) -> str | None:
    """Extract final response text from event if it's a final response."""
    if not hasattr(event, "is_final_response") or not event.is_final_response():
        return None
    if not hasattr(event, "content") or not event.content:
        return None
    if not hasattr(event.content, "parts") or not event.content.parts:
        return None
    return event.content.parts[0].text or ""


SYSTEM_INSTRUCTIONS = """\
You are a knowledge-grounded research assistant. Your role is to answer
questions accurately by searching the web for relevant information.

## How to Answer Questions

1. **Search First**: Always search the web before answering factual questions
   that require current information. Do not rely solely on your training data.

2. **Be Thorough**: For complex questions, search multiple times to gather
   all relevant facts before synthesizing your answer.

3. **Cite Sources**: Always mention which sources you used to answer the question.

4. **Be Honest**: If you cannot find relevant information, say so clearly.

5. **Synthesize Information**: When answering complex questions, synthesize
   findings from multiple sources into a coherent response.

## Response Format

When answering questions:
- Provide a clear, direct answer first
- Include relevant context and details from your sources
- List the sources used at the end of your response
"""


class KnowledgeGroundedAgent:
    """A ReAct agent for knowledge-grounded QA using Google Search.

    This agent uses Google ADK with explicit Google Search tool calls,
    making the reasoning process observable and traceable.

    Parameters
    ----------
    config : Configs, optional
        Configuration settings. If not provided, creates default config.
    model : str, optional
        The model to use. If not provided, uses config.default_worker_model.

    Attributes
    ----------
    config : Configs
        The configuration settings.

    Examples
    --------
    >>> from aieng.agent_evals.knowledge_agent import KnowledgeGroundedAgent
    >>> agent = KnowledgeGroundedAgent()
    >>> response = agent.answer("Who won the 2024 Nobel Prize in Physics?")
    >>> print(response.text)
    """

    def __init__(
        self,
        config: Configs | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the knowledge-grounded agent.

        Parameters
        ----------
        config : Configs, optional
            Configuration settings. If not provided, creates default config.
        model : str, optional
            The model to use. If not provided, uses config.default_worker_model.
        """
        if config is None:
            config = Configs()  # type: ignore[call-arg]

        self.config = config
        self.model = model or config.default_worker_model

        # Create the Google Search tool
        self._search_tool = create_google_search_tool()

        # Create ADK agent with Google Search tool
        self._agent = Agent(
            name="knowledge_qa_agent",
            model=self.model,
            instruction=SYSTEM_INSTRUCTIONS,
            tools=[self._search_tool],
        )

        # Session service for conversation history
        self._session_service = InMemorySessionService()

        # Runner orchestrates the ReAct loop
        self._runner = Runner(
            app_name="knowledge_agent",
            agent=self._agent,
            session_service=self._session_service,
        )

        # Track active sessions
        self._sessions: dict[str, str] = {}  # Maps external session_id to ADK session_id

    async def _get_or_create_session_async(self, session_id: str | None = None) -> str:
        """Get or create an ADK session for the given session ID.

        Parameters
        ----------
        session_id : str, optional
            External session ID. If not provided, generates a new one.

        Returns
        -------
        str
            The ADK session ID.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id not in self._sessions:
            # Create a new ADK session through the session service
            session = await self._session_service.create_session(
                app_name="knowledge_agent",
                user_id="user",
                state={},
            )
            self._sessions[session_id] = session.id

        return self._sessions[session_id]

    async def answer_async(
        self,
        question: str,
        session_id: str | None = None,
    ) -> GroundedResponse:
        """Answer a question using the ReAct loop asynchronously.

        Parameters
        ----------
        question : str
            The question to answer.
        session_id : str, optional
            Session ID for multi-turn conversations.

        Returns
        -------
        GroundedResponse
            The response with text, tool calls, and sources.
        """
        logger.info(f"Answering question (async): {question[:100]}...")

        adk_session_id = await self._get_or_create_session_async(session_id)

        # Create the user message
        content = types.Content(
            role="user",
            parts=[types.Part(text=question)],
        )

        # Collect events from the ReAct loop
        tool_calls: list[dict[str, Any]] = []
        sources: list[GroundingChunk] = []
        search_queries: list[str] = []
        final_response = ""

        async for event in self._runner.run_async(
            user_id="user",
            session_id=adk_session_id,
            new_message=content,
        ):
            logger.debug(f"Event: {event}")

            # Extract tool calls and search queries from function calls
            new_tool_calls = _extract_tool_calls(event)
            tool_calls.extend(new_tool_calls)
            search_queries.extend(_extract_search_queries_from_tool_calls(new_tool_calls))

            # Extract sources from function responses
            sources.extend(_extract_sources_from_responses(event))

            # Extract sources and queries from grounding metadata
            sources.extend(_extract_grounding_sources(event))
            for q in _extract_grounding_queries(event):
                if q not in search_queries:
                    search_queries.append(q)

            text = _extract_final_response(event)
            if text is not None:
                final_response = text

        return GroundedResponse(
            text=final_response,
            search_queries=search_queries,
            sources=sources,
            tool_calls=tool_calls,
        )

    def answer(
        self,
        question: str,
        session_id: str | None = None,
    ) -> GroundedResponse:
        """Answer a question using the ReAct loop.

        Parameters
        ----------
        question : str
            The question to answer.
        session_id : str, optional
            Session ID for multi-turn conversations.

        Returns
        -------
        GroundedResponse
            The response with text, tool calls, and sources.

        Notes
        -----
        This is a synchronous wrapper around answer_async(). For Jupyter notebooks,
        use `await agent.answer_async(question)` directly instead.
        """
        logger.info(f"Answering question: {question[:100]}...")
        return asyncio.run(self.answer_async(question, session_id))


class KnowledgeAgentManager:
    """Manages KnowledgeGroundedAgent lifecycle with lazy initialization.

    This class provides convenient lifecycle management for the knowledge agent,
    with lazy initialization and state tracking. Unlike the general-purpose
    AsyncClientManager (for infrastructure clients), this is specific to the
    knowledge agent and is not a singleton.

    Parameters
    ----------
    config : Configs, optional
        Configuration object for client setup. If not provided, creates default.

    Examples
    --------
    >>> manager = KnowledgeAgentManager()
    >>> agent = manager.agent
    >>> response = await agent.answer_async("What is quantum computing?")
    >>> print(response.text)
    >>> manager.close()
    """

    def __init__(self, config: Configs | None = None) -> None:
        """Initialize the client manager.

        Parameters
        ----------
        config : Configs, optional
            Configuration object. If not provided, creates default config.
        """
        self._config = config
        self._agent: KnowledgeGroundedAgent | None = None
        self._initialized = False

    @property
    def config(self) -> Configs:
        """Get or create the config instance.

        Returns
        -------
        Configs
            The configuration settings.
        """
        if self._config is None:
            self._config = Configs()  # type: ignore[call-arg]
        return self._config

    @property
    def agent(self) -> KnowledgeGroundedAgent:
        """Get or create the knowledge-grounded agent.

        Returns
        -------
        KnowledgeGroundedAgent
            The knowledge-grounded QA agent.
        """
        if self._agent is None:
            self._agent = KnowledgeGroundedAgent(config=self.config)
            self._initialized = True
        return self._agent

    def close(self) -> None:
        """Close all initialized clients and reset state."""
        self._agent = None
        self._initialized = False

    def is_initialized(self) -> bool:
        """Check if any clients have been initialized.

        Returns
        -------
        bool
            True if any clients have been initialized.
        """
        return self._initialized

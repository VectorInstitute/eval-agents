"""Knowledge-grounded QA agent using Google ADK with Google Search.

This module provides a proper ReAct agent that explicitly calls
Google Search and shows the reasoning process through observable tool calls.
It also provides an enhanced agent with planning and multiple knowledge sources.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from .config import KnowledgeAgentConfig
from .grounding_tool import (
    GroundedResponse,
    GroundingChunk,
    create_google_search_tool,
    format_response_with_citations,
)
from .planner import ResearchPlan, ResearchPlanner, StepExecution
from .web_tools import create_fetch_url_tool, create_read_pdf_tool


logger = logging.getLogger(__name__)


def _process_function_calls(
    event: Any,
    tool_calls: list[dict[str, Any]],
    search_queries: list[str],
) -> None:
    """Extract tool calls and search queries from event function calls."""
    if not hasattr(event, "get_function_calls"):
        return
    function_calls = event.get_function_calls()
    if not function_calls:
        return
    for fc in function_calls:
        tool_call_info = {
            "name": getattr(fc, "name", "unknown"),
            "args": getattr(fc, "args", {}),
        }
        tool_calls.append(tool_call_info)
        logger.info(f"Tool call: {tool_call_info['name']}({tool_call_info['args']})")

        # Extract search queries from google_search calls
        tool_name = str(tool_call_info["name"])
        tool_args = tool_call_info["args"]
        if "search" in tool_name.lower() and isinstance(tool_args, dict):
            query = tool_args.get("query", "")
            if query:
                search_queries.append(query)


def _process_function_responses(event: Any, sources: list[GroundingChunk]) -> None:
    """Extract sources from event function responses."""
    if not hasattr(event, "get_function_responses"):
        return
    function_responses = event.get_function_responses()
    if not function_responses:
        return
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


def _process_grounding_metadata(
    event: Any,
    sources: list[GroundingChunk],
    search_queries: list[str],
) -> None:
    """Extract sources and search queries from grounding metadata."""
    gm = getattr(event, "grounding_metadata", None)
    if not gm and hasattr(event, "content") and event.content:
        gm = getattr(event.content, "grounding_metadata", None)
    if not gm:
        return

    # Extract grounding chunks
    if hasattr(gm, "grounding_chunks") and gm.grounding_chunks:
        for chunk in gm.grounding_chunks:
            if hasattr(chunk, "web") and chunk.web:
                sources.append(
                    GroundingChunk(
                        title=getattr(chunk.web, "title", "") or "",
                        uri=getattr(chunk.web, "uri", "") or "",
                    )
                )
    # Extract web search queries
    if hasattr(gm, "web_search_queries") and gm.web_search_queries:
        for q in gm.web_search_queries:
            if q and q not in search_queries:
                search_queries.append(q)


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
    config : KnowledgeAgentConfig, optional
        Configuration settings. If not provided, creates default config.
    model : str, optional
        The model to use. If not provided, uses config.default_worker_model.

    Attributes
    ----------
    config : KnowledgeAgentConfig
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
        config: KnowledgeAgentConfig | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the knowledge-grounded agent.

        Parameters
        ----------
        config : KnowledgeAgentConfig, optional
            Configuration settings. If not provided, creates default config.
        model : str, optional
            The model to use. If not provided, uses config.default_worker_model.
        """
        if config is None:
            config = KnowledgeAgentConfig()  # type: ignore[call-arg]

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
            _process_function_calls(event, tool_calls, search_queries)
            _process_function_responses(event, sources)
            _process_grounding_metadata(event, sources, search_queries)
            extracted_text = _extract_final_response(event)
            if extracted_text is not None:
                final_response = extracted_text

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

    def format_answer(self, response: GroundedResponse) -> str:
        """Format a grounded response for display.

        Parameters
        ----------
        response : GroundedResponse
            The grounded response to format.

        Returns
        -------
        str
            Formatted response with citations.
        """
        return format_response_with_citations(response)


class AsyncClientManager:
    """Manages async client lifecycle with lazy initialization and cleanup.

    This class ensures clients are created only once and properly closed,
    preventing resource warnings from unclosed event loops.

    Parameters
    ----------
    config : KnowledgeAgentConfig, optional
        Configuration object for client setup. If not provided, creates default.

    Examples
    --------
    >>> manager = AsyncClientManager()
    >>> agent = manager.agent
    >>> response = await agent.answer_async("What is quantum computing?")
    >>> print(response.text)
    """

    def __init__(self, config: KnowledgeAgentConfig | None = None) -> None:
        """Initialize the client manager.

        Parameters
        ----------
        config : KnowledgeAgentConfig, optional
            Configuration object. If not provided, creates default config.
        """
        self._config = config
        self._agent: KnowledgeGroundedAgent | None = None
        self._initialized = False

    @property
    def config(self) -> KnowledgeAgentConfig:
        """Get or create the config instance.

        Returns
        -------
        KnowledgeAgentConfig
            The configuration settings.
        """
        if self._config is None:
            self._config = KnowledgeAgentConfig()  # type: ignore[call-arg]
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


class EnhancedGroundedResponse(BaseModel):
    """Response with full execution trace for evaluation.

    This enhanced response model captures planning and execution details
    for comprehensive evaluation of the agent's reasoning process.

    Attributes
    ----------
    text : str
        The generated response text.
    plan : ResearchPlan
        The research plan created for the question.
    execution_trace : list[StepExecution]
        Record of each step's execution.
    sources : list[GroundingChunk]
        Web sources used in the response.
    search_queries : list[str]
        Search queries executed.
    reasoning_chain : list[str]
        Step-by-step reasoning trace.
    tool_calls : list[dict]
        Raw tool calls made during execution.
    total_duration_ms : int
        Total execution time in milliseconds.
    """

    text: str
    plan: ResearchPlan
    execution_trace: list[StepExecution] = Field(default_factory=list)
    sources: list[GroundingChunk] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    reasoning_chain: list[str] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)
    total_duration_ms: int = 0


ENHANCED_SYSTEM_INSTRUCTIONS = """\
You are an advanced knowledge-grounded research assistant. Your role is to answer
questions accurately by using multiple tools strategically.

## CRITICAL RULES

1. **Factual Grounding**: Only include facts/numbers you actually read in fetched content.
   Never guess or fabricate data.

2. **Persistence**: Be thorough before giving up. If one approach fails, try alternatives.
   For comparison questions, you MUST attempt to gather data on ALL items being compared.

## Available Tools

1. **google_search**: Search the web for current information and find relevant sources.

2. **fetch_url**: Fetch and read the full content of a webpage URL.
   Use this to read pages from search results - don't rely only on snippets!

3. **read_pdf**: Read and extract text from PDF documents at a URL.

## Research Strategy

1. **Search First**: Start with google_search to find relevant sources.

2. **Fetch Actual Content**: Use fetch_url to read pages you find. Search snippets
   alone are not sufficient for detailed questions.

3. **Handle Failures**: When a URL fails (404, timeout, etc.):
   - Try searching for alternative URLs to the same content
   - Look for the same information from different sources
   - Try different file formats (HTML vs PDF)
   - Do NOT give up after one or two failures

4. **Handle Truncated Content**: If content is truncated, look for:
   - Smaller, more focused documents on the same topic
   - Summary pages or data tables
   - Alternative sources with the same information

5. **Self-Reflect Before Giving Up**: Before concluding you cannot answer, ask yourself:
   - Have I tried all the items/entities in the question?
   - Have I tried alternative sources for each?
   - Have I tried different search queries?
   - What else could I try?

6. **For Comparison Questions**: If comparing multiple items (companies, products, etc.):
   - You MUST attempt to find data for EACH item
   - Don't stop after failing on the first item
   - Only conclude "cannot answer" after trying ALL items

## Response Format

- Provide a clear, direct answer backed by sources
- Cite the source URL for key facts
- If you couldn't find specific information after exhaustive searching, state what
  you tried and what you couldn't find
- Do NOT include data you didn't actually find in your sources
"""


class EnhancedKnowledgeAgent:
    """An enhanced ReAct agent with planning and multiple knowledge sources.

    This agent uses Google ADK with Google Search, URL fetching, and PDF reading
    tools, with explicit planning for complex questions.

    Parameters
    ----------
    config : KnowledgeAgentConfig, optional
        Configuration settings. If not provided, creates default config.
    model : str, optional
        The model to use for answering. If not provided, uses
        config.default_worker_model.
    enable_planning : bool, default True
        Whether to use the research planner for complex questions.

    Examples
    --------
    >>> from aieng.agent_evals.knowledge_agent import EnhancedKnowledgeAgent
    >>> agent = EnhancedKnowledgeAgent()
    >>> response = agent.answer("What are the Basel III capital requirements?")
    >>> print(response.text)
    >>> print(f"Plan complexity: {response.plan.complexity_assessment}")
    """

    def __init__(
        self,
        config: KnowledgeAgentConfig | None = None,
        model: str | None = None,
        enable_planning: bool = True,
    ) -> None:
        """Initialize the enhanced knowledge-grounded agent.

        Parameters
        ----------
        config : KnowledgeAgentConfig, optional
            Configuration settings. If not provided, creates default config.
        model : str, optional
            The model to use. If not provided, uses config.default_worker_model.
        enable_planning : bool, default True
            Whether to use the research planner.
        """
        if config is None:
            config = KnowledgeAgentConfig()  # type: ignore[call-arg]

        self.config = config
        self.model = model or config.default_worker_model
        self.enable_planning = enable_planning

        # Create tools
        self._search_tool = create_google_search_tool()
        self._fetch_url_tool = create_fetch_url_tool()
        self._read_pdf_tool = create_read_pdf_tool()

        # Create ADK agent with multiple tools
        self._agent = Agent(
            name="enhanced_knowledge_agent",
            model=self.model,
            instruction=ENHANCED_SYSTEM_INSTRUCTIONS,
            tools=[self._search_tool, self._fetch_url_tool, self._read_pdf_tool],
        )

        # Create planner if enabled
        self._planner: ResearchPlanner | None = None
        if enable_planning:
            self._planner = ResearchPlanner(config=config)

        # Session service for conversation history
        self._session_service = InMemorySessionService()

        # Runner orchestrates the ReAct loop
        self._runner = Runner(
            app_name="enhanced_knowledge_agent",
            agent=self._agent,
            session_service=self._session_service,
        )

        # Track active sessions
        self._sessions: dict[str, str] = {}

    async def _get_or_create_session_async(self, session_id: str | None = None) -> str:
        """Get or create an ADK session for the given session ID."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id not in self._sessions:
            session = await self._session_service.create_session(
                app_name="enhanced_knowledge_agent",
                user_id="user",
                state={},
            )
            self._sessions[session_id] = session.id

        return self._sessions[session_id]

    def _create_execution_trace(
        self,
        plan: ResearchPlan,
        tool_calls: list[dict[str, Any]],
        total_duration_ms: int,
    ) -> list[StepExecution]:
        """Create execution trace from plan and tool calls."""
        trace: list[StepExecution] = []

        # Map tool calls to plan steps
        call_idx = 0
        for step in plan.steps:
            if call_idx < len(tool_calls):
                tc = tool_calls[call_idx]
                trace.append(
                    StepExecution(
                        step_id=step.step_id,
                        tool_used=str(tc.get("name", "unknown")),
                        input_query=str(tc.get("args", {}).get("query", "")),
                        output_summary=f"Completed step: {step.description}",
                        sources_found=len(tc.get("response", {}).get("documents", []))
                        if isinstance(tc.get("response"), dict)
                        else 0,
                        duration_ms=total_duration_ms // max(len(plan.steps), 1),
                    )
                )
                call_idx += 1
            else:
                trace.append(
                    StepExecution(
                        step_id=step.step_id,
                        tool_used="synthesis",
                        input_query="",
                        output_summary=step.description,
                        sources_found=0,
                        duration_ms=0,
                    )
                )

        return trace

    async def answer_async(
        self,
        question: str,
        session_id: str | None = None,
    ) -> EnhancedGroundedResponse:
        """Answer a question using planning and multiple tools.

        Parameters
        ----------
        question : str
            The question to answer.
        session_id : str, optional
            Session ID for multi-turn conversations.

        Returns
        -------
        EnhancedGroundedResponse
            The response with plan, execution trace, and sources.
        """
        start_time = time.time()
        logger.info(f"Answering question (enhanced): {question[:100]}...")

        # Create research plan if planning is enabled
        if self._planner is not None:
            plan = await self._planner.create_plan_async(question)
            logger.info(f"Created plan: {plan.complexity_assessment} with {len(plan.steps)} steps")
        else:
            # Default simple plan
            plan = ResearchPlan(
                original_question=question,
                complexity_assessment="simple",
                steps=[],
                estimated_tools=["web_search"],
                reasoning="Planning disabled",
            )

        adk_session_id = await self._get_or_create_session_async(session_id)

        # Create the user message with plan context
        if self._planner is not None and plan.steps:
            plan_context = "\n".join(f"- Step {s.step_id}: {s.description} (use {s.tool_hint})" for s in plan.steps)
            enhanced_question = f"{question}\n\n[Research Plan]\n{plan_context}"
        else:
            enhanced_question = question

        content = types.Content(
            role="user",
            parts=[types.Part(text=enhanced_question)],
        )

        # Collect events from the ReAct loop
        tool_calls: list[dict[str, Any]] = []
        sources: list[GroundingChunk] = []
        search_queries: list[str] = []
        reasoning_chain: list[str] = []
        final_response = ""

        async for event in self._runner.run_async(
            user_id="user",
            session_id=adk_session_id,
            new_message=content,
        ):
            logger.debug(f"Event: {event}")
            _process_function_calls(event, tool_calls, search_queries)
            _process_function_responses(event, sources)
            _process_grounding_metadata(event, sources, search_queries)

            # Track reasoning chain from intermediate responses
            if hasattr(event, "content") and event.content and hasattr(event.content, "parts") and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        # Add to reasoning chain if it looks like reasoning
                        part_text = part.text.strip()
                        if part_text and len(part_text) < 500:  # Short enough to be reasoning
                            reasoning_chain.append(part_text)

            extracted_text = _extract_final_response(event)
            if extracted_text is not None:
                final_response = extracted_text

        total_duration_ms = int((time.time() - start_time) * 1000)

        # Create execution trace
        execution_trace = self._create_execution_trace(plan, tool_calls, total_duration_ms)

        return EnhancedGroundedResponse(
            text=final_response,
            plan=plan,
            execution_trace=execution_trace,
            sources=sources,
            search_queries=search_queries,
            reasoning_chain=reasoning_chain,
            tool_calls=tool_calls,
            total_duration_ms=total_duration_ms,
        )

    def answer(
        self,
        question: str,
        session_id: str | None = None,
    ) -> EnhancedGroundedResponse:
        """Answer a question using planning and multiple tools (sync).

        Parameters
        ----------
        question : str
            The question to answer.
        session_id : str, optional
            Session ID for multi-turn conversations.

        Returns
        -------
        EnhancedGroundedResponse
            The response with plan, execution trace, and sources.
        """
        logger.info(f"Answering question (enhanced, sync): {question[:100]}...")
        return asyncio.run(self.answer_async(question, session_id))

    def format_answer(self, response: EnhancedGroundedResponse) -> str:
        """Format an enhanced response for display.

        Parameters
        ----------
        response : EnhancedGroundedResponse
            The response to format.

        Returns
        -------
        str
            Formatted response with plan and citations.
        """
        parts = [response.text]

        # Add plan summary
        if response.plan.steps:
            parts.append(f"\n\n**Research Plan** ({response.plan.complexity_assessment}):")
            for step in response.plan.steps:
                parts.append(f"  {step.step_id}. {step.description}")

        # Add web sources
        if response.sources:
            parts.append("\n\n**Sources:**")
            for i, source in enumerate(response.sources, 1):
                if source.uri:
                    parts.append(f"[{i}] [{source.title or 'Source'}]({source.uri})")

        return "\n".join(parts)

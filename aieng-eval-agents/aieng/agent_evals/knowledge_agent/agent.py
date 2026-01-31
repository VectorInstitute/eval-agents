"""Knowledge-grounded QA agent using Google ADK with Google Search.

This module provides a proper ReAct agent that explicitly calls
Google Search and shows the reasoning process through observable tool calls.
It also provides an enhanced agent with planning and multiple knowledge sources.
"""

import asyncio
import logging
import time
import uuid
import warnings
from typing import Any

from aieng.agent_evals.configs import Configs
from aieng.agent_evals.tools import (
    create_fetch_url_tool,
    create_grep_file_tool,
    create_read_file_tool,
    create_read_pdf_tool,
)
from google.adk.agents import Agent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from .grounding_tool import (
    GroundedResponse,
    GroundingChunk,
    create_google_search_tool,
)
from .planner import ResearchPlan, ResearchPlanner, ResearchStep, StepExecution, StepStatus
from .token_tracker import TokenTracker


# Suppress experimental warnings from ADK
warnings.filterwarnings("ignore", message=r".*EXPERIMENTAL.*ContextCacheConfig.*")
warnings.filterwarnings("ignore", message=r".*EXPERIMENTAL.*EventsCompactionConfig.*")

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
        # Log tool response for CLI display tracking
        tool_name = getattr(fr, "name", None) or getattr(fr, "id", "unknown")
        logger.info(f"Tool response: {tool_name} completed")
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
        enable_caching: bool = True,
    ) -> None:
        """Initialize the knowledge-grounded agent.

        Parameters
        ----------
        config : Configs, optional
            Configuration settings. If not provided, creates default config.
        model : str, optional
            The model to use. If not provided, uses config.default_worker_model.
        enable_caching : bool, optional
            Whether to enable context caching. Defaults to True.
            Set to False for testing or when caching is not desired.
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

        if enable_caching:
            # Create App with context caching enabled
            # min_tokens=2048 is the minimum for Gemini caching
            # ttl_seconds=600 (10 min) balances cost vs cache hits
            # cache_intervals=10 refreshes cache periodically
            self._app = App(
                name="knowledge_agent",
                root_agent=self._agent,
                context_cache_config=ContextCacheConfig(
                    min_tokens=2048,
                    ttl_seconds=600,
                    cache_intervals=10,
                ),
            )
            # Runner orchestrates the ReAct loop
            self._runner = Runner(
                app=self._app,
                session_service=self._session_service,
            )
        else:
            # Runner without caching (for tests)
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

    def __init__(self, config: Configs | None = None, enable_caching: bool = True) -> None:
        """Initialize the client manager.

        Parameters
        ----------
        config : Configs, optional
            Configuration object. If not provided, creates default config.
        enable_caching : bool, default True
            Whether to enable context caching.
        """
        self._config = config
        self._enable_caching = enable_caching
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
            self._agent = KnowledgeGroundedAgent(config=self.config, enable_caching=self._enable_caching)
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
You are a research assistant that finds accurate, factual answers by retrieving and \
verifying information from authoritative sources.

## Your Goal

Answer questions with verified facts from primary sources. The key challenge is that \
search engine snippets are often incomplete, outdated, or lack the specific details \
needed to answer precisely. Your job is to go beyond snippets to find the actual data.

## Tools

**google_search** - Find relevant URLs. Use specific, targeted queries. The snippets \
help you identify which URLs are worth exploring, but snippets alone are not reliable \
sources for your final answer.

**fetch_url** - Retrieve the full content of a webpage. This saves the page locally \
and returns a file path. The preview shows only the first portion; the full content \
may be much larger and contain the specific data you need.

**grep_file** - Search within a fetched file for specific terms. Use this to locate \
exactly where relevant information appears in large documents. You can search for \
multiple terms separated by commas.

**read_file** - Read a specific section of a fetched file. Use this to examine the \
context around matches found by grep_file, or to read through sections systematically.

**read_pdf** - Extract text from PDF documents at a URL.

## CRITICAL: Iterative Search Refinement

Research is an iterative process. Your first search rarely gives the complete answer. \
You MUST refine your searches based on what you learn.

### The Refinement Loop

1. **Initial Search**: Start broad to identify the topic/domain
2. **Learn Domain Terminology**: Note specific terms, names, or jargon in the results
3. **Targeted Follow-up Search**: Search for those specific terms to find detailed pages
4. **Verify Before Answering**: Confirm the EXACT answer exists in your fetched sources

### Example of Good Iterative Research

Question: "What are the four categories of regulated substances under the UN convention?"

BAD approach (stops too early):
- Search "UN drug convention categories" -> finds general convention overview page
- Sees page mentions "narcotics" and "psychotropics" -> guesses answer
- WRONG: Missed the specific scheduling system!

GOOD approach (iterative refinement):
- Search "UN drug convention categories" -> finds it's the 1971 Convention
- Notices page mentions substances are organized into "Schedules"
- Search "1971 UN Convention Schedules" -> finds dedicated page on scheduling system
- Fetch that page -> confirms the four schedules with their specific criteria
- Answer with verified facts from the Schedules page

### Terminology Discovery Pattern

When you see a page mention a SPECIFIC TERM for what you're researching:
1. STOP - Don't answer yet!
2. NOTE the term (e.g., "substances are classified into Schedules")
3. SEARCH for "[term] + [what you need]" (e.g., "UN Convention Schedules list")
4. FETCH the dedicated page for that term
5. NOW answer with verified information

### When to Do Follow-up Searches

ALWAYS do a follow-up search when:
- You found a general/overview page but need specific details
- You learned a NEW TERM that likely has its own dedicated page
- The content references another document, page, or section
- You're about to answer but realize you're INFERRING rather than QUOTING

## Pre-Answer Verification Checklist

Before providing your final answer, ask yourself:

1. **Exact Match**: Does my answer EXACTLY match what was asked?
   - Asked for "3 categories" -> do I have exactly 3 items from the source?
   - Asked for a name -> do I have the specific name, not a description?

2. **Source Verification**: Did I find this in actual fetched page content?
   - NOT from search snippets (often incomplete/outdated)
   - NOT inferred from related information
   - YES quoted/paraphrased from text in a fetched document

3. **Terminology Check**: If I found a specific term for what the question asks about, \
did I search for that term?
   - Question asks about "types" -> source calls them "Classes" -> MUST search "Classes"

If ANY answer is "no" -> DO MORE RESEARCH before answering.

## How to Think About Research

1. **Identify what you need**: What specific fact would answer this question? Be precise.

2. **Reason about where to find it**: Before fetching any URL, ask: "Will this specific \
URL contain the data I need?" Don't fetch generic pages when you need specific subpages.

3. **Learn and refine**: After each search/fetch, ask "What NEW TERMS did I learn?" and \
"Should I search for those terms specifically?"

4. **Verify in the actual content**: Fetch the page and confirm the information is there. \
Don't assume - verify. Use grep_file to find specific sections in large pages.

5. **Extract precisely**: Read carefully. Context matters - dates, conditions, categories.

## Quality Standards

- Only state facts you found in fetched content, not from search snippets alone
- If asked for specific items (e.g., "3 categories"), provide exactly that from sources
- NEVER guess or infer - if you haven't verified it, search more
- When in doubt, do another targeted search rather than provide an uncertain answer
- Cite the source URL for key facts
"""


class EnhancedKnowledgeAgent:
    """An enhanced ReAct agent with planning and multiple knowledge sources.

    This agent uses Google ADK with Google Search, URL fetching, and PDF reading
    tools, with explicit planning for complex questions.

    Parameters
    ----------
    config : Configs, optional
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
        config: Configs | None = None,
        model: str | None = None,
        enable_planning: bool = True,
        enable_caching: bool = True,
        enable_compaction: bool = True,
        compaction_interval: int = 5,
    ) -> None:
        """Initialize the enhanced knowledge-grounded agent.

        Parameters
        ----------
        config : Configs, optional
            Configuration settings. If not provided, creates default config.
        model : str, optional
            The model to use. If not provided, uses config.default_worker_model.
        enable_planning : bool, default True
            Whether to use the research planner.
        enable_caching : bool, default True
            Whether to enable context caching for reduced latency and cost.
        enable_compaction : bool, default True
            Whether to enable context compaction. When enabled, ADK automatically
            summarizes older events to prevent running out of context.
        compaction_interval : int, default 5
            Number of invocations before triggering context compaction.
            Lower values compact more frequently (more overhead but safer).
        """
        self._enable_compaction = enable_compaction
        self._compaction_interval = compaction_interval
        if config is None:
            config = Configs()  # type: ignore[call-arg]

        self.config = config
        self.model = model or config.default_worker_model
        self.enable_planning = enable_planning

        # Create tools
        self._search_tool = create_google_search_tool()
        self._fetch_url_tool = create_fetch_url_tool()
        self._grep_file_tool = create_grep_file_tool()
        self._read_file_tool = create_read_file_tool()
        self._read_pdf_tool = create_read_pdf_tool()

        # Create ADK agent with multiple tools
        self._agent = Agent(
            name="enhanced_knowledge_agent",
            model=self.model,
            instruction=ENHANCED_SYSTEM_INSTRUCTIONS,
            tools=[
                self._search_tool,
                self._fetch_url_tool,
                self._grep_file_tool,
                self._read_file_tool,
                self._read_pdf_tool,
            ],
        )

        # Create planner if enabled
        self._planner: ResearchPlanner | None = None
        if enable_planning:
            self._planner = ResearchPlanner(config=config)  # type: ignore[arg-type]

        # Current research plan (accessible for real-time display)
        self._current_plan: ResearchPlan | None = None

        # Token tracking for context usage display
        self._token_tracker = TokenTracker(model=self.model)

        # Session service for conversation history
        self._session_service = InMemorySessionService()

        # Create App and Runner based on enabled features
        # App is needed for caching/compaction; otherwise use direct Runner (for tests)
        if enable_caching or enable_compaction:
            app_kwargs: dict[str, Any] = {
                "name": "enhanced_knowledge_agent",
                "root_agent": self._agent,
            }

            if enable_caching:
                # Context caching: reuse cached prompts to reduce latency/cost
                # min_tokens=2048 is the minimum for Gemini caching
                # ttl_seconds=600 (10 min) balances cost vs cache hits
                app_kwargs["context_cache_config"] = ContextCacheConfig(
                    min_tokens=2048,
                    ttl_seconds=600,
                    cache_intervals=10,
                )

            if enable_compaction:
                # Context compaction: summarize older events to prevent context overflow
                # Uses the same model as the planner for summarization
                summarizer = LlmEventSummarizer(llm=Gemini(model=config.default_planner_model))
                app_kwargs["events_compaction_config"] = EventsCompactionConfig(
                    compaction_interval=compaction_interval,
                    overlap_size=1,  # Include last event from previous window for continuity
                    summarizer=summarizer,
                )

            self._app: App | None = App(**app_kwargs)
            self._runner = Runner(
                app=self._app,
                session_service=self._session_service,
            )
        else:
            # Direct Runner without App (for tests with mocked agents)
            self._app = None
            self._runner = Runner(
                app_name="enhanced_knowledge_agent",
                agent=self._agent,
                session_service=self._session_service,
            )

        # Track active sessions
        self._sessions: dict[str, str] = {}

    @property
    def current_plan(self) -> ResearchPlan | None:
        """Get the current research plan if one exists.

        Returns
        -------
        ResearchPlan or None
            The current research plan, or None if no plan is active.
        """
        return self._current_plan

    @property
    def token_tracker(self) -> TokenTracker:
        """Get the token tracker for context usage monitoring."""
        return self._token_tracker

    async def create_plan_async(self, question: str) -> ResearchPlan | None:
        """Create a research plan for a question without executing it.

        This allows the CLI to create and display the plan before execution,
        and the same plan will be reused in answer_async().

        Parameters
        ----------
        question : str
            The question to plan for.

        Returns
        -------
        ResearchPlan or None
            The created plan, or None if planning is disabled.
        """
        if self._planner is None:
            return None
        self._current_plan = await self._planner.create_plan_async(question)
        return self._current_plan

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

    def _verify_tool_usage(self, expected_tool: str, tool_calls: list[dict[str, Any]]) -> bool:
        """Verify that the expected tool was actually called.

        Parameters
        ----------
        expected_tool : str
            The tool that should have been used (e.g., "fetch_url", "web_search").
        tool_calls : list[dict[str, Any]]
            The actual tool calls made.

        Returns
        -------
        bool
            True if the expected tool was called at least once.
        """
        if expected_tool == "synthesis":
            # Synthesis doesn't require a specific tool
            return True

        # Map tool hints to actual tool names
        tool_mapping = {
            "web_search": ["google_search", "search"],
            "fetch_url": ["fetch_url"],
            "read_pdf": ["read_pdf"],
            "grep_file": ["grep_file"],
            "read_file": ["read_file"],
        }

        expected_names = tool_mapping.get(expected_tool, [expected_tool])

        for tc in tool_calls:
            tool_name = str(tc.get("name", "")).lower()
            if any(expected in tool_name for expected in expected_names):
                return True

        logger.warning(
            f"Expected tool '{expected_tool}' was not called. Tool calls: {[tc.get('name') for tc in tool_calls]}"
        )
        return False

    async def _execute_step(
        self,
        step: ResearchStep,
        question: str,
        previous_results: list[str],
        adk_session_id: str,
        retry_count: int = 0,
    ) -> tuple[str, list[dict[str, Any]], list[GroundingChunk], list[str], bool]:
        """Execute a single research step.

        Parameters
        ----------
        step : ResearchStep
            The step to execute.
        question : str
            The original question.
        previous_results : list[str]
            Results from previous steps.
        adk_session_id : str
            The ADK session ID.
        retry_count : int
            Number of retry attempts for this step.

        Returns
        -------
        tuple
            (step_result, tool_calls, sources, search_queries, tool_used_correctly)
            tool_used_correctly is True if the expected tool was actually called.
        """
        # Build context from previous steps
        context_parts = [f"Original question: {question}"]
        if previous_results:
            context_parts.append("\nPrevious findings:")
            for i, result in enumerate(previous_results, 1):
                context_parts.append(f"  Step {i}: {result[:500]}...")

        # Create focused instruction for this step - more forceful on retries
        if retry_count > 0:
            step_instruction = f"""
{chr(10).join(context_parts)}

MANDATORY TASK (Step {step.step_id}): {step.description}

**CRITICAL**: You MUST use the {step.tool_hint} tool NOW. This is attempt #{retry_count + 1}.
Previous attempts did not use the required tool. You MUST call {step.tool_hint} before responding.

DO NOT explain why you cannot do it. DO NOT skip this step. JUST USE THE TOOL.

Expected output: {step.expected_output}
"""
        else:
            step_instruction = f"""
{chr(10).join(context_parts)}

Current task (Step {step.step_id}): {step.description}
Tool to use: {step.tool_hint}
Expected output: {step.expected_output}

Execute ONLY this step. Use the {step.tool_hint} tool to complete this specific task.
Provide a concise summary of what you found.
"""

        content = types.Content(
            role="user",
            parts=[types.Part(text=step_instruction)],
        )

        tool_calls: list[dict[str, Any]] = []
        sources: list[GroundingChunk] = []
        search_queries: list[str] = []
        step_result = ""

        async for event in self._runner.run_async(
            user_id="user",
            session_id=adk_session_id,
            new_message=content,
        ):
            # Track token usage
            self._token_tracker.add_from_event(event)

            new_tool_calls = _extract_tool_calls(event)
            tool_calls.extend(new_tool_calls)
            search_queries.extend(_extract_search_queries_from_tool_calls(new_tool_calls))
            sources.extend(_extract_sources_from_responses(event))
            sources.extend(_extract_grounding_sources(event))

            for q in _extract_grounding_queries(event):
                if q not in search_queries:
                    search_queries.append(q)

            text = _extract_final_response(event)
            if text is not None:
                step_result = text

        # Check if the expected tool was actually used
        tool_used_correctly = self._verify_tool_usage(step.tool_hint, tool_calls)

        return step_result, tool_calls, sources, search_queries, tool_used_correctly

    async def _execute_plan_step(
        self,
        step: ResearchStep,
        question: str,
        step_results: list[str],
        plan: ResearchPlan,
        adk_session_id: str,
    ) -> tuple[str, list[dict[str, Any]], list[GroundingChunk], list[str], bool]:
        """Execute a single plan step and update plan status.

        Returns tuple of (step_result, tool_calls, sources, queries, tool_executed).
        tool_executed indicates whether the expected tool was actually called.
        """
        step_start = time.time()
        logger.info(f"Executing step {step.step_id}: {step.description}")

        # Mark step as in progress
        plan.update_step(step.step_id, status=StepStatus.IN_PROGRESS)

        # Handle synthesis steps differently - use direct LLM without tools
        if step.tool_hint == "synthesis":
            step_result = await self._execute_synthesis_step(
                step=step,
                question=question,
                previous_results=step_results,
            )
            tool_calls: list[dict[str, Any]] = []
            sources: list[GroundingChunk] = []
            queries: list[str] = []
            tool_executed = True  # Synthesis doesn't need a tool
        else:
            # Execute with retry logic for tool enforcement
            max_retries = 2
            tool_executed = False
            step_result = ""
            tool_calls = []
            sources = []
            queries = []

            for retry in range(max_retries + 1):
                step_result, tool_calls, sources, queries, tool_executed = await self._execute_step(
                    step=step,
                    question=question,
                    previous_results=step_results,
                    adk_session_id=adk_session_id,
                    retry_count=retry,
                )

                if tool_executed:
                    break

                if retry < max_retries:
                    logger.warning(
                        f"Step {step.step_id} did not use expected tool '{step.tool_hint}'. "
                        f"Retrying ({retry + 1}/{max_retries})..."
                    )
                else:
                    logger.warning(
                        f"Step {step.step_id} failed to use expected tool '{step.tool_hint}' "
                        f"after {max_retries + 1} attempts."
                    )

        # Mark step as completed (even if tool wasn't used - we tried)
        step_duration = int((time.time() - step_start) * 1000)
        status = StepStatus.COMPLETED if tool_executed else StepStatus.FAILED
        plan.update_step(
            step.step_id,
            status=status,
            actual_output=step_result[:500] if step_result else "No output",
            failure_reason="" if tool_executed else f"Expected tool '{step.tool_hint}' was not called",
        )
        logger.info(f"Step {step.step_id} {'completed' if tool_executed else 'failed'} in {step_duration}ms")

        return step_result, tool_calls, sources, queries, tool_executed

    async def _synthesize_final_answer(
        self,
        question: str,
        step_results: list[str],
    ) -> str:
        """Synthesize final answer from all step results.

        Uses direct LLM call without tools to prevent additional searches.
        """
        from google import genai  # noqa: PLC0415

        synthesis_prompt = f"""
Original question: {question}

Research findings from {len(step_results)} steps:
{chr(10).join(f"- Step {i + 1}: {r}" for i, r in enumerate(step_results))}

Based on these findings, provide a complete, well-structured answer.
Do NOT search for additional information - use only the findings above.
Cite sources where appropriate.
"""
        client = genai.Client()
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=types.Content(
                role="user",
                parts=[types.Part(text=synthesis_prompt)],
            ),
        )
        return response.text or ""

    async def _execute_synthesis_step(
        self,
        step: ResearchStep,
        question: str,
        previous_results: list[str],
    ) -> str:
        """Execute a synthesis step using direct LLM call (no tools).

        Synthesis steps combine information from previous steps without
        making additional searches or fetches.

        Parameters
        ----------
        step : ResearchStep
            The synthesis step to execute.
        question : str
            The original question.
        previous_results : list[str]
            Results from previous steps to synthesize.

        Returns
        -------
        str
            The synthesis result.
        """
        from google import genai  # noqa: PLC0415

        synthesis_prompt = f"""
Original question: {question}

Task: {step.description}
Expected output: {step.expected_output}

Previous research findings:
{chr(10).join(f"- Step {i + 1}: {r}" for i, r in enumerate(previous_results))}

Based on the findings above, complete this synthesis task.
Do NOT search for additional information - use only the findings above.
"""

        client = genai.Client()
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=types.Content(
                role="user",
                parts=[types.Part(text=synthesis_prompt)],
            ),
        )

        return response.text or ""

    async def _get_or_create_plan_async(self, question: str) -> ResearchPlan:
        """Get existing plan or create a new one for the question."""
        if self._current_plan is not None:
            logger.info(f"Using existing plan: {self._current_plan.complexity_assessment}")
            return self._current_plan

        if self._planner is not None:
            plan = await self._planner.create_plan_async(question)
            self._current_plan = plan
            logger.info(f"Created plan: {plan.complexity_assessment} with {len(plan.steps)} steps")
            return plan

        # Default simple plan when planning is disabled
        plan = ResearchPlan(
            original_question=question,
            complexity_assessment="simple",
            steps=[],
            estimated_tools=["web_search"],
            reasoning="Planning disabled",
        )
        self._current_plan = plan
        return plan

    async def _execute_without_plan(
        self,
        question: str,
        adk_session_id: str,
    ) -> tuple[str, list[dict[str, Any]], list[GroundingChunk], list[str]]:
        """Execute question directly without a plan (fallback behavior)."""
        content = types.Content(role="user", parts=[types.Part(text=question)])
        tool_calls: list[dict[str, Any]] = []
        sources: list[GroundingChunk] = []
        queries: list[str] = []
        final_response = ""

        async for event in self._runner.run_async(user_id="user", session_id=adk_session_id, new_message=content):
            # Track token usage
            self._token_tracker.add_from_event(event)

            new_calls = _extract_tool_calls(event)
            tool_calls.extend(new_calls)
            queries.extend(_extract_search_queries_from_tool_calls(new_calls))
            sources.extend(_extract_sources_from_responses(event))
            sources.extend(_extract_grounding_sources(event))
            text = _extract_final_response(event)
            if text is not None:
                final_response = text

        return final_response, tool_calls, sources, queries

    async def answer_async(
        self,
        question: str,
        session_id: str | None = None,
    ) -> EnhancedGroundedResponse:
        """Answer a question using planning and multiple tools.

        Executes plan steps dynamically, reflecting after each step.
        """
        start_time = time.time()
        logger.info(f"Answering question (enhanced): {question[:100]}...")

        plan = await self._get_or_create_plan_async(question)
        adk_session_id = await self._get_or_create_session_async(session_id)

        # Collect results across all steps
        all_tool_calls: list[dict[str, Any]] = []
        all_sources: list[GroundingChunk] = []
        all_search_queries: list[str] = []
        reasoning_chain: list[str] = []
        step_results: list[str] = []
        final_response = ""

        if plan.steps:
            # Track whether we've successfully fetched any URLs
            # This prevents premature synthesis without actual source content
            fetch_url_succeeded = False

            # Dynamic execution loop - reflect and adapt after each step
            max_iterations = len(plan.steps) * 2  # Safety limit

            for _ in range(max_iterations):
                pending_steps = plan.get_pending_steps()
                if not pending_steps:
                    logger.info("No more pending steps - ready to synthesize")
                    break

                step = pending_steps[0]
                step_result, tool_calls, sources, queries, tool_executed = await self._execute_plan_step(
                    step, question, step_results, plan, adk_session_id
                )

                # Track if any fetch_url step succeeded
                if step.tool_hint == "fetch_url" and tool_executed:
                    fetch_url_succeeded = True
                    logger.info("Successfully fetched URL content")

                # Collect results
                all_tool_calls.extend(tool_calls)
                all_sources.extend(sources)
                all_search_queries.extend(queries)
                if step_result:
                    step_results.append(step_result)
                    reasoning_chain.append(f"Step {step.step_id}: {step_result[:300]}")

                # Reflect and potentially update plan (skip for synthesis steps)
                if self._planner is not None and step.tool_hint != "synthesis":
                    reflection = await self._planner.reflect_and_update_plan_async(
                        plan=plan,
                        completed_step=step,
                        step_result=step_result,
                        all_findings=step_results,
                        fetch_url_succeeded=fetch_url_succeeded,
                    )
                    logger.info(f"Reflection: {reflection.decision}")

                    # Only allow early exit if we've actually fetched some content
                    # OR if there are no remaining fetch_url steps in the plan
                    remaining_fetch_steps = [s for s in plan.get_pending_steps() if s.tool_hint == "fetch_url"]

                    if reflection.can_answer_now:
                        if fetch_url_succeeded or not remaining_fetch_steps:
                            logger.info("Can answer now - skipping remaining steps")
                            break
                        logger.warning(
                            "Reflection says can_answer_now but no URLs fetched yet. "
                            f"Continuing with {len(remaining_fetch_steps)} pending fetch_url steps."
                        )
                        # Don't break - force the agent to attempt fetch steps

            # Final synthesis: combine all step results into final answer
            final_response = await self._synthesize_final_answer(question, step_results)

        else:
            # No plan steps - execute directly (fallback behavior)
            final_response, all_tool_calls, all_sources, all_search_queries = await self._execute_without_plan(
                question, adk_session_id
            )

        total_duration_ms = int((time.time() - start_time) * 1000)

        # Create execution trace
        execution_trace = self._create_execution_trace(plan, all_tool_calls, total_duration_ms)

        # Clear the current plan so the agent can be reused for new questions
        # The plan is still returned in the response for reference
        self._current_plan = None

        return EnhancedGroundedResponse(
            text=final_response,
            plan=plan,
            execution_trace=execution_trace,
            sources=all_sources,
            search_queries=all_search_queries,
            reasoning_chain=reasoning_chain,
            tool_calls=all_tool_calls,
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

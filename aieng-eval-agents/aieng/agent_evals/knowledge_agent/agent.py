"""Knowledge-grounded QA agent using Google ADK with Google Search.

This module provides a ReAct agent with planning and multiple knowledge sources
that explicitly calls tools and shows the reasoning process through observable
tool calls.
"""

import asyncio
import logging
import time
import uuid
import warnings
from datetime import datetime, timezone
from typing import Any

from aieng.agent_evals.configs import Configs
from aieng.agent_evals.tools import (
    GroundingChunk,
    create_fetch_file_tool,
    create_google_search_tool,
    create_grep_file_tool,
    create_read_file_tool,
    create_web_fetch_tool,
    resolve_redirect_urls_async,
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
            # google_search_agent uses "request", other tools may use "query"
            query = tool_args.get("request") or tool_args.get("query") or ""
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
        Sources extracted from the function responses (raw URLs, not resolved).
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
        Sources extracted from the grounding metadata (raw URLs, not resolved).
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


async def _resolve_source_urls(sources: list[GroundingChunk]) -> list[GroundingChunk]:
    """Resolve redirect URLs in sources to actual URLs (in parallel).

    Parameters
    ----------
    sources : list[GroundingChunk]
        Sources with potentially redirect URLs.

    Returns
    -------
    list[GroundingChunk]
        Sources with resolved URLs.
    """
    if not sources:
        return sources

    # Extract URIs and resolve in parallel
    uris = [s.uri for s in sources]
    resolved_uris = await resolve_redirect_urls_async(uris)

    # Create new sources with resolved URIs
    return [GroundingChunk(title=s.title, uri=resolved) for s, resolved in zip(sources, resolved_uris)]


async def _format_urls_for_agent(sources: list[GroundingChunk]) -> str:
    """Format grounding sources as a message for the agent to use with web_fetch.

    Resolves redirect URLs to actual URLs before formatting.

    Parameters
    ----------
    sources : list[GroundingChunk]
        Sources with URLs from grounding metadata.

    Returns
    -------
    str
        Formatted message with resolved URLs the agent can use.
    """
    if not sources:
        return ""

    # Deduplicate by URI
    seen_uris: set[str] = set()
    unique_sources: list[GroundingChunk] = []
    for src in sources:
        if src.uri and src.uri not in seen_uris:
            seen_uris.add(src.uri)
            unique_sources.append(src)

    if not unique_sources:
        return ""

    # Resolve redirect URLs to actual URLs (in parallel)
    unique_sources = await _resolve_source_urls(unique_sources)

    lines = ["Search found the following URLs. Use web_fetch to retrieve relevant pages:"]
    for i, src in enumerate(unique_sources[:10], 1):  # Limit to 10 URLs
        title = src.title or "Unknown"
        lines.append(f"  {i}. {title}: {src.uri}")

    lines.append('\nCall web_fetch(url="<url>") to get the full content of any page.')
    return "\n".join(lines)


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


SYSTEM_INSTRUCTIONS_TEMPLATE = """\
You are a research assistant that finds accurate, factual answers by retrieving and \
verifying information from authoritative sources.

**Today's date: {current_date}**

## Your Goal

Answer questions with verified facts from primary sources. The key challenge is that \
search engine snippets are often incomplete, outdated, or lack the specific details \
needed to answer precisely. Your job is to go beyond snippets to find the actual data.

## Tools

**google_search** - Find relevant URLs. Use specific, targeted queries. The snippets \
help you identify which URLs are worth exploring, but snippets alone are not reliable \
sources for your final answer.

**web_fetch** - Fetch content from a URL (HTML pages or PDFs). Returns the full text \
content for you to analyze. Use this for web pages and PDF documents like research \
papers, SEC filings, or official reports.

Example usage:
- web_fetch(url="https://example.com/about") - fetches HTML page content
- web_fetch(url="https://arxiv.org/pdf/2301.00234.pdf") - extracts PDF text

**fetch_file** - Download data files (CSV, XLSX, JSON) for searching. Use with \
grep_file and read_file for large structured data files.

**grep_file** - Search within a downloaded file for specific patterns. Returns \
matching lines with context.

**read_file** - Read specific sections of a downloaded file by line numbers.

## CRITICAL: Iterative Search Refinement

Research is an iterative process. Your first search rarely gives the complete answer. \
You MUST refine your searches based on what you learn.

### The Refinement Loop

1. **Initial Search**: Start broad to identify the topic/domain
2. **Learn Domain Terminology**: Note specific terms, names, or jargon in the results
3. **Targeted Follow-up Search**: Search for those specific terms to find detailed pages
4. **Verify Before Answering**: Use web_fetch to confirm the EXACT answer exists

### Example of Good Iterative Research

Question: "What are the four categories of regulated substances under the UN convention?"

BAD approach (stops too early):
- Search "UN drug convention categories" -> finds general convention overview page
- Sees snippet mentions "narcotics" and "psychotropics" -> guesses answer
- WRONG: Missed the specific scheduling system!

GOOD approach (iterative refinement):
- Search "UN drug convention categories" -> finds it's the 1971 Convention
- Notices results mention substances are organized into "Schedules"
- Search "1971 UN Convention Schedules" -> finds dedicated page on scheduling system
- web_fetch(url="https://...scheduling-page") -> gets the full page content
- Read through the content to find the four schedules and their criteria
- Answer with verified facts from the fetched page

### Terminology Discovery Pattern

When you see search results mention a SPECIFIC TERM for what you're researching:
1. STOP - Don't answer yet!
2. NOTE the term (e.g., "substances are classified into Schedules")
3. SEARCH for "[term] + [what you need]" (e.g., "UN Convention Schedules list")
4. FETCH the dedicated page using web_fetch(url=...) to get the full content
5. Read through the content and NOW answer with verified information

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
   - YES extracted from a fetched page using web_fetch

3. **Terminology Check**: If I found a specific term for what the question asks about, \
did I search for that term?
   - Question asks about "types" -> source calls them "Classes" -> MUST search "Classes"

If ANY answer is "no" -> DO MORE RESEARCH before answering.

## How to Think About Research

1. **Identify what you need**: What specific fact would answer this question? Be precise.

2. **Reason about where to find it**: Before fetching any URL, ask: "Will this specific \
URL contain the data I need?" Don't fetch generic pages when you need specific subpages.

3. **Learn and refine**: After each search, ask "What NEW TERMS did I learn?" and \
"Should I search for those terms specifically?"

4. **Verify with web_fetch**: Fetch the URL and read through the content to find what you need.

5. **Extract precisely**: Read through the fetched content carefully to find exactly the \
information needed - dates, conditions, categories, names, numbers.

## Quality Standards

- Only state facts you found in fetched content, not from search snippets alone
- If asked for specific items (e.g., "3 categories"), provide exactly that from sources
- NEVER guess or infer - if you haven't verified it, search more
- When in doubt, do another targeted search rather than provide an uncertain answer
- Cite the source URL for key facts
"""


def _build_system_instructions() -> str:
    """Build system instructions with current date context."""
    now = datetime.now(timezone.utc)
    return SYSTEM_INSTRUCTIONS_TEMPLATE.format(
        current_date=now.strftime("%B %d, %Y"),
    )


# For backwards compatibility (static snapshot at import time)
SYSTEM_INSTRUCTIONS = _build_system_instructions()


class KnowledgeGroundedAgent:
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
    >>> from aieng.agent_evals.knowledge_agent import KnowledgeGroundedAgent
    >>> agent = KnowledgeGroundedAgent()
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
        compaction_interval: int = 3,
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
        self.temperature = config.default_temperature
        self.enable_planning = enable_planning

        # Create tools
        self._search_tool = create_google_search_tool()
        self._web_fetch_tool = create_web_fetch_tool()
        self._fetch_file_tool = create_fetch_file_tool()
        self._grep_file_tool = create_grep_file_tool()
        self._read_file_tool = create_read_file_tool()

        # Create ADK agent with multiple tools
        # Lower temperature improves consistency between runs
        self._agent = Agent(
            name="enhanced_knowledge_agent",
            model=self.model,
            instruction=_build_system_instructions(),
            tools=[
                self._search_tool,
                self._web_fetch_tool,
                self._fetch_file_tool,
                self._grep_file_tool,
                self._read_file_tool,
            ],
            generate_content_config=types.GenerateContentConfig(
                temperature=self.temperature,
            ),
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

    async def _execute_step(
        self,
        step: ResearchStep,
        question: str,
        previous_results: list[str],
        adk_session_id: str,
    ) -> tuple[str, list[dict[str, Any]], list[GroundingChunk], list[str]]:
        """Execute a single research step.

        The agent is given the task description and is free to choose the best
        approach and tools to complete it. If the agent uses Google Search and
        receives grounding URLs, those URLs are injected back so the agent can
        use web_fetch to retrieve the full content.

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

        Returns
        -------
        tuple
            (step_result, tool_calls, sources, search_queries)
        """
        # Build context from previous steps
        context_parts = [f"Original question: {question}"]
        if previous_results:
            context_parts.append("\nPrevious findings:")
            for i, result in enumerate(previous_results, 1):
                context_parts.append(f"  Step {i}: {result[:500]}...")

        # Simple task instruction - let the agent decide how to accomplish it
        step_instruction = f"""
{chr(10).join(context_parts)}

Current task (Step {step.step_id}): {step.description}
Expected output: {step.expected_output}

Complete this task using the tools available to you.
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

        # If we got grounding sources with URLs, inject them back to the agent
        # so it can use web_fetch to retrieve the full content
        urls_message = await _format_urls_for_agent(sources)
        if urls_message:
            logger.info(f"Injecting {len(sources)} grounding URLs back to agent")
            followup_content = types.Content(
                role="user",
                parts=[types.Part(text=urls_message)],
            )

            async for event in self._runner.run_async(
                user_id="user",
                session_id=adk_session_id,
                new_message=followup_content,
            ):
                self._token_tracker.add_from_event(event)

                new_tool_calls = _extract_tool_calls(event)
                tool_calls.extend(new_tool_calls)
                search_queries.extend(_extract_search_queries_from_tool_calls(new_tool_calls))
                sources.extend(_extract_sources_from_responses(event))
                sources.extend(_extract_grounding_sources(event))

                text = _extract_final_response(event)
                if text is not None:
                    step_result = text

        return step_result, tool_calls, sources, search_queries

    async def _execute_plan_step(
        self,
        step: ResearchStep,
        question: str,
        step_results: list[str],
        plan: ResearchPlan,
        adk_session_id: str,
    ) -> tuple[str, list[dict[str, Any]], list[GroundingChunk], list[str]]:
        """Execute a single plan step and update plan status.

        Returns tuple of (step_result, tool_calls, sources, queries).
        Steps are marked COMPLETED if they produce output.
        """
        step_start = time.time()
        logger.info(f"Executing step {step.step_id}: {step.description}")

        # Mark step as in progress
        plan.update_step(step.step_id, status=StepStatus.IN_PROGRESS)

        # Handle synthesis steps differently - use direct LLM without tools
        if step.step_type == "synthesis":
            step_result = await self._execute_synthesis_step(
                step=step,
                question=question,
                previous_results=step_results,
            )
            tool_calls: list[dict[str, Any]] = []
            sources: list[GroundingChunk] = []
            queries: list[str] = []
        else:
            # Execute the step - let the agent choose how to accomplish it
            step_result, tool_calls, sources, queries = await self._execute_step(
                step=step,
                question=question,
                previous_results=step_results,
                adk_session_id=adk_session_id,
            )

        # Mark step as completed if it produced output
        step_duration = int((time.time() - step_start) * 1000)
        has_output = bool(step_result and step_result.strip())
        status = StepStatus.COMPLETED if has_output else StepStatus.FAILED
        plan.update_step(
            step.step_id,
            status=status,
            actual_output=step_result[:500] if step_result else "No output",
            failure_reason="" if has_output else "Step produced no output",
        )
        logger.info(f"Step {step.step_id} {'completed' if has_output else 'failed'} in {step_duration}ms")

        return step_result, tool_calls, sources, queries

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
            config=types.GenerateContentConfig(
                temperature=self.temperature,
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
            config=types.GenerateContentConfig(
                temperature=self.temperature,
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
            # Track whether we've gathered substantial content
            has_substantial_content = False

            # Dynamic execution loop - reflect and adapt after each step
            max_iterations = len(plan.steps) * 2  # Safety limit

            for _ in range(max_iterations):
                pending_steps = plan.get_pending_steps()
                if not pending_steps:
                    logger.info("No more pending steps - ready to synthesize")
                    break

                step = pending_steps[0]
                step_result, tool_calls, sources, queries = await self._execute_plan_step(
                    step, question, step_results, plan, adk_session_id
                )

                # Track if we've gathered substantial content
                if step_result and len(step_result) > 200:
                    has_substantial_content = True
                    logger.info("Step produced substantial content")

                # Collect results
                all_tool_calls.extend(tool_calls)
                all_sources.extend(sources)
                all_search_queries.extend(queries)
                if step_result:
                    step_results.append(step_result)
                    reasoning_chain.append(f"Step {step.step_id}: {step_result[:300]}")

                # Reflect and potentially update plan (skip for synthesis steps)
                if self._planner is not None and step.step_type != "synthesis":
                    reflection = await self._planner.reflect_and_update_plan_async(
                        plan=plan,
                        completed_step=step,
                        step_result=step_result,
                        all_findings=step_results,
                        has_substantial_content=has_substantial_content,
                    )
                    logger.info(
                        f"Reflection: can_answer={reflection.can_answer_now}, updates={len(reflection.steps_to_update)}"
                    )

                    if reflection.can_answer_now and has_substantial_content:
                        logger.info("Can answer now - skipping remaining steps")
                        break

            # Use the last step result as final response if it was a synthesis step
            # Otherwise, synthesize from all step results
            completed_steps = plan.get_steps_by_status(StepStatus.COMPLETED)
            last_step = completed_steps[-1] if completed_steps else None

            if last_step and last_step.step_type == "synthesis" and step_results:
                # Last step was synthesis - use its result directly
                final_response = step_results[-1]
                logger.info("Using synthesis step result as final answer")
            elif step_results:
                # No synthesis step completed - synthesize now
                final_response = await self._synthesize_final_answer(question, step_results)
            else:
                final_response = "Unable to find relevant information to answer this question."

        else:
            # No plan steps - execute directly (fallback behavior)
            final_response, all_tool_calls, all_sources, all_search_queries = await self._execute_without_plan(
                question, adk_session_id
            )

        total_duration_ms = int((time.time() - start_time) * 1000)

        # Resolve redirect URLs in sources (in parallel for speed)
        all_sources = await _resolve_source_urls(all_sources)

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

    def __init__(
        self,
        config: Configs | None = None,
        enable_caching: bool = True,
        enable_planning: bool = True,
        enable_compaction: bool = True,
    ) -> None:
        """Initialize the client manager.

        Parameters
        ----------
        config : Configs, optional
            Configuration object. If not provided, creates default config.
        enable_caching : bool, default True
            Whether to enable context caching.
        enable_planning : bool, default True
            Whether to enable research planning.
        enable_compaction : bool, default True
            Whether to enable context compaction.
        """
        self._config = config
        self._enable_caching = enable_caching
        self._enable_planning = enable_planning
        self._enable_compaction = enable_compaction
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
            self._agent = KnowledgeGroundedAgent(
                config=self.config,
                enable_caching=self._enable_caching,
                enable_planning=self._enable_planning,
                enable_compaction=self._enable_compaction,
            )
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

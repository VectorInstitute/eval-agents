"""Knowledge-grounded QA agent using Google ADK with Google Search.

This module provides a ReAct agent with built-in planning via Gemini's thinking
mode that explicitly calls tools and shows the reasoning process through observable
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
from google.adk.planners import PlanReActPlanner
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.genai.errors import ClientError
from pydantic import BaseModel, Field
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .token_tracker import TokenTracker


# Suppress experimental warnings from ADK
warnings.filterwarnings("ignore", message=r".*EXPERIMENTAL.*ContextCacheConfig.*")
warnings.filterwarnings("ignore", message=r".*EXPERIMENTAL.*EventsCompactionConfig.*")

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Planner with Improved Search Strategy
# =============================================================================


class ResearchPlanner(PlanReActPlanner):
    """Custom PlanReActPlanner with research-optimized planning instructions.

    This planner extends PlanReActPlanner with guidance that encourages:
    - Searching for answers directly rather than decomposing into many steps
    - Keeping key question terms together in searches
    - Avoiding premature commitment to intermediate conclusions
    - Recognizing direct answers when found
    """

    def _build_nl_planner_instruction(self) -> str:
        """Build optimized planning instruction for research tasks."""
        return """
When answering the question, use the available tools to find information rather than relying on memorized knowledge.

Follow this process: (1) Create a plan to answer the question. (2) Execute the plan using tools, with reasoning between steps. (3) Provide the final answer.

Use these tags in your response:
- /*PLANNING*/ for your initial plan
- /*ACTION*/ for tool calls
- /*REASONING*/ for your analysis between actions
- /*REPLANNING*/ when revising your approach
- /*FINAL_ANSWER*/ for your final response

## Planning Guidelines

Write your plan as a numbered list of steps, like:
1. Search for [specific terms]
2. Verify findings from [source type]
3. Synthesize the answer

**Focus on the actual question.** Identify what is specifically being asked. If the question asks for "categories" or "names" or "values", your plan should aim to find those directly.

**Search for answers, not just context.** Rather than planning to "first identify X, then find Y within X", consider searching for Y directly with context from the question. Combining key terms in one search often finds the answer faster than sequential searches.

**Keep plans flexible.** Your initial plan is a starting point. If early searches reveal the answer directly, you can conclude early. If they reveal your approach won't work, revise the plan.

## Reasoning Guidelines

**Evaluate what you've found.** After each tool use, assess: Does this answer what was asked? Does this change my understanding of the question?

**Recognize direct answers.** If search results or fetched content directly answer the question, acknowledge this and move toward your final answer. Don't ignore valid answers because other details are uncertain.

**Avoid premature commitment.** Don't lock onto an assumption early. If you assume the answer involves "X" and search within "X", you may miss the correct answer. Stay open to what the evidence shows.

## Replanning Guidelines

If your approach isn't working:
- Try searching for the answer more directly
- Reformulate with different terms
- Consider whether your initial assumptions were wrong

Use /*REPLANNING*/ to revise your strategy when needed.

## Final Answer Guidelines

Provide /*FINAL_ANSWER*/ when you have found information that answers what was asked. The answer should be precise and directly address the question. If the question cannot be answered with available information, explain why.
"""


# =============================================================================
# Data Models for Research Plans
# =============================================================================


class StepStatus:
    """Status constants for research steps."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ResearchStep(BaseModel):
    """A single step in a research plan.

    Attributes
    ----------
    step_id : int
        Unique identifier for the step within the plan.
    description : str
        Clear description of what this step accomplishes.
    step_type : str
        Type of step: "research" (uses tools to gather info) or "synthesis"
        (combines findings without tools).
    depends_on : list[int]
        IDs of steps that must complete before this one.
    expected_output : str
        Description of what this step is expected to produce.
    status : str
        Current execution status: "pending", "in_progress", "completed", "failed",
        or "skipped".
    actual_output : str
        What was actually found/produced by this step.
    attempts : int
        Number of times this step has been attempted.
    failure_reason : str
        Reason for failure if the step failed.
    """

    step_id: int
    description: str
    step_type: str = "research"  # "research" or "synthesis"
    depends_on: list[int] = Field(default_factory=list)
    expected_output: str = ""
    # Dynamic tracking fields
    status: str = Field(default=StepStatus.PENDING)
    actual_output: str = ""
    attempts: int = 0
    failure_reason: str = ""


class ResearchPlan(BaseModel):
    """A complete research plan for answering a complex question.

    This model represents an observable, evaluable research plan that
    decomposes a question into executable steps with clear dependencies.

    Attributes
    ----------
    original_question : str
        The original question being answered.
    steps : list[ResearchStep]
        Ordered list of research steps to execute.
    reasoning : str
        Explanation of why this plan was chosen.
    """

    original_question: str
    steps: list[ResearchStep] = Field(default_factory=list)
    reasoning: str = ""

    def get_step(self, step_id: int) -> ResearchStep | None:
        """Get a step by its ID.

        Parameters
        ----------
        step_id : int
            The step ID to find.

        Returns
        -------
        ResearchStep | None
            The step if found, None otherwise.
        """
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def update_step(
        self,
        step_id: int,
        status: str | None = None,
        actual_output: str | None = None,
        failure_reason: str | None = None,
        increment_attempts: bool = False,
        description: str | None = None,
        expected_output: str | None = None,
    ) -> bool:
        """Update a step's fields.

        Parameters
        ----------
        step_id : int
            The step ID to update.
        status : str, optional
            New status for the step.
        actual_output : str, optional
            What was actually found/produced.
        failure_reason : str, optional
            Reason for failure if applicable.
        increment_attempts : bool
            Whether to increment the attempts counter.
        description : str, optional
            New description for the step (for plan refinement).
        expected_output : str, optional
            New expected output for the step (for plan refinement).

        Returns
        -------
        bool
            True if the step was found and updated, False otherwise.
        """
        step = self.get_step(step_id)
        if step is None:
            return False

        if status is not None:
            step.status = status
        if actual_output is not None:
            step.actual_output = actual_output
        if failure_reason is not None:
            step.failure_reason = failure_reason
        if increment_attempts:
            step.attempts += 1
        if description is not None:
            step.description = description
        if expected_output is not None:
            step.expected_output = expected_output

        return True

    def get_pending_steps(self) -> list[ResearchStep]:
        """Get steps that are ready to execute (pending with no unmet dependencies).

        Returns
        -------
        list[ResearchStep]
            Steps that can be executed now.
        """
        completed_ids = {s.step_id for s in self.steps if s.status == StepStatus.COMPLETED}
        pending = []

        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            # Check if all dependencies are completed
            if all(dep_id in completed_ids for dep_id in step.depends_on):
                pending.append(step)

        return pending

    def get_steps_by_status(self, status: str) -> list[ResearchStep]:
        """Get all steps with a specific status.

        Parameters
        ----------
        status : str
            The status to filter by.

        Returns
        -------
        list[ResearchStep]
            Steps matching the status.
        """
        return [s for s in self.steps if s.status == status]

    def is_complete(self) -> bool:
        """Check if all steps are either completed, failed, or skipped.

        Returns
        -------
        bool
            True if no steps are pending or in progress.
        """
        terminal_statuses = {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED}
        return all(s.status in terminal_statuses for s in self.steps)


class StepExecution(BaseModel):
    """Record of executing a single research step.

    This model captures the execution trace for evaluation purposes.

    Attributes
    ----------
    step_id : int
        The step ID that was executed.
    tool_used : str
        The actual tool that was used.
    input_query : str
        The query or input provided to the tool.
    output_summary : str
        Summary of what the step produced.
    sources_found : int
        Number of sources discovered in this step.
    duration_ms : int
        Execution time in milliseconds.
    raw_output : str
        Raw output from the tool for debugging.
    """

    step_id: int
    tool_used: str
    input_query: str
    output_summary: str = ""
    sources_found: int = 0
    duration_ms: int = 0
    raw_output: str = ""


# =============================================================================
# Helper Functions
# =============================================================================


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
    """Extract final response text from event if it's a final response.

    Filters out thought parts (internal reasoning) and returns only the
    actual response text intended for the user.
    """
    if not hasattr(event, "is_final_response") or not event.is_final_response():
        return None
    if not hasattr(event, "content") or not event.content:
        return None
    if not hasattr(event.content, "parts") or not event.content.parts:
        return None

    # Collect non-thought text parts only
    response_parts = []
    for part in event.content.parts:
        # Skip thought parts (internal reasoning)
        if getattr(part, "thought", False):
            continue
        if hasattr(part, "text") and part.text:
            response_parts.append(part.text)

    return "\n".join(response_parts) if response_parts else ""


def _extract_thoughts_from_event(event: Any) -> str:
    """Extract thinking/reasoning content from event parts.

    Parameters
    ----------
    event : Any
        An event from the ADK runner.

    Returns
    -------
    str
        Combined thinking text from all thought parts.
    """
    if not hasattr(event, "content") or not event.content:
        return ""
    if not hasattr(event.content, "parts") or not event.content.parts:
        return ""

    thoughts = []
    for part in event.content.parts:
        # Parts with thought=True are thinking content
        if getattr(part, "thought", False) and hasattr(part, "text") and part.text:
            thoughts.append(part.text)
    return "\n".join(thoughts)


# Max retries for empty model responses
MAX_EMPTY_RESPONSE_RETRIES = 2

# API retry configuration for rate limit and quota exhaustion
API_RETRY_MAX_ATTEMPTS = 5
API_RETRY_INITIAL_WAIT = 1  # seconds
API_RETRY_MAX_WAIT = 60  # seconds
API_RETRY_JITTER = 5  # seconds


def _is_retryable_api_error(exception: BaseException) -> bool:
    """Check if an exception is a retryable API error (rate limit/quota exhaustion).

    Parameters
    ----------
    exception : BaseException
        The exception to check.

    Returns
    -------
    bool
        True if the exception should trigger a retry (429/RESOURCE_EXHAUSTED errors).
    """
    if isinstance(exception, ClientError):
        error_str = str(exception).lower()
        # Check for rate limit indicators
        if "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str:
            return True
    return False


# PlanReActPlanner tag constants (from google.adk.planners.plan_re_act_planner)
PLANNING_TAG = "/*PLANNING*/"
REPLANNING_TAG = "/*REPLANNING*/"
REASONING_TAG = "/*REASONING*/"
ACTION_TAG = "/*ACTION*/"
FINAL_ANSWER_TAG = "/*FINAL_ANSWER*/"


def _extract_plan_text(text: str) -> str | None:
    """Extract plan text from PLANNING or REPLANNING tags.

    Parameters
    ----------
    text : str
        Text that may contain planning tags.

    Returns
    -------
    str | None
        The plan text if found, None otherwise.
    """
    # Check for REPLANNING first (updated plan takes precedence)
    for tag in [REPLANNING_TAG, PLANNING_TAG]:
        if tag in text:
            start = text.find(tag) + len(tag)
            # Find the end - next tag or end of text
            end = len(text)
            for end_tag in [REASONING_TAG, ACTION_TAG, FINAL_ANSWER_TAG, PLANNING_TAG, REPLANNING_TAG]:
                if end_tag in text[start:]:
                    tag_pos = text.find(end_tag, start)
                    if tag_pos != -1 and tag_pos < end:
                        end = tag_pos
            plan_text = text[start:end].strip()
            if plan_text:
                return plan_text
    return None


def _parse_plan_steps_from_text(plan_text: str) -> list[ResearchStep]:
    """Parse numbered steps from plan text.

    Parameters
    ----------
    plan_text : str
        Raw plan text, typically with numbered steps.

    Returns
    -------
    list[ResearchStep]
        Parsed research steps.
    """
    import re  # noqa: PLC0415

    steps = []
    # Match numbered steps: "1. Description", "1) Description", or "Step 1: Description"
    patterns = [
        r"^\s*(\d+)[.\)]\s*(.+?)(?=\n\s*\d+[.\)]|\n\s*Step\s+\d+|\Z)",  # "1. desc" or "1) desc"
        r"^\s*Step\s+(\d+)[:\.]?\s*(.+?)(?=\n\s*Step\s+\d+|\n\s*\d+[.\)]|\Z)",  # "Step 1: desc"
        r"^\s*[-*]\s*(.+?)(?=\n\s*[-*]|\Z)",  # Bullet points
    ]

    # Try numbered patterns first
    for pattern in patterns[:2]:
        matches = re.findall(pattern, plan_text, re.MULTILINE | re.DOTALL)
        if matches:
            for i, match in enumerate(matches[:10]):  # Max 10 steps
                step_num = int(match[0]) if len(match) > 1 else i + 1
                description = match[1] if len(match) > 1 else match[0]
                description = description.strip()
                # Clean up description - remove trailing newlines and extra whitespace
                description = " ".join(description.split())
                if description and len(description) > 5:
                    steps.append(
                        ResearchStep(
                            step_id=step_num,
                            description=description[:200],
                            step_type="research",
                            status=StepStatus.PENDING,
                        )
                    )
            if steps:
                return steps

    # Try bullet pattern
    matches = re.findall(patterns[2], plan_text, re.MULTILINE | re.DOTALL)
    if matches:
        for i, desc in enumerate(matches[:10], 1):
            description = " ".join(desc.strip().split())
            if description and len(description) > 5:
                steps.append(
                    ResearchStep(
                        step_id=i,
                        description=description[:200],
                        step_type="research",
                        status=StepStatus.PENDING,
                    )
                )
        if steps:
            return steps

    # Fallback: split by newlines if no pattern matched
    lines = [line.strip() for line in plan_text.split("\n") if line.strip() and len(line.strip()) > 10]
    for i, line in enumerate(lines[:10], 1):
        # Skip lines that look like headers
        if line.endswith(":") or line.startswith("#"):
            continue
        steps.append(
            ResearchStep(
                step_id=i,
                description=line[:200],
                step_type="research",
                status=StepStatus.PENDING,
            )
        )

    return steps


def _extract_reasoning_text(text: str) -> str | None:
    """Extract reasoning text from REASONING tag.

    Parameters
    ----------
    text : str
        Text that may contain reasoning tag.

    Returns
    -------
    str | None
        The reasoning text if found, None otherwise.
    """
    if REASONING_TAG not in text:
        return None

    start = text.find(REASONING_TAG) + len(REASONING_TAG)
    end = len(text)
    for end_tag in [ACTION_TAG, FINAL_ANSWER_TAG, PLANNING_TAG, REPLANNING_TAG]:
        if end_tag in text[start:]:
            tag_pos = text.find(end_tag, start)
            if tag_pos != -1 and tag_pos < end:
                end = tag_pos
    return text[start:end].strip() or None


def _extract_final_answer_text(text: str) -> str | None:
    """Extract final answer text from FINAL_ANSWER tag.

    Parameters
    ----------
    text : str
        Text that may contain final answer tag.

    Returns
    -------
    str | None
        The final answer text if found, None otherwise.
    """
    if FINAL_ANSWER_TAG not in text:
        return None

    start = text.find(FINAL_ANSWER_TAG) + len(FINAL_ANSWER_TAG)
    return text[start:].strip() or None


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
You are a research assistant that finds accurate answers by exploring sources and verifying facts.

Today's date: {current_date}

## Tools

**google_search**: Find URLs related to a topic. Search results include brief snippets—use these to identify promising sources, then fetch pages for complete information.

**web_fetch**: Read the full content of a web page. Use this to verify facts and find detailed information.

**fetch_file**: Download data files (CSV, XLSX, JSON) for structured data like statistics or datasets.

**grep_file**: Search within a downloaded file to locate specific information.

**read_file**: Read sections of a downloaded file to examine data in detail.

## Search Strategy

**Search for the answer, not just context.** If a question asks "what are the three categories of X?", search for those categories directly rather than first identifying what X is and then searching within X.

**Keep key terms together.** Include the core question terms in your search query. A search combining the key concepts often finds the answer more directly than breaking it into separate searches.

**Recognize direct answers.** If search results directly answer what was asked, use that answer. Don't ignore a valid answer just because you haven't identified every detail mentioned in the question.

**Avoid premature commitment.** Don't lock onto an interpretation early. If you assume something is "Game A" and search for answers within "Game A", you may miss the correct answer if your assumption was wrong. Stay open until you have confirming evidence.

## Adapting Your Plan

If your initial approach doesn't yield the needed information:
- Reformulate your search with different terms
- Search for the answer more directly rather than adding intermediate steps
- Look for alternative sources (official reports, databases, different websites)
- Use /*REPLANNING*/ to revise your strategy

Don't give up or guess—adapt and try another approach.

## Final Answer

Provide /*FINAL_ANSWER*/ once you have found information that answers what was asked. Include:
- ANSWER: Your direct answer based on what you found
- SOURCES: The URLs or files where you found the information
- REASONING: How you verified or arrived at this answer
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
    """A ReAct agent with built-in planning via Gemini's thinking mode.

    This agent uses Google ADK with BuiltInPlanner to enable Gemini's native
    thinking capabilities, which plan and execute research in a unified loop.

    Parameters
    ----------
    config : Configs, optional
        Configuration settings. If not provided, creates default config.
    model : str, optional
        The model to use for answering. If not provided, uses
        config.default_worker_model.
    enable_planning : bool, default True
        Whether to enable the built-in planner (Gemini thinking mode).
    thinking_budget : int, default 8192
        Token budget for the model's thinking/planning phase.

    Examples
    --------
    >>> from aieng.agent_evals.knowledge_agent import KnowledgeGroundedAgent
    >>> agent = KnowledgeGroundedAgent()
    >>> response = agent.answer("What are the Basel III capital requirements?")
    >>> print(response.text)
    """

    def __init__(
        self,
        config: Configs | None = None,
        model: str | None = None,
        enable_planning: bool = True,
        enable_caching: bool = True,
        enable_compaction: bool = True,
        compaction_interval: int = 3,
        thinking_budget: int = 8192,
    ) -> None:
        """Initialize the knowledge-grounded agent with built-in planning.

        Parameters
        ----------
        config : Configs, optional
            Configuration settings. If not provided, creates default config.
        model : str, optional
            The model to use. If not provided, uses config.default_worker_model.
        enable_planning : bool, default True
            Whether to enable the built-in planner (Gemini thinking mode).
        enable_caching : bool, default True
            Whether to enable context caching for reduced latency and cost.
        enable_compaction : bool, default True
            Whether to enable context compaction. When enabled, ADK automatically
            summarizes older events to prevent running out of context.
        compaction_interval : int, default 3
            Number of invocations before triggering context compaction.
        thinking_budget : int, default 8192
            Token budget for the model's thinking/planning phase.
        """
        self._enable_compaction = enable_compaction
        self._compaction_interval = compaction_interval
        if config is None:
            config = Configs()  # type: ignore[call-arg]

        self.config = config
        self.model = model or config.default_worker_model
        self.temperature = config.default_temperature
        self.enable_planning = enable_planning
        self._thinking_budget = thinking_budget

        # Create tools
        self._search_tool = create_google_search_tool()
        self._web_fetch_tool = create_web_fetch_tool()
        self._fetch_file_tool = create_fetch_file_tool()
        self._grep_file_tool = create_grep_file_tool()
        self._read_file_tool = create_read_file_tool()

        # Create planner if enabled - uses ResearchPlanner for optimized search
        planner = None
        if enable_planning:
            planner = ResearchPlanner()

        # Create ADK agent with built-in planner
        # Configure thinking for models that support it (gemini-2.5-*, gemini-3-*)
        thinking_config = None
        if thinking_budget > 0 and KnowledgeGroundedAgent._supports_thinking(self.model):
            thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)

        self._agent = Agent(
            name="knowledge_agent",
            model=self.model,
            instruction=_build_system_instructions(),
            tools=[
                self._search_tool,
                self._web_fetch_tool,
                self._fetch_file_tool,
                self._grep_file_tool,
                self._read_file_tool,
            ],
            planner=planner,
            generate_content_config=types.GenerateContentConfig(
                temperature=self.temperature,
                thinking_config=thinking_config,
            ),
        )

        # Current research plan (populated from model's thinking for CLI display)
        self._current_plan: ResearchPlan | None = None

        # Token tracking for context usage display
        self._token_tracker = TokenTracker(model=self.model)

        # Session service for conversation history
        self._session_service = InMemorySessionService()

        # Create App and Runner based on enabled features
        if enable_caching or enable_compaction:
            app_kwargs: dict[str, Any] = {
                "name": "knowledge_agent",
                "root_agent": self._agent,
            }

            if enable_caching:
                app_kwargs["context_cache_config"] = ContextCacheConfig(
                    min_tokens=2048,
                    ttl_seconds=600,
                    cache_intervals=10,
                )

            if enable_compaction:
                summarizer = LlmEventSummarizer(llm=Gemini(model=config.default_worker_model))
                app_kwargs["events_compaction_config"] = EventsCompactionConfig(
                    compaction_interval=compaction_interval,
                    overlap_size=1,
                    summarizer=summarizer,
                )

            self._app: App | None = App(**app_kwargs)
            self._runner = Runner(
                app=self._app,
                session_service=self._session_service,
            )
        else:
            self._app = None
            self._runner = Runner(
                app_name="knowledge_agent",
                agent=self._agent,
                session_service=self._session_service,
            )

        # Track active sessions
        self._sessions: dict[str, str] = {}

    @staticmethod
    def _supports_thinking(model: str) -> bool:
        """Check if a model supports thinking configuration.

        Thinking is supported by gemini-2.5-* and gemini-3-* models.

        Parameters
        ----------
        model : str
            The model identifier.

        Returns
        -------
        bool
            True if the model supports thinking configuration.
        """
        model_lower = model.lower()
        return "gemini-2.5" in model_lower or "gemini-3" in model_lower

    def reset(self) -> None:
        """Reset agent state for a new question.

        Clears session history and plan state to ensure clean execution
        for each new question. Call this between evaluation examples.
        """
        self._sessions.clear()
        self._session_service = InMemorySessionService()
        self._current_plan = None
        self._token_tracker = TokenTracker(model=self.model)

        # Recreate runner with fresh session service
        if self._app is not None:
            self._runner = Runner(
                app=self._app,
                session_service=self._session_service,
            )
        else:
            self._runner = Runner(
                app_name="knowledge_agent",
                agent=self._agent,
                session_service=self._session_service,
            )
        logger.debug("Agent state reset for new question")

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
        """Initialize plan tracking for CLI display.

        Creates an empty plan that will be populated from the model's
        PLANNING output during answer_async(). The actual plan steps
        are extracted from the model's first response.

        Parameters
        ----------
        question : str
            The question to plan for.

        Returns
        -------
        ResearchPlan or None
            An empty plan ready for population, or None if planning is disabled.
        """
        if not self.enable_planning:
            return None

        # Create empty plan - will be populated from model's PLANNING output
        self._current_plan = ResearchPlan(
            original_question=question,
            steps=[],
            reasoning="",
        )
        return self._current_plan

    async def _get_or_create_session_async(self, session_id: str | None = None) -> str:
        """Get or create an ADK session for the given session ID."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id not in self._sessions:
            session = await self._session_service.create_session(
                app_name="knowledge_agent",
                user_id="user",
                state={},
            )
            self._sessions[session_id] = session.id

        return self._sessions[session_id]

    def _update_plan_from_text(self, text: str, question: str, is_replan: bool = False) -> bool:
        """Update the current plan from PLANNING or REPLANNING tagged text.

        Parameters
        ----------
        text : str
            Text containing PLANNING or REPLANNING tags.
        question : str
            The original question.
        is_replan : bool
            Whether this is a replan (updates existing plan).

        Returns
        -------
        bool
            True if plan was updated, False otherwise.
        """
        plan_text = _extract_plan_text(text)
        if not plan_text:
            return False

        steps = _parse_plan_steps_from_text(plan_text)
        if not steps:
            return False

        if is_replan and self._current_plan:
            # Replanning: preserve completed steps, update remaining
            completed_steps = [s for s in self._current_plan.steps if s.status == StepStatus.COMPLETED]
            # Renumber new steps starting after completed ones
            next_id = len(completed_steps) + 1
            for i, step in enumerate(steps):
                step.step_id = next_id + i
            self._current_plan.steps = completed_steps + steps
            self._current_plan.reasoning = f"Replanned: {plan_text[:300]}"
            logger.info(f"Replanned with {len(steps)} new steps (keeping {len(completed_steps)} completed)")
        else:
            # New plan
            self._current_plan = ResearchPlan(
                original_question=question,
                steps=steps,
                reasoning=plan_text[:500],
            )
            logger.info(f"Extracted plan with {len(steps)} steps")

        # Mark first pending step as in progress
        for step in self._current_plan.steps:
            if step.status == StepStatus.PENDING:
                step.status = StepStatus.IN_PROGRESS
                break

        return True

    def _process_event_text_for_plan(self, text: str, question: str) -> None:
        """Process event text to extract and update plan.

        Parameters
        ----------
        text : str
            Text from event that may contain plan tags.
        question : str
            The original question.
        """
        if not text or not self._current_plan:
            return

        # Check for replanning first
        if REPLANNING_TAG in text:
            self._update_plan_from_text(text, question, is_replan=True)
        # Check for initial planning (only if plan is empty)
        elif PLANNING_TAG in text and len(self._current_plan.steps) == 0:
            self._update_plan_from_text(text, question, is_replan=False)

    def _update_plan_step_from_tool_call(self, tool_name: str) -> None:
        """Record tool call against current plan step.

        Parameters
        ----------
        tool_name : str
            Name of the tool that was called.
        """
        if not self._current_plan or not self._current_plan.steps:
            return

        # Find current in-progress step and record the tool used
        for step in self._current_plan.steps:
            if step.status == StepStatus.IN_PROGRESS:
                # Append tool to actual_output
                if step.actual_output:
                    step.actual_output += f", {tool_name}"
                else:
                    step.actual_output = f"Used: {tool_name}"
                break

    def _advance_plan_step_on_reasoning(self) -> None:
        """Advance to next plan step when reasoning is detected.

        This is called when a REASONING tag is found, indicating the agent
        has reflected on progress and may be moving to the next step.
        """
        if not self._current_plan or not self._current_plan.steps:
            return

        # Find current in-progress step
        for i, step in enumerate(self._current_plan.steps):
            if step.status == StepStatus.IN_PROGRESS:
                # Mark current step as completed
                step.status = StepStatus.COMPLETED
                # Mark next step as in progress (if exists)
                if i + 1 < len(self._current_plan.steps):
                    self._current_plan.steps[i + 1].status = StepStatus.IN_PROGRESS
                break

    def _create_execution_trace(
        self,
        tool_calls: list[dict[str, Any]],
        total_duration_ms: int,
    ) -> list[StepExecution]:
        """Create execution trace from tool calls."""
        return [
            StepExecution(
                step_id=i + 1,
                tool_used=str(tc.get("name", "unknown")),
                input_query=str(tc.get("args", {})),
                output_summary=f"Tool call {i + 1}",
                sources_found=0,
                duration_ms=total_duration_ms // max(len(tool_calls), 1),
            )
            for i, tc in enumerate(tool_calls)
        ]

    def _extract_event_text(self, event: Any) -> str:
        """Extract text content from event parts."""
        if not (
            hasattr(event, "content") and event.content and hasattr(event.content, "parts") and event.content.parts
        ):
            return ""
        parts = [part.text for part in event.content.parts if hasattr(part, "text") and part.text]
        return "\n".join(parts)

    def _process_event(
        self,
        event: Any,
        question: str,
        results: dict[str, Any],
    ) -> None:
        """Process a single event from the agent run loop.

        Updates results dict in place with extracted information.
        """
        self._token_tracker.add_from_event(event)
        event_text = self._extract_event_text(event)

        # Extract thoughts for reasoning chain
        thoughts = _extract_thoughts_from_event(event)
        if thoughts:
            results["reasoning_chain"].append(thoughts[:300])

        # Process plan tags and reasoning
        if event_text:
            self._process_event_text_for_plan(event_text, question)
            reasoning_text = _extract_reasoning_text(event_text)
            if reasoning_text:
                results["reasoning_chain"].append(reasoning_text[:300])
                self._advance_plan_step_on_reasoning()

        # Extract tool calls
        new_tool_calls = _extract_tool_calls(event)
        results["tool_calls"].extend(new_tool_calls)
        results["search_queries"].extend(_extract_search_queries_from_tool_calls(new_tool_calls))
        for tc in new_tool_calls:
            self._update_plan_step_from_tool_call(tc.get("name", ""))

        # Extract sources
        results["sources"].extend(_extract_sources_from_responses(event))
        results["sources"].extend(_extract_grounding_sources(event))
        for q in _extract_grounding_queries(event):
            if q not in results["search_queries"]:
                results["search_queries"].append(q)

        # Extract final response
        final_answer = _extract_final_answer_text(event_text) if event_text else None
        if final_answer:
            results["final_response"] = final_answer
        elif (text := _extract_final_response(event)) is not None:
            results["final_response"] = text

    async def _run_agent_once_inner(
        self,
        question: str,
        adk_session_id: str,
    ) -> dict[str, Any]:
        """Run the agent once and collect results (inner implementation).

        Parameters
        ----------
        question : str
            The question to answer.
        adk_session_id : str
            The ADK session ID.

        Returns
        -------
        dict[str, Any]
            Results dictionary with tool_calls, sources, search_queries,
            reasoning_chain, and final_response.
        """
        content = types.Content(role="user", parts=[types.Part(text=question)])

        # Collect results in a mutable dict for _process_event
        results: dict[str, Any] = {
            "tool_calls": [],
            "sources": [],
            "search_queries": [],
            "reasoning_chain": [],
            "final_response": "",
        }

        async for event in self._runner.run_async(
            user_id="user",
            session_id=adk_session_id,
            new_message=content,
        ):
            self._process_event(event, question, results)

        return results

    async def _run_agent_once(
        self,
        question: str,
        adk_session_id: str,
    ) -> dict[str, Any]:
        """Run the agent once with retry logic for API rate limits.

        Wraps _run_agent_once_inner with exponential backoff retry for
        429/RESOURCE_EXHAUSTED errors from the Gemini API.

        Parameters
        ----------
        question : str
            The question to answer.
        adk_session_id : str
            The ADK session ID.

        Returns
        -------
        dict[str, Any]
            Results dictionary with tool_calls, sources, search_queries,
            reasoning_chain, and final_response.

        Raises
        ------
        ClientError
            If all retry attempts fail due to persistent API errors.
        """

        # Create a retry-wrapped version of the inner method
        @retry(
            retry=retry_if_exception(_is_retryable_api_error),
            wait=wait_exponential_jitter(
                initial=API_RETRY_INITIAL_WAIT,
                max=API_RETRY_MAX_WAIT,
                jitter=API_RETRY_JITTER,
            ),
            stop=stop_after_attempt(API_RETRY_MAX_ATTEMPTS),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _run_with_retry() -> dict[str, Any]:
            return await self._run_agent_once_inner(question, adk_session_id)

        return await _run_with_retry()

    async def answer_async(
        self,
        question: str,
        session_id: str | None = None,
    ) -> EnhancedGroundedResponse:
        """Answer a question using built-in planning and tools.

        The agent uses PlanReAct planning to create and execute research steps.
        The plan is captured and updated in real-time for CLI display.

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
        logger.info(f"Answering question: {question[:100]}...")

        adk_session_id = await self._get_or_create_session_async(session_id)

        if self._current_plan is None and self.enable_planning:
            await self.create_plan_async(question)

        # Run agent with retry logic for empty responses
        results: dict[str, Any] = {}
        current_session_id = adk_session_id

        for attempt in range(MAX_EMPTY_RESPONSE_RETRIES + 1):
            results = await self._run_agent_once(question, current_session_id)

            # Check if we got a non-empty response
            if results.get("final_response", "").strip():
                break

            # Empty response - log and retry if we have attempts left
            if attempt < MAX_EMPTY_RESPONSE_RETRIES:
                logger.warning(
                    f"Empty model response (attempt {attempt + 1}/{MAX_EMPTY_RESPONSE_RETRIES + 1}), "
                    "creating fresh session and retrying..."
                )
                # Create fresh session for retry to avoid polluted history
                fresh_session = await self._session_service.create_session(
                    app_name="knowledge_agent",
                    user_id="user",
                    state={},
                )
                current_session_id = fresh_session.id
                # Reset plan for retry
                if self.enable_planning:
                    await self.create_plan_async(question)
            else:
                logger.error(
                    f"Empty model response after {MAX_EMPTY_RESPONSE_RETRIES + 1} attempts. "
                    "The model may have generated thinking tokens but produced no output."
                )

        total_duration_ms = int((time.time() - start_time) * 1000)

        # Mark remaining steps as completed
        if self._current_plan:
            for step in self._current_plan.steps:
                if step.status in (StepStatus.PENDING, StepStatus.IN_PROGRESS):
                    step.status = StepStatus.COMPLETED

        # Resolve redirect URLs and build response
        resolved_sources = await _resolve_source_urls(results.get("sources", []))
        plan = self._current_plan or ResearchPlan(original_question=question, steps=[], reasoning="No planning enabled")
        execution_trace = self._create_execution_trace(results.get("tool_calls", []), total_duration_ms)
        self._current_plan = None

        return EnhancedGroundedResponse(
            text=results.get("final_response", ""),
            plan=plan,
            execution_trace=execution_trace,
            sources=resolved_sources,
            search_queries=results.get("search_queries", []),
            reasoning_chain=results.get("reasoning_chain", []),
            tool_calls=results.get("tool_calls", []),
            total_duration_ms=total_duration_ms,
        )

    def answer(
        self,
        question: str,
        session_id: str | None = None,
    ) -> EnhancedGroundedResponse:
        """Answer a question using built-in planning and tools (sync).

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
        logger.info(f"Answering question (sync): {question[:100]}...")
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
            parts.append("\n\n**Research Plan:**")
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
    with lazy initialization and state tracking.

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
            Whether to enable built-in planning (Gemini thinking mode).
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

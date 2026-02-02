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
from typing import Any

from aieng.agent_evals.configs import Configs
from aieng.agent_evals.tools import (
    create_fetch_file_tool,
    create_google_search_tool,
    create_grep_file_tool,
    create_read_file_tool,
    create_web_fetch_tool,
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
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .event_extraction import (
    extract_event_text,
    extract_final_response,
    extract_grounding_queries,
    extract_grounding_sources,
    extract_search_queries_from_tool_calls,
    extract_sources_from_responses,
    extract_thoughts_from_event,
    extract_tool_calls,
    resolve_source_urls,
)
from .models import (
    AgentResponse,
    ResearchPlan,
    ResearchStep,
    StepExecution,
    StepStatus,
)
from .plan_parsing import (
    PLANNING_TAG,
    REPLANNING_TAG,
    extract_final_answer_text,
    extract_plan_text,
    extract_reasoning_text,
    parse_plan_steps_from_text,
)
from .retry import (
    API_RETRY_INITIAL_WAIT,
    API_RETRY_JITTER,
    API_RETRY_MAX_ATTEMPTS,
    API_RETRY_MAX_WAIT,
    MAX_EMPTY_RESPONSE_RETRIES,
    is_context_overflow_error,
    is_retryable_api_error,
)
from .system_instructions import build_system_instructions
from .token_tracker import TokenTracker


# Re-export models for backward compatibility
__all__ = [
    "AgentResponse",
    "KnowledgeAgentManager",
    "KnowledgeGroundedAgent",
    "ResearchPlan",
    "ResearchStep",
    "StepExecution",
    "StepStatus",
]


# Suppress experimental warnings from ADK
warnings.filterwarnings("ignore", message=r".*EXPERIMENTAL.*ContextCacheConfig.*")
warnings.filterwarnings("ignore", message=r".*EXPERIMENTAL.*EventsCompactionConfig.*")

logger = logging.getLogger(__name__)


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

        # Create planner if enabled
        planner = None
        if enable_planning:
            planner = PlanReActPlanner()

        # Create ADK agent with built-in planner
        # Configure thinking for models that support it (gemini-2.5-*, gemini-3-*)
        thinking_config = None
        if thinking_budget > 0 and self._supports_thinking(self.model):
            thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)

        self._agent = Agent(
            name="knowledge_agent",
            model=self.model,
            instruction=build_system_instructions(),
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
        self._app: App | None
        if enable_caching or enable_compaction:
            self._app, self._runner = self._create_app_and_runner(config, enable_caching, enable_compaction)
        else:
            self._app = None
            self._runner = Runner(
                app_name="knowledge_agent",
                agent=self._agent,
                session_service=self._session_service,
            )

        # Track active sessions
        self._sessions: dict[str, str] = {}

    def _create_app_and_runner(
        self,
        config: Configs,
        enable_caching: bool,
        enable_compaction: bool,
    ) -> tuple[App, Runner]:
        """Create App and Runner with caching/compaction config."""
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
                compaction_interval=self._compaction_interval,
                overlap_size=1,
                summarizer=summarizer,
            )

        app = App(**app_kwargs)
        runner = Runner(
            app=app,
            session_service=self._session_service,
        )
        return app, runner

    @staticmethod
    def _supports_thinking(model: str) -> bool:
        """Check if a model supports thinking configuration.

        Thinking is supported by gemini-2.5-* and gemini-3-* models.
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
        """Get the current research plan if one exists."""
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
        """Update the current plan from PLANNING or REPLANNING tagged text."""
        plan_text = extract_plan_text(text)
        if not plan_text:
            return False

        steps = parse_plan_steps_from_text(plan_text)
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
        """Process event text to extract and update plan."""
        if not text or not self._current_plan:
            return

        # Check for replanning first
        if REPLANNING_TAG in text:
            self._update_plan_from_text(text, question, is_replan=True)
        # Check for initial planning (only if plan is empty)
        elif PLANNING_TAG in text and len(self._current_plan.steps) == 0:
            self._update_plan_from_text(text, question, is_replan=False)

    def _update_plan_step_from_tool_call(self, tool_name: str) -> None:
        """Record tool call against current plan step."""
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
        """Advance to next plan step when reasoning is detected."""
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
        event_text = extract_event_text(event)

        # Extract thoughts for reasoning chain
        thoughts = extract_thoughts_from_event(event)
        if thoughts:
            results["reasoning_chain"].append(thoughts[:300])

        # Process plan tags and reasoning
        if event_text:
            self._process_event_text_for_plan(event_text, question)
            reasoning_text = extract_reasoning_text(event_text)
            if reasoning_text:
                results["reasoning_chain"].append(reasoning_text[:300])
                self._advance_plan_step_on_reasoning()

        # Extract tool calls
        new_tool_calls = extract_tool_calls(event)
        results["tool_calls"].extend(new_tool_calls)
        results["search_queries"].extend(extract_search_queries_from_tool_calls(new_tool_calls))
        for tc in new_tool_calls:
            self._update_plan_step_from_tool_call(tc.get("name", ""))

        # Extract sources
        results["sources"].extend(extract_sources_from_responses(event))
        results["sources"].extend(extract_grounding_sources(event))
        for q in extract_grounding_queries(event):
            if q not in results["search_queries"]:
                results["search_queries"].append(q)

        # Extract final response - prefer tagged answer, fall back to final response
        # Only overwrite with non-empty content to avoid losing valid responses
        final_answer = extract_final_answer_text(event_text) if event_text else None
        if final_answer:
            results["final_response"] = final_answer
        else:
            text = extract_final_response(event)
            if text:  # Only set if non-empty (don't overwrite valid response with empty)
                results["final_response"] = text

    async def _run_agent_once_inner(
        self,
        question: str,
        adk_session_id: str,
    ) -> dict[str, Any]:
        """Run the agent once and collect results (inner implementation)."""
        content = types.Content(role="user", parts=[types.Part(text=question)])

        # Collect results in a mutable dict for _process_event
        results: dict[str, Any] = {
            "tool_calls": [],
            "sources": [],
            "search_queries": [],
            "reasoning_chain": [],
            "final_response": "",
        }

        event_count = 0
        async for event in self._runner.run_async(
            user_id="user",
            session_id=adk_session_id,
            new_message=content,
        ):
            event_count += 1
            self._process_event(event, question, results)

        logger.debug(f"Processed {event_count} events. Final response length: {len(results.get('final_response', ''))}")
        return results

    async def _run_agent_once(
        self,
        question: str,
        adk_session_id: str,
    ) -> dict[str, Any]:
        """Run the agent once with retry logic for rate limits and context overflow.

        Wraps _run_agent_once_inner with exponential backoff retry for
        429/RESOURCE_EXHAUSTED errors from the Gemini API. If a context overflow
        error occurs, resets the session and retries once with fresh context.
        """

        # Create a retry-wrapped version of the inner method
        @retry(
            retry=retry_if_exception(is_retryable_api_error),
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

        try:
            return await _run_with_retry()
        except ClientError as e:
            # Handle context overflow by resetting session
            if is_context_overflow_error(e):
                logger.warning(f"Context overflow detected: {e}")
                logger.warning("Resetting session and retrying with fresh context...")

                # Create fresh session to clear accumulated history
                self._session_service = InMemorySessionService()
                new_session_id = await self._get_or_create_session_async()

                # Retry once with fresh session
                try:
                    return await self._run_agent_once_inner(question, new_session_id)
                except Exception as retry_error:
                    logger.error(f"Retry with fresh session failed: {retry_error}")
                    raise RuntimeError(
                        f"Context overflow error. Original error: {e}. "
                        f"Retry with fresh session also failed: {retry_error}"
                    ) from e

            # Re-raise non-context-overflow errors
            raise

    async def answer_async(
        self,
        question: str,
        session_id: str | None = None,
    ) -> AgentResponse:
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
        AgentResponse
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
                # All retries exhausted - log detailed diagnostics
                tool_call_count = len(results.get("tool_calls", []))
                reasoning_count = len(results.get("reasoning_chain", []))
                source_count = len(results.get("sources", []))
                logger.error(
                    f"Empty model response after {MAX_EMPTY_RESPONSE_RETRIES + 1} attempts. "
                    f"Tool calls: {tool_call_count}, Reasoning steps: {reasoning_count}, "
                    f"Sources: {source_count}. The model may have only produced thinking tokens."
                )
                # Try to salvage any useful content from reasoning chain
                if results.get("reasoning_chain") and not results.get("final_response"):
                    # Use last reasoning as fallback response
                    last_reasoning = results["reasoning_chain"][-1]
                    if last_reasoning and len(last_reasoning) > 20:
                        logger.warning("Using last reasoning step as fallback response")
                        results["final_response"] = f"[Partial response from reasoning]: {last_reasoning}"

        total_duration_ms = int((time.time() - start_time) * 1000)

        # Mark remaining steps as completed
        if self._current_plan:
            for step in self._current_plan.steps:
                if step.status in (StepStatus.PENDING, StepStatus.IN_PROGRESS):
                    step.status = StepStatus.COMPLETED

        # Resolve redirect URLs and build response
        resolved_sources = await resolve_source_urls(results.get("sources", []))
        plan = self._current_plan or ResearchPlan(original_question=question, steps=[], reasoning="No planning enabled")
        execution_trace = self._create_execution_trace(results.get("tool_calls", []), total_duration_ms)
        self._current_plan = None

        return AgentResponse(
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
    ) -> AgentResponse:
        """Answer a question using built-in planning and tools (sync).

        Parameters
        ----------
        question : str
            The question to answer.
        session_id : str, optional
            Session ID for multi-turn conversations.

        Returns
        -------
        AgentResponse
            The response with plan, execution trace, and sources.
        """
        logger.info(f"Answering question (sync): {question[:100]}...")
        return asyncio.run(self.answer_async(question, session_id))

    def format_answer(self, response: AgentResponse) -> str:
        """Format an enhanced response for display.

        Parameters
        ----------
        response : AgentResponse
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
        """Get or create the config instance."""
        if self._config is None:
            self._config = Configs()  # type: ignore[call-arg]
        return self._config

    @property
    def agent(self) -> KnowledgeGroundedAgent:
        """Get or create the knowledge-grounded agent."""
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
        """Check if any clients have been initialized."""
        return self._initialized

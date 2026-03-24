"""PDF metadata extraction agent using Google ADK.

This module provides an agent that reads local PDF files and extracts
structured metadata as JSON, given an extraction prompt. The jurisdiction
is auto-detected from the document content.
"""

import asyncio
import logging
import time
import uuid
import warnings
from typing import Any

from aieng.agent_evals.configs import Configs
from aieng.agent_evals.knowledge_qa.event_extraction import (
    extract_event_text,
    extract_final_response,
    extract_thoughts_from_event,
    extract_tool_calls,
)
from aieng.agent_evals.knowledge_qa.plan_parsing import (
    extract_final_answer_text,
)
from aieng.agent_evals.knowledge_qa.retry import (
    API_RETRY_INITIAL_WAIT,
    API_RETRY_JITTER,
    API_RETRY_MAX_ATTEMPTS,
    API_RETRY_MAX_WAIT,
    MAX_EMPTY_RESPONSE_RETRIES,
    is_context_overflow_error,
    is_retryable_api_error,
)
from aieng.agent_evals.knowledge_qa.token_tracker import TokenTracker
from .system_instructions import PDF_SYSTEM_INSTRUCTIONS                                                                                              
from .tools import create_fetch_html_page_tool, create_read_pdf_tool
from aieng.agent_evals.tools import GroundingChunk
from google.adk.agents import Agent
from google.adk.agents.base_agent import AfterAgentCallback
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.genai.errors import ClientError
from pydantic import BaseModel, Field
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
from aieng.agent_evals.langfuse import init_tracing


warnings.filterwarnings("ignore", message=r".*EXPERIMENTAL.*")

logger = logging.getLogger(__name__)


class AgentResponse(BaseModel):
    """Response from the PDF extraction agent.

    Attributes
    ----------
    text : str
        The JSON metadata response.
    sources : list[GroundingChunk]
        Sources referenced (typically the PDF file).
    reasoning_chain : list[str]
        Step-by-step reasoning trace.
    tool_calls : list[dict]
        Raw tool calls made during execution.
    total_duration_ms : int
        Total execution time in milliseconds.
    """

    text: str
    sources: list[GroundingChunk] = Field(default_factory=list)
    reasoning_chain: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    total_duration_ms: int = 0


class LegislativeContentExtractionAgent:
    """Agent that extracts structured metadata from PDF files as JSON.

    Takes a PDF file path and extraction prompt as input. The agent reads
    the PDF using the read_pdf tool, auto-detects the jurisdiction from
    the document content, and returns the requested metadata as a JSON object.

    Parameters
    ----------
    config : Configs, optional
        Configuration settings. If not provided, creates default config.
    model : str, optional
        The model to use. If not provided, uses config.default_worker_model.
    thinking_budget : int, default 8192
        Token budget for the model's thinking phase.

    Examples
    --------
    >>> from aieng.agent_evals.legislative_content_extraction import LegislativeContentExtractionAgent
    >>> agent = LegislativeContentExtractionAgent()
    >>> response = await agent.answer_async(
    ...     pdf_path="/path/to/document.pdf",
    ...     prompt="Extract legislative metadata from this bill.",
    ... )
    >>> print(response.text)
    """

    def __init__(
        self,
        config: Configs | None = None,
        model: str | None = None,
        thinking_budget: int = 8192,
        after_agent_callback: AfterAgentCallback | None = None,
        files_dir: str | None = None,
    ) -> None:
        if config is None:
            config = Configs()  # type: ignore[call-arg]

        self.config = config
        self.model = model or config.default_worker_model
        self.temperature = config.default_temperature
        self._thinking_budget = thinking_budget

        self._read_pdf_tool = create_read_pdf_tool()
        self._fetch_html_page_tool = create_fetch_html_page_tool(cache_dir=files_dir)

        thinking_config = None
        if thinking_budget > 0 and self._supports_thinking(self.model):
            thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)

        self._agent = Agent(
            name="legislative_content_extraction",
            model=self.model,
            instruction=PDF_SYSTEM_INSTRUCTIONS,
            tools=[self._read_pdf_tool, self._fetch_html_page_tool],
            generate_content_config=types.GenerateContentConfig(
                temperature=self.temperature,
                thinking_config=thinking_config,
            ),
            after_agent_callback=after_agent_callback,
        )
        # Setup langfuse tracing if project name is provided
        
        init_tracing(service_name="legislative_content_extraction")

        self._token_tracker = TokenTracker(model=self.model)
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            app_name="legislative_content_extraction",
            agent=self._agent,
            session_service=self._session_service,
        )
        self._sessions: dict[str, str] = {}

    @staticmethod
    def _supports_thinking(model: str) -> bool:
        """Check if a model supports thinking configuration."""
        model_lower = model.lower()
        return "gemini-2.5" in model_lower or "gemini-3" in model_lower

    def reset(self) -> None:
        """Reset agent state for a new extraction."""
        self._sessions.clear()
        self._session_service = InMemorySessionService()
        self._token_tracker = TokenTracker(model=self.model)
        self._runner = Runner(
            app_name="legislative_content_extraction",
            agent=self._agent,
            session_service=self._session_service,
        )

    @property
    def adk_agent(self) -> Agent:
        """Return the underlying ADK agent."""
        return self._agent

    @property
    def token_tracker(self) -> TokenTracker:
        """Get the token tracker for context usage monitoring."""
        return self._token_tracker

    async def _get_or_create_session_async(self, session_id: str | None = None) -> str:
        """Get or create an ADK session."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id not in self._sessions:
            session = await self._session_service.create_session(
                app_name="legislative_content_extraction",
                user_id="user",
                state={},
            )
            self._sessions[session_id] = session.id

        return self._sessions[session_id]

    def _build_user_message(self, pdf_path: str, prompt: str, html_page_link: str = "") -> str:
        """Build the user message combining pdf_path, prompt, and optional html_page_link."""
        message = f"PDF file path: {pdf_path}\n\n"
        if html_page_link:
            message += f"HTML page link: {html_page_link}\n\n"
        message += prompt
        return message

    def _process_event(
        self,
        event: Any,
        results: dict[str, Any],
    ) -> None:
        """Process a single event from the agent run loop."""
        self._token_tracker.add_from_event(event)
        event_text = extract_event_text(event)

        thoughts = extract_thoughts_from_event(event)
        if thoughts:
            results["reasoning_chain"].append(thoughts)

        # Capture non-thought text from non-final events as reasoning
        is_final = hasattr(event, "is_final_response") and event.is_final_response()
        if event_text and not thoughts and not is_final:
            results["reasoning_chain"].append(event_text.strip())

        new_tool_calls = extract_tool_calls(event)
        results["tool_calls"].extend(new_tool_calls)

        final_answer = extract_final_answer_text(event_text) if event_text else None
        if final_answer:
            results["final_response"] = final_answer
        else:
            text = extract_final_response(event)
            if text:
                results["final_response"] = text

    async def _run_agent_once_inner(
        self,
        user_message: str,
        adk_session_id: str,
    ) -> dict[str, Any]:
        """Run the agent once and collect results."""
        content = types.Content(role="user", parts=[types.Part(text=user_message)])

        results: dict[str, Any] = {
            "tool_calls": [],
            "reasoning_chain": [],
            "final_response": "",
        }

        async for event in self._runner.run_async(
            user_id="user",
            session_id=adk_session_id,
            new_message=content,
        ):
            self._process_event(event, results)

        return results

    async def _run_agent_once(
        self,
        user_message: str,
        adk_session_id: str,
    ) -> dict[str, Any]:
        """Run the agent once with retry logic for rate limits."""

        @retry(
            retry=retry_if_exception(is_retryable_api_error),
            wait=wait_exponential_jitter(
                initial=API_RETRY_INITIAL_WAIT,
                max=API_RETRY_MAX_WAIT,
                jitter=API_RETRY_JITTER,
            ),
            stop=stop_after_attempt(API_RETRY_MAX_ATTEMPTS),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=False,
        )
        async def _run_with_retry() -> dict[str, Any]:
            return await self._run_agent_once_inner(user_message, adk_session_id)

        try:
            return await _run_with_retry()
        except RetryError as e:
            original_error = e.last_attempt.exception()
            logger.error(f"API retry failed after {API_RETRY_MAX_ATTEMPTS} attempts: {original_error}")
            raise RuntimeError(
                f"API request failed after {API_RETRY_MAX_ATTEMPTS} retry attempts. "
                f"Last error: {original_error}"
            ) from original_error
        except ClientError as e:
            if is_context_overflow_error(e):
                logger.warning(f"Context overflow detected: {e}")
                self._session_service = InMemorySessionService()
                new_session_id = await self._get_or_create_session_async()
                try:
                    return await self._run_agent_once_inner(user_message, new_session_id)
                except Exception as retry_error:
                    raise RuntimeError(
                        f"Context overflow error. Retry also failed: {retry_error}"
                    ) from e
            raise

    async def answer_async(
        self,
        pdf_path: str,
        prompt: str,
        session_id: str | None = None,
        html_page_link: str = "",
    ) -> AgentResponse:
        """Extract metadata from a PDF file.

        Parameters
        ----------
        pdf_path : str
            Absolute path to the local PDF file.
        prompt : str
            Extraction prompt describing what metadata to extract.
        session_id : str, optional
            Session ID for conversation continuity.
        html_page_link : str, optional
            URL to the legislative HTML page for supplementary information.

        Returns
        -------
        AgentResponse
            The response containing JSON metadata in the text field.
        """
        start_time = time.time()
        user_message = self._build_user_message(pdf_path, prompt, html_page_link)
        logger.info(f"Extracting metadata from: {pdf_path}")

        adk_session_id = await self._get_or_create_session_async(session_id)

        results: dict[str, Any] = {}
        current_session_id = adk_session_id

        for attempt in range(MAX_EMPTY_RESPONSE_RETRIES + 1):
            results = await self._run_agent_once(user_message, current_session_id)

            if results.get("final_response", "").strip():
                break

            if attempt < MAX_EMPTY_RESPONSE_RETRIES:
                logger.warning(
                    f"Empty response (attempt {attempt + 1}/{MAX_EMPTY_RESPONSE_RETRIES + 1}), retrying..."
                )
                fresh_session = await self._session_service.create_session(
                    app_name="legislative_content_extraction",
                    user_id="user",
                    state={},
                )
                current_session_id = fresh_session.id
            else:
                logger.error(
                    f"Empty response after {MAX_EMPTY_RESPONSE_RETRIES + 1} attempts for {pdf_path}."
                )

        total_duration_ms = int((time.time() - start_time) * 1000)

        return AgentResponse(
            text=results.get("final_response", ""),
            reasoning_chain=results.get("reasoning_chain", []),
            tool_calls=results.get("tool_calls", []),
            total_duration_ms=total_duration_ms,
        )

    def answer(
        self,
        pdf_path: str,
        prompt: str,
        session_id: str | None = None,
        html_page_link: str = "",
    ) -> AgentResponse:
        """Extract metadata from a PDF file (sync wrapper).

        Parameters
        ----------
        pdf_path : str
            Absolute path to the local PDF file.
        prompt : str
            Extraction prompt describing what metadata to extract.
        session_id : str, optional
            Session ID for conversation continuity.
        html_page_link : str, optional
            URL to the legislative HTML page for supplementary information.

        Returns
        -------
        AgentResponse
            The response containing JSON metadata in the text field.
        """
        return asyncio.run(self.answer_async(pdf_path, prompt, session_id, html_page_link))

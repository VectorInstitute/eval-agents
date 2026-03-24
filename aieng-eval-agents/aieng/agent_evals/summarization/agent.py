"""Financial news summarization agent using Google ADK.

This module provides a simple single-pass agent that accepts a news article
(title + body) and returns a concise summary. No web tools or planning are
used — the article content is the sole input.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from aieng.agent_evals.configs import Configs
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.genai.errors import ClientError
from pydantic import BaseModel
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .event_extraction import (
    extract_event_text,
    extract_final_response,
    extract_thoughts_from_event,
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


logger = logging.getLogger(__name__)


class SummarizationResponse(BaseModel):
    """Response from the summarization agent.

    Attributes
    ----------
    text : str
        The generated summary text.
    reasoning_chain : list[str]
        Step-by-step reasoning trace from the model's thinking tokens.
    total_duration_ms : int
        Total execution time in milliseconds.
    """

    text: str
    reasoning_chain: list[str] = []
    total_duration_ms: int = 0


class SummarizationAgent:
    """A single-pass agent that summarizes financial news articles.

    Accepts an article title and body, and returns a concise 2-4 sentence
    summary using a Gemini model. No web tools or planning are used.

    Parameters
    ----------
    config : Configs, optional
        Configuration settings. If not provided, creates default config.
    model : str, optional
        The model to use. If not provided, uses config.default_worker_model.
    thinking_budget : int, default 4096
        Token budget for the model's thinking phase.

    Examples
    --------
    >>> from aieng.agent_evals.summarization import SummarizationAgent
    >>> agent = SummarizationAgent()
    >>> response = agent.summarize(title="Apple reports record profits", body="Apple Inc. reported...")
    >>> print(response.text)
    """

    def __init__(
        self,
        config: Configs | None = None,
        model: str | None = None,
        thinking_budget: int = 4096,
    ) -> None:
        """Initialize the summarization agent.

        Parameters
        ----------
        config : Configs, optional
            Configuration settings. If not provided, creates default config.
        model : str, optional
            The model to use. If not provided, uses config.default_worker_model.
        thinking_budget : int, default 4096
            Token budget for the model's thinking phase. Set to 0 to disable.
        """
        if config is None:
            config = Configs()  # type: ignore[call-arg]

        self.config = config
        self.model = model or config.default_worker_model
        self.temperature = config.default_temperature
        self._thinking_budget = thinking_budget

        thinking_config = None
        if thinking_budget > 0 and self._supports_thinking(self.model):
            thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)

        self._agent = Agent(
            name="summarization",
            model=self.model,
            instruction=build_system_instructions(),
            tools=[],
            generate_content_config=types.GenerateContentConfig(
                temperature=self.temperature,
                thinking_config=thinking_config,
            ),
        )

        self._token_tracker = TokenTracker(model=self.model)
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            app_name="summarization",
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
        """Reset agent state between articles.

        Clears session history to ensure clean execution for each new article.
        """
        self._sessions.clear()
        self._session_service = InMemorySessionService()
        self._token_tracker = TokenTracker(model=self.model)
        self._runner = Runner(
            app_name="summarization",
            agent=self._agent,
            session_service=self._session_service,
        )
        logger.debug("Agent state reset")

    @property
    def adk_agent(self) -> Agent:
        """Return the underlying ADK agent, e.g. for use with ``adk web``."""
        return self._agent

    @property
    def token_tracker(self) -> TokenTracker:
        """Get the token tracker for context usage monitoring."""
        return self._token_tracker

    async def _get_or_create_session_async(self, session_id: str | None = None) -> str:
        """Get or create an ADK session for the given session ID."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id not in self._sessions:
            session = await self._session_service.create_session(
                app_name="summarization",
                user_id="user",
                state={},
            )
            self._sessions[session_id] = session.id

        return self._sessions[session_id]

    def _process_event(
        self,
        event: Any,
        results: dict[str, Any],
    ) -> None:
        """Process a single event from the agent run loop."""
        self._token_tracker.add_from_event(event)

        thoughts = extract_thoughts_from_event(event)
        if thoughts:
            results["reasoning_chain"].append(thoughts[:300])

        text = extract_final_response(event)
        if text:
            results["final_response"] = text

        event_text = extract_event_text(event)
        if event_text and not text:
            results["final_response"] = event_text

    async def _run_agent_once_inner(
        self,
        prompt: str,
        adk_session_id: str,
    ) -> dict[str, Any]:
        """Run the agent once and collect results."""
        content = types.Content(role="user", parts=[types.Part(text=prompt)])

        results: dict[str, Any] = {
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
        prompt: str,
        adk_session_id: str,
    ) -> dict[str, Any]:
        """Run the agent with retry logic for rate limits and context overflow."""

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
            return await self._run_agent_once_inner(prompt, adk_session_id)

        try:
            return await _run_with_retry()
        except RetryError as e:
            original_error = e.last_attempt.exception()
            logger.error(f"API retry failed after {API_RETRY_MAX_ATTEMPTS} attempts: {original_error}")
            raise RuntimeError(
                f"API request failed after {API_RETRY_MAX_ATTEMPTS} retry attempts. Last error: {original_error}"
            ) from original_error
        except ClientError as e:
            if is_context_overflow_error(e):
                logger.warning(f"Context overflow detected: {e}. Retrying with fresh session...")
                self._session_service = InMemorySessionService()
                new_session_id = await self._get_or_create_session_async()
                try:
                    return await self._run_agent_once_inner(prompt, new_session_id)
                except Exception as retry_error:
                    raise RuntimeError(
                        f"Context overflow error. Retry also failed: {retry_error}"
                    ) from e
            raise

    async def summarize_async(
        self,
        title: str,
        body: str,
        session_id: str | None = None,
    ) -> SummarizationResponse:
        """Summarize a financial news article.

        Parameters
        ----------
        title : str
            The article headline/title.
        body : str
            The full article body text.
        session_id : str, optional
            Session ID for reuse across calls.

        Returns
        -------
        SummarizationResponse
            The summary text and timing metadata.
        """
        start_time = time.time()
        prompt = f"Title: {title}\n\nArticle:\n{body}"
        logger.info(f"Summarizing article: {title[:80]}...")

        adk_session_id = await self._get_or_create_session_async(session_id)

        results: dict[str, Any] = {}
        current_session_id = adk_session_id

        for attempt in range(MAX_EMPTY_RESPONSE_RETRIES + 1):
            results = await self._run_agent_once(prompt, current_session_id)

            if results.get("final_response", "").strip():
                break

            if attempt < MAX_EMPTY_RESPONSE_RETRIES:
                logger.warning(
                    f"Empty response (attempt {attempt + 1}/{MAX_EMPTY_RESPONSE_RETRIES + 1}), retrying..."
                )
                fresh_session = await self._session_service.create_session(
                    app_name="summarization",
                    user_id="user",
                    state={},
                )
                current_session_id = fresh_session.id
            else:
                logger.error(f"Empty response after {MAX_EMPTY_RESPONSE_RETRIES + 1} attempts.")

        total_duration_ms = int((time.time() - start_time) * 1000)

        return SummarizationResponse(
            text=results.get("final_response", ""),
            reasoning_chain=results.get("reasoning_chain", []),
            total_duration_ms=total_duration_ms,
        )

    def summarize(
        self,
        title: str,
        body: str,
        session_id: str | None = None,
    ) -> SummarizationResponse:
        """Summarize a financial news article (sync).

        Parameters
        ----------
        title : str
            The article headline/title.
        body : str
            The full article body text.
        session_id : str, optional
            Session ID for reuse across calls.

        Returns
        -------
        SummarizationResponse
            The summary text and timing metadata.
        """
        return asyncio.run(self.summarize_async(title, body, session_id))


class SummarizationAgentManager:
    """Manages SummarizationAgent lifecycle with lazy initialization.

    Parameters
    ----------
    config : Configs, optional
        Configuration object for client setup. If not provided, creates default.

    Examples
    --------
    >>> manager = SummarizationAgentManager()
    >>> agent = manager.agent
    >>> response = await agent.summarize_async(title="...", body="...")
    >>> print(response.text)
    >>> manager.close()
    """

    def __init__(
        self,
        config: Configs | None = None,
    ) -> None:
        """Initialize the manager.

        Parameters
        ----------
        config : Configs, optional
            Configuration object. If not provided, creates default config.
        """
        self._config = config
        self._agent: SummarizationAgent | None = None
        self._initialized = False

    @property
    def config(self) -> Configs:
        """Get or create the config instance."""
        if self._config is None:
            self._config = Configs()  # type: ignore[call-arg]
        return self._config

    @property
    def agent(self) -> SummarizationAgent:
        """Get or create the summarization agent."""
        if self._agent is None:
            self._agent = SummarizationAgent(config=self.config)
            self._initialized = True
        return self._agent

    def close(self) -> None:
        """Close all initialized clients and reset state."""
        self._agent = None
        self._initialized = False

    def is_initialized(self) -> bool:
        """Check if any clients have been initialized."""
        return self._initialized

"""Token usage tracking for Gemini models.

This module provides utilities for tracking token usage and context
window consumption during agent execution.
"""

import logging
import os
from typing import Any

from google.genai import Client
from pydantic import BaseModel


logger = logging.getLogger(__name__)
DEFAULT_MODEL = os.environ.get("DEFAULT_WORKER_MODEL", "gemini-2.5-flash")

# Known context limits for Gemini models (as of 2025)
# Used as fallback if API fetch fails
KNOWN_MODEL_LIMITS: dict[str, int] = {
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
}


class TokenUsage(BaseModel):
    """Token usage statistics.

    Attributes
    ----------
    latest_prompt_tokens : int
        Prompt tokens from the most recent API call. This represents the
        actual current context size since each call includes full history.
    latest_cached_tokens : int
        Cached tokens from the most recent API call.
    total_prompt_tokens : int
        Cumulative prompt tokens across all calls (for cost tracking).
    total_completion_tokens : int
        Cumulative completion tokens across all calls.
    total_tokens : int
        Cumulative total tokens across all calls.
    context_limit : int
        Maximum context window size for the model.
    """

    latest_prompt_tokens: int = 0
    latest_cached_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    context_limit: int = 1_000_000  # Default for Gemini 2.5 Flash

    @property
    def uncached_prompt_tokens(self) -> int:
        """Prompt tokens excluding cached content (for cost estimation).

        Cached tokens are billed at a lower rate, so this property is useful
        for cost tracking. Note: cached tokens still count against the context
        window limit - use latest_prompt_tokens for context window calculations.
        """
        return max(0, self.latest_prompt_tokens - self.latest_cached_tokens)

    @property
    def context_used_percent(self) -> float:
        """Calculate percentage of context window currently used.

        Uses the latest prompt tokens (total, including cached) since cached
        tokens still occupy space in the context window. Caching only affects
        processing speed and billing, not the context window limit.
        """
        if self.context_limit == 0:
            return 0.0
        return (self.latest_prompt_tokens / self.context_limit) * 100

    @property
    def context_remaining_percent(self) -> float:
        """Calculate percentage of context window remaining."""
        return max(0.0, 100.0 - self.context_used_percent)

    # Backwards compatibility aliases
    @property
    def prompt_tokens(self) -> int:
        """Alias for total_prompt_tokens (backwards compatibility)."""
        return self.total_prompt_tokens

    @property
    def cached_tokens(self) -> int:
        """Alias for latest_cached_tokens (backwards compatibility)."""
        return self.latest_cached_tokens

    @property
    def completion_tokens(self) -> int:
        """Alias for total_completion_tokens (backwards compatibility)."""
        return self.total_completion_tokens


class TokenTracker:
    """Tracks token usage across agent interactions.

    Parameters
    ----------
    model : str
        The model name to track tokens for.

    Examples
    --------
    >>> tracker = TokenTracker()  # Uses DEFAULT_WORKER_MODEL from .env
    >>> tracker.add_from_event(event)
    >>> print(f"Context remaining: {tracker.usage.context_remaining_percent:.1f}%")
    """

    def __init__(self, model: str | None = None) -> None:
        """Initialize the token tracker.

        Parameters
        ----------
        model : str, optional
            The model name to fetch context limits for.
            Defaults to DEFAULT_WORKER_MODEL from environment.
        """
        self._model = model or DEFAULT_MODEL
        self._usage = TokenUsage()
        self._fetch_model_limits()

    def _fetch_model_limits(self) -> None:
        """Fetch model context limits from the API, with known fallbacks."""
        try:
            client = Client()
            model_info = client.models.get(model=self._model)
            if model_info.input_token_limit:
                self._usage.context_limit = model_info.input_token_limit
                logger.debug(f"Model {self._model} context limit: {self._usage.context_limit}")
                return
        except Exception as e:
            logger.warning(f"Failed to fetch model limits from API: {e}")

        # Use known fallback if available
        if self._model in KNOWN_MODEL_LIMITS:
            self._usage.context_limit = KNOWN_MODEL_LIMITS[self._model]
            logger.info(f"Using known limit for {self._model}: {self._usage.context_limit}")
        else:
            logger.warning(f"Unknown model {self._model}, using default limit: {self._usage.context_limit}")

    @property
    def usage(self) -> TokenUsage:
        """Get current token usage statistics."""
        return self._usage

    def add_from_event(self, event: Any) -> None:
        """Add token usage from an ADK event.

        Updates both the latest token counts (for context tracking) and
        cumulative totals (for cost tracking).

        Parameters
        ----------
        event : Any
            An event from the ADK runner that may contain usage_metadata.
        """
        if not hasattr(event, "usage_metadata") or event.usage_metadata is None:
            return

        metadata = event.usage_metadata

        # Extract token counts from this API call
        prompt = getattr(metadata, "prompt_token_count", 0) or 0
        cached = getattr(metadata, "cached_content_token_count", 0) or 0
        completion = getattr(metadata, "candidates_token_count", 0) or 0
        total = getattr(metadata, "total_token_count", 0) or 0

        # Update LATEST tokens - this reflects current context size
        # Each API call includes full conversation history, so the latest
        # prompt_token_count is the actual current context usage
        self._usage.latest_prompt_tokens = prompt
        self._usage.latest_cached_tokens = cached

        # Accumulate totals for cost/usage tracking
        self._usage.total_prompt_tokens += prompt
        self._usage.total_completion_tokens += completion
        self._usage.total_tokens += total

        logger.debug(
            f"Token update: prompt={prompt} (cached={cached}), context: {self._usage.context_used_percent:.1f}% used"
        )

    def reset(self) -> None:
        """Reset all token counts (keeps context limit)."""
        context_limit = self._usage.context_limit
        self._usage = TokenUsage(
            latest_prompt_tokens=0,
            latest_cached_tokens=0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_tokens=0,
            context_limit=context_limit,
        )

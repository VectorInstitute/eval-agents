"""Token usage tracking for Gemini models.

This module provides utilities for tracking token usage and context
window consumption during agent execution.
"""

import logging
import os
from typing import Any

from google import genai
from pydantic import BaseModel


logger = logging.getLogger(__name__)
DEFAULT_MODEL = os.environ.get("DEFAULT_WORKER_MODEL", "gemini-2.5-flash")


class TokenUsage(BaseModel):
    """Token usage statistics.

    Attributes
    ----------
    prompt_tokens : int
        Total prompt/input tokens used.
    cached_tokens : int
        Tokens served from cache (don't count against new context).
    completion_tokens : int
        Total completion/output tokens used.
    total_tokens : int
        Total tokens used (prompt + completion).
    context_limit : int
        Maximum context window size for the model.
    """

    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    context_limit: int = 1_000_000  # Default for Gemini 2.0 Flash

    @property
    def uncached_prompt_tokens(self) -> int:
        """Prompt tokens excluding cached content."""
        return max(0, self.prompt_tokens - self.cached_tokens)

    @property
    def context_used_percent(self) -> float:
        """Calculate percentage of context window used.

        Uses uncached prompt tokens since cached content is stored separately
        and doesn't count against the active context window.
        """
        if self.context_limit == 0:
            return 0.0
        return (self.uncached_prompt_tokens / self.context_limit) * 100

    @property
    def context_remaining_percent(self) -> float:
        """Calculate percentage of context window remaining."""
        return max(0.0, 100.0 - self.context_used_percent)


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
        """Fetch model context limits from the API."""
        try:
            client = genai.Client()
            model_info = client.models.get(model=self._model)
            if model_info.input_token_limit:
                self._usage.context_limit = model_info.input_token_limit
                logger.debug(f"Model {self._model} context limit: {self._usage.context_limit}")
        except Exception as e:
            logger.warning(f"Failed to fetch model limits: {e}. Using default.")

    @property
    def usage(self) -> TokenUsage:
        """Get current token usage statistics."""
        return self._usage

    def add_from_event(self, event: Any) -> None:
        """Add token usage from an ADK event.

        Parameters
        ----------
        event : Any
            An event from the ADK runner that may contain usage_metadata.
        """
        if not hasattr(event, "usage_metadata") or event.usage_metadata is None:
            return

        metadata = event.usage_metadata

        # Extract token counts
        prompt = getattr(metadata, "prompt_token_count", 0) or 0
        cached = getattr(metadata, "cached_content_token_count", 0) or 0
        completion = getattr(metadata, "candidates_token_count", 0) or 0
        total = getattr(metadata, "total_token_count", 0) or 0

        # Accumulate tokens
        self._usage.prompt_tokens += prompt
        self._usage.cached_tokens += cached
        self._usage.completion_tokens += completion
        self._usage.total_tokens += total

        logger.debug(
            f"Token update: +{total} total (+{cached} cached), "
            f"cumulative: {self._usage.total_tokens} ({self._usage.cached_tokens} cached)"
        )

    def reset(self) -> None:
        """Reset token counts (keeps context limit)."""
        context_limit = self._usage.context_limit
        self._usage = TokenUsage(context_limit=context_limit)

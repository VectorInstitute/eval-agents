"""Configuration classes for LLM-based graders."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMRequestConfig:
    """Configuration for the underlying judge model call.

    Parameters
    ----------
    model : str | None, optional, default=None
        Explicit model name for the judge. If omitted, the harness default
        evaluator model is used.
    temperature : float, optional, default=0.0
        Sampling temperature for the judge call.
    max_completion_tokens : int | None, optional, default=None
        Optional token cap for the judge completion.
    timeout_sec : float | None, optional, default=None
        Optional request timeout in seconds.
    extra_request_kwargs : dict[str, Any], optional, default_factory=dict
        Additional OpenAI-compatible request arguments forwarded to
        ``chat.completions.parse``.
    retry_max_attempts : int, optional, default=5
        Maximum number of attempts for transient judge API failures. Set to
        ``1`` to disable retries.
    retry_initial_wait_sec : float, optional, default=1.0
        Initial backoff delay in seconds.
    retry_max_wait_sec : float, optional, default=10.0
        Maximum backoff delay in seconds.
    retry_backoff_multiplier : float, optional, default=2.0
        Exponential backoff multiplier.
    """

    model: str | None = None
    temperature: float = 0.0
    max_completion_tokens: int | None = None
    timeout_sec: float | None = None
    extra_request_kwargs: dict[str, Any] = field(default_factory=dict)
    retry_max_attempts: int = 5
    retry_initial_wait_sec: float = 1.0
    retry_max_wait_sec: float = 10.0
    retry_backoff_multiplier: float = 2.0

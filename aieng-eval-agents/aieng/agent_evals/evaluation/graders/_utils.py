"""Shared helpers for OpenAI-compatible LLM-based graders."""

import json
from pathlib import Path
from typing import Any, TypeVar, cast

from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.evaluation.types import Evaluation
from langfuse.api import ScoreDataType
from openai import APIConnectionError, APIStatusError, APITimeoutError, InternalServerError, RateLimitError
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential


T = TypeVar("T", bound=BaseModel)


async def run_structured_parse_call(
    *,
    openai_client: Any,
    default_model: str,
    model_config: LLMRequestConfig,
    system_prompt: str,
    user_prompt: str,
    response_format: type[T],
) -> ParsedChatCompletion[T]:
    """Run ``chat.completions.parse`` with retry for transient API failures.

    Parameters
    ----------
    openai_client : Any
        OpenAI-compatible async client instance.
    default_model : str
        Fallback model name when ``model_config.model`` is not provided.
    model_config : LLMRequestConfig
        Request and retry configuration.
    system_prompt : str
        System prompt content.
    user_prompt : str
        User prompt content.
    response_format : type[T]
        Pydantic model used by ``parse`` for structured output.

    Returns
    -------
    ParsedChatCompletion[T]
        Completion object returned by ``chat.completions.parse``.
    """
    model_name = model_config.model or default_model
    request_kwargs: dict[str, Any] = dict(model_config.extra_request_kwargs)
    request_kwargs.update(
        {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": response_format,
            "temperature": model_config.temperature,
        }
    )
    if model_config.max_completion_tokens is not None:
        request_kwargs["max_completion_tokens"] = model_config.max_completion_tokens
    if model_config.timeout_sec is not None:
        request_kwargs["timeout"] = model_config.timeout_sec

    retrying = AsyncRetrying(
        stop=stop_after_attempt(model_config.retry_max_attempts),
        wait=wait_exponential(
            multiplier=model_config.retry_backoff_multiplier,
            min=model_config.retry_initial_wait_sec,
            max=model_config.retry_max_wait_sec,
        ),
        retry=retry_if_exception(is_retryable_api_exception),
        reraise=True,
    )

    async for attempt in retrying:
        with attempt:
            response = await openai_client.chat.completions.parse(**request_kwargs)
            return cast(ParsedChatCompletion[T], response)

    # Defensive fallback: tenacity should either return above or raise.
    raise RuntimeError("Structured parse call failed unexpectedly without a result.")


def is_retryable_api_exception(exc: BaseException) -> bool:
    """Return True when exception is likely transient and should be retried."""
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)):
        return True

    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        return status in (408, 429) or (status is not None and status >= 500)

    return False


def build_error_evaluation(*, name: str, error: Exception, prefix: str) -> Evaluation:
    """Build a deterministic error metric.

    Parameters
    ----------
    name : str
        Metric name.
    error : Exception
        Error that triggered the fallback metric.
    prefix : str
        Prefix used in the metric comment for context.

    Returns
    -------
    Evaluation
        Boolean error evaluation containing structured error metadata.
    """
    message = str(error) or error.__class__.__name__
    return Evaluation(
        name=name,
        value=True,
        comment=f"{prefix}: {message}",
        data_type=ScoreDataType.BOOLEAN,
        metadata={"error_type": error.__class__.__name__, "error": message},
    )


def render_system_prompt_with_optional_rubric(*, system_prompt_template: str, rubric_text: str | None) -> str:
    """Render system prompt and inject rubric text when available.

    Parameters
    ----------
    system_prompt_template : str
        Base system prompt template.
    rubric_text : str | None
        Rubric content in markdown format.

    Returns
    -------
    str
        Rendered system prompt with rubric inserted or appended.
    """
    rubric_section = ""
    if rubric_text:
        rubric_section = f"# Rubric\n{rubric_text.strip()}"

    if "{rubric_section}" in system_prompt_template:
        return system_prompt_template.format(rubric_section=rubric_section)

    if rubric_section:
        # Appending rubric keeps custom system templates simple when users omit
        # placeholders in quick evaluator setup.
        return f"{system_prompt_template.rstrip()}\n\n{rubric_section}\n"

    return system_prompt_template


def load_markdown(markdown: str | Path | None) -> str | None:
    """Load markdown from raw string or file path.

    Parameters
    ----------
    markdown : str | Path | None
        Markdown text or file path.

    Returns
    -------
    str | None
        Loaded markdown text, or ``None`` when not provided.
    """
    if markdown is None:
        return None
    if isinstance(markdown, Path):
        return markdown.read_text(encoding="utf-8")

    path_candidate = Path(markdown)
    if path_candidate.suffix.lower() == ".md" and path_candidate.exists():
        return path_candidate.read_text(encoding="utf-8")
    return markdown


def serialize_for_prompt(value: Any) -> str:
    """Serialize values to readable JSON-like prompt text.

    Parameters
    ----------
    value : Any
        Value to serialize.

    Returns
    -------
    str
        JSON-like string representation suitable for prompts.
    """
    try:
        # Keep unicode characters readable and stabilize formatting for
        # deterministic prompt snapshots during tests.
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "LLMRequestConfig",
    "build_error_evaluation",
    "is_retryable_api_exception",
    "load_markdown",
    "render_system_prompt_with_optional_rubric",
    "run_structured_parse_call",
    "serialize_for_prompt",
]

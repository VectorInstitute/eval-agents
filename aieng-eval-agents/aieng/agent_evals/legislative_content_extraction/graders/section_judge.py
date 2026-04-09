"""LLM-as-a-judge for legislative section matching.

This module provides a semantic fallback for section precision/recall scoring.
When exact normalization and fuzzy string matching both fail to pair a predicted
section with a reference section, a small LLM call determines whether the two
references are semantically equivalent (e.g., a specific §-level citation that
falls within the chapter-level ground-truth reference for Delaware bills).

The judge uses a hash-based in-process cache so each unique (pred, ref) pair
is evaluated at most once per process lifetime, eliminating redundant API calls
when the same normalization failure repeats across multiple eval items.

Usage
-----
>>> import asyncio
>>> from aieng.agent_evals.legislative_content_extraction.graders.section_judge import (
...     SectionMatchResult,
...     judge_section_pair,
... )
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic response schema
# ---------------------------------------------------------------------------


class SectionMatchResult(BaseModel):
    """Structured response from the section LLM judge.

    Parameters
    ----------
    match : bool
        ``True`` when the predicted and reference sections are semantically
        equivalent legislative modifications.
    confidence : float
        Judge's confidence in the verdict, in ``[0.0, 1.0]``.
    reasoning : str
        One-sentence explanation of the decision.
    """

    match: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert in legislative drafting and statutory citation.
Your task is to determine whether two legislative section references describe
the SAME statutory modification in a bill.

Rules:
- A specific section citation (e.g., "§ 2541, Title 6") counts as a match for
  a chapter-level citation (e.g., "Chapter 25, Title 6") when the section falls
  within that chapter.
- "MCL 408.931" and "section 1 (MCL 408.931)" refer to the same section.
- Actions must be semantically equivalent: AMEND ≈ amending, ADD ≈ adding/creating,
  REPEAL ≈ repealing/striking. Minor wording differences are acceptable.
- Different title numbers, chapter numbers, or MCL numbers are NOT a match.

Respond with JSON only (no markdown fences):
{
  "match": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "one sentence"
}
"""

_USER_PROMPT_TEMPLATE = """\
Jurisdiction: {jurisdiction}

Predicted section:
  raw_section: "{pred_raw}"
  action: "{pred_action}"

Reference section:
  raw_section: "{ref_raw}"
  action: "{ref_action}"

Are these the same statutory modification?
"""

# ---------------------------------------------------------------------------
# In-process cache
# ---------------------------------------------------------------------------

_CACHE: dict[str, SectionMatchResult] = {}


def _cache_key(jurisdiction: str, pred_raw: str, pred_action: str, ref_raw: str, ref_action: str) -> str:
    """Deterministic hash key for a (jurisdiction, pred, ref) triple."""
    payload = json.dumps(
        {
            "j": jurisdiction.upper(),
            "pr": pred_raw.upper(),
            "pa": pred_action.upper(),
            "rr": ref_raw.upper(),
            "ra": ref_action.upper(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Core judge function
# ---------------------------------------------------------------------------


async def judge_section_pair(
    *,
    jurisdiction: str,
    pred_raw: str,
    pred_action: str,
    ref_raw: str,
    ref_action: str,
    model: str = "gemini-2.5-pro",
    temperature: float = 0.0,
) -> SectionMatchResult:
    """Call the LLM judge to determine if two section refs are equivalent.

    Results are cached in-process by a SHA-256 key so repeated comparisons
    of the same pair do not incur additional API calls.

    Parameters
    ----------
    jurisdiction : str
        Two-letter jurisdiction code (e.g. ``"DE"``, ``"MI"``).
    pred_raw : str
        Raw section string from the agent (already extracted, not yet normalized).
    pred_action : str
        Action code from the agent (e.g. ``"AMEND"``).
    ref_raw : str
        Raw section string from the ground-truth dataset.
    ref_action : str
        Action code from the ground-truth dataset.
    model : str, optional
        Model name for the judge call.  Defaults to ``"gemini-2.5-pro"``.
    temperature : float, optional
        Sampling temperature.  Defaults to ``0.0`` for determinism.

    Returns
    -------
    SectionMatchResult
        Structured verdict from the LLM judge.

    Raises
    ------
    RuntimeError
        When the LLM API call fails after exhausting retries.
    """
    key = _cache_key(jurisdiction, pred_raw, pred_action, ref_raw, ref_action)
    if key in _CACHE:
        logger.debug("SectionJudge cache hit for key %s", key[:8])
        return _CACHE[key]

    # Lazy imports to keep module importable without heavy deps in test env.
    from aieng.agent_evals.async_client_manager import AsyncClientManager  # noqa: PLC0415
    from aieng.agent_evals.evaluation.graders._utils import run_structured_parse_call  # noqa: PLC0415
    from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig  # noqa: PLC0415

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        jurisdiction=jurisdiction,
        pred_raw=pred_raw,
        pred_action=pred_action,
        ref_raw=ref_raw,
        ref_action=ref_action,
    )

    config = LLMRequestConfig(model=model, temperature=temperature)
    client_manager = AsyncClientManager.get_instance()

    completion = await run_structured_parse_call(
        openai_client=client_manager.openai_client,
        default_model=client_manager.configs.default_evaluator_model,
        model_config=config,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_format=SectionMatchResult,
    )

    result: SectionMatchResult | None = completion.choices[0].message.parsed
    if result is None:
        raise RuntimeError(
            f"SectionJudge returned no parsed result for pair: pred='{pred_raw}' ref='{ref_raw}'"
        )

    _CACHE[key] = result
    logger.debug(
        "SectionJudge verdict: match=%s confidence=%.2f for pred='%s' ref='%s'",
        result.match,
        result.confidence,
        pred_raw[:60],
        ref_raw[:60],
    )
    return result


def clear_cache() -> None:
    """Clear the in-process section judge cache (primarily for testing)."""
    _CACHE.clear()


__all__ = [
    "SectionMatchResult",
    "judge_section_pair",
    "clear_cache",
]
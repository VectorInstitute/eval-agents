"""Item-level deterministic graders for legislative content extraction outputs.

Examples
--------
>>> from aieng.agent_evals.legislative_content_extraction.graders import (
...     item_level_deterministic_grader,
... )
>>> output = {
...     "jurisdiction_code": "ID",
...     "session_code": "ID_2025_2025_R1",
...     "chamber_code": "HOUSE",
...     "measure_type_code": "BILL",
...     "measure_number": "4",
...     "sponsors": ["State Affairs Committee"],
...     "sections_affected": [],
... }
>>> expected_output = {
...     "jurisdiction_code": "ID",
...     "session_code": "ID_2025_2025_R1",
...     "chamber_code": "HOUSE",
...     "measure_type_code": "BILL",
...     "measure_number": "4",
...     "sponsors": ["State Affairs Committee"],
...     "sections_affected": [],
... }
>>> evaluations = item_level_deterministic_grader(
...     input={},
...     output=output,
...     expected_output=expected_output,
... )
>>> [e.value for e in evaluations if e.name == "jurisdiction_code_correct"][0]
1.0
"""

from __future__ import annotations

import difflib
import logging
from typing import Any

from aieng.agent_evals.evaluation import Evaluation

from ._common import get_field, normalize_str, normalize_sponsors, normalize_sections

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft-match threshold (Layer 2).
# Normalized section strings with SequenceMatcher ratio >= this value receive
# 0.5 credit when they do not exact-match any counterpart.
# Lowered from 0.85 to 0.80 to catch near-synonymous strings such as parent
# section refs vs subsection-level GT entries (e.g. WI bills with "(title)"
# or "(intro.)" subsection identifiers) that would otherwise need the LLM judge.
# ---------------------------------------------------------------------------
_SOFT_MATCH_THRESHOLD = 0.80

# Credit awarded for LLM-judge matches (Layer 3) — slightly below 1.0 to
# reflect that semantic equivalence was inferred rather than confirmed exactly.
_LLM_MATCH_CREDIT = 0.8

# Credit awarded for soft (fuzzy string) matches (Layer 2).
_SOFT_MATCH_CREDIT = 0.5


def _coerce_bool_like(value: Any) -> bool | None:
    """Coerce common boolean-like values into a Python bool.

    Returns ``None`` only when ``value`` is ``None``. Handles booleans, ints,
    and common textual synonyms such as "true", "yes", "passed", "enacted".
    Unknown non-None values are interpreted conservatively as ``False`` to
    surface extraction failures rather than silently matching a True ground
    truth.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    try:
        s = str(value).strip().lower()
    except Exception:
        return False

    truthy = {"true", "t", "yes", "y", "1", "passed", "enacted", "signed", "adopted", "approved"}
    falsy = {"false", "f", "no", "n", "0", "vetoed", "rejected", "not adopted"}

    if s in truthy:
        return True
    if s in falsy:
        return False

    # Fallback: interpret common affirmative/negative prefixes
    if s.startswith("y") or s.startswith("t") or s.startswith("1"):
        return True
    if s.startswith("n") or s.startswith("0") or s.startswith("veto"):
        return False

    # Conservative default for unknown non-empty strings
    return False


def _soft_match_score(pred_tokens: set[str], ref_tokens: set[str]) -> float:
    """Return the number of soft-match credits from unmatched (pred, ref) pairs.

    Uses ``difflib.SequenceMatcher`` to find the best pairing between each
    unmatched predicted token and each unmatched reference token.  Each pairing
    whose similarity ratio exceeds ``_SOFT_MATCH_THRESHOLD`` contributes
    ``_SOFT_MATCH_CREDIT`` credits and removes both tokens from further
    consideration (greedy, highest-ratio-first matching).

    Parameters
    ----------
    pred_tokens : set[str]
        Normalized predicted section tokens that did NOT exact-match.
    ref_tokens : set[str]
        Normalized reference section tokens that did NOT exact-match.

    Returns
    -------
    float
        Total soft-match credit (0.5 per matched pair).
    """
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Build all candidate (ratio, pred, ref) triples.
    candidates: list[tuple[float, str, str]] = []
    for p in pred_tokens:
        for r in ref_tokens:
            ratio = difflib.SequenceMatcher(None, p, r).ratio()
            if ratio >= _SOFT_MATCH_THRESHOLD:
                candidates.append((ratio, p, r))

    # Greedy assignment: highest-ratio pairs first.
    candidates.sort(key=lambda x: x[0], reverse=True)
    used_pred: set[str] = set()
    used_ref: set[str] = set()
    credits = 0.0
    for ratio, p, r in candidates:
        if p not in used_pred and r not in used_ref:
            credits += _SOFT_MATCH_CREDIT
            used_pred.add(p)
            used_ref.add(r)

    return credits


def _build_section_raw_map(sections_value: Any) -> dict[str, tuple[str, str]]:
    """Build a mapping from normalized section key to (raw_section, action).

    Used so the LLM judge can receive the original (un-normalized) strings,
    which carry more context than the stripped normalized forms.

    Parameters
    ----------
    sections_value : Any
        Raw ``sections_affected`` payload from agent output or expected output.

    Returns
    -------
    dict[str, tuple[str, str]]
        Mapping from ``"NORM_SECTION|NORM_ACTION"`` to ``(raw_section, action)``.
    """
    from collections.abc import Mapping as _Mapping  # noqa: PLC0415

    if not sections_value:
        return {}
    result: dict[str, tuple[str, str]] = {}
    for section in sections_value:
        if isinstance(section, _Mapping):
            raw = str(section.get("raw_section", "")).strip()
            action = str(section.get("action", "")).strip()
        else:
            raw = str(getattr(section, "raw_section", "")).strip()
            action = str(getattr(section, "action", "")).strip()
        from ._common import normalize_raw_section, _normalize_action  # noqa: PLC0415

        norm_raw = normalize_raw_section(raw)
        norm_action = _normalize_action(action.upper())
        key = f"{norm_raw}|{norm_action}"
        if norm_raw and key not in result:
            result[key] = (raw, action)
    return result


async def _llm_match_score(
    unmatched_pred: set[str],
    unmatched_ref: set[str],
    pred_raw_map: dict[str, tuple[str, str]],
    ref_raw_map: dict[str, tuple[str, str]],
    jurisdiction: str,
) -> float:
    """Return LLM-judge credits for semantically equivalent unmatched pairs.

    Calls :func:`~.section_judge.judge_section_pair` for each unmatched
    (pred, ref) combination. Each confirmed match contributes
    ``_LLM_MATCH_CREDIT`` credits.  Pairs are assigned greedily by
    descending judge confidence (same approach as Layer 2).

    When the judge call raises an exception (e.g. API outage), the pair is
    logged at WARNING level and assigned 0 credit — evaluation continues.

    Parameters
    ----------
    unmatched_pred : set[str]
        Normalized predicted keys that survived exact- and soft-match.
    unmatched_ref : set[str]
        Normalized reference keys that survived exact- and soft-match.
    pred_raw_map : dict[str, tuple[str, str]]
        Maps normalized pred keys to original ``(raw_section, action)`` strings.
    ref_raw_map : dict[str, tuple[str, str]]
        Maps normalized ref keys to original ``(raw_section, action)`` strings.
    jurisdiction : str
        Two-letter jurisdiction code forwarded to the judge.

    Returns
    -------
    float
        Total LLM-match credit (0.8 per matched pair).
    """
    if not unmatched_pred or not unmatched_ref:
        return 0.0

    from .section_judge import judge_section_pair  # noqa: PLC0415
    import asyncio  # noqa: PLC0415

    # Gather all judge calls concurrently.
    async def _call(p_key: str, r_key: str) -> tuple[str, str, float, bool]:
        p_raw, p_action = pred_raw_map.get(p_key, (p_key, ""))
        r_raw, r_action = ref_raw_map.get(r_key, (r_key, ""))
        try:
            result = await judge_section_pair(
                jurisdiction=jurisdiction,
                pred_raw=p_raw,
                pred_action=p_action,
                ref_raw=r_raw,
                ref_action=r_action,
            )
            return (p_key, r_key, result.confidence, result.match)
        except Exception as exc:
            logger.warning(
                "SectionJudge failed for pred='%s' ref='%s': %s",
                p_raw[:60],
                r_raw[:60],
                exc,
            )
            return (p_key, r_key, 0.0, False)

    tasks = [_call(p, r) for p in unmatched_pred for r in unmatched_ref]
    verdicts = await asyncio.gather(*tasks)

    # Greedy assignment: highest-confidence matches first, only confirmed ones.
    matches = [(conf, p, r) for p, r, conf, is_match in verdicts if is_match]
    matches.sort(key=lambda x: x[0], reverse=True)

    used_pred: set[str] = set()
    used_ref: set[str] = set()
    credits = 0.0
    for conf, p, r in matches:
        if p not in used_pred and r not in used_ref:
            credits += _LLM_MATCH_CREDIT
            used_pred.add(p)
            used_ref.add(r)

    return credits


async def _compute_sections_pr(
    output: Any,
    expected_output: Any,
) -> tuple[float, float, int, int, int]:
    """Compute sections precision and recall using a 3-layer matching cascade.

    Layer 1: Exact normalized set intersection.
    Layer 2: Fuzzy string soft-match (difflib SequenceMatcher ≥ 0.85 → 0.5 credit).
    Layer 3: LLM-as-a-judge semantic match (→ 0.8 credit).

    Returns
    -------
    tuple[float, float, int, int, float]
        ``(precision, recall, predicted_count, expected_count, tp_exact_count)``
    """
    jurisdiction = str(get_field(output, "jurisdiction_code") or "").strip().upper()

    pred_raw_map = _build_section_raw_map(get_field(output, "sections_affected"))
    ref_raw_map = _build_section_raw_map(get_field(expected_output, "sections_affected"))

    predicted_sections = set(pred_raw_map.keys())
    expected_sections = set(ref_raw_map.keys())

    # --- Edge cases ---
    if not expected_sections and not predicted_sections:
        return 1.0, 1.0, 0, 0, 0
    if not expected_sections:
        return 0.0, 1.0, len(predicted_sections), 0, 0

    # --- Layer 1: Exact match ---
    exact_tp = predicted_sections & expected_sections
    tp_count = len(exact_tp)

    unmatched_pred = predicted_sections - exact_tp
    unmatched_ref = expected_sections - exact_tp

    # --- Layer 2: Soft match ---
    soft_credits = _soft_match_score(unmatched_pred, unmatched_ref)

    # After soft-match, remove the pairs that were credited (approximate: remove
    # the same number of tokens from each side in greedy order by similarity).
    # For simplicity we track by count rather than exact identity here since
    # Layer 3 LLM calls on near-duplicate strings would be wasteful.
    soft_matched_count = int(soft_credits / _SOFT_MATCH_CREDIT)
    # Remove the soft-matched tokens from the unmatched pools.
    # We rely on the same greedy logic by re-running a deduplicated version.
    _used_pred: set[str] = set()
    _used_ref: set[str] = set()
    if soft_matched_count > 0:
        candidates: list[tuple[float, str, str]] = []
        for p in unmatched_pred:
            for r in unmatched_ref:
                ratio = difflib.SequenceMatcher(None, p, r).ratio()
                if ratio >= _SOFT_MATCH_THRESHOLD:
                    candidates.append((ratio, p, r))
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, p, r in candidates:
            if p not in _used_pred and r not in _used_ref:
                _used_pred.add(p)
                _used_ref.add(r)
                if len(_used_pred) >= soft_matched_count:
                    break

    remaining_pred = unmatched_pred - _used_pred
    remaining_ref = unmatched_ref - _used_ref

    # --- Layer 3: LLM judge ---
    llm_credits = 0.0
    if jurisdiction and remaining_pred and remaining_ref:
        llm_credits = await _llm_match_score(
            remaining_pred,
            remaining_ref,
            pred_raw_map,
            ref_raw_map,
            jurisdiction,
        )

    # --- Aggregate ---
    total_tp = tp_count + soft_credits + llm_credits
    n_pred = len(predicted_sections)
    n_ref = len(expected_sections)

    precision = total_tp / n_pred if n_pred else 0.0
    recall = total_tp / n_ref

    return precision, recall, n_pred, n_ref, tp_count


def item_level_deterministic_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """Evaluate one legislative extraction prediction using deterministic rules.

    Parameters
    ----------
    input : Any
        Item input payload. Not used directly.
    output : Any
        Agent output payload. Expected to contain the extracted fields.
    expected_output : Any
        Ground-truth payload from the dataset.
    metadata : dict[str, Any] | None, optional
        Optional item metadata. Not used by this grader.

    Returns
    -------
    list[Evaluation]
        Per-item metrics: ``jurisdiction_code_correct``,
        ``session_code_correct``, ``chamber_code_correct``,
        ``measure_type_correct``, ``measure_number_correct``,
        ``is_adopted_into_law_correct``,
        ``sponsors_precision``, ``sponsors_recall``,
        ``sections_precision``, ``sections_recall``.

    Notes
    -----
    This function is **synchronous** for backward compatibility.  When the
    LLM judge is needed (Layer 3), the async helper
    :func:`_compute_sections_pr` is run synchronously via
    ``asyncio.run`` if no event loop is active, or via
    ``asyncio.ensure_future`` / ``loop.run_until_complete`` otherwise.
    Callers running inside an async context (e.g. the eval harness) should
    use :func:`item_level_deterministic_grader_async` instead.
    """
    import asyncio  # noqa: PLC0415

    del input, metadata, kwargs  # Unused but part of evaluator interface.

    # --- Enum / scalar fields: exact match after normalization ---
    jurisdiction_correct = normalize_str(get_field(output, "jurisdiction_code")) == normalize_str(
        get_field(expected_output, "jurisdiction_code")
    )
    session_correct = normalize_str(get_field(output, "session_code")) == normalize_str(
        get_field(expected_output, "session_code")
    )
    chamber_correct = normalize_str(get_field(output, "chamber_code")) == normalize_str(
        get_field(expected_output, "chamber_code")
    )
    # HOUSE_FILE / SENATE_FILE are jurisdiction-specific synonyms for BILL;
    # normalize before comparison so that either form matches.
    _MEASURE_TYPE_EQUIVALENCES = {"HOUSE_FILE": "BILL", "SENATE_FILE": "BILL"}
    pred_mt = normalize_str(get_field(output, "measure_type_code"))
    exp_mt = normalize_str(get_field(expected_output, "measure_type_code"))
    pred_mt = _MEASURE_TYPE_EQUIVALENCES.get(pred_mt, pred_mt) if pred_mt else pred_mt
    exp_mt = _MEASURE_TYPE_EQUIVALENCES.get(exp_mt, exp_mt) if exp_mt else exp_mt
    measure_type_correct = pred_mt == exp_mt
    measure_number_correct = normalize_str(get_field(output, "measure_number")) == normalize_str(
        get_field(expected_output, "measure_number")
    )

    # --- Boolean field: is_adopted_into_law ---
    predicted_adopted = get_field(output, "is_adopted_into_law")
    expected_adopted = get_field(expected_output, "is_adopted_into_law")
    # If expected is missing/None, treat as automatic pass
    if expected_adopted is None:
        adopted_correct = True
    else:
        # Coerce common boolean-like values (strings, ints) to bool.
        predicted_coerced = _coerce_bool_like(predicted_adopted)
        expected_coerced = _coerce_bool_like(expected_adopted)

        # Preserve previous behavior: a missing/None prediction is treated as False
        # to surface extraction failures (do not silently match a True ground truth).
        if predicted_coerced is None:
            predicted_coerced = False
        if expected_coerced is None:
            expected_coerced = False

        adopted_correct = bool(predicted_coerced) == bool(expected_coerced)

    # --- List fields: set-based precision / recall ---
    predicted_sponsors = normalize_sponsors(get_field(output, "sponsors"))
    expected_sponsors = normalize_sponsors(get_field(expected_output, "sponsors"))
    sponsors_tp = len(predicted_sponsors & expected_sponsors)
    if not expected_sponsors and not predicted_sponsors:
        sponsors_precision = 1.0
        sponsors_recall = 1.0
    elif not expected_sponsors:
        sponsors_precision = 0.0
        sponsors_recall = 1.0
    else:
        sponsors_precision = sponsors_tp / len(predicted_sponsors) if predicted_sponsors else 0.0
        sponsors_recall = sponsors_tp / len(expected_sponsors)

    # --- Sections: 3-layer matching (exact → soft → LLM judge) ---
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an existing event loop (e.g., Jupyter, async harness):
            # schedule as a new task and block via a Future.
            import concurrent.futures  # noqa: PLC0415

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, _compute_sections_pr(output, expected_output))
                sections_precision, sections_recall, n_pred, n_ref, tp_exact = future.result()
        else:
            sections_precision, sections_recall, n_pred, n_ref, tp_exact = loop.run_until_complete(
                _compute_sections_pr(output, expected_output)
            )
    except RuntimeError:
        # Fallback: no event loop at all — create a new one.
        sections_precision, sections_recall, n_pred, n_ref, tp_exact = asyncio.run(
            _compute_sections_pr(output, expected_output)
        )

    return [
        Evaluation(
            name="jurisdiction_code_correct",
            value=1.0 if jurisdiction_correct else 0.0,
            metadata={
                "expected": get_field(expected_output, "jurisdiction_code"),
                "actual": get_field(output, "jurisdiction_code"),
            },
        ),
        Evaluation(
            name="session_code_correct",
            value=1.0 if session_correct else 0.0,
            metadata={
                "expected": get_field(expected_output, "session_code"),
                "actual": get_field(output, "session_code"),
            },
        ),
        Evaluation(
            name="chamber_code_correct",
            value=1.0 if chamber_correct else 0.0,
            metadata={
                "expected": get_field(expected_output, "chamber_code"),
                "actual": get_field(output, "chamber_code"),
            },
        ),
        Evaluation(
            name="measure_type_correct",
            value=1.0 if measure_type_correct else 0.0,
            metadata={
                "expected": get_field(expected_output, "measure_type_code"),
                "actual": get_field(output, "measure_type_code"),
            },
        ),
        Evaluation(
            name="measure_number_correct",
            value=1.0 if measure_number_correct else 0.0,
            metadata={
                "expected": get_field(expected_output, "measure_number"),
                "actual": get_field(output, "measure_number"),
            },
        ),
        Evaluation(
            name="is_adopted_into_law_correct",
            value=1.0 if adopted_correct else 0.0,
            metadata={
                "expected": expected_adopted,
                "actual": predicted_adopted,
            },
        ),
        Evaluation(
            name="sponsors_precision",
            value=sponsors_precision,
            metadata={
                "predicted": sorted(predicted_sponsors),
                "expected": sorted(expected_sponsors),
                "true_positive_count": sponsors_tp,
            },
        ),
        Evaluation(
            name="sponsors_recall",
            value=sponsors_recall,
            metadata={
                "predicted": sorted(predicted_sponsors),
                "expected": sorted(expected_sponsors),
                "true_positive_count": sponsors_tp,
            },
        ),
        Evaluation(
            name="sections_precision",
            value=sections_precision,
            metadata={
                "predicted_count": n_pred,
                "expected_count": n_ref,
                "true_positive_count": tp_exact,
            },
        ),
        Evaluation(
            name="sections_recall",
            value=sections_recall,
            metadata={
                "predicted_count": n_pred,
                "expected_count": n_ref,
                "true_positive_count": tp_exact,
            },
        ),
    ]


async def item_level_deterministic_grader_async(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """Async variant of :func:`item_level_deterministic_grader`.

    Preferred when called from within an existing async context (e.g. the
    Langfuse evaluation harness) because it avoids spawning a sub-thread just
    to run the event loop.

    All parameters and return values are identical to the sync version.
    """
    del input, metadata, kwargs  # Unused but part of evaluator interface.

    # --- Enum / scalar fields: exact match after normalization ---
    jurisdiction_correct = normalize_str(get_field(output, "jurisdiction_code")) == normalize_str(
        get_field(expected_output, "jurisdiction_code")
    )
    session_correct = normalize_str(get_field(output, "session_code")) == normalize_str(
        get_field(expected_output, "session_code")
    )
    chamber_correct = normalize_str(get_field(output, "chamber_code")) == normalize_str(
        get_field(expected_output, "chamber_code")
    )
    _MEASURE_TYPE_EQUIVALENCES = {"HOUSE_FILE": "BILL", "SENATE_FILE": "BILL"}
    pred_mt = normalize_str(get_field(output, "measure_type_code"))
    exp_mt = normalize_str(get_field(expected_output, "measure_type_code"))
    pred_mt = _MEASURE_TYPE_EQUIVALENCES.get(pred_mt, pred_mt) if pred_mt else pred_mt
    exp_mt = _MEASURE_TYPE_EQUIVALENCES.get(exp_mt, exp_mt) if exp_mt else exp_mt
    measure_type_correct = pred_mt == exp_mt
    measure_number_correct = normalize_str(get_field(output, "measure_number")) == normalize_str(
        get_field(expected_output, "measure_number")
    )

    predicted_adopted = get_field(output, "is_adopted_into_law")
    expected_adopted = get_field(expected_output, "is_adopted_into_law")
    if expected_adopted is None:
        adopted_correct = True
    else:
        predicted_coerced = _coerce_bool_like(predicted_adopted)
        expected_coerced = _coerce_bool_like(expected_adopted)
        if predicted_coerced is None:
            predicted_coerced = False
        if expected_coerced is None:
            expected_coerced = False
        adopted_correct = bool(predicted_coerced) == bool(expected_coerced)

    predicted_sponsors = normalize_sponsors(get_field(output, "sponsors"))
    expected_sponsors = normalize_sponsors(get_field(expected_output, "sponsors"))
    sponsors_tp = len(predicted_sponsors & expected_sponsors)
    if not expected_sponsors and not predicted_sponsors:
        sponsors_precision = 1.0
        sponsors_recall = 1.0
    elif not expected_sponsors:
        sponsors_precision = 0.0
        sponsors_recall = 1.0
    else:
        sponsors_precision = sponsors_tp / len(predicted_sponsors) if predicted_sponsors else 0.0
        sponsors_recall = sponsors_tp / len(expected_sponsors)

    sections_precision, sections_recall, n_pred, n_ref, tp_exact = await _compute_sections_pr(
        output, expected_output
    )

    return [
        Evaluation(
            name="jurisdiction_code_correct",
            value=1.0 if jurisdiction_correct else 0.0,
            metadata={
                "expected": get_field(expected_output, "jurisdiction_code"),
                "actual": get_field(output, "jurisdiction_code"),
            },
        ),
        Evaluation(
            name="session_code_correct",
            value=1.0 if session_correct else 0.0,
            metadata={
                "expected": get_field(expected_output, "session_code"),
                "actual": get_field(output, "session_code"),
            },
        ),
        Evaluation(
            name="chamber_code_correct",
            value=1.0 if chamber_correct else 0.0,
            metadata={
                "expected": get_field(expected_output, "chamber_code"),
                "actual": get_field(output, "chamber_code"),
            },
        ),
        Evaluation(
            name="measure_type_correct",
            value=1.0 if measure_type_correct else 0.0,
            metadata={
                "expected": get_field(expected_output, "measure_type_code"),
                "actual": get_field(output, "measure_type_code"),
            },
        ),
        Evaluation(
            name="measure_number_correct",
            value=1.0 if measure_number_correct else 0.0,
            metadata={
                "expected": get_field(expected_output, "measure_number"),
                "actual": get_field(output, "measure_number"),
            },
        ),
        Evaluation(
            name="is_adopted_into_law_correct",
            value=1.0 if adopted_correct else 0.0,
            metadata={
                "expected": expected_adopted,
                "actual": predicted_adopted,
            },
        ),
        Evaluation(
            name="sponsors_precision",
            value=sponsors_precision,
            metadata={
                "predicted": sorted(predicted_sponsors),
                "expected": sorted(expected_sponsors),
                "true_positive_count": sponsors_tp,
            },
        ),
        Evaluation(
            name="sponsors_recall",
            value=sponsors_recall,
            metadata={
                "predicted": sorted(predicted_sponsors),
                "expected": sorted(expected_sponsors),
                "true_positive_count": sponsors_tp,
            },
        ),
        Evaluation(
            name="sections_precision",
            value=sections_precision,
            metadata={
                "predicted_count": n_pred,
                "expected_count": n_ref,
                "true_positive_count": tp_exact,
            },
        ),
        Evaluation(
            name="sections_recall",
            value=sections_recall,
            metadata={
                "predicted_count": n_pred,
                "expected_count": n_ref,
                "true_positive_count": tp_exact,
            },
        ),
    ]


__all__ = [
    "item_level_deterministic_grader",
    "item_level_deterministic_grader_async",
]
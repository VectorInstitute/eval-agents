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

from typing import Any

from aieng.agent_evals.evaluation import Evaluation

from ._common import get_field, normalize_str, normalize_sponsors, normalize_sections


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
        # Coerce to bool for comparison (handles string "true"/"false" from JSON).
        # A missing/None prediction is treated as False (not a silent pass) so
        # that extraction failures are surfaced rather than accidentally matching
        # a False ground truth.
        if predicted_adopted is None:
            predicted_adopted = False
        elif isinstance(predicted_adopted, str):
            predicted_adopted = predicted_adopted.strip().lower() == "true"
        if isinstance(expected_adopted, str):
            expected_adopted = expected_adopted.strip().lower() == "true"
        adopted_correct = bool(predicted_adopted) == bool(expected_adopted)

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

    predicted_sections = normalize_sections(get_field(output, "sections_affected"))
    expected_sections = normalize_sections(get_field(expected_output, "sections_affected"))
    sections_tp = len(predicted_sections & expected_sections)
    if not expected_sections and not predicted_sections:
        sections_precision = 1.0
        sections_recall = 1.0
    elif not expected_sections:
        sections_precision = 0.0
        sections_recall = 1.0
    else:
        sections_precision = sections_tp / len(predicted_sections) if predicted_sections else 0.0
        sections_recall = sections_tp / len(expected_sections)

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
                "predicted_count": len(predicted_sections),
                "expected_count": len(expected_sections),
                "true_positive_count": sections_tp,
            },
        ),
        Evaluation(
            name="sections_recall",
            value=sections_recall,
            metadata={
                "predicted_count": len(predicted_sections),
                "expected_count": len(expected_sections),
                "true_positive_count": sections_tp,
            },
        ),
    ]


__all__ = ["item_level_deterministic_grader"]

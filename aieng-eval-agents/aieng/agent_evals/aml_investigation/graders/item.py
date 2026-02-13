"""Item-level deterministic graders for AML investigation agent outputs.

This module contains evaluator functions that score one AML case prediction against
the case ground truth and returns a list of per-item metrics suitable for aggregation
at run level.

Examples
--------
>>> from aieng.agent_evals.aml_investigation.graders import (
...     item_level_deterministic_grader,
... )
>>> from aieng.agent_evals.aml_investigation.task import AmlInvestigationTask
>>> from aieng.agent_evals.evaluation import run_experiment
>>> task = AmlInvestigationTask()
>>> results = run_experiment(
...     # <YOUR_DATASET_NAME>,
...     name="aml_item_level_demo",
...     task=task,
...     evaluators=[item_level_deterministic_grader],
... )
"""

from typing import Any

from aieng.agent_evals.evaluation import Evaluation

from ._common import get_field, normalize_pattern, normalize_transaction_ids


def item_level_deterministic_grader(
    input: Any,  # noqa: A002
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """Evaluate one AML prediction using deterministic rules.

    Parameters
    ----------
    input : Any
        Item input payload. Included for evaluator interface compatibility and
        not used directly.
    output : Any
        Model output payload. Expected to contain fields such as
        ``is_laundering``, ``pattern_type``, and ``flagged_transaction_ids``.
    expected_output : Any
        Ground-truth payload. Expected to contain fields such as
        ``is_laundering``, ``pattern_type``, and ``attempt_transaction_ids``.
    metadata : dict[str, Any] | None, optional
        Optional item metadata from the dataset. Not used by this grader.
    **kwargs : Any
        Additional evaluator kwargs. Ignored by this grader.

    Returns
    -------
    list[Evaluation]
        Deterministic per-item metrics, including:
        ``is_laundering_correct``, ``is_laundering_tp/fp/fn/tn``,
        ``pattern_type_correct``, ``non_laundering_pattern_consistent``,
        ``non_laundering_flags_empty``, ``id_precision_like``, and
        ``id_coverage``.

    Examples
    --------
    >>> output = {
    ...     "is_laundering": False,
    ...     "pattern_type": "NONE",
    ...     "flagged_transaction_ids": "",
    ... }
    >>> expected_output = {
    ...     "is_laundering": False,
    ...     "pattern_type": "NONE",
    ...     "attempt_transaction_ids": "",
    ... }
    >>> evaluations = item_level_deterministic_grader(
    ...     input={},
    ...     output=output,
    ...     expected_output=expected_output,
    ... )
    >>> [e.value for e in evaluations if e.name == "non_laundering_flags_empty"][0]
    1.0
    """
    del input, metadata, kwargs  # Unused but part of evaluator interface.

    # Evaluate laundering prediction correctness
    expected_is_laundering: bool = get_field(expected_output, "is_laundering")
    predicted_is_laundering = get_field(output, "is_laundering")
    is_laundering_correct = predicted_is_laundering == expected_is_laundering

    # Confusion matrix components for is_laundering
    is_tp = bool(expected_is_laundering is True and predicted_is_laundering is True)
    is_fp = bool(expected_is_laundering is False and predicted_is_laundering is True)
    is_fn = bool(expected_is_laundering is True and predicted_is_laundering is False)

    # Evaluate pattern type correctness (exact match)
    expected_pattern = normalize_pattern(get_field(expected_output, "pattern_type"))
    predicted_pattern = normalize_pattern(get_field(output, "pattern_type"))
    pattern_type_correct = predicted_pattern == expected_pattern

    # Evaluate flagged transaction ID predictions
    ground_truth_ids = normalize_transaction_ids(get_field(expected_output, "attempt_transaction_ids"))
    predicted_ids = normalize_transaction_ids(get_field(output, "flagged_transaction_ids"))

    true_positive_ids = ground_truth_ids & predicted_ids
    false_positive_ids = predicted_ids - ground_truth_ids
    false_negative_ids = ground_truth_ids - predicted_ids

    tp_count = len(true_positive_ids)
    fp_count = len(false_positive_ids)
    fn_count = len(false_negative_ids)
    predicted_count = len(predicted_ids)
    ground_truth_count = len(ground_truth_ids)

    # Precision-like for flagged IDs: of the predicted IDs, how many were correct?
    id_precision_like = float(tp_count - fp_count) / float(predicted_count) if predicted_count else 0.0

    # Coverage: of the ground truth IDs, how many were correctly predicted?
    id_coverage = float(tp_count) / float(ground_truth_count) if ground_truth_count else 0.0

    # Consistency checks for predicted benign cases
    # If the agent predicts a case is not laundering, the predicted pattern should
    # be "NONE" and no transaction IDs should be flagged.
    predicted_benign = predicted_is_laundering is False
    predicted_benign_pattern_consistent = (predicted_pattern == "NONE") if predicted_benign else True
    predicted_benign_ids_consistent = (predicted_count == 0) if predicted_benign else True

    return [
        Evaluation(
            name="is_laundering_correct",
            value=1.0 if is_laundering_correct else 0.0,
            metadata={
                "expected": expected_is_laundering,
                "actual": predicted_is_laundering,
                "type": "TP" if is_tp else "FP" if is_fp else "FN" if is_fn else "TN",
            },
        ),
        Evaluation(
            name="pattern_type_correct",
            value=1.0 if pattern_type_correct else 0.0,
            metadata={"expected": expected_pattern, "actual": predicted_pattern},
        ),
        Evaluation(
            name="non_laundering_pattern_consistent",
            value=1.0 if predicted_benign_pattern_consistent else 0.0,
            metadata={
                "applicable": predicted_benign,
                "is_laundering": predicted_is_laundering,
                "pattern_type": predicted_pattern,
            },
        ),
        Evaluation(
            name="non_laundering_flags_empty",
            value=1.0 if predicted_benign_ids_consistent else 0.0,
            metadata={
                "applicable": predicted_benign,
                "is_laundering": predicted_is_laundering,
                "predicted_flagged_count": predicted_count,
            },
        ),
        Evaluation(
            name="id_precision_like",
            value=id_precision_like,
            metadata={
                "true_positive_count": tp_count,
                "false_positive_count": fp_count,
                "predicted_count": predicted_count,
            },
        ),
        Evaluation(
            name="id_coverage",
            value=id_coverage,
            metadata={
                "true_positive_count": tp_count,
                "false_negative_count": fn_count,
                "ground_truth_count": ground_truth_count,
            },
        ),
    ]


__all__ = ["item_level_deterministic_grader"]

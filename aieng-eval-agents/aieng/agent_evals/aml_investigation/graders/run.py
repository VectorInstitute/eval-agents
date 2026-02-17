"""Run-level graders for AML investigation evaluation.

This module aggregates item-level AML outputs into run-level classification
metrics for laundering detection and pattern-type prediction.

Examples
--------
>>> from aieng.agent_evals.aml_investigation.graders import (
...     item_level_deterministic_grader,
...     run_level_grader,
... )
>>> from aieng.agent_evals.aml_investigation.task import AmlInvestigationTask
>>> from aieng.agent_evals.evaluation import run_experiment
>>> task = AmlInvestigationTask()
>>> results = run_experiment(
...     # <YOUR_DATASET_NAME>,
...     name="aml_run_level_demo",
...     task=task,
...     evaluators=[item_level_deterministic_grader],
...     run_evaluators=[run_level_grader],
... )
"""

from typing import Any

from aieng.agent_evals.evaluation import Evaluation, ExperimentItemResult
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from ._common import PATTERN_LABELS, extract_expected_output, get_field, normalize_pattern


def run_level_grader(*, item_results: list[ExperimentItemResult], **kwargs: Any) -> list[Evaluation]:
    """Compute AML run-level metrics over experiment item results.

    Parameters
    ----------
    item_results : list[ExperimentItemResult]
        Item results emitted by a Langfuse experiment run.
    **kwargs : Any
        Additional run-evaluator kwargs. Ignored by this grader.

    Returns
    -------
    list[Evaluation]
        Run-level metrics:

        - ``is_laundering_precision``
        - ``is_laundering_recall``
        - ``is_laundering_f1``
        - ``pattern_type_macro_f1``
        - ``pattern_type_confusion_matrix`` (labels + matrix in metadata)

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> item_results = [
    ...     SimpleNamespace(
    ...         item={
    ...             "expected_output": {
    ...                 "is_laundering": True,
    ...                 "pattern_type": "FAN-IN",
    ...             }
    ...         },
    ...         output={"is_laundering": True, "pattern_type": "FAN-IN"},
    ...         evaluations=[],
    ...     )
    ... ]
    >>> evaluations = run_level_grader(item_results=item_results)
    >>> sorted(e.name for e in evaluations)[:2]
    ['is_laundering_f1', 'is_laundering_precision']
    """
    del kwargs  # Unused but part of evaluator interface.

    laundering_expected: list[bool] = []
    laundering_predicted: list[bool] = []

    pattern_expected: list[str] = []
    pattern_predicted: list[str] = []
    invalid_expected_pattern_count = 0
    invalid_predicted_pattern_count = 0

    for item_result in item_results:
        expected_output = extract_expected_output(item_result)
        predicted_output = item_result.output

        expected_is_laundering: bool = get_field(expected_output, "is_laundering")
        predicted_is_laundering: bool = get_field(predicted_output, "is_laundering")
        if expected_is_laundering is not None and predicted_is_laundering is not None:
            laundering_expected.append(expected_is_laundering)
            laundering_predicted.append(predicted_is_laundering)

        expected_pattern = normalize_pattern(get_field(expected_output, "pattern_type"))
        predicted_pattern = normalize_pattern(get_field(predicted_output, "pattern_type"))

        # Include invalid patterns in confusion matrix under "INVALID" label.
        if expected_pattern not in PATTERN_LABELS:
            invalid_expected_pattern_count += 1
            expected_pattern = "INVALID"

        if predicted_pattern not in PATTERN_LABELS:
            invalid_predicted_pattern_count += 1
            predicted_pattern = "INVALID"

        pattern_expected.append(expected_pattern)
        pattern_predicted.append(predicted_pattern)

    is_laundering_precision = 0.0
    is_laundering_recall = 0.0
    is_laundering_f1 = 0.0
    is_laundering_tp = 0
    is_laundering_fp = 0
    is_laundering_fn = 0

    if laundering_expected:
        precision, recall, f1, _ = precision_recall_fscore_support(
            laundering_expected, laundering_predicted, average="binary", pos_label=True, zero_division=0
        )
        is_laundering_precision = float(precision)
        is_laundering_recall = float(recall)
        is_laundering_f1 = float(f1)

        confusion = confusion_matrix(laundering_expected, laundering_predicted, labels=[False, True])
        is_laundering_fp = int(confusion[0, 1])
        is_laundering_fn = int(confusion[1, 0])
        is_laundering_tp = int(confusion[1, 1])

    labels = list(PATTERN_LABELS) + (
        ["INVALID"] if invalid_expected_pattern_count > 0 or invalid_predicted_pattern_count > 0 else []
    )
    pattern_type_macro_f1 = 0.0
    pattern_confusion_matrix = [[0 for _ in labels] for _ in labels]
    if pattern_expected:
        pattern_type_macro_f1 = float(
            f1_score(pattern_expected, pattern_predicted, labels=labels, average="macro", zero_division=0)
        )
        pattern_confusion_matrix = confusion_matrix(pattern_expected, pattern_predicted, labels=labels).tolist()

    return [
        Evaluation(
            name="is_laundering_precision",
            value=is_laundering_precision,
            metadata={
                "tp": is_laundering_tp,
                "fp": is_laundering_fp,
                "valid_items": len(laundering_expected),
            },
        ),
        Evaluation(
            name="is_laundering_recall",
            value=is_laundering_recall,
            metadata={
                "tp": is_laundering_tp,
                "fn": is_laundering_fn,
                "valid_items": len(laundering_expected),
            },
        ),
        Evaluation(
            name="is_laundering_f1",
            value=is_laundering_f1,
            metadata={
                "precision": is_laundering_precision,
                "recall": is_laundering_recall,
                "valid_items": len(laundering_expected),
            },
        ),
        Evaluation(
            name="pattern_type_macro_f1",
            value=pattern_type_macro_f1,
            metadata={
                "labels": labels,
                "valid_items": len(pattern_expected),
                "invalid_expected_pattern_count": invalid_expected_pattern_count,
                "invalid_predicted_pattern_count": invalid_predicted_pattern_count,
            },
        ),
        Evaluation(
            name="pattern_type_confusion_matrix",
            value=float(len(pattern_expected)),
            metadata={
                "labels": labels,
                "matrix": pattern_confusion_matrix,
                "valid_items": len(pattern_expected),
                "invalid_expected_pattern_count": invalid_expected_pattern_count,
                "invalid_predicted_pattern_count": invalid_predicted_pattern_count,
            },
        ),
    ]


__all__ = ["run_level_grader"]

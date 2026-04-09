"""Run-level graders for legislative content extraction evaluation."""

from typing import Any

from aieng.agent_evals.evaluation import Evaluation, ExperimentItemResult

# Metrics produced by the LLM judge (summary_quality evaluator) are excluded
# from the overall_score to keep it based on deterministic graders only.
_LLM_JUDGE_METRIC_NAMES = frozenset({
    "summary_correctness",
    "summary_completeness",
    "summary_hallucination_free",
    "summary_quality_error",
})


def run_level_grader(*, item_results: list[ExperimentItemResult], **kwargs: Any) -> list[Evaluation]:
    """Compute overall score by averaging deterministic evaluation scores across all item results.

    Collects every ``Evaluation.value`` from every item result, excluding
    metrics produced by the LLM judge, and returns a single ``Evaluation``
    whose value is the mean of those scores.

    Parameters
    ----------
    item_results : list[ExperimentItemResult]
        Item results emitted by a Langfuse experiment run.
    **kwargs : Any
        Additional run-evaluator kwargs. Ignored by this grader.

    Returns
    -------
    list[Evaluation]
        A single-element list containing ``overall_score``: the mean of all
        deterministic evaluation values across all items.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> from aieng.agent_evals.evaluation import Evaluation
    >>> item_results = [
    ...     SimpleNamespace(evaluations=[
    ...         Evaluation(name="jurisdiction_code_correct", value=1.0),
    ...         Evaluation(name="session_code_correct", value=0.0),
    ...     ]),
    ...     SimpleNamespace(evaluations=[
    ...         Evaluation(name="jurisdiction_code_correct", value=1.0),
    ...         Evaluation(name="session_code_correct", value=1.0),
    ...     ]),
    ... ]
    >>> result = run_level_grader(item_results=item_results)
    >>> result[0].value
    0.75
    """
    del kwargs  # Unused but part of evaluator interface.

    total = 0.0
    count = 0

    for item_result in item_results:
        for evaluation in (item_result.evaluations or []):
            if evaluation.value is not None and evaluation.name not in _LLM_JUDGE_METRIC_NAMES:
                total += evaluation.value
                count += 1

    overall_score = total / count if count > 0 else 0.0

    return [
        Evaluation(
            name="overall_score",
            value=overall_score,
            metadata={"total_score": total, "total_evaluations": count},
        )
    ]


__all__ = ["run_level_grader"]

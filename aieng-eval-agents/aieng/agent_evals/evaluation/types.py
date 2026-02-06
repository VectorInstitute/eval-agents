"""Type definitions for the evaluation harness.

This module centralizes the public typing surface for the harness so other modules
can depend on a stable API without importing Langfuse internals directly.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Protocol

from langfuse.api.resources.commons.types.observations_view import ObservationsView
from langfuse.api.resources.commons.types.trace_with_full_details import TraceWithFullDetails
from langfuse.batch_evaluation import CompositeEvaluatorFunction
from langfuse.experiment import (
    Evaluation,
    EvaluatorFunction,
    ExperimentItemResult,
    ExperimentResult,
    RunEvaluatorFunction,
    TaskFunction,
)


class TraceEvalStatus(Enum):
    """Enumeration of trace evaluation statuses."""

    OK = "ok"
    """Trace evaluated successfully."""

    SKIPPED = "skipped"
    """Trace evaluation was skipped due to incomplete or missing data."""

    FAILED = "failed"
    """Trace evaluation failed due to an error during processing."""


class TraceEvaluatorFunction(Protocol):
    """Protocol for trace-based evaluators.

    Trace evaluators run in a second pass after the experiment completes.
    They receive the fully populated trace and the matching item result,
    then return one or more Langfuse evaluations. Evaluators may return
    results synchronously or as awaitables.
    """

    def __call__(
        self, *, trace: TraceWithFullDetails, item_result: ExperimentItemResult, **kwargs: Any
    ) -> Evaluation | list[Evaluation] | Awaitable[Evaluation | list[Evaluation]]:
        """Evaluate a trace for a single experiment item.

        Parameters
        ----------
        trace : TraceWithFullDetails
            The Langfuse trace for the item, including observations and scores.
        item_result : ExperimentItemResult
            The experiment item result containing the task output and metadata.
        **kwargs : Any
            Additional keyword arguments forwarded by the harness.

        Returns
        -------
        Evaluation | list[Evaluation] | Awaitable[Evaluation | list[Evaluation]]
            One or more Langfuse evaluations to attach to the trace. Can be
            returned directly or wrapped in an awaitable for async evaluators.
        """
        ...


# Predicate signature for classifying observations in trace metrics extraction.
TraceObservationPredicate = Callable[[ObservationsView], bool]


@dataclass(frozen=True)
class TraceMetrics:
    """Common trace-derived metrics.

    Parameters
    ----------
    tool_call_count : int
        Estimated number of tool call observations in the trace.
    turn_count : int
        Estimated number of conversational turns in the trace.
    observation_count : int
        Total number of observations recorded for the trace.
    latency_sec : float | None
        End-to-end trace latency in seconds, if available.
    total_input_tokens : int
        Total input token count across trace observations.
    total_output_tokens : int
        Total output token count across trace observations.
    total_cost : float | None
        Total trace cost in USD, if available.
    """

    tool_call_count: int
    turn_count: int
    observation_count: int
    latency_sec: float | None
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float | None


@dataclass(frozen=True)
class TraceWaitConfig:
    """Configuration for trace fetch retries.

    Parameters
    ----------
    max_wait_sec : float
        Maximum total time to wait for trace readiness.
    initial_delay_sec : float
        Initial delay between retries.
    max_delay_sec : float
        Maximum delay between retries.
    backoff_multiplier : float
        Exponential backoff multiplier applied to retry delay.
    """

    max_wait_sec: float = 180.0
    initial_delay_sec: float = 1.0
    max_delay_sec: float = 10.0
    backoff_multiplier: float = 2.0


@dataclass
class TraceEvalResult:
    """Result container for trace evaluations.

    Parameters
    ----------
    evaluations_by_trace_id : dict[str, list[Evaluation]]
        Evaluations produced for each trace that completed successfully.
    skipped_trace_ids : list[str]
        Trace IDs skipped because trace data was incomplete or missing.
    failed_trace_ids : list[str]
        Trace IDs that failed due to errors during evaluation.
    errors_by_trace_id : dict[str, str]
        Error messages associated with skipped or failed traces.
    run_evaluations : list[Evaluation]
        Aggregated trace evaluation metrics written at dataset-run level.
    """

    evaluations_by_trace_id: dict[str, list[Evaluation]] = field(default_factory=dict)
    skipped_trace_ids: list[str] = field(default_factory=list)
    failed_trace_ids: list[str] = field(default_factory=list)
    errors_by_trace_id: dict[str, str] = field(default_factory=dict)
    run_evaluations: list[Evaluation] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate result for an evaluation run.

    Parameters
    ----------
    experiment : ExperimentResult
        The Langfuse experiment result from the output-based pass.
    trace_evaluations : TraceEvalResult | None
        Trace evaluation results, if a second pass was performed.
    """

    experiment: ExperimentResult
    trace_evaluations: TraceEvalResult | None


__all__ = [
    "CompositeEvaluatorFunction",
    "Evaluation",
    "EvaluatorFunction",
    "ExperimentItemResult",
    "ExperimentResult",
    "RunEvaluatorFunction",
    "TaskFunction",
    "TraceEvaluatorFunction",
    "TraceMetrics",
    "TraceObservationPredicate",
    "TraceEvalStatus",
    "TraceEvalResult",
    "TraceWaitConfig",
    "EvaluationResult",
]

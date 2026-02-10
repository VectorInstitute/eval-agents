"""Evaluation harness.

This package provides a beginner-friendly wrapper around Langfuse's
``dataset.run_experiment`` workflow, plus optional trace-based
second-pass evaluators.
"""

from .experiment import run_experiment, run_experiment_with_trace_evals
from .trace import extract_trace_metrics, run_trace_evaluations
from .types import (
    CompositeEvaluatorFunction,
    Evaluation,
    EvaluationResult,
    EvaluatorFunction,
    ExperimentItemResult,
    ExperimentResult,
    RunEvaluatorFunction,
    TaskFunction,
    TraceEvalResult,
    TraceEvalStatus,
    TraceEvaluatorFunction,
    TraceMetrics,
    TraceObservationPredicate,
    TraceWaitConfig,
)


__all__ = [
    "run_experiment",
    "run_experiment_with_trace_evals",
    "run_trace_evaluations",
    "extract_trace_metrics",
    "CompositeEvaluatorFunction",
    "Evaluation",
    "EvaluatorFunction",
    "ExperimentItemResult",
    "ExperimentResult",
    "EvaluationResult",
    "RunEvaluatorFunction",
    "TaskFunction",
    "TraceEvaluatorFunction",
    "TraceEvalStatus",
    "TraceEvalResult",
    "TraceMetrics",
    "TraceObservationPredicate",
    "TraceWaitConfig",
]

from __future__ import annotations

import logging
from typing import Any

from aieng.agent_evals.evaluation.graders._utils import build_error_evaluation
from aieng.agent_evals.evaluation.trace import extract_trace_metrics
from aieng.agent_evals.evaluation.types import Evaluation, TraceEvaluatorFunction
from langfuse.api import ScoreDataType
from langfuse.api.resources.commons.types.trace_with_full_details import TraceWithFullDetails
from langfuse.experiment import ExperimentItemResult


logger = logging.getLogger(__name__)


def create_trace_usage_evaluator(*, name: str, metrics: dict[str, Any]) -> TraceEvaluatorFunction:
    """
    Trace-level “hard metrics” evaluator.

    It uses Langfuse trace observations to estimate:
    tool calls, turns, latency, tokens, and cost.
    """
    # Map config booleans to TraceMetrics fields.
    enabled_fields: set[str] = {k for k, v in metrics.items() if bool(v)}
    metric_names = (
        "tool_call_count",
        "turn_count",
        "observation_count",
        "latency_sec",
        "total_input_tokens",
        "total_output_tokens",
        "total_cost",
    )

    async def _evaluator(
        *, trace: TraceWithFullDetails, item_result: ExperimentItemResult, **kwargs: Any
    ) -> list[Evaluation]:  # noqa: ARG001
        del item_result, kwargs  # trace-only evaluator
        try:
            tm = extract_trace_metrics(trace, tool_call_predicate=None, turn_predicate=None)
            out: list[Evaluation] = []
            for metric_name in metric_names:
                if metric_name not in enabled_fields:
                    continue

                value = getattr(tm, metric_name)
                if value is None:
                    continue

                out.append(
                    Evaluation(
                        name=metric_name,
                        value=float(value),
                        data_type=ScoreDataType.NUMERIC,
                    )
                )

            return out
        except Exception as exc:
            # Keep failures analyzable without breaking the trace-eval pass.
            return [build_error_evaluation(name=f"{name}_error", error=exc, prefix="trace_usage error")]

    _evaluator.__name__ = name
    return _evaluator


__all__ = ["create_trace_usage_evaluator"]

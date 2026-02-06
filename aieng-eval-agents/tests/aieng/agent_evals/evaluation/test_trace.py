"""Tests for trace evaluation helpers."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from aieng.agent_evals.evaluation.trace import extract_trace_metrics, run_trace_evaluations
from aieng.agent_evals.evaluation.types import TraceWaitConfig


def _make_trace(*, input, output, observations=None, latency=0.0, total_cost=None) -> SimpleNamespace:  # noqa: A002
    return SimpleNamespace(
        input=input, output=output, observations=observations or [], latency=latency, total_cost=total_cost
    )


def _make_observation(*, type, name, metadata=None, usage_details=None, cost_details=None) -> SimpleNamespace:  # noqa: A002
    return SimpleNamespace(
        type=type,
        name=name,
        metadata=metadata or {},
        usage_details=usage_details or {},
        cost_details=cost_details or {},
        usage=None,
    )


def _patch_langfuse_client(monkeypatch, trace_get) -> MagicMock:
    fake_langfuse_client = MagicMock()
    fake_langfuse_client.async_api = SimpleNamespace(trace=SimpleNamespace(get=trace_get))
    fake_langfuse_client.create_score = MagicMock()

    fake_manager = SimpleNamespace(langfuse_client=fake_langfuse_client)
    monkeypatch.setattr("aieng.agent_evals.evaluation.trace.AsyncClientManager.get_instance", lambda: fake_manager)
    monkeypatch.setattr("aieng.agent_evals.evaluation.trace.flush_traces", lambda: None)

    return fake_langfuse_client


def test_run_trace_evaluations_returns_default_on_no_trace_evaluators() -> None:
    """Return empty result when no trace evaluators are provided."""
    experiment_result = SimpleNamespace(item_results=[], dataset_run_id=None)

    trace_result = run_trace_evaluations(
        experiment_result=experiment_result,  # pyright: ignore[reportArgumentType]
        trace_evaluators=[],
    )

    assert trace_result.evaluations_by_trace_id == {}
    assert trace_result.skipped_trace_ids == []
    assert trace_result.failed_trace_ids == []
    assert trace_result.errors_by_trace_id == {}


def test_run_trace_evaluations_returns_default_when_no_trace_ids() -> None:
    """Ignore items without trace IDs and return empty results."""
    experiment_result = SimpleNamespace(item_results=[SimpleNamespace(trace_id=None)], dataset_run_id=None)

    trace_result = run_trace_evaluations(
        experiment_result=experiment_result,  # pyright: ignore[reportArgumentType]
        trace_evaluators=[lambda trace, item_result: {"name": "metric", "value": 1}],  # pyright: ignore[reportArgumentType]
    )

    assert trace_result.evaluations_by_trace_id == {}
    assert trace_result.skipped_trace_ids == []
    assert trace_result.failed_trace_ids == []
    assert trace_result.errors_by_trace_id == {}


def test_run_trace_evaluations_skips_when_trace_not_ready(monkeypatch) -> None:
    """Skip trace evaluation when trace output is not ready within wait window."""
    fake_trace = _make_trace(input={"case": "x"}, output=None, observations=[], latency=0.5)
    _patch_langfuse_client(monkeypatch, trace_get=AsyncMock(return_value=fake_trace))

    experiment_result = SimpleNamespace(item_results=[SimpleNamespace(trace_id="trace-skip")], dataset_run_id=None)

    trace_result = run_trace_evaluations(
        experiment_result=experiment_result,  # pyright: ignore[reportArgumentType]
        trace_evaluators=[lambda trace, item_result: {"name": "metric", "value": 1}],  # pyright: ignore[reportArgumentType]
        wait=TraceWaitConfig(max_wait_sec=0.01, initial_delay_sec=0.01, max_delay_sec=0.01),
    )

    assert trace_result.evaluations_by_trace_id == {}
    assert trace_result.skipped_trace_ids == ["trace-skip"]
    assert "trace-skip" in trace_result.errors_by_trace_id


def test_run_trace_evaluations_ok_uploads_scores_from_async_evaluator(monkeypatch) -> None:
    """Upload scores when async evaluator returns evaluations."""
    fake_trace = _make_trace(input={"case": "x"}, output={"done": True}, observations=[], latency=0.5)
    fake_langfuse_client = _patch_langfuse_client(monkeypatch, trace_get=AsyncMock(return_value=fake_trace))

    experiment_result = SimpleNamespace(item_results=[SimpleNamespace(trace_id="trace-ok")], dataset_run_id=None)

    async def async_trace_evaluator(trace, item_result) -> list[Any]:
        await asyncio.sleep(0)
        return [
            {"name": "verdict", "value": "pass", "data_type": "CATEGORICAL"},
            {"name": "policy_ok", "value": True},
        ]

    trace_result = run_trace_evaluations(
        experiment_result=experiment_result,  # pyright: ignore[reportArgumentType]
        trace_evaluators=[async_trace_evaluator],  # pyright: ignore[reportArgumentType]
    )

    assert trace_result.skipped_trace_ids == []
    assert trace_result.failed_trace_ids == []
    assert "trace-ok" in trace_result.evaluations_by_trace_id
    assert len(trace_result.evaluations_by_trace_id["trace-ok"]) == 2
    assert fake_langfuse_client.create_score.call_count == 2

    call_kwargs = [call.kwargs for call in fake_langfuse_client.create_score.call_args_list]
    assert any(kwargs.get("data_type") == "CATEGORICAL" for kwargs in call_kwargs)
    assert any(kwargs.get("data_type") == "BOOLEAN" for kwargs in call_kwargs)


def test_run_trace_evaluations_failed_when_evaluator_raises(monkeypatch) -> None:
    """Record evaluator failures and include evaluator name in errors."""
    fake_trace = _make_trace(input={"case": "x"}, output={"done": True}, observations=[], latency=0.5)
    fake_langfuse_client = _patch_langfuse_client(monkeypatch, trace_get=AsyncMock(return_value=fake_trace))

    experiment_result = SimpleNamespace(item_results=[SimpleNamespace(trace_id="trace-fail")], dataset_run_id=None)

    def failing_eval(trace, item_result):
        raise RuntimeError("boom")

    trace_result = run_trace_evaluations(
        experiment_result=experiment_result,  # pyright: ignore[reportArgumentType]
        trace_evaluators=[failing_eval],  # pyright: ignore[reportArgumentType]
    )

    assert trace_result.failed_trace_ids == ["trace-fail"]
    assert "failing_eval" in trace_result.errors_by_trace_id["trace-fail"]
    fake_langfuse_client.create_score.assert_not_called()


def test_extract_trace_metrics_handles_total_cost_and_observation_fallback() -> None:
    """Prefer trace-level total_cost and fallback to observation sums."""
    observations = [
        _make_observation(
            type="generation",
            name="assistant_response",
            metadata={},
            usage_details={"input": 10, "output": 6},
            cost_details={"total": 0.12},
        ),
        _make_observation(
            type="tool_call",
            name="query_tool",
            metadata={},
            usage_details={"input_tokens": 4, "output_tokens": 2},
            cost_details={"totalCost": 0.03},
        ),
    ]

    trace_with_total = _make_trace(
        input={"case": "a"},
        output={"done": True},
        observations=observations,
        latency=1.5,
        total_cost=0.5,
    )
    metrics = extract_trace_metrics(trace_with_total)  # pyright: ignore[reportArgumentType]
    assert metrics.tool_call_count == 1
    assert metrics.turn_count == 1
    assert metrics.observation_count == 2
    assert metrics.total_input_tokens == 14
    assert metrics.total_output_tokens == 8
    assert metrics.total_cost == 0.5
    assert metrics.latency_sec == 1.5

    trace_without_total = _make_trace(
        input={"case": "b"},
        output={"done": True},
        observations=observations,
        latency=0.7,
        total_cost=None,
    )
    metrics = extract_trace_metrics(trace_without_total)  # pyright: ignore[reportArgumentType]
    assert metrics.total_cost == 0.15
    assert metrics.latency_sec == 0.7

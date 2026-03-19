from datetime import UTC, datetime
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from implementations.misalignment_qa.config_types import (
    AgentOverrideSpec,
    EvalSpec,
    ExperimentConfig,
    LLMJudgeSpec,
    VariantSpec,
)
from implementations.misalignment_qa.preparation import ExecutionIdentity, prepare_variant_runs
from implementations.misalignment_qa.report_metrics import (
    TraceRecord,
    _filter_to_run_instance,
    aggregate_condition_summaries,
)


def test_aggregate_condition_summaries_groups_trace_metrics_and_scores() -> None:
    traces = [
        TraceRecord(
            trace_id="t1",
            timestamp=datetime(2026, 3, 19, 16, 0, tzinfo=UTC),
            metadata={"variant_id": "baseline"},
        ),
        TraceRecord(
            trace_id="t2",
            timestamp=datetime(2026, 3, 19, 16, 1, tzinfo=UTC),
            metadata={"variant_id": "baseline"},
        ),
        TraceRecord(
            trace_id="t3",
            timestamp=datetime(2026, 3, 19, 16, 2, tzinfo=UTC),
            metadata={"variant_id": "treatment"},
        ),
    ]

    score_rows = [
        {"traceId": "t1", "name": "harmful", "avg_value": 1},
        {"traceId": "t2", "name": "harmful", "avg_value": 0},
        {"traceId": "t3", "name": "harmful", "avg_value": 1},
        {"traceId": "t1", "name": "helpful", "avg_value": 0},
        {"traceId": "t2", "name": "helpful", "avg_value": 1},
        {"traceId": "t3", "name": "helpful", "avg_value": 0},
        {"traceId": "t1", "name": "turn_count", "avg_value": 3},
        {"traceId": "t2", "name": "turn_count", "avg_value": 1},
        {"traceId": "t3", "name": "turn_count", "avg_value": 2},
    ]
    trace_metric_rows = [
        {"id": "t1", "avg_latency": 4000, "sum_totalCost": 0.1, "sum_totalTokens": "100", "count_count": "1"},
        {"id": "t2", "avg_latency": 2000, "sum_totalCost": 0.2, "sum_totalTokens": "200", "count_count": "1"},
        {"id": "t3", "avg_latency": 1000, "sum_totalCost": 0.3, "sum_totalTokens": "300", "count_count": "1"},
    ]

    summaries, metric_names = aggregate_condition_summaries(
        traces=traces,
        score_rows=score_rows,
        trace_metric_rows=trace_metric_rows,
        condition_key="variant_id",
    )

    assert metric_names == ["harmful", "helpful", "turn_count"]
    assert [summary.condition for summary in summaries] == ["baseline", "treatment"]

    baseline = summaries[0]
    assert baseline.traces == 2
    assert baseline.total_tokens == 300
    assert baseline.total_cost_usd == pytest.approx(0.3)
    assert baseline.latency_ms.average == 3000
    assert baseline.score_metrics["harmful"].average == 0.5
    assert baseline.score_metrics["helpful"].average == 0.5
    assert baseline.score_metrics["turn_count"].average == 2

    treatment = summaries[1]
    assert treatment.traces == 1
    assert treatment.total_tokens == 300
    assert treatment.total_cost_usd == pytest.approx(0.3)
    assert treatment.latency_ms.average == 1000
    assert treatment.score_metrics["harmful"].average == 1
    assert treatment.score_metrics["helpful"].average == 0
    assert treatment.score_metrics["turn_count"].average == 2


def test_filter_to_run_instance_defaults_to_latest_run_instance() -> None:
    traces = [
        TraceRecord(
            trace_id="old-1",
            timestamp=datetime(2026, 3, 19, 15, 0, tzinfo=UTC),
            metadata={"exp_id": "demo", "run_instance_id": "20260319T150000Z_old"},
        ),
        TraceRecord(
            trace_id="new-1",
            timestamp=datetime(2026, 3, 19, 16, 0, tzinfo=UTC),
            metadata={"exp_id": "demo", "run_instance_id": "20260319T160000Z_new"},
        ),
        TraceRecord(
            trace_id="new-2",
            timestamp=datetime(2026, 3, 19, 16, 1, tzinfo=UTC),
            metadata={"exp_id": "demo", "run_instance_id": "20260319T160000Z_new"},
        ),
    ]

    filtered, selected = _filter_to_run_instance(
        traces,
        run_instance_id=None,
        all_run_instances=False,
    )

    assert selected == "20260319T160000Z_new"
    assert [trace.trace_id for trace in filtered] == ["new-1", "new-2"]


def test_prepare_variant_runs_reuses_one_execution_identity_across_variants() -> None:
    config = ExperimentConfig(
        id="demo-exp",
        display_label="Demo Experiment",
        langfuse_dataset_name="demo-dataset",
        description="demo",
        base_agent=AgentOverrideSpec(system_prompt="Be helpful", model="gemini-2.5-flash"),
        examples=[],
        variants=[
            VariantSpec(id="v1", agent=AgentOverrideSpec(model="gemini-2.5-flash")),
            VariantSpec(id="v2", agent=AgentOverrideSpec(model="gemini-3-flash-preview")),
        ],
        tasks=[],
        evaluation=EvalSpec(llm_judge=LLMJudgeSpec(rubric_markdown="Return JSON only.")),
    )
    execution = ExecutionIdentity(
        run_instance_id="20260319T160000Z_abcd1234",
        run_started_at="2026-03-19T16:00:00+00:00",
    )

    prepared = prepare_variant_runs(config, execution=execution)

    assert [variant.run_instance_id for variant in prepared] == [
        "20260319T160000Z_abcd1234",
        "20260319T160000Z_abcd1234",
    ]
    assert prepared[0].run_name == "demo-exp__20260319T160000Z_abcd1234__v1"
    assert prepared[1].run_name == "demo-exp__20260319T160000Z_abcd1234__v2"
    assert prepared[0].run_metadata["run_instance_id"] == "20260319T160000Z_abcd1234"
    assert prepared[1].run_metadata["run_started_at"] == "2026-03-19T16:00:00+00:00"

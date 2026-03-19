"""Terminal reporting for misalignment QA experiments backed by Langfuse metrics.

This script combines two Langfuse surfaces:

1. ``trace.list`` to recover experiment metadata such as ``variant_id`` and
   ``model`` for each trace.
2. The Metrics API to fetch numeric scores and core trace metrics keyed by
   ``traceId``.

The join lets us aggregate results by any stored condition key while still
staying close to Langfuse's reporting primitives.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import json
from typing import Any

from dotenv import load_dotenv
from langfuse import Langfuse

from aieng.agent_evals.configs import Configs
from implementations.misalignment_qa.experiment import load_experiment_config


DEFAULT_HOURS_BACK = 24
DEFAULT_TRACE_PAGE_SIZE = 100
BOOLEAN_RATE_METRICS = {"harmful", "helpful", "follows_reckless_pattern"}
RUN_INSTANCE_METADATA_KEY = "run_instance_id"
PRIMARY_METRIC_ORDER = [
    "harmful",
    "helpful",
    "follows_reckless_pattern",
    "turn_count",
    "tool_call_count",
]


@dataclass(frozen=True)
class TraceRecord:
    """Minimal trace representation needed for aggregation."""

    trace_id: str
    timestamp: datetime
    metadata: dict[str, Any]


@dataclass
class NumericAccumulator:
    """Tracks sums and counts for averages."""

    total: float = 0.0
    count: int = 0

    def add(self, value: float, weight: int = 1) -> None:
        self.total += value * weight
        self.count += weight

    @property
    def average(self) -> float | None:
        if self.count == 0:
            return None
        return self.total / self.count


@dataclass
class ConditionSummary:
    """Aggregated metrics for one experimental condition."""

    condition: str
    traces: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    latency_ms: NumericAccumulator = field(default_factory=NumericAccumulator)
    score_metrics: dict[str, NumericAccumulator] = field(default_factory=lambda: defaultdict(NumericAccumulator))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report Langfuse experiment metrics as a terminal table.")
    parser.add_argument("--config", help="Optional experiment YAML config. Used to infer exp_id.")
    parser.add_argument("--experiment-id", help="Explicit exp_id filter. Overrides --config.")
    parser.add_argument(
        "--run-instance-id",
        help="Restrict to a single execution instance. If omitted, the latest run instance is used when available.",
    )
    parser.add_argument(
        "--all-run-instances",
        action="store_true",
        help="Aggregate all matching historical runs instead of defaulting to the latest run instance.",
    )
    parser.add_argument(
        "--condition-key",
        default="variant_id",
        help="Trace metadata key to group by, e.g. variant_id, model, condition_model, experiment_name.",
    )
    parser.add_argument(
        "--hours-back",
        type=int,
        default=DEFAULT_HOURS_BACK,
        help=f"Lookback window when --from is omitted. Default: {DEFAULT_HOURS_BACK} hours.",
    )
    parser.add_argument("--from", dest="from_timestamp", help="Inclusive start time in ISO-8601, e.g. 2026-03-19T15:45:00Z.")
    parser.add_argument("--to", dest="to_timestamp", help="Exclusive end time in ISO-8601. Defaults to now (UTC).")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of matching traces after filtering.",
    )
    return parser.parse_args()


def _parse_datetime(raw: str) -> datetime:
    value = raw.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _resolve_time_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    end = _parse_datetime(args.to_timestamp) if args.to_timestamp else datetime.now(tz=UTC)
    start = _parse_datetime(args.from_timestamp) if args.from_timestamp else end - timedelta(hours=args.hours_back)
    return start, end


def _resolve_experiment_id(args: argparse.Namespace) -> str | None:
    if args.experiment_id:
        return args.experiment_id
    if args.config:
        return load_experiment_config(args.config).id
    return None


def _build_client() -> Langfuse:
    load_dotenv(verbose=True)
    config = Configs()
    return Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key.get_secret_value() if config.langfuse_secret_key else None,
        host=config.langfuse_host,
    )


def _iter_traces(
    client: Langfuse,
    *,
    start: datetime,
    end: datetime,
    page_size: int = DEFAULT_TRACE_PAGE_SIZE,
) -> list[TraceRecord]:
    traces: list[TraceRecord] = []
    page = 1

    while True:
        response = client.api.trace.list(
            page=page,
            limit=page_size,
            from_timestamp=start,
            to_timestamp=end,
            order_by="timestamp.desc",
        )
        rows = list(getattr(response, "data", []))
        if not rows:
            break

        for row in rows:
            traces.append(
                TraceRecord(
                    trace_id=str(row.id),
                    timestamp=row.timestamp,
                    metadata=dict(getattr(row, "metadata", {}) or {}),
                )
            )

        if len(rows) < page_size:
            break
        page += 1

    return traces


def _filter_trace_records(
    traces: list[TraceRecord],
    *,
    experiment_id: str | None,
    limit: int | None,
) -> list[TraceRecord]:
    filtered = traces
    if experiment_id is not None:
        filtered = [trace for trace in filtered if trace.metadata.get("exp_id") == experiment_id]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def _metrics_query(
    *,
    view: str,
    metrics: list[dict[str, Any]],
    dimensions: list[dict[str, str]],
    start: datetime,
    end: datetime,
) -> str:
    return json.dumps(
        {
            "view": view,
            "metrics": metrics,
            "dimensions": dimensions,
            "filters": [],
            "fromTimestamp": start.isoformat().replace("+00:00", "Z"),
            "toTimestamp": end.isoformat().replace("+00:00", "Z"),
        }
    )


def _fetch_score_rows(client: Langfuse, *, start: datetime, end: datetime) -> list[dict[str, Any]]:
    response = client.api.metrics.metrics(
        query=_metrics_query(
            view="scores-numeric",
            metrics=[{"measure": "value", "aggregation": "avg"}],
            dimensions=[{"field": "traceId"}, {"field": "name"}],
            start=start,
            end=end,
        )
    )
    return list(response.data)


def _fetch_trace_metric_rows(client: Langfuse, *, start: datetime, end: datetime) -> list[dict[str, Any]]:
    response = client.api.metrics.metrics(
        query=_metrics_query(
            view="traces",
            metrics=[
                {"measure": "count", "aggregation": "count"},
                {"measure": "latency", "aggregation": "avg"},
                {"measure": "totalCost", "aggregation": "sum"},
                {"measure": "totalTokens", "aggregation": "sum"},
            ],
            dimensions=[{"field": "id"}],
            start=start,
            end=end,
        )
    )
    return list(response.data)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(float(value))


def _condition_value(trace: TraceRecord, condition_key: str) -> str:
    value = trace.metadata.get(condition_key)
    if value is None:
        return f"<missing:{condition_key}>"
    return str(value)


def _filter_to_run_instance(
    traces: list[TraceRecord],
    *,
    run_instance_id: str | None,
    all_run_instances: bool,
) -> tuple[list[TraceRecord], str | None]:
    if all_run_instances:
        return traces, None

    if run_instance_id is not None:
        return (
            [trace for trace in traces if trace.metadata.get(RUN_INSTANCE_METADATA_KEY) == run_instance_id],
            run_instance_id,
        )

    available_run_instances = {
        str(trace.metadata[RUN_INSTANCE_METADATA_KEY])
        for trace in traces
        if trace.metadata.get(RUN_INSTANCE_METADATA_KEY) is not None
    }
    if not available_run_instances:
        return traces, None

    latest_trace = max(
        (
            trace
            for trace in traces
            if trace.metadata.get(RUN_INSTANCE_METADATA_KEY) is not None
        ),
        key=lambda trace: trace.timestamp,
    )
    latest_run_instance_id = str(latest_trace.metadata[RUN_INSTANCE_METADATA_KEY])
    return (
        [trace for trace in traces if trace.metadata.get(RUN_INSTANCE_METADATA_KEY) == latest_run_instance_id],
        latest_run_instance_id,
    )


def aggregate_condition_summaries(
    *,
    traces: list[TraceRecord],
    score_rows: list[dict[str, Any]],
    trace_metric_rows: list[dict[str, Any]],
    condition_key: str,
) -> tuple[list[ConditionSummary], list[str]]:
    """Aggregate metrics rows into one summary per condition."""

    trace_index = {trace.trace_id: trace for trace in traces}
    summaries: dict[str, ConditionSummary] = {}
    seen_metric_names: set[str] = set()

    def get_summary(trace_id: str) -> ConditionSummary | None:
        trace = trace_index.get(trace_id)
        if trace is None:
            return None
        label = _condition_value(trace, condition_key)
        summary = summaries.get(label)
        if summary is None:
            summary = ConditionSummary(condition=label)
            summaries[label] = summary
        return summary

    for trace in traces:
        summary = get_summary(trace.trace_id)
        if summary is not None:
            summary.traces += 1

    for row in trace_metric_rows:
        trace_id = str(row["id"])
        summary = get_summary(trace_id)
        if summary is None:
            continue

        latency = _coerce_float(row.get("avg_latency"))
        total_cost = _coerce_float(row.get("sum_totalCost"))
        total_tokens = _coerce_int(row.get("sum_totalTokens"))
        count = _coerce_int(row.get("count_count")) or 0

        if latency is not None and count > 0:
            summary.latency_ms.add(latency, weight=count)
        if total_cost is not None:
            summary.total_cost_usd += total_cost
        if total_tokens is not None:
            summary.total_tokens += total_tokens

    for row in score_rows:
        trace_id = str(row["traceId"])
        metric_name = str(row["name"])
        summary = get_summary(trace_id)
        if summary is None:
            continue

        value = _coerce_float(row.get("avg_value"))
        if value is None:
            continue

        summary.score_metrics[metric_name].add(value)
        seen_metric_names.add(metric_name)

    ordered_metric_names = [
        metric_name for metric_name in PRIMARY_METRIC_ORDER if metric_name in seen_metric_names
    ] + sorted(metric_name for metric_name in seen_metric_names if metric_name not in PRIMARY_METRIC_ORDER)

    ordered_summaries = sorted(summaries.values(), key=lambda summary: summary.condition)
    return ordered_summaries, ordered_metric_names


def _format_metric_value(metric_name: str, value: float | None) -> str:
    if value is None:
        return "-"
    if metric_name in BOOLEAN_RATE_METRICS:
        return f"{value * 100:.1f}%"
    if metric_name in {"turn_count", "tool_call_count"}:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _table_rows(summaries: list[ConditionSummary], metric_names: list[str]) -> tuple[list[str], list[list[str]]]:
    headers = [
        "condition",
        "n",
        *[
            {
                "harmful": "harmful%",
                "helpful": "helpful%",
                "follows_reckless_pattern": "reckless%",
                "turn_count": "avg_turns",
                "tool_call_count": "avg_tools",
            }.get(metric_name, metric_name)
            for metric_name in metric_names
        ],
        "avg_latency_s",
        "avg_tokens",
        "total_tokens",
        "total_cost_usd",
    ]

    rows: list[list[str]] = []
    for summary in summaries:
        avg_latency_ms = summary.latency_ms.average
        avg_tokens = (summary.total_tokens / summary.traces) if summary.traces else None
        rows.append(
            [
                summary.condition,
                str(summary.traces),
                *[
                    _format_metric_value(metric_name, summary.score_metrics[metric_name].average)
                    for metric_name in metric_names
                ],
                "-" if avg_latency_ms is None else f"{avg_latency_ms / 1000:.2f}",
                "-" if avg_tokens is None else f"{avg_tokens:.1f}",
                str(summary.total_tokens),
                f"{summary.total_cost_usd:.6f}",
            ]
        )
    return headers, rows


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format a simple ASCII table."""

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    parts = [render_row(headers), separator]
    parts.extend(render_row(row) for row in rows)
    return "\n".join(parts)


def build_report(
    *,
    client: Langfuse,
    start: datetime,
    end: datetime,
    experiment_id: str | None,
    run_instance_id: str | None,
    all_run_instances: bool,
    condition_key: str,
    limit: int | None,
) -> str:
    """Build the terminal report text."""

    traces = _filter_trace_records(
        _iter_traces(client, start=start, end=end),
        experiment_id=experiment_id,
        limit=limit,
    )
    traces, selected_run_instance_id = _filter_to_run_instance(
        traces,
        run_instance_id=run_instance_id,
        all_run_instances=all_run_instances,
    )
    trace_ids = {trace.trace_id for trace in traces}
    if not trace_ids:
        if run_instance_id is not None:
            scope = f"run_instance_id={run_instance_id}"
        else:
            scope = f"exp_id={experiment_id}" if experiment_id else "provided time window"
        return f"No matching traces found for {scope}."

    score_rows = _fetch_score_rows(client, start=start, end=end)
    trace_metric_rows = _fetch_trace_metric_rows(client, start=start, end=end)

    summaries, metric_names = aggregate_condition_summaries(
        traces=traces,
        score_rows=score_rows,
        trace_metric_rows=trace_metric_rows,
        condition_key=condition_key,
    )
    headers, rows = _table_rows(summaries, metric_names)

    prelude = [
        "Misalignment QA Langfuse Report",
        f"time window: {start.isoformat()} -> {end.isoformat()}",
        f"matching traces: {len(trace_ids)}",
        f"grouped by: metadata.{condition_key}",
    ]
    if experiment_id:
        prelude.append(f"exp_id: {experiment_id}")
    if selected_run_instance_id is not None:
        prelude.append(f"run_instance_id: {selected_run_instance_id}")
    elif all_run_instances:
        prelude.append("run_scope: all matching run instances")
    elif run_instance_id is None:
        prelude.append("run_scope: all matching traces (no run_instance_id metadata found)")

    return "\n".join([*prelude, "", format_table(headers, rows)])


def main() -> None:
    """CLI entrypoint."""

    args = _parse_args()
    start, end = _resolve_time_window(args)
    experiment_id = _resolve_experiment_id(args)
    client = _build_client()
    report = build_report(
        client=client,
        start=start,
        end=end,
        experiment_id=experiment_id,
        run_instance_id=args.run_instance_id,
        all_run_instances=args.all_run_instances,
        condition_key=args.condition_key,
        limit=args.limit,
    )
    print(report)  # noqa: T201


if __name__ == "__main__":
    main()

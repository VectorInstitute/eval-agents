"""Notebook-friendly exploration helpers for misalignment QA Langfuse results."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import json
from typing import Any

from dotenv import load_dotenv
from langfuse import Langfuse
import pandas as pd

from implementations.misalignment_qa.experiment import load_experiment_config


DEFAULT_TRACE_PAGE_SIZE = 100
DEFAULT_DATASET_PAGE_SIZE = 100
RUN_INSTANCE_METADATA_KEY = "run_instance_id"
BOOLEAN_RATE_METRICS = {"harmful", "helpful", "follows_reckless_pattern"}
PRIMARY_METRIC_ORDER = [
    "harmful",
    "helpful",
    "follows_reckless_pattern",
    "turn_count",
    "tool_call_count",
]
DEFAULT_GROUP_KEY_ORDER = [
    "variant_id",
    "condition_model",
    "model",
    "condition_provider",
    "run_family",
    "exp_id",
    "run_instance_id",
]


@dataclass(frozen=True)
class LocalDatasetConfig:
    """Local experiment config keyed by Langfuse dataset name."""

    dataset_name: str
    experiment_id: str
    display_label: str
    description: str
    config_path: Path


@dataclass(frozen=True)
class TraceRecord:
    """Minimal trace representation needed for notebook aggregation."""

    trace_id: str
    timestamp: datetime
    metadata: dict[str, Any]


@dataclass
class NumericAccumulator:
    """Tracks sums and counts for averages."""

    total: float = 0.0
    count: int = 0

    def add(self, value: float, *, weight: int = 1) -> None:
        self.total += value * weight
        self.count += weight

    @property
    def average(self) -> float | None:
        if self.count == 0:
            return None
        return self.total / self.count


@dataclass
class ConditionSummary:
    """Aggregated metrics for one condition grouping."""

    condition: str
    traces: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    latency_ms: NumericAccumulator = field(default_factory=NumericAccumulator)
    score_metrics: dict[str, NumericAccumulator] = field(default_factory=lambda: defaultdict(NumericAccumulator))


@dataclass(frozen=True)
class AnalysisBundle:
    """Structured notebook output for one dataset/run selection."""

    dataset_name: str
    dataset_runs_df: pd.DataFrame
    run_instances_df: pd.DataFrame
    selected_runs_df: pd.DataFrame
    traces_df: pd.DataFrame
    summary_df: pd.DataFrame
    metric_names: list[str]
    available_group_keys: list[str]
    selected_execution: str
    experiment_ids: list[str]
    run_instance_ids: list[str]
    start: datetime | None
    end: datetime | None


def _parse_datetime(raw: str) -> datetime:
    value = raw.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(float(value))


def _normalize_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _condition_value(trace: TraceRecord, condition_key: str) -> str:
    value = trace.metadata.get(condition_key)
    if value is None:
        return f"<missing:{condition_key}>"
    return str(value)


def _config_root() -> Path:
    return Path(__file__).resolve().parent / "configs"


def _repo_root() -> Path:
    return next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / "implementations").exists())


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _build_client() -> Langfuse:
    load_dotenv(dotenv_path=_repo_root() / ".env", verbose=False)
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")
    if not public_key or not secret_key or not host:
        raise RuntimeError(
            "Missing Langfuse credentials. Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST."
        )
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )


def discover_local_dataset_configs(config_dir: Path | None = None) -> list[LocalDatasetConfig]:
    """Load all local misalignment QA configs for dataset-to-experiment mapping."""

    root = config_dir or _config_root()
    records: list[LocalDatasetConfig] = []
    for path in sorted(root.glob("*.yaml")):
        config = load_experiment_config(path)
        records.append(
            LocalDatasetConfig(
                dataset_name=config.langfuse_dataset_name,
                experiment_id=config.id,
                display_label=config.display_label,
                description=config.description or "",
                config_path=path,
            )
        )
    return records


def _list_all_datasets(client: Langfuse) -> list[Any]:
    datasets: list[Any] = []
    page = 1
    while True:
        response = client.api.datasets.list(page=page, limit=DEFAULT_DATASET_PAGE_SIZE)
        rows = list(getattr(response, "data", []))
        if not rows:
            break
        datasets.extend(rows)
        if len(rows) < DEFAULT_DATASET_PAGE_SIZE:
            break
        page += 1
    return datasets


def _list_all_dataset_runs(client: Langfuse, dataset_name: str) -> list[Any]:
    runs: list[Any] = []
    page = 1
    while True:
        response = client.api.datasets.get_runs(dataset_name, page=page, limit=DEFAULT_DATASET_PAGE_SIZE)
        rows = list(getattr(response, "data", []))
        if not rows:
            break
        runs.extend(rows)
        if len(rows) < DEFAULT_DATASET_PAGE_SIZE:
            break
        page += 1
    return runs


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
                    metadata=_normalize_metadata(getattr(row, "metadata", {})),
                )
            )

        if len(rows) < page_size:
            break
        page += 1

    return traces


def _fetch_score_rows(client: Langfuse, *, start: datetime, end: datetime) -> list[dict[str, Any]]:
    response = client.api.metrics.metrics(
        query=json.dumps(
            {
                "view": "scores-numeric",
                "metrics": [{"measure": "value", "aggregation": "avg"}],
                "dimensions": [{"field": "traceId"}, {"field": "name"}],
                "filters": [],
                "fromTimestamp": start.isoformat().replace("+00:00", "Z"),
                "toTimestamp": end.isoformat().replace("+00:00", "Z"),
            }
        )
    )
    return list(response.data)


def _fetch_trace_metric_rows(client: Langfuse, *, start: datetime, end: datetime) -> list[dict[str, Any]]:
    response = client.api.metrics.metrics(
        query=json.dumps(
            {
                "view": "traces",
                "metrics": [
                    {"measure": "count", "aggregation": "count"},
                    {"measure": "latency", "aggregation": "avg"},
                    {"measure": "totalCost", "aggregation": "sum"},
                    {"measure": "totalTokens", "aggregation": "sum"},
                ],
                "dimensions": [{"field": "id"}],
                "filters": [],
                "fromTimestamp": start.isoformat().replace("+00:00", "Z"),
                "toTimestamp": end.isoformat().replace("+00:00", "Z"),
            }
        )
    )
    return list(response.data)


def aggregate_condition_summaries(
    *,
    traces: list[TraceRecord],
    score_rows: list[dict[str, Any]],
    trace_metric_rows: list[dict[str, Any]],
    condition_key: str,
) -> tuple[list[ConditionSummary], list[str]]:
    """Aggregate trace and score rows into one summary per selected condition."""

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


def _summary_dataframe(summaries: list[ConditionSummary], metric_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        avg_latency_ms = summary.latency_ms.average
        avg_tokens = (summary.total_tokens / summary.traces) if summary.traces else None
        row: dict[str, Any] = {
            "condition": summary.condition,
            "traces": summary.traces,
            "avg_latency_s": None if avg_latency_ms is None else avg_latency_ms / 1000,
            "avg_tokens": avg_tokens,
            "total_tokens": summary.total_tokens,
            "total_cost_usd": summary.total_cost_usd,
        }
        for metric_name in metric_names:
            value = summary.score_metrics[metric_name].average
            if metric_name in BOOLEAN_RATE_METRICS and value is not None:
                row[f"{metric_name}_pct"] = value * 100
            else:
                row[metric_name] = value
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "condition",
                "traces",
                *[
                    f"{metric_name}_pct" if metric_name in BOOLEAN_RATE_METRICS else metric_name
                    for metric_name in metric_names
                ],
                "avg_latency_s",
                "avg_tokens",
                "total_tokens",
                "total_cost_usd",
            ]
        )

    summary_df = pd.DataFrame(rows)
    preferred_columns = [
        "condition",
        "traces",
        *[
            f"{metric_name}_pct" if metric_name in BOOLEAN_RATE_METRICS else metric_name
            for metric_name in metric_names
        ],
        "avg_latency_s",
        "avg_tokens",
        "total_tokens",
        "total_cost_usd",
    ]
    available_columns = [column for column in preferred_columns if column in summary_df.columns]
    return summary_df.loc[:, available_columns].sort_values(["traces", "condition"], ascending=[False, True]).reset_index(
        drop=True
    )


def _trace_dataframe(traces: list[TraceRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        metadata = trace.metadata
        rows.append(
            {
                "trace_id": trace.trace_id,
                "timestamp": trace.timestamp,
                "exp_id": metadata.get("exp_id"),
                "run_instance_id": metadata.get("run_instance_id"),
                "variant_id": metadata.get("variant_id"),
                "model": metadata.get("model"),
                "condition_model": metadata.get("condition_model"),
                "condition_provider": metadata.get("condition_provider"),
                "run_family": metadata.get("run_family"),
                "metadata": metadata,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "trace_id",
                "timestamp",
                "exp_id",
                "run_instance_id",
                "variant_id",
                "model",
                "condition_model",
                "condition_provider",
                "run_family",
                "metadata",
            ]
        )
    return pd.DataFrame(rows).sort_values("timestamp", ascending=False).reset_index(drop=True)


def _stringify_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [_stringify_value(part) for part in content]
            joined = "\n".join(part for part in text_parts if part)
            return joined or json.dumps(value, ensure_ascii=False)
        parts = value.get("parts")
        if isinstance(parts, list):
            text_parts = [_stringify_value(part) for part in parts]
            joined = "\n".join(part for part in text_parts if part)
            return joined or json.dumps(value, ensure_ascii=False)
        text = value.get("text")
        if isinstance(text, str):
            return text
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        text_parts = [_stringify_value(part) for part in value]
        joined = "\n".join(part for part in text_parts if part)
        return joined or json.dumps(value, ensure_ascii=False)
    return str(value)


def _truncate_text(value: str | None, *, limit: int = 240) -> str | None:
    if value is None or len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def _extract_judge_payload(trace_detail: Any) -> dict[str, Any]:
    for observation in getattr(trace_detail, "observations", []) or []:
        if getattr(observation, "type", None) != "GENERATION":
            continue
        text = _stringify_value(getattr(observation, "output", None))
        if not text:
            continue
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "metrics" in payload and "explanation" in payload:
            return payload
    return {}


def _trace_url(trace_detail: Any) -> str | None:
    html_path = getattr(trace_detail, "html_path", None)
    host = os.getenv("LANGFUSE_HOST")
    if not html_path or not host:
        return None
    return f"{host.rstrip('/')}{html_path}"


def _trace_detail_row(trace_detail: Any) -> dict[str, Any]:
    metadata = _normalize_metadata(getattr(trace_detail, "metadata", None))
    judge_payload = _extract_judge_payload(trace_detail)
    scores = getattr(trace_detail, "scores", []) or []

    row: dict[str, Any] = {
        "trace_id": str(getattr(trace_detail, "id")),
        "timestamp": getattr(trace_detail, "timestamp", None),
        "task_id": metadata.get("task_id"),
        "exp_id": metadata.get("exp_id"),
        "run_instance_id": metadata.get("run_instance_id"),
        "variant_id": metadata.get("variant_id"),
        "model": metadata.get("model"),
        "condition_model": metadata.get("condition_model"),
        "condition_provider": metadata.get("condition_provider"),
        "run_family": metadata.get("run_family"),
        "trace_input": _stringify_value(getattr(trace_detail, "input", None)),
        "model_output": _stringify_value(getattr(trace_detail, "output", None)),
        "judge_explanation": judge_payload.get("explanation"),
        "langfuse_url": _trace_url(trace_detail),
        "metadata": metadata,
    }
    row["trace_input_preview"] = _truncate_text(row["trace_input"])
    row["model_output_preview"] = _truncate_text(row["model_output"])
    row["judge_explanation_preview"] = _truncate_text(row["judge_explanation"])

    for score in scores:
        name = str(getattr(score, "name", ""))
        if not name:
            continue
        row[name] = getattr(score, "value", None)
        row[f"{name}_comment"] = getattr(score, "comment", None)

    return row


def _collect_group_keys(traces: list[TraceRecord], dataset_runs_df: pd.DataFrame) -> list[str]:
    keys: set[str] = set()
    for trace in traces:
        keys.update(trace.metadata.keys())

    if "metadata" in dataset_runs_df.columns:
        for metadata in dataset_runs_df["metadata"].tolist():
            if isinstance(metadata, dict):
                keys.update(metadata.keys())

    ordered = [key for key in DEFAULT_GROUP_KEY_ORDER if key in keys]
    ordered.extend(sorted(key for key in keys if key not in DEFAULT_GROUP_KEY_ORDER))
    return ordered


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


class MisalignmentResultsExplorer:
    """Dataset-first Langfuse explorer for the notebook UI."""

    def __init__(self, *, config_dir: Path | None = None, client: Langfuse | None = None) -> None:
        self.config_dir = config_dir or _config_root()
        self.client = client or _build_client()
        self.local_configs = discover_local_dataset_configs(self.config_dir)

    def list_datasets_frame(self) -> pd.DataFrame:
        """Return remote Langfuse datasets annotated with matching local configs."""

        config_rows_by_dataset: dict[str, list[LocalDatasetConfig]] = defaultdict(list)
        for record in self.local_configs:
            config_rows_by_dataset[record.dataset_name].append(record)

        live_rows = {dataset.name: dataset for dataset in _list_all_datasets(self.client)}
        dataset_names = sorted(
            set(live_rows) | {record.dataset_name for record in self.local_configs},
            key=lambda name: (
                live_rows[name].updated_at if name in live_rows else datetime.min.replace(tzinfo=UTC),
                name,
            ),
            reverse=True,
        )

        rows: list[dict[str, Any]] = []
        for dataset_name in dataset_names:
            dataset = live_rows.get(dataset_name)
            configs = config_rows_by_dataset.get(dataset_name, [])
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_exists": dataset is not None,
                    "created_at": getattr(dataset, "created_at", None),
                    "updated_at": getattr(dataset, "updated_at", None),
                    "description": getattr(dataset, "description", None) or (configs[0].description if configs else ""),
                    "local_experiment_ids": ", ".join(sorted(record.experiment_id for record in configs)),
                    "local_display_labels": ", ".join(sorted(record.display_label for record in configs)),
                    "local_config_paths": ", ".join(sorted(_display_path(record.config_path) for record in configs)),
                }
            )
        return pd.DataFrame(rows)

    def list_dataset_runs_frame(self, dataset_name: str) -> pd.DataFrame:
        """Return one row per Langfuse dataset run."""

        rows: list[dict[str, Any]] = []
        for run in _list_all_dataset_runs(self.client, dataset_name):
            metadata = _normalize_metadata(getattr(run, "metadata", None))
            run_started_at_raw = metadata.get("run_started_at")
            execution_id = metadata.get("run_instance_id") or getattr(run, "name", None)
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "run_name": run.name,
                    "execution_id": execution_id,
                    "run_instance_id": metadata.get("run_instance_id"),
                    "exp_id": metadata.get("exp_id"),
                    "variant_id": metadata.get("variant_id"),
                    "model": metadata.get("model"),
                    "condition_model": metadata.get("condition_model"),
                    "condition_provider": metadata.get("condition_provider"),
                    "run_family": metadata.get("run_family"),
                    "run_started_at": _parse_datetime(run_started_at_raw) if isinstance(run_started_at_raw, str) else None,
                    "created_at": run.created_at,
                    "updated_at": run.updated_at,
                    "metadata": metadata,
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "dataset_name",
                    "run_name",
                    "execution_id",
                    "run_instance_id",
                    "exp_id",
                    "variant_id",
                    "model",
                    "condition_model",
                    "condition_provider",
                    "run_family",
                    "run_started_at",
                    "created_at",
                    "updated_at",
                    "metadata",
                ]
            )

        return pd.DataFrame(rows).sort_values(["created_at", "run_name"], ascending=[False, True]).reset_index(drop=True)

    def list_run_instances_frame(self, dataset_name: str) -> pd.DataFrame:
        """Collapse dataset runs into one row per execution instance."""

        dataset_runs_df = self.list_dataset_runs_frame(dataset_name)
        if dataset_runs_df.empty:
            return pd.DataFrame(
                columns=[
                    "execution_id",
                    "run_instance_id",
                    "started_at",
                    "created_at",
                    "updated_at",
                    "run_count",
                    "variant_count",
                    "experiment_ids",
                    "variant_ids",
                    "models",
                ]
            )

        grouped_rows: list[dict[str, Any]] = []
        for execution_id, group in dataset_runs_df.groupby("execution_id", dropna=False):
            experiment_ids = sorted({value for value in group["exp_id"].dropna().astype(str).tolist() if value})
            variant_ids = sorted({value for value in group["variant_id"].dropna().astype(str).tolist() if value})
            models = sorted({value for value in group["model"].dropna().astype(str).tolist() if value})
            started_candidates = [value for value in group["run_started_at"].tolist() if value is not None]
            grouped_rows.append(
                {
                    "execution_id": execution_id,
                    "run_instance_id": next(
                        (value for value in group["run_instance_id"].tolist() if isinstance(value, str) and value),
                        None,
                    ),
                    "started_at": min(started_candidates) if started_candidates else group["created_at"].min(),
                    "created_at": group["created_at"].min(),
                    "updated_at": group["updated_at"].max(),
                    "run_count": len(group),
                    "variant_count": len(variant_ids),
                    "experiment_ids": ", ".join(experiment_ids),
                    "variant_ids": ", ".join(variant_ids),
                    "models": ", ".join(models),
                }
            )

        return pd.DataFrame(grouped_rows).sort_values("created_at", ascending=False).reset_index(drop=True)

    def analyze_dataset(
        self,
        *,
        dataset_name: str,
        execution_id: str = "latest",
        condition_key: str = "variant_id",
        trace_limit: int | None = None,
        time_buffer_minutes: int = 5,
    ) -> AnalysisBundle:
        """Build notebook tables for one dataset selection and execution scope."""

        dataset_runs_df = self.list_dataset_runs_frame(dataset_name)
        run_instances_df = self.list_run_instances_frame(dataset_name)
        selected_execution = execution_id

        if dataset_runs_df.empty:
            return AnalysisBundle(
                dataset_name=dataset_name,
                dataset_runs_df=dataset_runs_df,
                run_instances_df=run_instances_df,
                selected_runs_df=dataset_runs_df,
                traces_df=_trace_dataframe([]),
                summary_df=pd.DataFrame(),
                metric_names=[],
                available_group_keys=DEFAULT_GROUP_KEY_ORDER.copy(),
                selected_execution=execution_id,
                experiment_ids=[],
                run_instance_ids=[],
                start=None,
                end=None,
            )

        if execution_id == "latest":
            latest_execution_id = str(run_instances_df.iloc[0]["execution_id"])
            selected_runs_df = dataset_runs_df[dataset_runs_df["execution_id"] == latest_execution_id].copy()
            selected_execution = latest_execution_id
        elif execution_id == "all":
            selected_runs_df = dataset_runs_df.copy()
        else:
            selected_runs_df = dataset_runs_df[dataset_runs_df["execution_id"] == execution_id].copy()
            if selected_runs_df.empty:
                raise ValueError(f"Unknown execution_id '{execution_id}' for dataset '{dataset_name}'.")

        experiment_ids = sorted({value for value in selected_runs_df["exp_id"].dropna().astype(str).tolist() if value})
        run_instance_ids = sorted(
            {value for value in selected_runs_df["run_instance_id"].dropna().astype(str).tolist() if value}
        )
        start = selected_runs_df["created_at"].min().to_pydatetime() - timedelta(minutes=time_buffer_minutes)
        end = selected_runs_df["updated_at"].max().to_pydatetime() + timedelta(minutes=time_buffer_minutes)

        traces = _iter_traces(self.client, start=start, end=end)
        filtered_traces = [
            trace
            for trace in traces
            if (not experiment_ids or str(trace.metadata.get("exp_id")) in experiment_ids)
            and (not run_instance_ids or str(trace.metadata.get("run_instance_id")) in run_instance_ids)
        ]
        if trace_limit is not None:
            filtered_traces = filtered_traces[:trace_limit]

        score_rows = _fetch_score_rows(self.client, start=start, end=end)
        trace_metric_rows = _fetch_trace_metric_rows(self.client, start=start, end=end)
        summaries, metric_names = aggregate_condition_summaries(
            traces=filtered_traces,
            score_rows=score_rows,
            trace_metric_rows=trace_metric_rows,
            condition_key=condition_key,
        )

        available_group_keys = _collect_group_keys(filtered_traces, dataset_runs_df)
        if condition_key not in available_group_keys:
            available_group_keys = [condition_key, *available_group_keys]

        return AnalysisBundle(
            dataset_name=dataset_name,
            dataset_runs_df=dataset_runs_df,
            run_instances_df=run_instances_df,
            selected_runs_df=selected_runs_df.reset_index(drop=True),
            traces_df=_trace_dataframe(filtered_traces),
            summary_df=_summary_dataframe(summaries, metric_names),
            metric_names=metric_names,
            available_group_keys=available_group_keys,
            selected_execution=selected_execution,
            experiment_ids=experiment_ids,
            run_instance_ids=run_instance_ids,
            start=start,
            end=end,
        )

    def build_master_traces_frame(
        self,
        *,
        dataset_name: str,
        execution_id: str = "latest",
        condition_key: str = "variant_id",
        trace_limit: int | None = None,
        time_buffer_minutes: int = 5,
    ) -> pd.DataFrame:
        """Return one joined row per trace with outputs, scores, and judge comments."""

        bundle = self.analyze_dataset(
            dataset_name=dataset_name,
            execution_id=execution_id,
            condition_key=condition_key,
            trace_limit=trace_limit,
            time_buffer_minutes=time_buffer_minutes,
        )
        if bundle.traces_df.empty:
            return pd.DataFrame()

        rows = [
            _trace_detail_row(self.client.api.trace.get(str(trace_id)))
            for trace_id in bundle.traces_df["trace_id"].tolist()
        ]
        master_df = pd.DataFrame(rows)
        if master_df.empty:
            return master_df

        metric_columns = _dedupe_preserving_order([
            column
            for column in PRIMARY_METRIC_ORDER
            if column in master_df.columns
        ] + sorted(
            column
            for column in master_df.columns
            if column not in {
                "trace_id",
                "timestamp",
                "task_id",
                "exp_id",
                "run_instance_id",
                "variant_id",
                "model",
                "condition_model",
                "condition_provider",
                "run_family",
                "trace_input",
                "trace_input_preview",
                "model_output",
                "model_output_preview",
                "judge_explanation",
                "judge_explanation_preview",
                "langfuse_url",
                "metadata",
            }
            and not column.endswith("_comment")
        ))
        comment_columns = sorted(column for column in master_df.columns if column.endswith("_comment"))
        preferred_columns = [
            "timestamp",
            "trace_id",
            "task_id",
            "variant_id",
            "model",
            "condition_model",
            "condition_provider",
            "exp_id",
            "run_instance_id",
            *metric_columns,
            "judge_explanation",
            *comment_columns,
            "trace_input",
            "model_output",
            "trace_input_preview",
            "model_output_preview",
            "judge_explanation_preview",
            "langfuse_url",
            "metadata",
        ]
        available_columns = _dedupe_preserving_order([column for column in preferred_columns if column in master_df.columns])
        return master_df.loc[:, available_columns].sort_values("timestamp", ascending=False).reset_index(drop=True)


__all__ = [
    "AnalysisBundle",
    "BOOLEAN_RATE_METRICS",
    "DEFAULT_GROUP_KEY_ORDER",
    "LocalDatasetConfig",
    "MisalignmentResultsExplorer",
    "discover_local_dataset_configs",
]

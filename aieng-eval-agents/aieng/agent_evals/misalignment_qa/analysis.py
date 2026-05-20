"""Notebook-friendly exploration helpers for misalignment QA Langfuse results."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from aieng.agent_evals.misalignment_qa.experiment import load_experiment_config
from dotenv import load_dotenv
from langfuse import Langfuse


_PAGE_SIZE = 100

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
    """Mapping from a local YAML config to its Langfuse dataset name."""

    dataset_name: str
    experiment_id: str
    display_label: str
    description: str
    config_path: Path


@dataclass(frozen=True)
class AnalysisBundle:
    """Summary of one dataset/execution analysis returned by analyze_dataset."""

    selected_execution: str
    """Execution ID selected (resolved from 'latest' / 'all' / a specific ID)."""
    summary_df: pd.DataFrame
    """One row per condition group with metric averages. Shown in the notebook."""
    dataset_runs_df: pd.DataFrame
    """All dataset runs for this dataset (useful for inspection)."""
    trace_ids: list[str]
    """Trace IDs matched to this execution (used by build_master_traces_frame)."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "implementations").exists())


def _build_client() -> Langfuse:
    load_dotenv(dotenv_path=_repo_root() / ".env", verbose=False)
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")
    if not public_key or not secret_key or not host:
        raise RuntimeError(
            "Missing Langfuse credentials. Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST."
        )
    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


def _normalize_metadata(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _parse_datetime(raw: str) -> datetime:
    value = raw.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


# ---------------------------------------------------------------------------
# Langfuse data fetchers
# ---------------------------------------------------------------------------


def _list_all_datasets(client: Langfuse) -> list[Any]:
    items, page = [], 1
    while True:
        rows = list(getattr(client.api.datasets.list(page=page, limit=_PAGE_SIZE), "data", []))
        items.extend(rows)
        if len(rows) < _PAGE_SIZE:
            break
        page += 1
    return items


def _list_all_dataset_runs(client: Langfuse, dataset_name: str) -> list[Any]:
    items, page = [], 1
    while True:
        rows = list(getattr(client.api.datasets.get_runs(dataset_name, page=page, limit=_PAGE_SIZE), "data", []))
        items.extend(rows)
        if len(rows) < _PAGE_SIZE:
            break
        page += 1
    return items


def _iter_traces(client: Langfuse, *, start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Fetch all traces in a time window; return lightweight dicts."""
    traces: list[dict[str, Any]] = []
    page = 1
    while True:
        rows = list(
            getattr(
                client.api.trace.list(
                    page=page,
                    limit=_PAGE_SIZE,
                    from_timestamp=start,
                    to_timestamp=end,
                    order_by="timestamp.desc",
                ),
                "data",
                [],
            )
        )
        if not rows:
            break
        traces.extend(
            {
                "trace_id": str(r.id),
                "timestamp": r.timestamp,
                "metadata": _normalize_metadata(getattr(r, "metadata", {})),
            }
            for r in rows
        )
        if len(rows) < _PAGE_SIZE:
            break
        page += 1
    return traces


def _query_metrics_api(client: Langfuse, *, payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(client.api.metrics.metrics(query=json.dumps(payload)).data)


def _fetch_scores_df(client: Langfuse, *, start: datetime, end: datetime) -> pd.DataFrame:
    """Return a DataFrame (trace_id, metric_name, value) from Langfuse scores API."""
    rows = _query_metrics_api(
        client,
        payload={
            "view": "scores-numeric",
            "metrics": [{"measure": "value", "aggregation": "avg"}],
            "dimensions": [{"field": "traceId"}, {"field": "name"}],
            "filters": [],
            "fromTimestamp": start.isoformat().replace("+00:00", "Z"),
            "toTimestamp": end.isoformat().replace("+00:00", "Z"),
        },
    )
    records = [
        {"trace_id": str(r["traceId"]), "metric_name": str(r["name"]), "value": float(r["avg_value"])}
        for r in rows
        if r.get("avg_value") is not None
    ]
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["trace_id", "metric_name", "value"])


def _fetch_trace_metrics_df(client: Langfuse, *, start: datetime, end: datetime) -> pd.DataFrame:
    """Return a DataFrame with per-trace usage metrics (latency, cost, tokens)."""
    rows = _query_metrics_api(
        client,
        payload={
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
        },
    )
    if not rows:
        return pd.DataFrame(columns=["trace_id", "avg_latency_ms", "total_cost_usd", "total_tokens"])

    def _to_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    df = pd.DataFrame(
        [
            {
                "trace_id": str(r["id"]),
                "avg_latency_ms": _to_float(r.get("avg_latency")),
                "total_cost_usd": _to_float(r.get("sum_totalCost")),
                "total_tokens": _to_float(r.get("sum_totalTokens")),
            }
            for r in rows
        ]
    )
    for col in ("avg_latency_ms", "total_cost_usd", "total_tokens"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Trace detail parsing (used by build_master_traces_frame)
# ---------------------------------------------------------------------------


def _stringify_value(value: Any) -> str | None:
    """Recursively extract readable text from varied trace output shapes."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        items = value.values() if isinstance(value, dict) else value  # type: ignore[assignment]
        if isinstance(value, dict):
            for key in ("content", "parts", "text"):
                inner = value.get(key)
                if isinstance(inner, str):
                    return inner
                if isinstance(inner, list):
                    items = inner
                    break
        joined = "\n".join(p for p in (_stringify_value(x) for x in items) if p)
        return joined or json.dumps(value, ensure_ascii=False)
    return str(value)


def _truncate(value: str | None, *, limit: int = 240) -> str | None:
    if value is None or len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def _extract_judge_payload(trace_detail: Any) -> dict[str, Any]:
    """Find the LLM judge JSON output in a trace's observations."""
    for obs in getattr(trace_detail, "observations", []) or []:
        if getattr(obs, "type", None) != "GENERATION":
            continue
        text = _stringify_value(getattr(obs, "output", None))
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
    return f"{host.rstrip('/')}{html_path}" if html_path and host else None


def _trace_detail_row(trace_detail: Any) -> dict[str, Any]:
    """Flatten one full trace object into a dict row for the master DataFrame."""
    metadata = _normalize_metadata(getattr(trace_detail, "metadata", None))
    judge_payload = _extract_judge_payload(trace_detail)
    scores = getattr(trace_detail, "scores", []) or []

    trace_input = _stringify_value(getattr(trace_detail, "input", None))
    model_output = _stringify_value(getattr(trace_detail, "output", None))
    judge_explanation = judge_payload.get("explanation")

    row: dict[str, Any] = {
        "trace_id": str(trace_detail.id),
        "timestamp": getattr(trace_detail, "timestamp", None),
        "task_id": metadata.get("task_id"),
        "exp_id": metadata.get("exp_id"),
        "run_instance_id": metadata.get("run_instance_id"),
        "variant_id": metadata.get("variant_id"),
        "model": metadata.get("model"),
        # condition_ prefix is added by prepare_variant_runs for condition_metadata keys
        "condition": metadata.get("condition_condition"),
        "condition_model": metadata.get("condition_model"),
        "condition_provider": metadata.get("condition_provider"),
        "run_family": metadata.get("run_family"),
        "trace_input": trace_input,
        "model_output": model_output,
        "judge_explanation": judge_explanation,
        "trace_input_preview": _truncate(trace_input),
        "model_output_preview": _truncate(model_output),
        "judge_explanation_preview": _truncate(judge_explanation),
        "langfuse_url": _trace_url(trace_detail),
        "metadata": metadata,
    }
    for score in scores:
        name = str(getattr(score, "name", ""))
        if name:
            row[name] = getattr(score, "value", None)
            row[f"{name}_comment"] = getattr(score, "comment", None)
    return row


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------


def discover_local_dataset_configs(config_dir: Path) -> list[LocalDatasetConfig]:
    """Load all local YAML configs and map them to their Langfuse dataset names."""
    configs = []
    for path in sorted(config_dir.glob("*.yaml")):
        config = load_experiment_config(path)
        configs.append(
            LocalDatasetConfig(
                dataset_name=config.langfuse_dataset_name,
                experiment_id=config.id,
                display_label=config.display_label,
                description=config.description or "",
                config_path=path,
            )
        )
    return configs


def _build_summary_df(
    traces: list[dict[str, Any]],
    scores_df: pd.DataFrame,
    trace_metrics_df: pd.DataFrame,
    *,
    condition_key: str,
) -> pd.DataFrame:
    """Aggregate per-trace data into one summary row per condition group."""
    if not traces:
        return pd.DataFrame()

    base_df = pd.DataFrame(
        [
            {
                "trace_id": t["trace_id"],
                "condition": str(t["metadata"].get(condition_key, f"<missing:{condition_key}>")),
            }
            for t in traces
        ]
    )

    if not scores_df.empty:
        score_pivot = scores_df.pivot_table(
            index="trace_id", columns="metric_name", values="value", aggfunc="mean"
        ).reset_index()
    else:
        score_pivot = pd.DataFrame(columns=["trace_id"])

    merged = base_df.merge(score_pivot, on="trace_id", how="left")
    if not trace_metrics_df.empty:
        merged = merged.merge(
            trace_metrics_df[["trace_id", "avg_latency_ms", "total_cost_usd", "total_tokens"]],
            on="trace_id",
            how="left",
        )

    score_metric_names = [
        c
        for c in merged.columns
        if c not in {"trace_id", "condition", "avg_latency_ms", "total_cost_usd", "total_tokens"}
    ]
    agg: dict[str, Any] = {"trace_id": "count"}
    for col in score_metric_names:
        agg[col] = "mean"
    for col, fn in [("avg_latency_ms", "mean"), ("total_cost_usd", "sum"), ("total_tokens", "sum")]:
        if col in merged.columns:
            agg[col] = fn

    summary = merged.groupby("condition", as_index=False).agg(agg).rename(columns={"trace_id": "traces"})

    if "avg_latency_ms" in summary.columns:
        summary["avg_latency_s"] = (summary["avg_latency_ms"] / 1000).round(2)
        summary = summary.drop(columns=["avg_latency_ms"])
    if "total_tokens" in summary.columns:
        summary["avg_tokens"] = (summary["total_tokens"] / summary["traces"].replace(0, pd.NA)).round(0)

    for metric in score_metric_names:
        if metric in BOOLEAN_RATE_METRICS and metric in summary.columns:
            summary[f"{metric}_pct"] = (summary[metric] * 100).round(1)
            summary = summary.drop(columns=[metric])

    def pct_or_orig(m: str) -> str:
        return f"{m}_pct" if m in BOOLEAN_RATE_METRICS else m

    ordered_metrics = [m for m in PRIMARY_METRIC_ORDER if m in score_metric_names] + sorted(
        m for m in score_metric_names if m not in PRIMARY_METRIC_ORDER
    )
    preferred_cols = (
        ["condition", "traces"]
        + [pct_or_orig(m) for m in ordered_metrics if pct_or_orig(m) in summary.columns]
        + [c for c in ["avg_latency_s", "avg_tokens", "total_tokens", "total_cost_usd"] if c in summary.columns]
    )

    return (
        summary[[c for c in preferred_cols if c in summary.columns]]
        .sort_values(["traces", "condition"], ascending=[False, True])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MisalignmentResultsExplorer:
    """Dataset-first Langfuse explorer for the notebook UI.

    Instantiate, then call ``analyze_dataset`` or ``build_master_traces_frame``
    with a Langfuse dataset name to pull scores and trace metadata.
    """

    def __init__(self, *, config_dir: Path, client: Langfuse | None = None) -> None:
        self.config_dir = config_dir
        self.client = client or _build_client()
        self.local_configs = discover_local_dataset_configs(self.config_dir)

    def list_datasets_frame(self) -> pd.DataFrame:
        """Return all known datasets (remote + local YAML configs) as a DataFrame."""
        config_by_dataset: dict[str, list[LocalDatasetConfig]] = {}
        for cfg in self.local_configs:
            config_by_dataset.setdefault(cfg.dataset_name, []).append(cfg)

        live = {d.name: d for d in _list_all_datasets(self.client)}
        all_names = sorted(
            set(live) | set(config_by_dataset),
            key=lambda n: (live[n].updated_at if n in live else datetime.min.replace(tzinfo=UTC), n),
            reverse=True,
        )
        rows = []
        for name in all_names:
            dataset = live.get(name)
            cfgs = config_by_dataset.get(name, [])
            rows.append(
                {
                    "dataset_name": name,
                    "dataset_exists": dataset is not None,
                    "created_at": getattr(dataset, "created_at", None),
                    "updated_at": getattr(dataset, "updated_at", None),
                    "description": getattr(dataset, "description", None) or (cfgs[0].description if cfgs else ""),
                    "local_experiment_ids": ", ".join(sorted(c.experiment_id for c in cfgs)),
                    "local_config_paths": ", ".join(sorted(c.config_path.name for c in cfgs)),
                }
            )
        return pd.DataFrame(rows)

    def list_dataset_runs_frame(self, dataset_name: str) -> pd.DataFrame:
        """Return one row per Langfuse dataset run for the given dataset."""
        rows = []
        for run in _list_all_dataset_runs(self.client, dataset_name):
            metadata = _normalize_metadata(getattr(run, "metadata", None))
            run_started_raw = metadata.get("run_started_at")
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "run_name": run.name,
                    "execution_id": metadata.get("run_instance_id") or run.name,
                    "run_instance_id": metadata.get("run_instance_id"),
                    "exp_id": metadata.get("exp_id"),
                    "variant_id": metadata.get("variant_id"),
                    "model": metadata.get("model"),
                    "condition_model": metadata.get("condition_model"),
                    "condition_provider": metadata.get("condition_provider"),
                    "run_family": metadata.get("run_family"),
                    "run_started_at": _parse_datetime(run_started_raw) if isinstance(run_started_raw, str) else None,
                    "created_at": run.created_at,
                    "updated_at": run.updated_at,
                    "metadata": metadata,
                }
            )
        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(rows).sort_values(["created_at", "run_name"], ascending=[False, True]).reset_index(drop=True)
        )

    def list_run_instances_frame(self, dataset_name: str) -> pd.DataFrame:
        """Collapse dataset runs into one row per execution instance."""
        runs_df = self.list_dataset_runs_frame(dataset_name)
        if runs_df.empty:
            return pd.DataFrame()

        def join_unique(series: pd.Series) -> str:
            return ", ".join(sorted({str(v) for v in series.dropna() if v}))

        grouped = (
            runs_df.groupby("execution_id", dropna=False)
            .agg(
                run_instance_id=("run_instance_id", "first"),
                started_at=("run_started_at", lambda s: next((v for v in s if v is not None), None)),
                created_at=("created_at", "min"),
                updated_at=("updated_at", "max"),
                run_count=("run_name", "count"),
                experiment_ids=("exp_id", join_unique),
                variant_ids=("variant_id", join_unique),
                models=("model", join_unique),
            )
            .reset_index()
        )
        grouped["variant_count"] = grouped["variant_ids"].apply(
            lambda s: len([x for x in s.split(", ") if x]) if s else 0
        )
        return grouped.sort_values("created_at", ascending=False).reset_index(drop=True)

    def analyze_dataset(
        self,
        *,
        dataset_name: str,
        execution_id: str = "latest",
        condition_key: str = "variant_id",
        trace_limit: int | None = None,
        time_buffer_minutes: int = 5,
    ) -> AnalysisBundle:
        """Build a summary table for one dataset/execution selection.

        Args:
            dataset_name: Langfuse dataset name.
            execution_id: "latest" (default), "all", or a specific execution ID from
                list_run_instances_frame().
            condition_key: Trace metadata key to group by (e.g. "variant_id", "model").
            trace_limit: Cap on traces fetched (useful for quick iteration).
            time_buffer_minutes: Extra minutes added around run timestamps when
                fetching traces.
        """
        dataset_runs_df = self.list_dataset_runs_frame(dataset_name)
        run_instances_df = self.list_run_instances_frame(dataset_name)

        if dataset_runs_df.empty:
            return AnalysisBundle(
                selected_execution=execution_id,
                summary_df=pd.DataFrame(),
                dataset_runs_df=dataset_runs_df,
                trace_ids=[],
            )

        if execution_id == "latest":
            selected_id = str(run_instances_df.iloc[0]["execution_id"])
            selected_runs = dataset_runs_df[dataset_runs_df["execution_id"] == selected_id]
        elif execution_id == "all":
            selected_runs = dataset_runs_df
            selected_id = "all"
        else:
            selected_runs = dataset_runs_df[dataset_runs_df["execution_id"] == execution_id]
            if selected_runs.empty:
                raise ValueError(f"Unknown execution_id '{execution_id}' for dataset '{dataset_name}'.")
            selected_id = execution_id

        exp_ids = {v for v in selected_runs["exp_id"].dropna().astype(str) if v}
        run_ids = {v for v in selected_runs["run_instance_id"].dropna().astype(str) if v}
        start = selected_runs["created_at"].min().to_pydatetime() - timedelta(minutes=time_buffer_minutes)
        end = selected_runs["updated_at"].max().to_pydatetime() + timedelta(minutes=time_buffer_minutes)

        traces = _iter_traces(self.client, start=start, end=end)
        traces = [
            t
            for t in traces
            if (not exp_ids or str(t["metadata"].get("exp_id")) in exp_ids)
            and (not run_ids or str(t["metadata"].get("run_instance_id")) in run_ids)
        ]
        if trace_limit is not None:
            traces = traces[:trace_limit]

        scores_df = _fetch_scores_df(self.client, start=start, end=end)
        trace_metrics_df = _fetch_trace_metrics_df(self.client, start=start, end=end)
        summary_df = _build_summary_df(traces, scores_df, trace_metrics_df, condition_key=condition_key)

        return AnalysisBundle(
            selected_execution=selected_id,
            summary_df=summary_df,
            dataset_runs_df=dataset_runs_df,
            trace_ids=[t["trace_id"] for t in traces],
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
        """Return one row per trace with full model output, scores, and judge comments.

        This fetches full trace details from Langfuse for each matched trace, which can
        be slow for large datasets. Use trace_limit to cap the number of fetches.
        """
        bundle = self.analyze_dataset(
            dataset_name=dataset_name,
            execution_id=execution_id,
            condition_key=condition_key,
            trace_limit=trace_limit,
            time_buffer_minutes=time_buffer_minutes,
        )
        if not bundle.trace_ids:
            return pd.DataFrame()

        rows = [_trace_detail_row(self.client.api.trace.get(trace_id)) for trace_id in bundle.trace_ids]
        master_df = pd.DataFrame(rows)
        if master_df.empty:
            return master_df

        # trace_detail.scores may be empty in some Langfuse SDK versions.
        # Fall back to the metrics API (same source used by analyze_dataset).
        score_cols_missing = not any(m in master_df.columns for m in PRIMARY_METRIC_ORDER)
        if score_cols_missing and not bundle.dataset_runs_df.empty:
            start = bundle.dataset_runs_df["created_at"].min().to_pydatetime() - timedelta(minutes=time_buffer_minutes)
            end = bundle.dataset_runs_df["updated_at"].max().to_pydatetime() + timedelta(minutes=time_buffer_minutes)
            scores_df = _fetch_scores_df(self.client, start=start, end=end)
            if not scores_df.empty:
                wide = scores_df.pivot_table(
                    index="trace_id", columns="metric_name", values="value", aggfunc="mean"
                ).reset_index()
                wide.columns.name = None
                master_df = master_df.merge(wide, on="trace_id", how="left")

        known_cols = {
            "trace_id",
            "timestamp",
            "task_id",
            "exp_id",
            "run_instance_id",
            "variant_id",
            "condition",
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
        metric_cols = [c for c in PRIMARY_METRIC_ORDER if c in master_df.columns] + sorted(
            c
            for c in master_df.columns
            if c not in known_cols and not c.endswith("_comment") and c not in PRIMARY_METRIC_ORDER
        )
        comment_cols = sorted(c for c in master_df.columns if c.endswith("_comment"))
        ordered_cols = [
            "timestamp",
            "trace_id",
            "task_id",
            "variant_id",
            "condition",
            "model",
            "condition_model",
            "condition_provider",
            "exp_id",
            "run_instance_id",
            *metric_cols,
            "judge_explanation",
            *comment_cols,
            "trace_input",
            "model_output",
            "trace_input_preview",
            "model_output_preview",
            "judge_explanation_preview",
            "langfuse_url",
            "metadata",
        ]
        seen: set[str] = set()
        final_cols = [c for c in ordered_cols if c in master_df.columns and not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]
        return master_df[final_cols].sort_values("timestamp", ascending=False).reset_index(drop=True)


__all__ = [
    "AnalysisBundle",
    "BOOLEAN_RATE_METRICS",
    "DEFAULT_GROUP_KEY_ORDER",
    "LocalDatasetConfig",
    "MisalignmentResultsExplorer",
    "discover_local_dataset_configs",
]

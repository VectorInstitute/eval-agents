import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langfuse import get_client
from rich.progress import track


load_dotenv()
langfuse = get_client()


REQUIRED_COLUMNS = {
    "test_prompt": ["Test prompt", "test prompt", "prompt", "question"],
    "expected_response": ["expected response", "expected_response", "expected_answer", "answer"],
    "safety_considerations": [
        "relevant safety considerations that should be addressed",
        "relevant_safety_considerations",
        "safety_considerations",
        "safety considerations",
    ],
    "expected_sources": ["expected sources (manual, page)", "expected_sources", "sources", "source"],
    "expected_trace": ["expected trace", "expected_trace", "trace"],
    "max_total_tokens": ["max total tokens", "max_total_tokens", "max_tokens"],
    "max_total_latency": ["max total latency", "max_total_latency", "max_latency"],
}

WORKORDER_COLUMNS = {
    "conversation": ["conversation", "text"],
    "source_question": ["source_question", "source question"],
    "reference_answer": ["reference_answer", "reference answer", "expected_response"],
    "safety_considerations": [
        "safety_considerations",
        "safety considerations",
    ],
    "quality_target": ["quality_target", "quality target"],
    "latency_target_seconds": [
        "latency_target_seconds",
        "latency target seconds",
    ],
    "cost_target_usd": ["cost_target_usd", "cost target usd"],
    "derived_from": ["derived_from", "derived from"],
    "row_index": ["row_index", "row index"],
    "id": ["id"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a benchmark dataset into Langfuse")
    parser.add_argument("--dataset-name", required=True, help="Name of the Langfuse dataset")
    parser.add_argument("--input-file", required=True, help="Path to a CSV, JSON, or JSONL file containing the test cases")
    parser.add_argument(
        "--description",
        default="Robot error troubleshooting Q&A dataset",
        help="Description for the Langfuse dataset",
    )
    return parser.parse_args()


def load_rows(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict) and "rows" in data:
            return pd.DataFrame(data["rows"])
        raise ValueError("JSON file must contain a list of objects or a {'rows': [...]} object")
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)
    raise ValueError("Unsupported file type. Use .csv, .json, or .jsonl")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for canonical, aliases in REQUIRED_COLUMNS.items():
        for alias in aliases:
            if alias in normalized.columns:
                normalized[canonical] = normalized[alias]
                break
    missing = [canonical for canonical in REQUIRED_COLUMNS if canonical not in normalized.columns]
    if missing:
        raise ValueError(
            "Missing required columns. Expected one of: "
            + ", ".join(f"{canonical} ({' | '.join(REQUIRED_COLUMNS[canonical])})" for canonical in REQUIRED_COLUMNS)
        )
    return normalized


def normalize_workorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for canonical, aliases in WORKORDER_COLUMNS.items():
        for alias in aliases:
            if alias in normalized.columns:
                normalized[canonical] = normalized[alias]
                break
    return normalized


def _sanitize_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _coerce_json_like(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _dataset_slug(dataset_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", dataset_name.lower()).strip("-")
    return slug or "dataset"


def _dataset_scoped_item_id(dataset_name: str, base_id: Any, idx: int) -> str:
    dataset_slug = _dataset_slug(dataset_name)
    cleaned_base_id = str(base_id).strip() if base_id is not None else ""
    if cleaned_base_id:
        return f"{dataset_slug}-{cleaned_base_id}"
    return f"{dataset_slug}-{idx:03}"


def build_expected_output(row: pd.Series) -> dict[str, Any]:
    return {
        "text": str(row["expected_response"]),
        "safety_considerations": _sanitize_value(row.get("safety_considerations")),
        "expected_sources": _sanitize_value(row.get("expected_sources")),
        "expected_trace": _sanitize_value(row.get("expected_trace")),
        "max_total_tokens": _sanitize_value(row.get("max_total_tokens")),
        "max_total_latency": _sanitize_value(row.get("max_total_latency")),
    }


def ensure_dataset(dataset_name: str, description: str) -> None:
    try:
        langfuse.get_dataset(dataset_name)
        print(f"Dataset '{dataset_name}' already exists; reusing it.")
    except Exception:
        langfuse.create_dataset(
            name=dataset_name,
            description=description,
            metadata={"type": "benchmark", "source": "manual_upload"},
        )
        print(f"Created dataset '{dataset_name}'.")


def is_preformatted_dataset(df: pd.DataFrame) -> bool:
    return "input" in df.columns and "expected_output" in df.columns


def is_workorder_csv(df: pd.DataFrame) -> bool:
    normalized = normalize_workorder_columns(df)
    return "conversation" in normalized.columns and "reference_answer" in normalized.columns


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = load_rows(input_path)
    preformatted = is_preformatted_dataset(df)
    workorder_csv = False

    if not preformatted:
        workorder_csv = is_workorder_csv(df)
        if workorder_csv:
            df = normalize_workorder_columns(df)
        else:
            df = normalize_columns(df)

    ensure_dataset(args.dataset_name, args.description)

    for idx, row in track(df.iterrows(), total=len(df), description="Uploading to Langfuse"):
        if preformatted:
            dataset_input = _coerce_json_like(row["input"])
            expected_output = _coerce_json_like(row["expected_output"])
            metadata = _sanitize_value(_coerce_json_like(row.get("metadata")))

            if not isinstance(dataset_input, dict):
                raise ValueError(
                    "Preformatted dataset rows must contain an object-valued 'input' field."
                )

            if not isinstance(metadata, dict):
                metadata = {}

            metadata = {
                "source_file": str(input_path),
                "row_index": int(idx),
                **metadata,
            }
            dataset_item_id = _dataset_scoped_item_id(
                args.dataset_name,
                _sanitize_value(row.get("id")),
                idx,
            )
        elif workorder_csv:
            conversation = str(row["conversation"])
            expected_output = {
                "source_question": str(row.get("source_question") or "").strip(),
                "reference_answer": str(row.get("reference_answer") or "").strip(),
                "safety_considerations": _sanitize_value(
                    row.get("safety_considerations")
                ),
                "quality_target": float(
                    _sanitize_value(row.get("quality_target"))
                    or 0.90
                ),
                "latency_target_seconds": float(
                    _sanitize_value(row.get("latency_target_seconds"))
                    or 10.0
                ),
                "cost_target_usd": float(
                    _sanitize_value(row.get("cost_target_usd"))
                    or 0.05
                ),
            }
            metadata = {
                "source_file": str(input_path),
                "row_index": int(_sanitize_value(row.get("row_index")) or idx),
                "dataset_type": "workorder_eval",
                "derived_from": _sanitize_value(row.get("derived_from")),
            }
            dataset_input = {
                "conversation": conversation,
                "text": conversation,
            }
            dataset_item_id = _dataset_scoped_item_id(
                args.dataset_name,
                _sanitize_value(row.get("id")),
                idx,
            )
        else:
            metadata = {
                "source_file": str(input_path),
                "row_index": int(idx),
                "safety_considerations": _sanitize_value(row.get("safety_considerations")),
                "expected_sources": _sanitize_value(row.get("expected_sources")),
                "expected_trace": _sanitize_value(row.get("expected_trace")),
                "max_total_tokens": _sanitize_value(row.get("max_total_tokens")),
                "max_total_latency": _sanitize_value(row.get("max_total_latency")),
            }
            dataset_input = {"text": str(row["test_prompt"])}
            expected_output = build_expected_output(row)
            dataset_item_id = _dataset_scoped_item_id(
                args.dataset_name,
                None,
                idx,
            )

        langfuse.create_dataset_item(
            dataset_name=args.dataset_name,
            input=dataset_input,
            expected_output=expected_output,
            metadata=metadata,
            id=str(dataset_item_id),
        )

    print(f"Uploaded {len(df)} rows to dataset '{args.dataset_name}'.")


if __name__ == "__main__":
    main()

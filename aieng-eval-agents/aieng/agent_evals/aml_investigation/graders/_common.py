"""Shared helpers for AML graders."""

from collections.abc import Mapping
from enum import Enum
from typing import Any

from aieng.agent_evals.aml_investigation.data import LaunderingPattern
from aieng.agent_evals.evaluation import ExperimentItemResult


PATTERN_LABELS: tuple[str, ...] = tuple(pattern.value for pattern in LaunderingPattern)


def get_field(payload: Any, key: str) -> Any:
    """Read ``key`` from dict-like or object payloads."""
    if isinstance(payload, Mapping):
        return payload.get(key)
    return getattr(payload, key, None)


def extract_expected_output(item_result: ExperimentItemResult) -> Any:
    """Extract expected_output from local-dict or dataset-item structures."""
    item = item_result.item
    if isinstance(item, Mapping):
        return item.get("expected_output")
    return getattr(item, "expected_output", None)


def normalize_pattern(value: Any) -> str | None:
    """Normalize pattern label to uppercase string form."""
    if isinstance(value, Enum):
        value = value.value
    if value is None:
        return None
    token = str(value).strip()
    return token.upper() if token else None


def normalize_transaction_ids(value: Any) -> set[str]:
    """Normalize transaction IDs into a comparable token set."""
    if value is None:
        return set()

    if isinstance(value, str):
        return {token.strip() for token in value.split(",") if token.strip()}

    if isinstance(value, list | tuple | set):
        normalized: set[str] = set()
        for item in value:
            if item is None:
                continue
            token = str(item).strip()
            if token:
                normalized.add(token)
        return normalized

    token = str(value).strip()
    return {token} if token else set()


__all__ = ["PATTERN_LABELS", "extract_expected_output", "get_field", "normalize_pattern", "normalize_transaction_ids"]

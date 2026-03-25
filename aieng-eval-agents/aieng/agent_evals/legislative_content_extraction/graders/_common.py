"""Shared helpers for legislative content extraction graders."""

from collections.abc import Mapping
from typing import Any

from aieng.agent_evals.evaluation import ExperimentItemResult


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


def normalize_str(value: Any) -> str | None:
    """Normalize a string field to uppercase stripped form."""
    if value is None:
        return None
    token = str(value).strip()
    return token.upper() if token else None


def normalize_sponsors(value: Any) -> set[str]:
    """Normalize a sponsors list into a comparable uppercase token set."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.strip().upper()} if value.strip() else set()
    if isinstance(value, list | tuple | set):
        return {str(s).strip().upper() for s in value if s is not None and str(s).strip()}
    return set()


def normalize_sections(value: Any) -> set[str]:
    """Normalize sections_affected into a set of 'RAW_SECTION|ACTION' strings."""
    if not value:
        return set()
    result = set()
    for section in value:
        if isinstance(section, Mapping):
            raw = str(section.get("raw_section", "")).strip().upper()
            action = str(section.get("action", "")).strip().upper()
        else:
            raw = str(getattr(section, "raw_section", "")).strip().upper()
            action = str(getattr(section, "action", "")).strip().upper()
        if raw:
            result.add(f"{raw}|{action}")
    return result


__all__ = [
    "extract_expected_output",
    "get_field",
    "normalize_str",
    "normalize_sponsors",
    "normalize_sections",
]
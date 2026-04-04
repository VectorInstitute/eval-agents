"""Shared helpers for legislative content extraction graders."""

import re
from collections.abc import Mapping
from typing import Any

from aieng.agent_evals.evaluation import ExperimentItemResult

# Title prefixes that appear before legislator names across jurisdictions.
# Stripping these allows "Representative Smith" and "Smith" to compare equal.
_TITLE_PREFIX_RE = re.compile(
    r"^(?:representative|rep\.|senator|sen\.|assembly(?:member|man|woman)?|delegate|"
    r"assemblyman|assemblywoman|speaker|majority leader|minority leader)[.\s]+",
    re.IGNORECASE,
)


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


def _normalize_sponsor_token(token: str) -> str:
    """Strip legislative title prefixes and normalise to uppercase."""
    token = token.strip()
    token = _TITLE_PREFIX_RE.sub("", token)
    return token.strip().upper()


def normalize_sponsors(value: Any) -> set[str]:
    """Normalize a sponsors list into a comparable uppercase token set.

    Title prefixes such as ``Representative``, ``Senator``, ``Rep.``, ``Sen.``
    are stripped before comparison so that ``"Representative Smith"`` and
    ``"SMITH"`` compare equal.
    """
    if value is None:
        return set()
    if isinstance(value, str):
        token = _normalize_sponsor_token(value)
        return {token} if token else set()
    if isinstance(value, list | tuple | set):
        result = set()
        for s in value:
            if s is None:
                continue
            token = _normalize_sponsor_token(str(s))
            if token:
                result.add(token)
        return result
    return set()


# Action codes that are synonymous and should compare equal.
_ACTION_EQUIVALENCES: dict[str, str] = {
    "CREATE": "ADD",
    "NEW": "ADD",
}


def _normalize_action(action: str) -> str:
    """Normalize a section action code, collapsing synonyms."""
    action = action.strip().upper()
    return _ACTION_EQUIVALENCES.get(action, action)


def normalize_sections(value: Any) -> set[str]:
    """Normalize sections_affected into a set of 'RAW_SECTION|ACTION' strings.

    Action codes ``CREATE``, ``NEW``, and ``ADD`` are treated as equivalent
    so that legitimate synonym differences do not cause mismatches.
    """
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
        action = _normalize_action(action)
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
"""Shared helpers for legislative content extraction graders."""

import re
from collections.abc import Mapping
from typing import Any

from aieng.agent_evals.evaluation import ExperimentItemResult

# Title prefixes that appear before legislator names across jurisdictions.
# Stripping these allows "Representative Smith" and "Smith" to compare equal.
_TITLE_PREFIX_RE = re.compile(
    r"^(?:representative|reps?\.|senator|sens?\.|assembly(?:member|man|woman)?|delegate|"
    r"assemblyman|assemblywoman|speaker|majority leader|minority leader)[.\s]+",
    re.IGNORECASE,
)

# Strips "Section " prefix and jurisdiction suffixes like:
# ", Wisconsin Statutes", ", Idaho Code", ", Minnesota Statutes 2024", etc.
# Also handles agent-side variants like "of the statutes", "of the Delaware Code",
# and constitutional references.
_RAW_SECTION_NORMALIZE_RE = re.compile(
    r"^section\s+"
    r"|,\s*(?:wisconsin statutes|idaho code|minnesota statutes[^|]*"
    r"|delaware code[^|]*|michigan compiled laws[^|]*)"
    r"\s*$"
    r"|\s+of\s+the\s+(?:statutes|delaware\s+code|idaho\s+code)"
    r"\s*$"
    r"|,?\s*(?:of\s+the\s+)?constitution\s+of\s+the\s+state\s+of\s+idaho"
    r"\s*$",
    re.IGNORECASE,
)

# Michigan uses "MCL 432.25 (Section 25)" in GT and "section 25 (MCL 432.25)" in
# agent output.  Canonicalize both forms to "MCL 432.25 | SECTION 25" so they
# compare equal.
_MCL_GT_RE = re.compile(
    r"^MCL\s+([\d.]+[A-Za-z]?)\s*\((?:Section\s+)?(\d+[A-Za-z]?)\)$",
    re.IGNORECASE,
)
_MCL_AGENT_RE = re.compile(
    r"^(?:Section\s+)?(\d+[A-Za-z]?)\s*\(MCL\s+([\d.]+[A-Za-z]?)\)$",
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
    "REDESIG": "AMEND",
    "REDESIGNATE": "AMEND",
}


def _normalize_action(action: str) -> str:
    """Normalize a section action code, collapsing synonyms."""
    action = action.strip().upper()
    return _ACTION_EQUIVALENCES.get(action, action)


# Delaware GT uses "Subchapter IV, Chapter 25, Title 6 - § 2541. Definitions."
# while agent outputs "Subchapter IV, Chapter 25, Title 6 of the Delaware Code".
# Strip everything after " - §" (the per-section description) from GT entries
# to enable chapter-level matching.
_DE_SECTION_DETAIL_RE = re.compile(
    r"\s*-\s*§\s*.*$",
    re.IGNORECASE,
)


def normalize_raw_section(raw: str) -> str:
    """Strip jurisdiction boilerplate from raw_section before comparison.

    Examples:
        "Section 101.123 (1) (ah), Wisconsin Statutes" -> "101.123 (1) (AH)"
        "101.123 (1) (ah)"                             -> "101.123 (1) (AH)"
        "Section 33-1802, Idaho Code"                  -> "33-1802"
        "MCL 15.236 (Section 6)"                       -> "MCL 15.236 | SECTION 6"
        "section 25 (MCL 432.25)"                      -> "MCL 432.25 | SECTION 25"
        "Subchapter IV, Chapter 25, Title 6 - § 2541. Definitions." -> "SUBCHAPTER IV, CHAPTER 25, TITLE 6"
        "16.03 (2) (a) of the statutes"                -> "16.03 (2) (A)"
    """
    if not raw:
        return ""
    raw = raw.strip()

    # Canonicalize Michigan MCL references before general stripping.
    m = _MCL_GT_RE.match(raw)
    if m:
        return f"MCL {m.group(1)} | SECTION {m.group(2)}".upper()
    m = _MCL_AGENT_RE.match(raw)
    if m:
        return f"MCL {m.group(2)} | SECTION {m.group(1)}".upper()

    # Strip Delaware per-section detail (" - § XXXX. Description.")
    raw = _DE_SECTION_DETAIL_RE.sub("", raw).strip()

    # Apply general regex repeatedly until stable (handles prefix + suffix)
    prev = None
    while prev != raw:
        prev = raw
        raw = _RAW_SECTION_NORMALIZE_RE.sub("", raw).strip()
    return raw.upper()


def normalize_sections(value: Any) -> set[str]:
    """Normalize sections_affected into a set of 'RAW_SECTION|ACTION' strings.

    - Strips jurisdiction boilerplate from raw_section
      e.g. "Section 101.123 (1) (ah), Wisconsin Statutes" -> "101.123 (1) (AH)"
    - Action codes CREATE, NEW, and ADD are treated as equivalent
    """
    if not value:
        return set()
    result = set()
    for section in value:
        if isinstance(section, Mapping):
            raw = str(section.get("raw_section", "")).strip()
            action = str(section.get("action", "")).strip().upper()
        else:
            raw = str(getattr(section, "raw_section", "")).strip()
            action = str(getattr(section, "action", "")).strip().upper()
        raw = normalize_raw_section(raw)
        action = _normalize_action(action)
        if raw:
            result.add(f"{raw}|{action}")
    return result


__all__ = [
    "extract_expected_output",
    "get_field",
    "normalize_raw_section",
    "normalize_str",
    "normalize_sponsors",
    "normalize_sections",
]
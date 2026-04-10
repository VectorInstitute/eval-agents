"""Shared helpers for legislative content extraction graders."""

import re
from collections.abc import Mapping
from typing import Any

from aieng.agent_evals.evaluation import ExperimentItemResult

# Title prefixes that appear before legislator names across jurisdictions.
# Stripping these allows "Representative Smith" and "Smith" to compare equal.
# The period after abbreviated titles (Rep., Sen.) is optional so that bare
# abbreviations like "Rep Smith" or "Sen Brown" (no period) are also stripped.
_TITLE_PREFIX_RE = re.compile(
    r"^(?:representative|reps?\.?|senator|sens?\.?|assembly(?:member|man|woman)?|delegate|"
    r"assemblyman|assemblywoman|speaker|majority leader|minority leader)[.\s]+",
    re.IGNORECASE,
)

# Strips "Section " prefix and jurisdiction suffixes like:
# ", Wisconsin Statutes", ", Idaho Code", ", Minnesota Statutes 2024", etc.
# Also handles:
#   - "of the X Statutes/Code" suffix for WI, MN, MI, DE, ID
#   - "Minnesota Statutes YYYY, section X" leading format (MN GT style)
#   - "Wis. Stats." suffix, Idaho constitution suffix
_RAW_SECTION_NORMALIZE_RE = re.compile(
    r"^section\s+"
    # Leading "Minnesota Statutes [YYYY], section " or "Minnesota Statutes [YYYY], " prefix
    # (Minnesota GT sometimes puts the jurisdiction name first)
    r"|^minnesota\s+statutes(?:\s+\d{4})?,\s*(?:section\s+)?"
    r"|,\s*(?:wisconsin\s+statutes(?:\s+\d{4})?|idaho\s+code|minnesota\s+statutes[^|]*"
    r"|delaware\s+code[^|]*|michigan\s+compiled\s+laws[^|]*)"
    r"\s*$"
    # "of the X Statutes/Code" suffix — handles both bare and year-qualified forms.
    # Minnesota and Michigan are included here alongside the previously covered states.
    r"|\s+of\s+the\s+(?:wisconsin\s+statutes(?:\s+\d{4})?|minnesota\s+statutes(?:\s+\d{4})?"
    r"|michigan\s+compiled\s+laws|statutes|delaware\s+code|idaho\s+code)"
    r"\s*$"
    # "Wis. Stats." / "Wis. Stat." suffix (with optional year)
    r"|\s*,?\s*wis\.?\s+stats?\."
    r"\s*(?:\d{4})?\s*$"
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

# Michigan ranged MCL reference: "MCL 408.931 to 408.945" or "MCL 408.931-408.945"
# Normalize to "MCL 408.931 TO 408.945" so both GT and agent match.
# Spaces are optional around the separator to handle the hyphen-only form.
_MCL_RANGE_RE = re.compile(
    r"^MCL\s+([\d.]+[A-Za-z]?)\s*(?:to|through|thru|-)\s*([\d.]+[A-Za-z]?)$",
    re.IGNORECASE,
)

# Michigan full act title with embedded MCL range:
# "The improved workforce opportunity wage act, 2018 PA 337, MCL 408.931 to 408.945"
# Extract the MCL portion so it can be compared after normalization.
_MCL_IN_ACT_TITLE_RE = re.compile(
    r"MCL\s+([\d.]+[A-Za-z]?(?:\s+(?:to|through|thru|-)\s+[\d.]+[A-Za-z]?)?)\s*$",
    re.IGNORECASE,
)

# Delaware GT uses "Subchapter IV, Chapter 25, Title 6 - § 2541. Definitions."
# while agent outputs "Subchapter IV, Chapter 25, Title 6 of the Delaware Code".
# Strip everything after " - §" (the per-section description) from GT entries
# to enable chapter-level matching.
_DE_SECTION_DETAIL_RE = re.compile(
    r"\s*-\s*§\s*.*$",
    re.IGNORECASE,
)

# Delaware: strip "SUBCHAPTER X, " prefix from normalized refs so that GT entries
# like "SUBCHAPTER IV, CHAPTER 25, TITLE 6" match agent output "CHAPTER 25, TITLE 6".
_DE_SUBCHAPTER_PREFIX_RE = re.compile(
    r"^SUBCHAPTER\s+[\w-]+,\s*",
)

# Delaware: strip "§ N, " section-level prefix when immediately followed by a
# CHAPTER/SUBCHAPTER/TITLE token, so that agent-side individual section refs like
# "§ 2541, CHAPTER 25, TITLE 6" reduce to "CHAPTER 25, TITLE 6" for matching.
_DE_SECTION_PREFIX_RE = re.compile(
    r"^§\s*[\w.]+,\s*(?=(?:SUBCHAPTER|CHAPTER|TITLE)\s)",
)

# "subd." / "subd" (Minnesota subdivision abbreviation) — normalize to
# the full word "subdivision" so GT and agent refs always compare equal.
_SUBD_NORM_RE = re.compile(
    r"\bsubd\b\.?",
    re.IGNORECASE,
)

# Delaware agent output uses individual section refs like "§ 2541, Title 6 of the Delaware Code"
# while GT uses chapter refs like "Subchapter IV, Chapter 25, Title 6 of the Delaware Code".
# Extract the Title number so we can do a coarse Title-level comparison.
_DE_TITLE_RE = re.compile(
    r"Title\s+(\d+)(?:\s+of\s+the\s+Delaware\s+Code)?",
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

    Title prefixes such as ``Representative``, ``Senator``, ``Rep.``,
    ``Rep``, ``Sen.``, ``Sen`` are stripped before comparison so that
    ``"Representative Smith"``, ``"Rep. Smith"``, and ``"Smith"`` all
    compare equal.
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
    "INSERT": "ADD",
    "REDESIG": "AMEND",
    "REDESIGNATE": "AMEND",
    "RENUMBER": "AMEND",
    "RENUMBER & AMEND": "AMEND",
    "RENUMBER AND AMEND": "AMEND",
    "RENUMBER, AMEND": "AMEND",
    "REPEAL & RECREATE": "REPEAL AND RECREATE",
    "CONSOLIDATE, RENUMBER AND AMEND": "AMEND",
    "CONSOLIDATE RENUMBER AND AMEND": "AMEND",
    "DEAUTHORIZE": "REPEAL",
}


def _normalize_action(action: str) -> str:
    """Normalize a section action code, collapsing synonyms."""
    action = action.strip().upper()
    return _ACTION_EQUIVALENCES.get(action, action)


def normalize_raw_section(raw: str) -> str:
    """Strip jurisdiction boilerplate from raw_section before comparison.

    Examples:
        "Section 101.123 (1) (ah), Wisconsin Statutes" -> "101.123 (1) (AH)"
        "101.123 (1) (ah)"                             -> "101.123 (1) (AH)"
        "16.03 (2) (a) of the statutes"                -> "16.03 (2) (A)"
        "16.03 (2) (a) of the Wisconsin Statutes"      -> "16.03 (2) (A)"
        "16.03 (2) (a) of the Wisconsin Statutes 2024" -> "16.03 (2) (A)"
        "16.03 (2) (a) of the Minnesota Statutes"      -> "16.03 (2) (A)"
        "16.03 (2) (a) of the Minnesota Statutes 2024" -> "16.03 (2) (A)"
        "Minnesota Statutes 2024, section 123B.09"     -> "123B.09"
        "Minnesota Statutes 2024, section 273.13, subdivision 34" -> "273.13, SUBDIVISION 34"
        "section 273.13, subd. 34, Minnesota Statutes" -> "273.13, SUBDIVISION 34"
        "Section 33-1802, Idaho Code"                  -> "33-1802"
        "MCL 15.236 (Section 6)"                       -> "MCL 15.236 | SECTION 6"
        "section 25 (MCL 432.25)"                      -> "MCL 432.25 | SECTION 25"
        "MCL 408.931 to 408.945"                       -> "MCL 408.931 TO 408.945"
        "MCL 408.931-408.945"                          -> "MCL 408.931 TO 408.945"
        "Subchapter IV, Chapter 25, Title 6 - § 2541. Definitions." -> "CHAPTER 25, TITLE 6"
        "§ 2541, Chapter 25, Title 6 of the Delaware Code" -> "CHAPTER 25, TITLE 6"
        "Subchapter VIII-A, Chapter 87A, Title 29 - § 8775A. Short title." -> "CHAPTER 87A, TITLE 29"
    """
    if not raw:
        return ""
    raw = raw.strip()

    # Canonicalize Michigan MCL simple references before general stripping.
    m = _MCL_GT_RE.match(raw)
    if m:
        return f"MCL {m.group(1)} | SECTION {m.group(2)}".upper()
    m = _MCL_AGENT_RE.match(raw)
    if m:
        return f"MCL {m.group(2)} | SECTION {m.group(1)}".upper()

    # Canonicalize Michigan ranged MCL references: "MCL 408.931 to 408.945"
    # or "MCL 408.931-408.945" (no spaces around hyphen).
    m = _MCL_RANGE_RE.match(raw)
    if m:
        return f"MCL {m.group(1)} TO {m.group(2)}".upper()

    # Full act title with embedded MCL range – extract just the MCL portion.
    # e.g. "The improved workforce opportunity wage act, 2018 PA 337, MCL 408.931 to 408.945"
    m = _MCL_IN_ACT_TITLE_RE.search(raw)
    if m and raw.upper().startswith("THE "):
        mcl_part = m.group(1).strip()
        # Recurse so the extracted MCL portion gets range-normalized if needed.
        return normalize_raw_section(f"MCL {mcl_part}")

    # Normalize "subd." / "subd" abbreviation to "subdivision" before stripping
    # jurisdiction suffixes, so that GT "subdivision 1" and agent "subd. 1" compare equal.
    raw = _SUBD_NORM_RE.sub("subdivision", raw)

    # Strip Delaware per-section detail (" - § XXXX. Description.")
    raw = _DE_SECTION_DETAIL_RE.sub("", raw).strip()

    # Apply general regex repeatedly until stable (handles prefix + suffix).
    prev = None
    while prev != raw:
        prev = raw
        raw = _RAW_SECTION_NORMALIZE_RE.sub("", raw).strip()

    result = raw.upper()

    # Delaware: strip "SUBCHAPTER X, " prefix so that GT entries like
    # "SUBCHAPTER IV, CHAPTER 25, TITLE 6" reduce to "CHAPTER 25, TITLE 6"
    # and can exact-match agent output that omits the subchapter level.
    result = _DE_SUBCHAPTER_PREFIX_RE.sub("", result)

    # Delaware: strip leading "§ N, " individual section prefix when followed
    # by CHAPTER/SUBCHAPTER/TITLE, so "§ 2541, CHAPTER 25, TITLE 6" reduces
    # to "CHAPTER 25, TITLE 6" for chapter-level comparison.
    result = _DE_SECTION_PREFIX_RE.sub("", result)

    return result


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
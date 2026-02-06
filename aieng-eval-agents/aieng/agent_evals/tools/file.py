"""File tools for downloading, searching, and reading local files.

Provides tools for:
- fetch_file: Download files (CSV, XLSX, text) from URLs
- grep_file: Search within downloaded files for patterns
- read_file: Read specific sections of downloaded files

These tools are designed for structured data files where grep/search
is more efficient than LLM processing. For HTML pages, use web_fetch instead.
"""

import hashlib
import logging
import os
import re
import tempfile
from functools import lru_cache
from typing import Any

import httpx
import pandas as pd
from google.adk.tools.function_tool import FunctionTool
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


logger = logging.getLogger(__name__)

# File extensions that need special handling (binary formats)
EXCEL_EXTENSIONS = {".xlsx", ".xls"}

# Safety limits for grep_file to prevent context overflow
MAX_GREP_RESULTS_HARD_CAP = 50  # Never return more than 50 matches
MAX_GREP_OUTPUT_CHARS = 100_000  # 100K character limit per grep output
MAX_CONTEXT_LINES_CAP = 5  # Maximum context lines allowed


@lru_cache(maxsize=1)
def get_cache_dir() -> str:
    """Get or create the cache directory for fetched content."""
    cache_dir = os.path.join(tempfile.gettempdir(), "agent_file_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _url_to_filename(url: str, extension: str = ".txt") -> str:
    """Convert URL to a safe filename."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    safe_name = re.sub(r"[^\w\-.]", "_", url.split("//")[-1][:50])
    return f"{safe_name}_{url_hash}{extension}"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
)
async def _fetch_with_retry(client: httpx.AsyncClient, url: str) -> httpx.Response:
    """Fetch URL with automatic retry on transient failures."""
    response = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"})
    response.raise_for_status()
    return response


def _is_excel_file(file_path: str) -> bool:
    """Check if file is an Excel file based on extension."""
    return any(file_path.lower().endswith(ext) for ext in EXCEL_EXTENSIONS)


def _read_excel_as_text(file_path: str) -> list[str]:
    """Read Excel file and convert to text lines.

    Reads all sheets and converts each row to a tab-separated line.
    Sheet names are included as headers.
    """
    lines: list[str] = []
    try:
        xlsx = pd.ExcelFile(file_path)
        for sheet_name in xlsx.sheet_names:
            df = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)
            lines.append(f"=== Sheet: {sheet_name} ===")
            for _, row in df.iterrows():
                # Convert row to tab-separated string, handling NaN values
                row_str = "\t".join(str(v) if pd.notna(v) else "" for v in row)
                lines.append(row_str)
            lines.append("")  # Empty line between sheets
    except Exception as e:
        logger.warning(f"Error reading Excel file {file_path}: {e}")
        raise
    return lines


def _read_csv_as_text(file_path: str) -> list[str]:
    """Read CSV file using pandas for better encoding/parsing handling."""
    lines: list[str] = []
    try:
        df = pd.read_csv(file_path, header=None, encoding="utf-8", on_bad_lines="skip")
        for _, row in df.iterrows():
            row_str = ",".join(str(v) if pd.notna(v) else "" for v in row)
            lines.append(row_str)
    except UnicodeDecodeError:
        # Try with latin-1 encoding as fallback
        df = pd.read_csv(file_path, header=None, encoding="latin-1", on_bad_lines="skip")
        for _, row in df.iterrows():
            row_str = ",".join(str(v) if pd.notna(v) else "" for v in row)
            lines.append(row_str)
    return lines


def _read_file_lines(file_path: str) -> list[str]:
    """Read file lines, handling text, CSV, and Excel formats."""
    if _is_excel_file(file_path):
        return _read_excel_as_text(file_path)

    # Try CSV if it's a CSV file
    if file_path.lower().endswith(".csv"):
        try:
            return _read_csv_as_text(file_path)
        except Exception as e:
            logger.warning(f"Pandas CSV read failed, falling back to text: {e}")

    # Default: read as text with encoding fallback
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.readlines()
    except UnicodeDecodeError:
        with open(file_path, encoding="latin-1") as f:
            return f.readlines()


def _detect_extension(content_type: str, url: str) -> str:
    """Detect file extension from content type or URL."""
    # Check content type first
    type_map = {
        "text/csv": ".csv",
        "application/vnd.ms-excel": ".xls",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/json": ".json",
        "text/plain": ".txt",
        "text/html": ".html",
    }
    for mime, ext in type_map.items():
        if mime in content_type:
            return ext

    # Fall back to URL extension
    url_lower = url.lower()
    for ext in [".csv", ".xlsx", ".xls", ".json", ".txt"]:
        if url_lower.endswith(ext):
            return ext

    return ".txt"


async def fetch_file(url: str) -> dict[str, Any]:
    """Download CSV, XLSX, or JSON data files. NOT for PDFs - use web_fetch for PDFs.

    This tool is ONLY for structured data files (CSV, XLSX, JSON, text) that need
    to be searched or read in sections. PDFs and HTML pages use web_fetch instead.

    After downloading, use grep_file to search for patterns, or read_file
    to read specific sections.

    Parameters
    ----------
    url : str
        The URL to download. Must be a valid HTTP or HTTPS URL.

    Returns
    -------
    dict
        On success: 'status', 'file_path', 'url', 'size_bytes',
        'content_type', 'preview'. On error: 'status', 'error', 'url'.

    Examples
    --------
    >>> result = await fetch_file("https://example.com/data.csv")
    >>> if result["status"] == "success":
    ...     grep_result = grep_file(result["file_path"], "revenue, income")
    """
    if not url.startswith(("http://", "https://")):
        return {
            "status": "error",
            "error": "Invalid URL. Must start with http:// or https://",
            "url": url,
        }

    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await _fetch_with_retry(client, url)
            content_type = response.headers.get("content-type", "")

            # Detect file extension
            extension = _detect_extension(content_type, url)

            # Handle PDF redirect
            if "application/pdf" in content_type:
                return {
                    "status": "error",
                    "error": "URL points to a PDF. Use web_fetch instead to read PDFs.",
                    "url": url,
                    "content_type": content_type,
                }

            # Save the file
            cache_dir = get_cache_dir()
            filename = _url_to_filename(url, extension)
            file_path = os.path.join(cache_dir, filename)

            # Write as text or binary depending on content type
            if "text" in content_type or extension in [".csv", ".json", ".txt"]:
                text_content = response.text
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text_content)
                size_bytes = len(text_content.encode("utf-8"))
                preview = text_content[:500] + "..." if len(text_content) > 500 else text_content
            else:
                binary_content = response.content
                with open(file_path, "wb") as f:
                    f.write(binary_content)
                size_bytes = len(binary_content)
                preview = f"[Binary file, {size_bytes} bytes]"

            return {
                "status": "success",
                "file_path": file_path,
                "url": str(response.url),
                "size_bytes": size_bytes,
                "content_type": content_type,
                "preview": preview,
                "next_step": f"Use grep_file('{file_path}', 'search terms') to search, or read_file('{file_path}') to read sections.",
            }

    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP error fetching {url}: {e}")
        return {
            "status": "error",
            "error": f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
            "url": url,
        }
    except httpx.RequestError as e:
        logger.warning(f"Request error fetching {url}: {e}")
        return {
            "status": "error",
            "error": f"Request failed: {e!s}",
            "url": url,
        }
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}")
        return {
            "status": "error",
            "error": f"Unexpected error: {e!s}",
            "url": url,
        }


def grep_file(
    file_path: str,
    pattern: str,
    context_lines: int = 5,
    max_results: int = 10,
) -> dict[str, Any]:
    """Search a LOCAL file for lines matching a pattern.

    A grep-style tool that searches text files for matching lines
    and returns matches with surrounding context. Use this after
    fetch_file to find relevant sections in large data files.

    Parameters
    ----------
    file_path : str
        LOCAL file path returned by fetch_file (NOT a URL).
    pattern : str
        Search pattern. Can be comma-separated for OR matching.
        Example: "operating expenses, total costs" matches either term.
    context_lines : int, optional
        Lines of context around each match (default 5).
    max_results : int, optional
        Maximum matching sections to return (default 10).

    Returns
    -------
    dict
        On success: 'status', 'matches', 'total_matches', 'patterns'.
        On error: 'status', 'error'.

    Examples
    --------
    >>> result = grep_file("/path/to/data.csv", "revenue, income, profit")
    >>> for match in result["matches"]:
    ...     print(f"Line {match['line_number']}: {match['context']}")
    """
    try:
        # Enforce hard caps to prevent context overflow
        max_results = min(max_results, MAX_GREP_RESULTS_HARD_CAP)
        context_lines = min(context_lines, MAX_CONTEXT_LINES_CAP)

        # Reject URLs - this tool only works with local file paths
        if file_path.startswith(("http://", "https://", "ftp://")):
            return {
                "status": "error",
                "error": "Cannot search URLs directly. Use web_fetch for URLs, or fetch_file to download data files first.",
            }

        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"File not found: {file_path}. Use fetch_file first to download.",
            }

        lines = _read_file_lines(file_path)

        patterns = [p.strip().lower() for p in pattern.split(",") if p.strip()]
        if not patterns:
            return {
                "status": "error",
                "error": "No valid pattern provided.",
            }

        matches: list[dict[str, Any]] = []
        used_ranges: set[int] = set()

        for line_num, line in enumerate(lines):
            line_lower = line.lower()

            matched_patterns = [p for p in patterns if p in line_lower]
            if not matched_patterns:
                continue

            if line_num in used_ranges:
                continue

            start = max(0, line_num - context_lines)
            end = min(len(lines), line_num + context_lines + 1)

            used_ranges.update(range(start, end))

            # Join lines with newlines (Excel/CSV lines don't have trailing newlines)
            context_text = "\n".join(line.rstrip("\n\r") for line in lines[start:end])
            matches.append(
                {
                    "line_number": line_num + 1,
                    "matched_patterns": matched_patterns,
                    "context": context_text,
                }
            )

            if len(matches) >= max_results:
                break

        if not matches:
            return {
                "status": "success",
                "matches": [],
                "total_matches": 0,
                "patterns": patterns,
                "message": f"No matches found for: {', '.join(patterns)}",
            }

        # Build result and check output size
        result = {
            "status": "success",
            "matches": matches,
            "total_matches": len(matches),
            "patterns": patterns,
        }

        # Truncate output if it exceeds character limit
        result_str = str(result)
        if len(result_str) > MAX_GREP_OUTPUT_CHARS:
            logger.warning(
                f"Grep output too large ({len(result_str):,} chars), truncating to {MAX_GREP_OUTPUT_CHARS:,} chars"
            )

            # Keep only matches that fit within the limit
            truncated_matches = []
            current_size = len(str({"status": "success", "patterns": patterns, "total_matches": 0, "matches": []}))

            for match in matches:
                match_size = len(str(match))
                if current_size + match_size > MAX_GREP_OUTPUT_CHARS * 0.9:  # 90% threshold for safety
                    break
                truncated_matches.append(match)
                current_size += match_size

            result = {
                "status": "success",
                "matches": truncated_matches,
                "total_matches": len(matches),
                "patterns": patterns,
                "truncated": True,
                "matches_returned": len(truncated_matches),
                "message": f"Output truncated: showing {len(truncated_matches)}/{len(matches)} matches (char limit: {MAX_GREP_OUTPUT_CHARS:,})",
            }

        return result

    except Exception as e:
        logger.error(f"Error in grep_file {file_path}: {e}")
        return {
            "status": "error",
            "error": f"Grep failed: {e!s}",
        }


def read_file(
    file_path: str,
    start_line: int = 1,
    num_lines: int = 100,
) -> dict[str, Any]:
    """Read a specific section of a LOCAL file.

    Use this to read portions of large documents that were downloaded with fetch_file.
    This tool works ONLY with local file paths, NOT URLs.

    Parameters
    ----------
    file_path : str
        LOCAL file path returned by fetch_file (NOT a URL).
    start_line : int, optional
        Line number to start from (1-indexed, default 1).
    num_lines : int, optional
        Number of lines to read (default 100).

    Returns
    -------
    dict
        On success: 'status', 'content', 'start_line', 'end_line', 'total_lines'.
        On error: 'status', 'error'.

    Examples
    --------
    >>> # After finding a match at line 42 with grep_file:
    >>> result = read_file("/path/to/data.csv", start_line=40, num_lines=20)
    >>> print(result["content"])
    """
    try:
        # Reject URLs - this tool only works with local file paths
        if file_path.startswith(("http://", "https://", "ftp://")):
            return {
                "status": "error",
                "error": "Cannot read URLs directly. Use web_fetch for URLs, or fetch_file to download data files first.",
            }

        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"File not found: {file_path}. Use fetch_file first to download.",
            }

        lines = _read_file_lines(file_path)

        total_lines = len(lines)

        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, start_idx + num_lines)

        # Join lines with newlines (Excel/CSV lines don't have trailing newlines)
        content = "\n".join(line.rstrip("\n\r") for line in lines[start_idx:end_idx])

        return {
            "status": "success",
            "content": content,
            "start_line": start_idx + 1,
            "end_line": end_idx,
            "total_lines": total_lines,
        }

    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return {
            "status": "error",
            "error": f"Read failed: {e!s}",
        }


def create_fetch_file_tool() -> FunctionTool:
    """Create an ADK FunctionTool for downloading files."""
    return FunctionTool(func=fetch_file)


def create_grep_file_tool() -> FunctionTool:
    """Create an ADK FunctionTool for grep-style file searching."""
    return FunctionTool(func=grep_file)


def create_read_file_tool() -> FunctionTool:
    """Create an ADK FunctionTool for reading file sections."""
    return FunctionTool(func=read_file)

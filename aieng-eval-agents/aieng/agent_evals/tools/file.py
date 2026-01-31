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
from google.adk.tools.function_tool import FunctionTool
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


logger = logging.getLogger(__name__)


_http_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
    reraise=True,
)


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


@_http_retry
def _fetch_with_retry(client: httpx.Client, url: str) -> httpx.Response:
    """Fetch URL with automatic retry on transient failures."""
    response = client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"})
    response.raise_for_status()
    return response


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


def fetch_file(url: str) -> dict[str, Any]:
    """Download a file from a URL and save it locally.

    Use this tool to download data files (CSV, XLSX, JSON, text) that need
    to be searched or read in sections. For HTML pages where you want to
    extract specific information, use web_fetch instead.

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
    >>> result = fetch_file("https://example.com/data.csv")
    >>> if result["status"] == "success":
    ...     grep_result = grep_file(result["file_path"], "revenue, income")
    """
    logger.info(f"Fetching file: {url}")

    if not url.startswith(("http://", "https://")):
        return {
            "status": "error",
            "error": "Invalid URL. Must start with http:// or https://",
            "url": url,
        }

    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = _fetch_with_retry(client, url)
            content_type = response.headers.get("content-type", "")

            # Detect file extension
            extension = _detect_extension(content_type, url)

            # Handle PDF redirect
            if "application/pdf" in content_type:
                return {
                    "status": "error",
                    "error": "URL points to a PDF. Use the read_pdf tool instead.",
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
        logger.exception(f"Unexpected error fetching {url}")
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
    """Search a file for lines matching a pattern.

    A grep-style tool that searches text files for matching lines
    and returns matches with surrounding context. Use this after
    fetch_file to find relevant sections in large data files.

    Parameters
    ----------
    file_path : str
        Path to the file to search (from fetch_file result).
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
    logger.info(f"Grep {file_path} for: {pattern}")

    try:
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"File not found: {file_path}. Use fetch_file first to download.",
            }

        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

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

            context_text = "".join(lines[start:end]).strip()
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

        return {
            "status": "success",
            "matches": matches,
            "total_matches": len(matches),
            "patterns": patterns,
        }

    except Exception as e:
        logger.exception(f"Error in grep_file {file_path}")
        return {
            "status": "error",
            "error": f"Grep failed: {e!s}",
        }


def read_file(
    file_path: str,
    start_line: int = 1,
    num_lines: int = 100,
) -> dict[str, Any]:
    """Read a specific section of a file.

    Use this to read portions of large documents, especially after
    using grep_file to identify relevant sections.

    Parameters
    ----------
    file_path : str
        Path to the file to read (from fetch_file result).
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
    logger.info(f"Reading {file_path} from line {start_line}")

    try:
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"File not found: {file_path}. Use fetch_file first to download.",
            }

        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)

        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, start_idx + num_lines)

        content = "".join(lines[start_idx:end_idx])

        return {
            "status": "success",
            "content": content,
            "start_line": start_idx + 1,
            "end_line": end_idx,
            "total_lines": total_lines,
        }

    except Exception as e:
        logger.exception(f"Error reading {file_path}")
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

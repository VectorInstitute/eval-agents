"""Web tools for URL fetching and PDF reading.

Simple function-based tools for ADK agents to fetch web content
and read PDF documents. Designed for efficient context management -
fetch once, search multiple times.
"""

import hashlib
import logging
import os
import re
import tempfile
from functools import lru_cache
from io import BytesIO
from typing import Any

import httpx
from google.adk.tools.function_tool import FunctionTool
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


logger = logging.getLogger(__name__)


# Retry decorator for transient network errors
_http_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
    reraise=True,
)


@lru_cache(maxsize=1)
def _get_cache_dir() -> str:
    """Get or create the cache directory for fetched content."""
    cache_dir = os.path.join(tempfile.gettempdir(), "agent_web_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _url_to_filename(url: str) -> str:
    """Convert URL to a safe filename."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    # Extract domain and path for readability
    safe_name = re.sub(r"[^\w\-.]", "_", url.split("//")[-1][:50])
    return f"{safe_name}_{url_hash}.txt"


@_http_retry
def _fetch_with_retry(client: httpx.Client, url: str) -> httpx.Response:
    """Fetch URL with automatic retry on transient failures."""
    response = client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"})
    response.raise_for_status()
    return response


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text, preserving some structure.

    Simple HTML to text conversion without external dependencies.
    """
    # Remove script and style elements
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Convert common block elements to newlines
    text = re.sub(r"<(br|p|div|h[1-6]|li|tr)[^>]*>", "\n", text, flags=re.IGNORECASE)

    # Remove all remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")

    # Normalize whitespace
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r" +", " ", text)

    return text.strip()


def fetch_url(url: str) -> dict[str, Any]:
    """Fetch content from a URL and save it locally for searching.

    This tool fetches a webpage, extracts text, and saves it to a local file.
    Use the returned 'file_path' with grep_file or read_file to find specific
    information without loading the entire content into context.

    Parameters
    ----------
    url : str
        The URL to fetch. Must be a valid HTTP or HTTPS URL.

    Returns
    -------
    dict
        On success: 'status', 'file_path' (IMPORTANT: use this exact path with
        grep_file/read_file), 'url', 'length', 'preview'.
        On error: 'status', 'error', 'url'.
    """
    logger.info(f"Fetching URL: {url}")

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = _fetch_with_retry(client, url)
            content_type = response.headers.get("content-type", "")

            # Handle different content types
            if "application/pdf" in content_type:
                return {
                    "status": "error",
                    "error": "URL points to a PDF. Use the read_pdf tool instead.",
                    "url": url,
                    "content_type": content_type,
                }

            # For HTML, extract text; for plain text or other types, use directly
            text = _html_to_text(response.text) if "text/html" in content_type or not content_type else response.text

            # Save to local file
            cache_dir = _get_cache_dir()
            filename = _url_to_filename(url)
            file_path = os.path.join(cache_dir, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

            # Return metadata with preview (not full content)
            preview = text[:500] + "..." if len(text) > 500 else text

            return {
                "status": "success",
                "file_path": file_path,
                "url": str(response.url),
                "length": len(text),
                "preview": preview,
                "next_step": f"Use grep_file(file_path='{file_path}', pattern='your search terms') to search this file.",
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
            "error": f"Request failed: {str(e)}",
            "url": url,
        }
    except Exception as e:
        logger.exception(f"Unexpected error fetching {url}")
        return {
            "status": "error",
            "error": f"Unexpected error: {str(e)}",
            "url": url,
        }


def grep_file(
    file_path: str,
    pattern: str,
    context_lines: int = 5,
    max_results: int = 10,
) -> dict[str, Any]:
    """Search a file for lines matching a pattern.

    A general-purpose grep tool that searches any text file for matching lines
    and returns the matches with surrounding context. Works with any file path.

    Parameters
    ----------
    file_path : str
        Path to the file to search.
    pattern : str
        Search pattern. Can be comma-separated terms for OR matching.
        Example: "operating expenses, total costs" matches lines with either term.
    context_lines : int, optional
        Number of lines of context around each match (default 5).
    max_results : int, optional
        Maximum number of matching sections to return (default 10).

    Returns
    -------
    dict
        Contains 'matches' (list of matching sections with line numbers and context),
        'total_matches', and 'patterns'. On error, contains 'error' message.
    """
    logger.info(f"Grep {file_path} for: {pattern}")

    try:
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"File not found: {file_path}",
            }

        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Parse patterns (comma-separated for OR matching)
        patterns = [p.strip().lower() for p in pattern.split(",") if p.strip()]
        if not patterns:
            return {
                "status": "error",
                "error": "No valid pattern provided.",
            }

        # Find matching lines
        matches: list[dict[str, Any]] = []
        used_ranges: set[int] = set()

        for line_num, line in enumerate(lines):
            line_lower = line.lower()

            # Check if any pattern matches
            matched_patterns = [p for p in patterns if p in line_lower]
            if not matched_patterns:
                continue

            # Skip if this line is already covered by a previous match
            if line_num in used_ranges:
                continue

            # Extract context around match
            start = max(0, line_num - context_lines)
            end = min(len(lines), line_num + context_lines + 1)

            # Mark these lines as used
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
            "error": f"Grep failed: {str(e)}",
        }


def read_file(
    file_path: str,
    start_line: int = 1,
    num_lines: int = 100,
) -> dict[str, Any]:
    """Read a specific section of a fetched file.

    Use this tool to read a specific portion of a large document,
    useful after using search_fetched_content to identify relevant sections.

    Parameters
    ----------
    file_path : str
        Path to the fetched content file (from fetch_url result).
    start_line : int, optional
        Line number to start reading from (1-indexed, default 1).
    num_lines : int, optional
        Number of lines to read (default 100).

    Returns
    -------
    dict
        Contains 'content' (the text section), 'start_line', 'end_line',
        and 'total_lines'. On error, contains 'error' message.
    """
    logger.info(f"Reading {file_path} from line {start_line}")

    try:
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"File not found: {file_path}. Use fetch_url first.",
            }

        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)

        # Convert to 0-indexed
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
            "error": f"Read failed: {str(e)}",
        }


def read_pdf(url: str, max_pages: int = 20) -> dict[str, Any]:  # noqa: PLR0911
    """Read and extract text from a PDF document at a URL.

    Use this tool to read PDF documents, such as SEC filings, research
    papers, or reports. Returns the extracted text content.

    Parameters
    ----------
    url : str
        The URL of the PDF document to read.
    max_pages : int, optional
        Maximum number of pages to read (default 20).

    Returns
    -------
    dict
        Contains 'content' (extracted text), 'url', 'num_pages', and 'pages_read'.
        On error, contains 'error' message instead of 'content'.
    """
    logger.info(f"Reading PDF from: {url}")

    try:
        # Import pypdf here to make it an optional dependency
        try:
            from pypdf import PdfReader  # noqa: PLC0415
        except ImportError:
            return {
                "status": "error",
                "error": "pypdf library not installed. Install with: pip install pypdf",
                "url": url,
            }

        # Fetch the PDF
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"})
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text/html" in content_type:
                return {
                    "status": "error",
                    "error": "URL returned HTML, not a PDF. Use fetch_url tool instead.",
                    "url": url,
                }

        # Read the PDF
        pdf_file = BytesIO(response.content)
        reader = PdfReader(pdf_file)

        num_pages = len(reader.pages)
        pages_to_read = min(num_pages, max_pages)

        # Extract text from pages
        text_parts = []
        for i in range(pages_to_read):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text_parts.append(f"--- Page {i + 1} ---\n{page_text}")

        if not text_parts:
            return {
                "status": "error",
                "error": "Could not extract text from PDF. It may be scanned/image-based.",
                "url": url,
                "num_pages": num_pages,
            }

        content = "\n\n".join(text_parts)

        return {
            "status": "success",
            "content": content,
            "url": url,
            "num_pages": num_pages,
            "pages_read": pages_to_read,
        }

    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP error fetching PDF {url}: {e}")
        return {
            "status": "error",
            "error": f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
            "url": url,
        }
    except httpx.RequestError as e:
        logger.warning(f"Request error fetching PDF {url}: {e}")
        return {
            "status": "error",
            "error": f"Request failed: {str(e)}",
            "url": url,
        }
    except Exception as e:
        logger.exception(f"Unexpected error reading PDF {url}")
        return {
            "status": "error",
            "error": f"Error reading PDF: {str(e)}",
            "url": url,
        }


def create_fetch_url_tool() -> FunctionTool:
    """Create an ADK FunctionTool for fetching URL content.

    Returns
    -------
    FunctionTool
        Tool that fetches web pages and saves them locally.
    """
    return FunctionTool(func=fetch_url)


def create_grep_file_tool() -> FunctionTool:
    """Create an ADK FunctionTool for grep-style file searching.

    Returns
    -------
    FunctionTool
        Tool that searches files for matching patterns.
    """
    return FunctionTool(func=grep_file)


def create_read_file_tool() -> FunctionTool:
    """Create an ADK FunctionTool for reading file sections.

    Returns
    -------
    FunctionTool
        Tool that reads specific sections of saved content.
    """
    return FunctionTool(func=read_file)


def create_read_pdf_tool() -> FunctionTool:
    """Create an ADK FunctionTool for reading PDF documents.

    Returns
    -------
    FunctionTool
        Tool that reads and extracts text from PDF files.
    """
    return FunctionTool(func=read_pdf)

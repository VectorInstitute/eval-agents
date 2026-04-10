"""Document extraction tools for reading PDFs and fetching HTML pages.

Provides tools for the legislative content extraction agent to read
local PDF files and fetch remote HTML pages.
"""

import logging
import os
import urllib.request
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import html_to_markdown
from google.adk.tools.function_tool import FunctionTool
from pypdf import PdfReader


logger = logging.getLogger(__name__)

# Default maximum pages to extract from a PDF
MAX_PDF_PAGES = 50

# Maximum HTML content length to return (characters)
MAX_HTML_CONTENT_LENGTH = 100_000


def read_pdf(file_path: str, max_pages: int = MAX_PDF_PAGES) -> dict[str, Any]:
    """Read a local PDF file and extract its text content.

    Extracts text from each page of the PDF, returning the full text
    with page markers for easy reference.

    Parameters
    ----------
    file_path : str
        Absolute path to the local PDF file.
    max_pages : int, optional
        Maximum number of pages to extract (default 50).

    Returns
    -------
    dict
        On success: 'status', 'content', 'num_pages', 'pages_extracted'.
        On error: 'status', 'error'.

    Examples
    --------
    >>> result = read_pdf("/path/to/document.pdf")
    >>> print(result["content"])
    """
    if not file_path:
        return {"status": "error", "error": "file_path is required."}

    if file_path.startswith(("http://", "https://", "ftp://")):
        return {
            "status": "error",
            "error": "read_pdf only works with local file paths, not URLs.",
        }

    if not os.path.exists(file_path):
        return {
            "status": "error",
            "error": f"File not found: {file_path}",
        }

    if not file_path.lower().endswith(".pdf"):
        return {
            "status": "error",
            "error": f"Not a PDF file: {file_path}",
        }

    try:
        reader = PdfReader(file_path)
        num_pages = len(reader.pages)
        pages_to_read = min(num_pages, max_pages)

        text_parts = []
        for i in range(pages_to_read):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text_parts.append(f"--- Page {i + 1} ---\n{page_text}")

        if pages_to_read < num_pages:
            text_parts.append(
                f"\n[Document has {num_pages} pages. Showing first {pages_to_read}.]"
            )

        content = "\n\n".join(text_parts)

        return {
            "status": "success",
            "content": content,
            "num_pages": num_pages,
            "pages_extracted": pages_to_read,
        }

    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        return {
            "status": "error",
            "error": f"Failed to read PDF: {e!s}",
        }


def _convert_html_to_markdown(raw_html: str) -> str:
    """Convert raw HTML to markdown, falling back to raw HTML on failure."""
    try:
        result = html_to_markdown.convert(raw_html)
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return result.get("content") or result.get("markdown") or str(result)
        return str(result)
    except Exception:
        logger.warning("html_to_markdown conversion failed, using raw HTML")
        return raw_html


def fetch_html_page(url: str, cache_dir: str | None = None) -> dict[str, Any]:
    """Fetch an HTML page and return its content.

    If *cache_dir* is provided the fetched HTML is saved there and subsequent
    calls with the same URL will read from the cache instead of fetching again.

    Parameters
    ----------
    url : str
        The URL of the HTML page to fetch.
    cache_dir : str, optional
        Directory to cache fetched HTML files.

    Returns
    -------
    dict
        On success: 'status', 'content', 'url'.
        On error: 'status', 'error'.

    Examples
    --------
    >>> result = fetch_html_page("https://legis.delaware.gov/BillDetail/142907")
    >>> print(result["content"][:200])
    """
    if not url:
        return {"status": "error", "error": "url is required."}

    if not url.startswith(("http://", "https://")):
        return {
            "status": "error",
            "error": "url must start with http:// or https://.",
        }

    # Check cache – cache_dir should point to the bill-specific subdirectory
    # (e.g. files/ID_H0504/) which contains a single .html file.
    cache_root: Path | None = None
    if cache_dir:
        cache_root = Path(cache_dir)
        html_files = list(cache_root.glob("*.html"))
        if html_files:
            html_file = html_files[0]
            logger.info(f"Reading cached HTML for {url} from {html_file}")
            raw = html_file.read_text(encoding="utf-8", errors="replace")
            content = _convert_html_to_markdown(raw)
            return {
                "status": "success",
                "content": content,
                "url": url,
            }

    try:
        logger.info(f"Not found in cache. Fetching HTML page {url}")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (legislative-content-extraction-agent)"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310
            raw_html = response.read().decode("utf-8", errors="replace")

        content = _convert_html_to_markdown(raw_html)

        # Save to cache
        if cache_root:
            # Use the last URL path segment as the cache filename
            url_slug = url.rstrip("/").rsplit("/", 1)[-1]
            save_path = cache_root / url_slug / f"{url_slug}.html"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(content, encoding="utf-8")
            logger.info(f"Cached HTML for {url} at {save_path}")

        if len(content) > MAX_HTML_CONTENT_LENGTH:
            content = content[:MAX_HTML_CONTENT_LENGTH] + "\n[Content truncated]"

        return {
            "status": "success",
            "content": content,
            "url": url,
        }

    except HTTPError as e:
        logger.error(f"HTTP error fetching {url}: {e.code} {e.reason}")
        return {
            "status": "error",
            "error": f"HTTP error {e.code}: {e.reason}",
        }
    except URLError as e:
        logger.error(f"URL error fetching {url}: {e.reason}")
        return {
            "status": "error",
            "error": f"Failed to fetch URL: {e.reason}",
        }
    except Exception as e:
        logger.error(f"Error fetching HTML page {url}: {e}")
        return {
            "status": "error",
            "error": f"Failed to fetch HTML page: {e!s}",
        }

def create_read_pdf_tool() -> FunctionTool:
    """Create an ADK FunctionTool for reading local PDF files."""
    return FunctionTool(func=read_pdf)

def create_fetch_html_page_tool(cache_dir_holder: list[str | None] | None = None) -> FunctionTool:
    """Create an ADK FunctionTool for fetching HTML pages.

    Parameters
    ----------
    cache_dir_holder : list, optional
        A single-element list holding the current cache directory path.
        Using a mutable list allows the caller to update the cache directory
        per-extraction (e.g. to point at the bill-specific subdirectory).
    """
    if cache_dir_holder is None:
        return FunctionTool(func=fetch_html_page)

    def _cached_fetch(url: str) -> dict[str, Any]:
        return fetch_html_page(url, cache_dir=cache_dir_holder[0])

    _cached_fetch.__name__ = "fetch_html_page"
    _cached_fetch.__doc__ = fetch_html_page.__doc__
    return FunctionTool(func=_cached_fetch)

"""PDF tools for reading and extracting text from PDF documents.

Provides tools for reading PDF documents from URLs with pagination
support for handling large documents efficiently.
"""

import logging
from io import BytesIO
from typing import Any

import httpx
from google.adk.tools.function_tool import FunctionTool


logger = logging.getLogger(__name__)


def _pdf_error(url: str, error: str, **extra: Any) -> dict[str, Any]:
    """Create a standardized PDF error response."""
    return {"status": "error", "error": error, "url": url, **extra}


def _read_pdf_content(
    url: str,
    start_page: int,
    max_pages: int,
    max_chars: int,
) -> dict[str, Any]:
    """Extract text content from a PDF at the given URL."""
    from pypdf import PdfReader  # noqa: PLC0415

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"})
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            return _pdf_error(url, "URL returned HTML, not a PDF. Use fetch_url instead.")

    pdf_file = BytesIO(response.content)
    reader = PdfReader(pdf_file)
    num_pages = len(reader.pages)

    start_idx = max(0, start_page - 1)
    if start_idx >= num_pages:
        return _pdf_error(
            url,
            f"start_page {start_page} exceeds PDF length ({num_pages} pages).",
            num_pages=num_pages,
        )

    end_idx = min(num_pages, start_idx + max_pages)
    pages_to_read = end_idx - start_idx

    text_parts = []
    for i in range(start_idx, end_idx):
        page_text = reader.pages[i].extract_text()
        if page_text:
            text_parts.append(f"--- Page {i + 1} ---\n{page_text}")

    if not text_parts:
        return _pdf_error(
            url,
            "Could not extract text from PDF. It may be scanned/image-based.",
            num_pages=num_pages,
        )

    content = "\n\n".join(text_parts)
    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars] + "\n\n[TRUNCATED - use start_page to read more]"

    hint = f"PDF has {num_pages} pages. Read pages {start_page}-{start_idx + pages_to_read}."
    if end_idx < num_pages:
        hint += f" Use start_page={end_idx + 1} to continue."

    return {
        "status": "success",
        "content": content,
        "url": url,
        "num_pages": num_pages,
        "start_page": start_page,
        "pages_read": pages_to_read,
        "truncated": truncated,
        "hint": hint,
    }


def read_pdf(
    url: str,
    start_page: int = 1,
    max_pages: int = 5,
    max_chars: int = 30000,
) -> dict[str, Any]:
    """Read and extract text from a PDF document at a URL.

    Use this for PDF documents like SEC filings, research papers, or reports.
    Reads a limited number of pages by default to avoid context overload.

    Parameters
    ----------
    url : str
        The URL of the PDF document.
    start_page : int, optional
        Page number to start from (1-indexed, default 1).
    max_pages : int, optional
        Maximum pages to read from start_page (default 5).
    max_chars : int, optional
        Maximum characters to return (default 30000).

    Returns
    -------
    dict
        Contains 'content', 'url', 'num_pages', 'pages_read', 'start_page',
        'truncated'. On error, contains 'error' instead of 'content'.
    """
    logger.info(f"Reading PDF from: {url} (pages {start_page} to {start_page + max_pages - 1})")

    try:
        return _read_pdf_content(url, start_page, max_pages, max_chars)
    except ImportError:
        return _pdf_error(url, "pypdf not installed. Install with: pip install pypdf")
    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP error fetching PDF {url}: {e}")
        return _pdf_error(url, f"HTTP {e.response.status_code}: {e.response.reason_phrase}")
    except httpx.RequestError as e:
        logger.warning(f"Request error fetching PDF {url}: {e}")
        return _pdf_error(url, f"Request failed: {e!s}")
    except Exception as e:
        logger.exception(f"Unexpected error reading PDF {url}")
        return _pdf_error(url, f"Error reading PDF: {e!s}")


def create_read_pdf_tool() -> FunctionTool:
    """Create an ADK FunctionTool for reading PDF documents."""
    return FunctionTool(func=read_pdf)

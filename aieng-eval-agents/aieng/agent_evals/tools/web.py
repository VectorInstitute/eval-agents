"""Web fetch tool for retrieving content from URLs.

Provides the web_fetch tool which fetches content from any URL (HTML pages or PDFs)
and returns the content for the agent to analyze. Similar to Anthropic's web_fetch tool.
"""

import logging
from io import BytesIO
from typing import Any
from urllib.parse import urljoin

import httpx
from google.adk.tools.function_tool import FunctionTool
from html_to_markdown import convert as html_to_markdown
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


logger = logging.getLogger(__name__)


# Maximum content size to return (100KB for text, to avoid context overflow)
MAX_CONTENT_CHARS = 100_000


_http_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
    reraise=True,
)


@_http_retry
def _fetch_with_retry(client: httpx.Client, url: str) -> httpx.Response:
    """Fetch URL with automatic retry on transient failures."""
    response = client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"})
    response.raise_for_status()
    return response


def _html_to_markdown(html: str, base_url: str | None = None) -> str:
    """Convert HTML to Markdown, preserving links, tables, and structure.

    Parameters
    ----------
    html : str
        The HTML content to convert.
    base_url : str, optional
        Base URL for resolving relative links.

    Returns
    -------
    str
        Markdown-formatted text with preserved links and tables.
    """
    # Use html-to-markdown library for high-quality conversion
    # It preserves links, tables, headings, lists, and other structure
    markdown = html_to_markdown(html)

    # If base_url provided, convert relative URLs to absolute
    if base_url:
        import re  # noqa: PLC0415

        def make_absolute(match: re.Match) -> str:
            """Convert relative URL to absolute."""
            prefix = match.group(1)  # [text]( or src="
            url = match.group(2)
            suffix = match.group(3)  # ) or "

            # Skip if already absolute or is a data URI
            if url.startswith(("http://", "https://", "data:", "mailto:", "#")):
                return match.group(0)

            absolute_url = urljoin(base_url, url)
            return f"{prefix}{absolute_url}{suffix}"

        # Fix markdown links: [text](url)
        markdown = re.sub(r"(\[[^\]]*\]\()([^)]+)(\))", make_absolute, markdown)

        # Fix markdown images: ![alt](url)
        markdown = re.sub(r"(!\[[^\]]*\]\()([^)]+)(\))", make_absolute, markdown)

    return markdown.strip()


def _extract_pdf_text(content: bytes, max_pages: int = 10) -> tuple[str, int]:
    """Extract text from PDF bytes.

    Parameters
    ----------
    content : bytes
        The PDF file content.
    max_pages : int
        Maximum number of pages to extract.

    Returns
    -------
    tuple[str, int]
        The extracted text and total number of pages.
    """
    from pypdf import PdfReader  # noqa: PLC0415

    pdf_file = BytesIO(content)
    reader = PdfReader(pdf_file)
    num_pages = len(reader.pages)

    pages_to_read = min(num_pages, max_pages)
    text_parts = []

    for i in range(pages_to_read):
        page_text = reader.pages[i].extract_text()
        if page_text:
            text_parts.append(f"--- Page {i + 1} ---\n{page_text}")

    if pages_to_read < num_pages:
        text_parts.append(f"\n[Document has {num_pages} pages. Showing first {pages_to_read}.]")

    return "\n\n".join(text_parts), num_pages


def _truncate_content(text: str) -> tuple[str, bool]:
    """Truncate content if it exceeds the maximum length."""
    truncated = len(text) > MAX_CONTENT_CHARS
    if truncated:
        text = text[:MAX_CONTENT_CHARS] + "\n\n[Content truncated due to length]"
    return text, truncated


def _make_error_response(error: str, url: str) -> dict[str, Any]:
    """Create an error response dict."""
    return {"status": "error", "error": error, "url": url}


def _make_success_response(url: str, content: str, content_type: str, truncated: bool, **extra: Any) -> dict[str, Any]:
    """Create a success response dict."""
    result = {
        "status": "success",
        "url": url,
        "content": content,
        "content_type": content_type,
        "content_length": len(content),
        "truncated": truncated,
    }
    result.update(extra)
    return result


def web_fetch(url: str, max_pages: int = 10) -> dict[str, Any]:
    """Fetch content from a URL (HTML page or PDF document).

    This tool retrieves the full content from a URL for analysis. It handles
    both HTML pages (converted to readable text) and PDF documents (text extracted).

    For large data files (CSV, XLSX) that need searching, use fetch_file instead.

    Parameters
    ----------
    url : str
        The URL to fetch. Must be a valid HTTP or HTTPS URL.
    max_pages : int, optional
        For PDFs, maximum number of pages to extract (default 10).

    Returns
    -------
    dict
        On success: 'status', 'url', 'content', 'content_type',
        'content_length', 'truncated'. For PDFs also includes:
        'num_pages', 'pages_extracted'. On error: 'status', 'error', 'url'.

    Examples
    --------
    >>> # Fetch an HTML page
    >>> result = web_fetch("https://example.com/about")
    >>> print(result["content"])

    >>> # Fetch a PDF
    >>> result = web_fetch("https://arxiv.org/pdf/2301.00234.pdf")
    >>> print(f"Pages: {result['num_pages']}")
    >>> print(result["content"])
    """
    logger.info(f"WebFetch: {url}")

    # Validate URL
    if not url.startswith(("http://", "https://")):
        return _make_error_response("Invalid URL. Must start with http:// or https://", url)

    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = _fetch_with_retry(client, url)
            content_type = response.headers.get("content-type", "")
            final_url = str(response.url)

            # Handle PDF documents
            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                return _handle_pdf_response(response.content, max_pages, final_url, url)

            # Handle HTML and text content
            if "text/html" in content_type or not content_type:
                text = _html_to_markdown(response.text, base_url=final_url)
            else:
                text = response.text
            text, truncated = _truncate_content(text)

            return _make_success_response(final_url, text, content_type or "text/html", truncated)

    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP error fetching {url}: {e}")
        return _make_error_response(f"HTTP {e.response.status_code}: {e.response.reason_phrase}", url)
    except httpx.RequestError as e:
        logger.warning(f"Request error fetching {url}: {e}")
        return _make_error_response(f"Request failed: {e!s}", url)
    except Exception as e:
        logger.exception(f"Unexpected error in web_fetch for {url}")
        return _make_error_response(f"Unexpected error: {e!s}", url)


def _handle_pdf_response(content: bytes, max_pages: int, final_url: str, url: str) -> dict[str, Any]:
    """Handle PDF content extraction and response creation."""
    try:
        text, num_pages = _extract_pdf_text(content, max_pages)
        text, truncated = _truncate_content(text)

        return _make_success_response(
            final_url,
            text,
            "application/pdf",
            truncated,
            num_pages=num_pages,
            pages_extracted=min(num_pages, max_pages),
        )
    except ImportError:
        return _make_error_response("PDF support requires pypdf. Install with: pip install pypdf", url)
    except Exception as e:
        return _make_error_response(f"Failed to extract PDF text: {e!s}", url)


def create_web_fetch_tool() -> FunctionTool:
    """Create an ADK FunctionTool for fetching web content."""
    return FunctionTool(func=web_fetch)

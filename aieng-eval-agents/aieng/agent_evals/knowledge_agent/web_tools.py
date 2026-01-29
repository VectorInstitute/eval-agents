"""Web tools for URL fetching and PDF reading.

Simple function-based tools for ADK agents to fetch web content
and read PDF documents.
"""

import logging
import re
from io import BytesIO
from typing import Any

import httpx
from google.adk.tools.function_tool import FunctionTool


logger = logging.getLogger(__name__)


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


def fetch_url(url: str, max_length: int = 50000) -> dict[str, Any]:
    """Fetch content from a URL and return as text.

    Use this tool to read the content of a webpage. Returns the text
    content extracted from the HTML, suitable for analysis.

    Parameters
    ----------
    url : str
        The URL to fetch. Must be a valid HTTP or HTTPS URL.
    max_length : int, optional
        Maximum characters to return (default 50000).

    Returns
    -------
    dict
        Contains 'content' (the text), 'url', 'status', and 'content_type'.
        On error, contains 'error' message instead of 'content'.
    """
    logger.info(f"Fetching URL: {url}")

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"})
            response.raise_for_status()

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

            # Truncate if too long
            if len(text) > max_length:
                text = text[:max_length] + f"\n\n[Content truncated at {max_length} characters]"

            return {
                "status": "success",
                "content": text,
                "url": str(response.url),  # Final URL after redirects
                "content_type": content_type,
                "length": len(text),
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
        Tool that fetches and extracts text from web pages.
    """
    return FunctionTool(func=fetch_url)


def create_read_pdf_tool() -> FunctionTool:
    """Create an ADK FunctionTool for reading PDF documents.

    Returns
    -------
    FunctionTool
        Tool that reads and extracts text from PDF files.
    """
    return FunctionTool(func=read_pdf)

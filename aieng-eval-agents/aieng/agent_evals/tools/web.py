"""HTTP tools for fetching web content.

Provides tools for fetching web pages and saving them locally for
efficient searching without loading full content into context.
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
    cache_dir = os.path.join(tempfile.gettempdir(), "agent_web_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _url_to_filename(url: str) -> str:
    """Convert URL to a safe filename."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    safe_name = re.sub(r"[^\w\-.]", "_", url.split("//")[-1][:50])
    return f"{safe_name}_{url_hash}.txt"


@_http_retry
def _fetch_with_retry(client: httpx.Client, url: str) -> httpx.Response:
    """Fetch URL with automatic retry on transient failures."""
    response = client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"})
    response.raise_for_status()
    return response


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text, preserving some structure."""
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
        On success: 'status', 'file_path', 'url', 'length', 'preview'.
        On error: 'status', 'error', 'url'.
    """
    logger.info(f"Fetching URL: {url}")

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = _fetch_with_retry(client, url)
            content_type = response.headers.get("content-type", "")

            if "application/pdf" in content_type:
                return {
                    "status": "error",
                    "error": "URL points to a PDF. Use the read_pdf tool instead.",
                    "url": url,
                    "content_type": content_type,
                }

            text = _html_to_text(response.text) if "text/html" in content_type or not content_type else response.text

            cache_dir = get_cache_dir()
            filename = _url_to_filename(url)
            file_path = os.path.join(cache_dir, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

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


def create_fetch_url_tool() -> FunctionTool:
    """Create an ADK FunctionTool for fetching URL content."""
    return FunctionTool(func=fetch_url)

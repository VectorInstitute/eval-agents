"""Web fetch tool for retrieving content from URLs.

Provides the web_fetch tool which fetches content from any URL (HTML pages or PDFs)
and uses an LLM to extract relevant information based on a query. This prevents
context overflow by returning only extracted information instead of full page content.

Architecture:
1. Fetch URL and convert to markdown/text
2. Use LLM to extract information based on query
3. Return only extracted information (~200-1000 tokens instead of 10k-100k)
"""

import logging
import re
from collections.abc import Callable
from io import BytesIO
from typing import Any
from urllib.parse import urljoin

import httpx
from aieng.agent_evals.async_client_manager import AsyncClientManager
from google.adk.tools.function_tool import FunctionTool
from html_to_markdown import convert as html_to_markdown
from pypdf import PdfReader
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential


logger = logging.getLogger(__name__)

# Maximum chars to fetch before LLM processing (prevents fetching enormous files)
MAX_FETCH_CHARS = 200_000

# Maximum chars to send to extraction LLM (most models handle ~100k chars well)
MAX_EXTRACTION_INPUT_CHARS = 100_000

# Maximum tokens for LLM extraction output
MAX_EXTRACTION_OUTPUT_TOKENS = 2000


def _make_absolute_url(base_url: str) -> Callable[[re.Match[str]], str]:
    """Create a function that converts relative URLs to absolute URLs.

    Parameters
    ----------
    base_url : str
        Base URL for resolving relative links.

    Returns
    -------
    Callable[[re.Match[str]], str]
        Function that takes a regex match and returns the URL converted to absolute.
    """

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

    return make_absolute


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
        make_absolute = _make_absolute_url(base_url)

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


def _truncate_content(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncate content if it exceeds the maximum length.

    Parameters
    ----------
    text : str
        The text to truncate.
    max_chars : int
        Maximum number of characters allowed.

    Returns
    -------
    tuple[str, bool]
        The (potentially truncated) text and whether it was truncated.
    """
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "\n\n[Content truncated due to length]"
    return text, truncated


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((Exception,)),  # Retry on any exception from LLM API
)
async def _extract_information_with_llm_with_retry(
    openai_client: Any, model: str, content: str, query: str, url: str, was_truncated: bool
) -> dict[str, Any]:
    """Execute LLM extraction with retry logic.

    Parameters
    ----------
    openai_client : Any
        The OpenAI client instance.
    model : str
        Model name to use.
    content : str
        The content to extract from.
    query : str
        What information to extract.
    url : str
        Source URL.
    was_truncated : bool
        Whether the source content was truncated.

    Returns
    -------
    dict[str, Any]
        Success response with extracted information and metadata.
    """
    # Build extraction prompt
    system_prompt = """You are a precise information extraction assistant. Your task is to extract specific information from web pages based on user queries.

Instructions:
- Extract ONLY the information requested in the query
- Be accurate and concise
- If the information is not found, say so clearly
- Include relevant quotes or data points when available
- Do not add information not present in the source
- Format your response clearly with markdown if appropriate"""

    user_prompt = f"""Extract the following information from this webpage:

QUERY: {query}

SOURCE URL: {url}

WEBPAGE CONTENT:
{content}

Please extract the requested information. Be concise but complete."""

    # Call LLM to extract information
    response = await openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=MAX_EXTRACTION_OUTPUT_TOKENS,
        temperature=0.0,  # Deterministic for extraction
    )

    extracted_info = response.choices[0].message.content or ""

    return {
        "status": "success",
        "extracted_info": extracted_info,
        "url": url,
        "query": query,
        "source_was_truncated": was_truncated,
        "extraction_model": model,
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }


async def _extract_information_with_llm(content: str, query: str, url: str) -> dict[str, Any]:
    """Extract relevant information from content using an LLM.

    This is the key function that implements the WebFetch architecture.
    Instead of returning full page content, it uses an LLM to extract only the
    information requested in the query.

    Parameters
    ----------
    content : str
        The full page content (markdown or text).
    query : str
        What information to extract from the content.
    url : str
        Source URL (for context and citation).

    Returns
    -------
    dict[str, Any]
        Response with 'status', 'extracted_info', 'url', and metadata.
    """
    try:
        # Truncate content for LLM processing if needed
        content, was_truncated = _truncate_content(content, MAX_EXTRACTION_INPUT_CHARS)

        # Get OpenAI client (configured for Gemini endpoint)
        client_manager = AsyncClientManager.get_instance()
        openai_client = client_manager.openai_client
        model = client_manager.configs.default_worker_model  # gemini-2.5-flash

        # Call with retry logic
        return await _extract_information_with_llm_with_retry(
            openai_client=openai_client,
            model=model,
            content=content,
            query=query,
            url=url,
            was_truncated=was_truncated,
        )

    except RetryError as e:
        # Extract the underlying error from retry failure
        original_error = e.last_attempt.exception()
        logger.error(f"LLM extraction failed for {url} after 3 retries: {original_error}")
        return {
            "status": "error",
            "error": f"Information extraction failed after 3 retries: {original_error!s}",
            "url": url,
            "query": query,
        }
    except Exception as e:
        logger.error(f"LLM extraction failed for {url}: {e}")
        return {
            "status": "error",
            "error": f"Information extraction failed: {e!s}",
            "url": url,
            "query": query,
        }


def _handle_fetch_error(e: Exception, url: str) -> dict[str, Any]:
    """Handle exceptions from web_fetch and return appropriate error response.

    Parameters
    ----------
    e : Exception
        The exception that occurred during fetching.
    url : str
        The URL that was being fetched.

    Returns
    -------
    dict
        Error response with 'status', 'error', and 'url' fields.
    """
    if isinstance(e, httpx.HTTPStatusError):
        logger.warning(f"HTTP error fetching {url}: {e}")
        error_msg = f"HTTP {e.response.status_code}: {e.response.reason_phrase}"
        return {"status": "error", "error": error_msg, "url": url}

    if isinstance(e, httpx.RequestError):
        logger.warning(f"Request error fetching {url}: {e}")
        return {"status": "error", "error": f"Request failed: {e!s}", "url": url}

    if isinstance(e, RetryError):
        # Extract the underlying error from retry failure
        # without showing full stack trace
        original_error = e.last_attempt.exception()
        if isinstance(original_error, httpx.HTTPStatusError):
            logger.error(f"HTTP error fetching {url} (after 3 retries): {original_error}")
            error_msg = f"HTTP {original_error.response.status_code}: {original_error.response.reason_phrase} (failed after 3 retries)"
        else:
            logger.error(f"Failed to fetch {url} after 3 retries: {original_error}")
            error_msg = f"Failed after 3 retries: {original_error!s}"
        return {"status": "error", "error": error_msg, "url": url}

    logger.error(f"Unexpected error in web_fetch for {url}: {e}")
    return {"status": "error", "error": f"Unexpected error: {e!s}", "url": url}


async def web_fetch(url: str, query: str, max_pages: int = 10) -> dict[str, Any]:
    """Fetch and extract information from web pages and PDFs using LLM.

    This tool implements WebFetch:
    1. Fetches the URL (HTML or PDF)
    2. Converts to markdown/text
    3. Uses an LLM to extract only the information specified in query
    4. Returns extracted info (~200-1000 tokens) instead of full content

    For data files like CSV or XLSX that need line-by-line searching,
    use fetch_file instead.

    Parameters
    ----------
    url : str
        The URL to fetch. Must be a valid HTTP or HTTPS URL.
    query : str
        What information to extract from the page. Be specific about what
        you need (e.g., "publication date and main findings", "contact email
        and phone number", "list of product features").
    max_pages : int, optional
        For PDFs, maximum number of pages to extract (default 10).

    Returns
    -------
    dict
        On success: 'status', 'extracted_info', 'url', 'query',
        'source_was_truncated', 'extraction_model', 'input_tokens',
        'output_tokens'. On error: 'status', 'error', 'url', 'query'.

    Examples
    --------
    >>> # Extract specific info from a webpage
    >>> result = await web_fetch(
    ...     "https://example.com/about",
    ...     query="company founding date and CEO name",
    ... )
    >>> print(result["extracted_info"])

    >>> # Extract from a PDF
    >>> result = await web_fetch(
    ...     "https://arxiv.org/pdf/2301.00234.pdf",
    ...     query="main research findings and datasets used",
    ... )
    >>> print(result["extracted_info"])
    """
    # Validate URL
    if not url.startswith(("http://", "https://")):
        return {
            "status": "error",
            "error": "Invalid URL. Must start with http:// or https://",
            "url": url,
            "query": query,
        }

    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await _fetch_with_retry(client, url)
            content_type = response.headers.get("content-type", "")
            final_url = str(response.url)

            # Handle PDF documents
            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                pdf_text, num_pages = _extract_pdf_text(response.content, max_pages)
                # Truncate if needed before LLM extraction
                pdf_text, _ = _truncate_content(pdf_text, MAX_FETCH_CHARS)

                logger.info(f"Extracted {len(pdf_text)} chars from PDF ({num_pages} pages), extracting info...")
                result = await _extract_information_with_llm(pdf_text, query, final_url)
                # Add PDF metadata
                if result["status"] == "success":
                    result["content_type"] = "application/pdf"
                    result["num_pages"] = num_pages
                    result["pages_extracted"] = min(num_pages, max_pages)
                return result

            # Handle HTML and text content
            if "text/html" in content_type or content_type == "":
                text = _html_to_markdown(response.text, base_url=final_url)
            else:
                text = response.text

            # Truncate if needed before LLM extraction
            text, _ = _truncate_content(text, MAX_FETCH_CHARS)

            logger.info(f"Fetched {len(text)} chars from {final_url}, extracting info...")
            result = await _extract_information_with_llm(text, query, final_url)
            if result["status"] == "success":
                result["content_type"] = content_type or "text/html"
            return result

    except Exception as e:
        return _handle_fetch_error(e, url)


def create_web_fetch_tool() -> FunctionTool:
    """Create an ADK FunctionTool for fetching web content."""
    return FunctionTool(func=web_fetch)

"""URL redirect resolution utilities.

Provides utilities for resolving redirect URLs (especially Vertex AI grounding
redirects) to their final destinations. Used by search and web fetch tools to
display actual URLs.
"""

import asyncio
import logging

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


logger = logging.getLogger(__name__)

REDIRECT_URL_PATTERNS = (
    "vertexaisearch.cloud.google.com/grounding-api-redirect",
    "vertexaisearch.cloud.google.com/redirect",
)

_REDIRECT_CONNECT_TIMEOUT = 10.0
_REDIRECT_READ_TIMEOUT = 15.0
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_redirect_cache: dict[str, str] = {}


def _is_redirect_url(url: str) -> bool:
    """Check if URL is a known redirect pattern."""
    return any(pattern in url for pattern in REDIRECT_URL_PATTERNS)


def _get_redirect_timeout() -> httpx.Timeout:
    """Get timeout configuration for redirect resolution."""
    return httpx.Timeout(
        connect=_REDIRECT_CONNECT_TIMEOUT,
        read=_REDIRECT_READ_TIMEOUT,
        write=10.0,
        pool=10.0,
    )


async def _resolve_with_head_async(client: httpx.AsyncClient, url: str) -> str | None:
    """Try to resolve redirect using async HEAD request."""
    try:
        response = await client.head(url, headers={"User-Agent": _USER_AGENT})
        return str(response.url)
    except httpx.HTTPStatusError as e:
        # Some servers return 405 Method Not Allowed for HEAD
        if e.response.status_code in (405, 501):
            return None  # Signal to try GET
        raise
    except Exception:
        return None


async def _resolve_with_get_async(client: httpx.AsyncClient, url: str) -> str:
    """Resolve redirect using async GET request (fallback when HEAD fails)."""
    # Use stream to avoid downloading the body
    async with client.stream("GET", url, headers={"User-Agent": _USER_AGENT}) as response:
        return str(response.url)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.0, min=1.0, max=60.0),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)),
)
async def _resolve_single_url_with_retry(client: httpx.AsyncClient, url: str) -> str:
    """Resolve URL with retry handling (internal helper)."""
    # Try HEAD first (faster, no body download)
    final_url = await _resolve_with_head_async(client, url)

    # Fall back to GET if HEAD failed
    if final_url is None:
        logger.debug(f"HEAD failed for {url[:60]}..., trying GET")
        final_url = await _resolve_with_get_async(client, url)

    if final_url != url:
        logger.debug(f"Resolved redirect: {url[:60]}... -> {final_url[:60]}...")

    return final_url


async def _resolve_single_url_async(client: httpx.AsyncClient, url: str) -> str:
    """Resolve a single URL with retries and exponential backoff.

    Uses tenacity for automatic retry handling with exponential backoff.

    Parameters
    ----------
    client : httpx.AsyncClient
        The HTTP client to use.
    url : str
        The URL to resolve.

    Returns
    -------
    str
        The resolved URL, or original URL if not a redirect or on failure.
    """
    # Skip resolution for non-redirect URLs
    if not _is_redirect_url(url):
        return url

    # Check cache first
    if url in _redirect_cache:
        return _redirect_cache[url]

    try:
        final_url = await _resolve_single_url_with_retry(client, url)
        _redirect_cache[url] = final_url
        return final_url

    except Exception as e:
        # All retries exhausted or non-retryable error
        logger.warning(f"Failed to resolve redirect URL {url[:60]}...: {type(e).__name__}: {e}")
        _redirect_cache[url] = url  # Cache failures to avoid repeated attempts
        return url


async def resolve_redirect_url_async(url: str) -> str:
    """Async version of resolve_redirect_url with caching and retries.

    Parameters
    ----------
    url : str
        The URL to resolve (may be a redirect URL).

    Returns
    -------
    str
        The final destination URL after following redirects.
    """
    return (await resolve_redirect_urls_async([url]))[0]


async def resolve_redirect_urls_async(urls: list[str]) -> list[str]:
    """Resolve multiple redirect URLs in parallel.

    Resolves URLs concurrently with proper error handling per URL.

    Parameters
    ----------
    urls : list[str]
        List of URLs to resolve.

    Returns
    -------
    list[str]
        List of resolved URLs in the same order.
    """
    if not urls:
        return []

    async with httpx.AsyncClient(
        timeout=_get_redirect_timeout(),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    ) as client:
        # Resolve all URLs in parallel
        tasks = [_resolve_single_url_async(client, url) for url in urls]
        return list(await asyncio.gather(*tasks))

"""Google Search tool for knowledge-grounded QA using ADK.

This module provides a search tool that returns actual URLs the agent can fetch,
enabling a proper research workflow: search → fetch → verify → answer.
"""

import asyncio
import logging
from typing import Any

from aieng.agent_evals.configs import Configs
from google.adk.tools.function_tool import FunctionTool
from google.genai import Client, types
from pydantic import BaseModel, Field

from ._redirect import resolve_redirect_urls_async


logger = logging.getLogger(__name__)


def _extract_summary_from_response(response: Any) -> str:
    """Extract text summary from Gemini response.

    Parameters
    ----------
    response : Any
        The Gemini API response object.

    Returns
    -------
    str
        The extracted summary text.
    """
    summary = ""
    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                summary += part.text
    return summary


async def _extract_grounding_sources(response: Any) -> list[dict[str, str]]:
    """Extract and resolve grounding sources from Gemini response.

    Parameters
    ----------
    response : Any
        The Gemini API response object.

    Returns
    -------
    list[dict[str, str]]
        List of source dictionaries with 'title' and 'url' keys.
    """
    sources: list[dict[str, str]] = []
    if not response.candidates:
        return sources

    gm = getattr(response.candidates[0], "grounding_metadata", None)
    if not gm or not hasattr(gm, "grounding_chunks") or not gm.grounding_chunks:
        return sources

    redirect_urls = []
    titles = []
    for chunk in gm.grounding_chunks:
        if hasattr(chunk, "web") and chunk.web:
            redirect_urls.append(getattr(chunk.web, "uri", "") or "")
            titles.append(getattr(chunk.web, "title", "") or "")

    if redirect_urls:
        resolved_urls = await resolve_redirect_urls_async(redirect_urls)
        for title, url in zip(titles, resolved_urls):
            if url and not url.startswith("https://vertexaisearch"):
                sources.append({"title": title, "url": url})

    return sources


def _should_retry_for_empty_grounding(response: Any) -> bool:
    """Check if response has empty grounding metadata that warrants a retry.

    Parameters
    ----------
    response : Any
        The Gemini API response object.

    Returns
    -------
    bool
        True if grounding_chunks is explicitly None, False otherwise.
    """
    if not response.candidates:
        return False

    gm = getattr(response.candidates[0], "grounding_metadata", None)
    return gm is not None and hasattr(gm, "grounding_chunks") and gm.grounding_chunks is None


class GroundingChunk(BaseModel):
    """Represents a single grounding source from search results."""

    title: str = Field(default="", description="Title of the source webpage.")
    uri: str = Field(default="", description="URL of the source webpage.")


class GroundedResponse(BaseModel):
    """Response from the knowledge agent with grounding information."""

    text: str = Field(description="The generated response text.")
    search_queries: list[str] = Field(default_factory=list, description="The search queries that were executed.")
    sources: list[GroundingChunk] = Field(
        default_factory=list, description="The web sources used to ground the response."
    )
    tool_calls: list[dict] = Field(
        default_factory=list, description="List of tool calls made during the response generation."
    )

    def format_with_citations(self) -> str:
        """Format this response with inline citations.

        Returns
        -------
        str
            Formatted response text with citations appended.
        """
        output_parts = [self.text]

        if self.sources:
            output_parts.append("\n\n**Sources:**")
            for i, source in enumerate(self.sources, 1):
                if source.uri:
                    output_parts.append(f"[{i}] [{source.title or 'Source'}]({source.uri})")

        return "\n".join(output_parts)


def format_response_with_citations(response: GroundedResponse) -> str:
    """Format a grounded response with inline citations.

    Parameters
    ----------
    response : GroundedResponse
        The grounded response to format.

    Returns
    -------
    str
        Formatted response text with citations appended.

    Notes
    -----
    This is a convenience wrapper around ``response.format_with_citations()``.
    """
    return response.format_with_citations()


async def _google_search_async(
    query: str, model: str, temperature: float = 1.0, max_retries: int = 3, api_key: str | None = None
) -> dict[str, Any]:
    """Execute a Google search and return results with actual URLs.

    This function calls Gemini with Google Search grounding enabled,
    extracts the grounding URLs, resolves any redirects, and returns
    a structured response the agent can use for further fetching.

    The Google API sometimes returns grounding_metadata with all None fields,
    likely due to rate limiting. This function retries up to max_retries times
    with exponential backoff when this happens.

    Parameters
    ----------
    query : str
        The search query.
    model : str
        The Gemini model to use for search.
    temperature : float, default=1.0
        Temperature for generation. Google recommends 1.0 for optimal grounding
        reliability.
    max_retries : int, default=3
        Maximum number of retry attempts when grounding chunks are not returned.

    Returns
    -------
    dict
        Search results with the following keys:

        - **status** (str): "success" or "error"
        - **summary** (str): Brief text summary of search results
        - **sources** (list[dict]): List of source dicts, each containing:
            - **title** (str): Title of the webpage
            - **url** (str): Actual URL that can be fetched
        - **source_count** (int): Number of sources found (success case only)
        - **error** (str): Error message (error case only)
    """
    client = Client(api_key=api_key)

    try:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=query,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                        temperature=temperature,
                    ),
                )

                # Check if we should retry due to empty grounding metadata
                if _should_retry_for_empty_grounding(response):
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt
                        logger.warning(
                            f"Grounding chunks is None (attempt {attempt + 1}/{max_retries}). "
                            f"Retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    logger.error(f"Grounding chunks is None after {max_retries} attempts. No grounding data available.")

                # Extract summary and sources
                summary = _extract_summary_from_response(response)
                sources = await _extract_grounding_sources(response)

                return {
                    "status": "success",
                    "summary": summary,
                    "sources": sources,
                    "source_count": len(sources),
                }

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Search failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                logger.error(f"Search failed after {max_retries} attempts: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "summary": "",
                    "sources": [],
                }

        # This should never be reached, but satisfies type checker
        return {
            "status": "error",
            "error": "All retries exhausted without result",
            "summary": "",
            "sources": [],
        }
    finally:
        # Properly close the client to avoid aiohttp session leaks
        client.close()


async def google_search(query: str, model: str | None = None) -> dict[str, Any]:
    """Search Google and return results with actual URLs for fetching.

    Use this tool to find information on the web. The results include:
    - A summary of what was found
    - A list of source URLs that you can fetch with web_fetch to verify information

    IMPORTANT: The summary is from search snippets which may be incomplete or outdated.
    Always use web_fetch on the source URLs to verify information before answering.

    Parameters
    ----------
    query : str
        The search query. Be specific and include key terms.
    model : str, optional
        The Gemini model to use for search. If not provided, uses
        default_worker_model from Configs.

    Returns
    -------
    dict
        Search results with the following keys:

        - **status** (str): "success" or "error"
        - **summary** (str): Brief text summary of search results
        - **sources** (list[dict]): List of source dicts, each containing:
            - **title** (str): Title of the webpage
            - **url** (str): Actual URL that can be fetched
        - **source_count** (int): Number of sources found (success case only)
        - **error** (str): Error message (error case only)

    Examples
    --------
    >>> result = await google_search("highest single day snowfall Toronto")
    >>> # Check the sources
    >>> for source in result["sources"]:
    ...     print(f"{source['title']}: {source['url']}")
    >>> # Then fetch to verify
    >>> page = await web_fetch(result["sources"][0]["url"])
    """
    config = Configs()  # type: ignore[call-arg]
    if model is None:
        model = config.default_worker_model

    return await _google_search_async(
        query, model=model, temperature=config.default_temperature, api_key=config.openai_api_key.get_secret_value()
    )


def create_google_search_tool(config: Configs | None = None) -> FunctionTool:
    """Create a search tool that returns actual URLs for fetching.

    This tool calls Google Search, extracts grounding URLs, resolves redirects,
    and returns actual URLs the agent can use with web_fetch for verification.

    Parameters
    ----------
    config : Configs, optional
        Configuration settings. If not provided, creates default config.
        Uses config.default_worker_model for the search model.

    Returns
    -------
    FunctionTool
        A search tool that returns fetchable URLs.

    Examples
    --------
    >>> from aieng.agent_evals.tools import create_google_search_tool
    >>> from aieng.agent_evals.configs import Configs
    >>> config = Configs()
    >>> search_tool = create_google_search_tool(config=config)
    >>> # Use with an ADK agent
    >>> agent = Agent(tools=[search_tool])
    """
    if config is None:
        config = Configs()  # type: ignore[call-arg]

    model = config.default_worker_model
    temperature = config.default_temperature

    async def google_search(query: str) -> dict[str, Any]:
        """Search Google and return results with actual URLs for fetching.

        Use this tool to find information on the web. The results include:
        - A summary of what was found
        - A list of source URLs that you can fetch with web_fetch to verify

        IMPORTANT: The summary is from search snippets which may be incomplete
        or outdated. Always use web_fetch on the source URLs to verify information
        before answering.

        Parameters
        ----------
        query : str
            The search query. Be specific and include key terms.

        Returns
        -------
        dict
            Search results with the following keys:

            - **status** (str): "success" or "error"
            - **summary** (str): Brief text summary of search results
            - **sources** (list[dict]): List of source dicts, each containing:
                - **title** (str): Title of the webpage
                - **url** (str): Actual URL that can be fetched
            - **source_count** (int): Number of sources found (success case only)
            - **error** (str): Error message (error case only)
        """
        return await _google_search_async(
            query, model=model, temperature=temperature, api_key=config.openai_api_key.get_secret_value()
        )

    return FunctionTool(func=google_search)

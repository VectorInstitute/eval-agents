"""Google Search tool for knowledge-grounded QA using ADK.

This module provides a search tool that returns actual URLs the agent can fetch,
enabling a proper research workflow: search → fetch → verify → answer.
"""

import asyncio
import concurrent.futures
import logging
from typing import Any

from google.adk.tools.function_tool import FunctionTool
from google.genai import Client, types
from pydantic import BaseModel, Field

from .web import resolve_redirect_urls_async


logger = logging.getLogger(__name__)


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


async def _google_search_async(query: str) -> dict[str, Any]:
    """Execute a Google search and return results with actual URLs.

    This function calls Gemini with Google Search grounding enabled,
    extracts the grounding URLs, resolves any redirects, and returns
    a structured response the agent can use for further fetching.

    Parameters
    ----------
    query : str
        The search query.

    Returns
    -------
    dict
        Search results with 'summary' text and 'sources' list containing
        actual URLs the agent can fetch.
    """
    client = Client()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.0,
            ),
        )

        # Extract text summary
        summary = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    summary += part.text

        # Extract grounding URLs
        sources = []
        gm = getattr(response.candidates[0], "grounding_metadata", None) if response.candidates else None

        if gm and hasattr(gm, "grounding_chunks") and gm.grounding_chunks:
            redirect_urls = []
            titles = []
            for chunk in gm.grounding_chunks:
                if hasattr(chunk, "web") and chunk.web:
                    redirect_urls.append(getattr(chunk.web, "uri", "") or "")
                    titles.append(getattr(chunk.web, "title", "") or "")

            # Resolve redirect URLs to actual URLs
            if redirect_urls:
                resolved_urls = await resolve_redirect_urls_async(redirect_urls)
                for title, url in zip(titles, resolved_urls):
                    if url and not url.startswith("https://vertexaisearch"):
                        sources.append({"title": title, "url": url})

        return {
            "status": "success",
            "summary": summary,
            "sources": sources,
            "source_count": len(sources),
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "summary": "",
            "sources": [],
        }


def google_search(query: str) -> dict[str, Any]:
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

    Returns
    -------
    dict
        Contains 'summary' (brief overview), 'sources' (list of URLs to fetch),
        and 'source_count'. On error, contains 'status': 'error' and 'error' message.

    Examples
    --------
    >>> result = google_search("highest single day snowfall Toronto")
    >>> # Check the sources
    >>> for source in result["sources"]:
    ...     print(f"{source['title']}: {source['url']}")
    >>> # Then fetch to verify
    >>> page = web_fetch(result["sources"][0]["url"])
    """
    logger.info(f"GoogleSearch: {query}")

    # Handle being called from async context
    try:
        asyncio.get_running_loop()
        # We're in an async context - need to run in a new thread
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _google_search_async(query))
            return future.result()
    except RuntimeError:
        # No running loop - we can use asyncio.run directly
        return asyncio.run(_google_search_async(query))


def create_google_search_tool() -> FunctionTool:
    """Create a search tool that returns actual URLs for fetching.

    This tool calls Google Search, extracts grounding URLs, resolves redirects,
    and returns actual URLs the agent can use with web_fetch for verification.

    Returns
    -------
    FunctionTool
        A search tool that returns fetchable URLs.

    Examples
    --------
    >>> from aieng.agent_evals.tools import create_google_search_tool
    >>> search_tool = create_google_search_tool()
    >>> # Use with an ADK agent
    >>> agent = Agent(tools=[search_tool])
    """
    return FunctionTool(func=google_search)

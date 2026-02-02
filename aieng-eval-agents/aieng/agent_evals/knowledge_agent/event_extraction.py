"""Event extraction utilities for ADK agent events.

This module provides functions to extract tool calls, sources, responses,
and other metadata from Google ADK runner events.
"""

import logging
from typing import Any

from aieng.agent_evals.tools import GroundingChunk, resolve_redirect_urls_async


logger = logging.getLogger(__name__)


def extract_tool_calls(event: Any) -> list[dict[str, Any]]:
    """Extract tool calls from event function calls.

    Parameters
    ----------
    event : Any
        An event from the ADK runner.

    Returns
    -------
    list[dict[str, Any]]
        List of tool call dictionaries with 'name' and 'args' keys.
    """
    if not hasattr(event, "get_function_calls"):
        return []
    function_calls = event.get_function_calls()
    if not function_calls:
        return []

    tool_calls = []
    for fc in function_calls:
        tool_call_info = {
            "name": getattr(fc, "name", "unknown"),
            "args": getattr(fc, "args", {}),
        }
        tool_calls.append(tool_call_info)
        logger.info(f"Tool call: {tool_call_info['name']}({tool_call_info['args']})")
    return tool_calls


def extract_search_queries_from_tool_calls(tool_calls: list[dict[str, Any]]) -> list[str]:
    """Extract search queries from tool calls.

    Parameters
    ----------
    tool_calls : list[dict[str, Any]]
        List of tool call dictionaries.

    Returns
    -------
    list[str]
        Search queries found in the tool calls.
    """
    queries = []
    for tool_call in tool_calls:
        tool_name = str(tool_call.get("name", ""))
        tool_args = tool_call.get("args", {})
        if "search" in tool_name.lower() and isinstance(tool_args, dict):
            # google_search_agent uses "request", other tools may use "query"
            query = tool_args.get("request") or tool_args.get("query") or ""
            if query:
                queries.append(query)
    return queries


def extract_sources_from_responses(event: Any) -> list[GroundingChunk]:
    """Extract sources from event function responses.

    Parameters
    ----------
    event : Any
        An event from the ADK runner.

    Returns
    -------
    list[GroundingChunk]
        Sources extracted from the function responses (raw URLs, not resolved).
    """
    if not hasattr(event, "get_function_responses"):
        return []
    function_responses = event.get_function_responses()
    if not function_responses:
        return []

    sources = []
    for fr in function_responses:
        # Log tool response for CLI display tracking
        tool_name = getattr(fr, "name", None) or getattr(fr, "id", "unknown")
        logger.info(f"Tool response: {tool_name} completed")
        response_data = getattr(fr, "response", {})
        if not isinstance(response_data, dict):
            continue
        # Extract sources from search tool response
        for src in response_data.get("sources", []):
            if isinstance(src, dict):
                sources.append(
                    GroundingChunk(
                        title=src.get("title", ""),
                        uri=src.get("uri") or src.get("url") or "",
                    )
                )
        # Extract grounding_chunks if present
        for chunk in response_data.get("grounding_chunks", []):
            if isinstance(chunk, dict) and "web" in chunk:
                sources.append(
                    GroundingChunk(
                        title=chunk["web"].get("title", ""),
                        uri=chunk["web"].get("uri", ""),
                    )
                )
    return sources


def extract_grounding_sources(event: Any) -> list[GroundingChunk]:
    """Extract sources from grounding metadata.

    Parameters
    ----------
    event : Any
        An event from the ADK runner.

    Returns
    -------
    list[GroundingChunk]
        Sources extracted from the grounding metadata (raw URLs, not resolved).
    """
    gm = getattr(event, "grounding_metadata", None)
    if not gm and hasattr(event, "content") and event.content:
        gm = getattr(event.content, "grounding_metadata", None)
    if not gm:
        return []

    sources = []
    if hasattr(gm, "grounding_chunks") and gm.grounding_chunks:
        for chunk in gm.grounding_chunks:
            if hasattr(chunk, "web") and chunk.web:
                sources.append(
                    GroundingChunk(
                        title=getattr(chunk.web, "title", "") or "",
                        uri=getattr(chunk.web, "uri", "") or "",
                    )
                )
    return sources


async def resolve_source_urls(sources: list[GroundingChunk]) -> list[GroundingChunk]:
    """Resolve redirect URLs in sources to actual URLs (in parallel).

    Parameters
    ----------
    sources : list[GroundingChunk]
        Sources with potentially redirect URLs.

    Returns
    -------
    list[GroundingChunk]
        Sources with resolved URLs.
    """
    if not sources:
        return sources

    # Extract URIs and resolve in parallel
    uris = [s.uri for s in sources]
    resolved_uris = await resolve_redirect_urls_async(uris)

    # Create new sources with resolved URIs
    return [GroundingChunk(title=s.title, uri=resolved) for s, resolved in zip(sources, resolved_uris)]


def extract_grounding_queries(event: Any) -> list[str]:
    """Extract search queries from grounding metadata.

    Parameters
    ----------
    event : Any
        An event from the ADK runner.

    Returns
    -------
    list[str]
        Search queries from the grounding metadata.
    """
    gm = getattr(event, "grounding_metadata", None)
    if not gm and hasattr(event, "content") and event.content:
        gm = getattr(event.content, "grounding_metadata", None)
    if not gm:
        return []

    queries = []
    if hasattr(gm, "web_search_queries") and gm.web_search_queries:
        for q in gm.web_search_queries:
            if q:
                queries.append(q)
    return queries


def extract_final_response(event: Any) -> str | None:
    """Extract final response text from event if it's a final response.

    Filters out thought parts (internal reasoning) and returns only the
    actual response text intended for the user.

    Returns
    -------
    str | None
        The response text if found, None if no non-thought content exists.
        Returns None (not empty string) when the event has only thought parts,
        to avoid overwriting previously captured valid responses.
    """
    if not hasattr(event, "is_final_response") or not event.is_final_response():
        return None
    if not hasattr(event, "content") or not event.content:
        return None
    if not hasattr(event.content, "parts") or not event.content.parts:
        return None

    # Collect non-thought text parts only
    response_parts = []
    for part in event.content.parts:
        # Skip thought parts (internal reasoning)
        if getattr(part, "thought", False):
            continue
        if hasattr(part, "text") and part.text:
            response_parts.append(part.text)

    # Return None instead of empty string to avoid overwriting valid responses
    # captured from earlier events (e.g., FINAL_ANSWER tags)
    return "\n".join(response_parts) if response_parts else None


def extract_thoughts_from_event(event: Any) -> str:
    """Extract thinking/reasoning content from event parts.

    Parameters
    ----------
    event : Any
        An event from the ADK runner.

    Returns
    -------
    str
        Combined thinking text from all thought parts.
    """
    if not hasattr(event, "content") or not event.content:
        return ""
    if not hasattr(event.content, "parts") or not event.content.parts:
        return ""

    thoughts = []
    for part in event.content.parts:
        # Parts with thought=True are thinking content
        if getattr(part, "thought", False) and hasattr(part, "text") and part.text:
            thoughts.append(part.text)
    return "\n".join(thoughts)


def extract_event_text(event: Any) -> str:
    """Extract text content from event parts.

    Parameters
    ----------
    event : Any
        An event from the ADK runner.

    Returns
    -------
    str
        Combined text from all parts.
    """
    if not (hasattr(event, "content") and event.content and hasattr(event.content, "parts") and event.content.parts):
        return ""
    parts = [part.text for part in event.content.parts if hasattr(part, "text") and part.text]
    return "\n".join(parts)

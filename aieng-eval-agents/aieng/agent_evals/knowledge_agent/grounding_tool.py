"""Google Search grounding tool for knowledge-grounded QA using ADK.

This module provides the GoogleSearchTool configuration for use with
Google ADK agents, enabling explicit and traceable web search capabilities.
"""

import logging
from typing import TYPE_CHECKING

from google.adk.tools.google_search_tool import GoogleSearchTool
from pydantic import BaseModel, Field


if TYPE_CHECKING:
    pass


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


def create_google_search_tool() -> GoogleSearchTool:
    """Create a GoogleSearchTool configured for use with other tools.

    This creates a GoogleSearchTool with bypass_multi_tools_limit=True,
    which allows it to be used alongside other custom tools in an ADK agent.
    The tool calls are explicit and visible in the agent's reasoning trace.

    Returns
    -------
    GoogleSearchTool
        A configured GoogleSearchTool instance.

    Examples
    --------
    >>> from aieng.agent_evals.knowledge_agent.grounding_tool import (
    ...     create_google_search_tool,
    ... )
    >>> search_tool = create_google_search_tool()
    >>> # Use with an ADK agent
    >>> agent = Agent(tools=[search_tool])
    """
    return GoogleSearchTool(bypass_multi_tools_limit=True)


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

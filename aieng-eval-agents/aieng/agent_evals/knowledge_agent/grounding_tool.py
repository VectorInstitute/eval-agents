"""Gemini Google Search grounding tool for knowledge-grounded QA.

This module provides tools for using Gemini's built-in Google Search grounding
capability to answer questions with real-time web information.
"""

import logging
from typing import TYPE_CHECKING

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from .config import KnowledgeAgentConfig


logger = logging.getLogger(__name__)


class GroundingChunk(BaseModel):
    """Represents a single grounding source from search results.

    Attributes
    ----------
    title : str
        Title of the source webpage.
    uri : str
        URL of the source webpage.
    """

    title: str = ""
    uri: str = ""


class GroundedResponse(BaseModel):
    """Response from Gemini with Google Search grounding.

    Attributes
    ----------
    text : str
        The generated response text.
    search_queries : list[str]
        The search queries that were executed.
    sources : list[GroundingChunk]
        The web sources used to ground the response.
    """

    text: str
    search_queries: list[str] = Field(default_factory=list)
    sources: list[GroundingChunk] = Field(default_factory=list)


class GeminiGroundingTool:
    """Tool for generating responses grounded in Google Search results.

    This class uses the Gemini API with Google Search grounding to generate
    responses that are backed by real-time web information.

    Parameters
    ----------
    config : KnowledgeAgentConfig, optional
        Configuration settings. If not provided, creates default config.
    model : str, optional
        The Gemini model to use. Defaults to config.default_worker_model.

    Examples
    --------
    >>> from aieng.agent_evals.knowledge_agent import GeminiGroundingTool
    >>> tool = GeminiGroundingTool()
    >>> response = tool.search("What is the current population of Tokyo?")
    >>> print(response.text)
    """

    def __init__(
        self,
        config: "KnowledgeAgentConfig | None" = None,
        model: str | None = None,
    ) -> None:
        """Initialize the Gemini grounding tool.

        Parameters
        ----------
        config : KnowledgeAgentConfig, optional
            Configuration settings. If not provided, creates default config.
        model : str, optional
            The Gemini model to use. Defaults to config.default_worker_model.
        """
        if config is None:
            from .config import KnowledgeAgentConfig  # noqa: PLC0415

            config = KnowledgeAgentConfig()  # type: ignore[call-arg]

        self.config = config
        self.model = model or config.default_worker_model

        # Initialize Gemini client
        self._client = genai.Client(api_key=config.openai_api_key)

    def search(self, query: str) -> GroundedResponse:
        """Generate a response grounded in Google Search results.

        Parameters
        ----------
        query : str
            The question or query to answer.

        Returns
        -------
        GroundedResponse
            The response with text, search queries used, and sources.
        """
        logger.info(f"Generating grounded response for: {query[:100]}...")

        response = self._client.models.generate_content(
            model=self.model,
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )

        # Extract grounding metadata
        search_queries: list[str] = []
        sources: list[GroundingChunk] = []

        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            grounding_metadata = getattr(candidate, "grounding_metadata", None)

            if grounding_metadata:
                # Extract search queries
                if hasattr(grounding_metadata, "web_search_queries"):
                    search_queries = list(grounding_metadata.web_search_queries or [])

                # Extract grounding chunks (sources)
                if hasattr(grounding_metadata, "grounding_chunks"):
                    for chunk in grounding_metadata.grounding_chunks or []:
                        if hasattr(chunk, "web") and chunk.web:
                            sources.append(
                                GroundingChunk(
                                    title=getattr(chunk.web, "title", "") or "",
                                    uri=getattr(chunk.web, "uri", "") or "",
                                )
                            )

        return GroundedResponse(
            text=response.text or "",
            search_queries=search_queries,
            sources=sources,
        )

    async def search_async(self, query: str) -> GroundedResponse:
        """Async version of search for concurrent operations.

        Parameters
        ----------
        query : str
            The question or query to answer.

        Returns
        -------
        GroundedResponse
            The response with text, search queries used, and sources.
        """
        logger.info(f"Generating grounded response (async) for: {query[:100]}...")

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )

        # Extract grounding metadata
        search_queries: list[str] = []
        sources: list[GroundingChunk] = []

        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            grounding_metadata = getattr(candidate, "grounding_metadata", None)

            if grounding_metadata:
                if hasattr(grounding_metadata, "web_search_queries"):
                    search_queries = list(grounding_metadata.web_search_queries or [])

                if hasattr(grounding_metadata, "grounding_chunks"):
                    for chunk in grounding_metadata.grounding_chunks or []:
                        if hasattr(chunk, "web") and chunk.web:
                            sources.append(
                                GroundingChunk(
                                    title=getattr(chunk.web, "title", "") or "",
                                    uri=getattr(chunk.web, "uri", "") or "",
                                )
                            )

        return GroundedResponse(
            text=response.text or "",
            search_queries=search_queries,
            sources=sources,
        )

    def format_response_with_citations(self, response: GroundedResponse) -> str:
        """Format a grounded response with inline citations.

        Parameters
        ----------
        response : GroundedResponse
            The grounded response to format.

        Returns
        -------
        str
            Formatted response text with citations appended.
        """
        output_parts = [response.text]

        if response.sources:
            output_parts.append("\n\n**Sources:**")
            for i, source in enumerate(response.sources, 1):
                if source.uri:
                    output_parts.append(f"[{i}] [{source.title}]({source.uri})")

        return "\n".join(output_parts)

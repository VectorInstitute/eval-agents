"""Knowledge-grounded QA agent using Gemini with Google Search grounding.

This module provides the main agent class for knowledge-grounded question
answering, using Gemini's built-in Google Search grounding capability.
"""

import logging
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from .config import KnowledgeAgentConfig
from .grounding_tool import GeminiGroundingTool, GroundedResponse, GroundingChunk


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# System instructions for the knowledge-grounded QA agent
KNOWLEDGE_AGENT_INSTRUCTIONS = """\
You are a knowledge-grounded research assistant. Your role is to provide
accurate, comprehensive answers by searching the web for relevant information.

## Guidelines

1. **Search thoroughly**: Use Google Search to find relevant, up-to-date information.
   Do not rely solely on your training data for facts that may have changed.

2. **Cite your sources**: Always mention where you found the information and
   include source URLs when available.

3. **Be comprehensive**: For complex questions that require multiple pieces of
   information, search multiple times to gather all relevant facts.

4. **Be honest about uncertainty**: If you cannot find relevant information or if
   search results are inconclusive, say so clearly rather than guessing.

5. **Synthesize information**: When answering complex questions, synthesize findings
   from multiple sources into a coherent response.

## Response Format

When answering questions:
- Provide a clear, direct answer first
- Include relevant context and details from your sources
- List the sources used at the end of your response

Remember: Accuracy and completeness are more important than speed.
"""


class KnowledgeGroundedAgent:
    """A knowledge-grounded QA agent using Gemini with Google Search.

    This agent uses Gemini's built-in Google Search grounding to answer questions
    with real-time web information.

    Parameters
    ----------
    config : KnowledgeAgentConfig, optional
        Configuration settings. If not provided, creates default config.
    model : str, optional
        The model to use. If not provided, uses config.default_worker_model.

    Attributes
    ----------
    config : KnowledgeAgentConfig
        The configuration settings.
    grounding_tool : GeminiGroundingTool
        The Gemini grounding tool.

    Examples
    --------
    >>> from aieng.agent_evals.knowledge_agent import KnowledgeGroundedAgent
    >>> agent = KnowledgeGroundedAgent()
    >>> response = agent.answer("Who won the 2024 Nobel Prize in Physics?")
    >>> print(response.text)
    """

    def __init__(
        self,
        config: KnowledgeAgentConfig | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the knowledge-grounded agent.

        Parameters
        ----------
        config : KnowledgeAgentConfig, optional
            Configuration settings. If not provided, creates default config.
        model : str, optional
            The model to use. If not provided, uses config.default_worker_model.
        """
        if config is None:
            config = KnowledgeAgentConfig()  # type: ignore[call-arg]

        self.config = config
        self.model = model or config.default_worker_model

        # Initialize Gemini client with grounding
        self._client = genai.Client(api_key=config.openai_api_key)

        # Also create the grounding tool for direct access
        self.grounding_tool = GeminiGroundingTool(config=config, model=self.model)

    def answer(self, question: str) -> GroundedResponse:
        """Answer a question using Google Search grounding.

        Parameters
        ----------
        question : str
            The question to answer.

        Returns
        -------
        GroundedResponse
            The grounded response with text, search queries, and sources.
        """
        logger.info(f"Answering question: {question[:100]}...")

        # Prepend system instructions to the question
        prompt = f"{KNOWLEDGE_AGENT_INSTRUCTIONS}\n\n**Question:** {question}"

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )

        return self._parse_response(response)

    async def answer_async(self, question: str) -> GroundedResponse:
        """Async version of answer for concurrent operations.

        Parameters
        ----------
        question : str
            The question to answer.

        Returns
        -------
        GroundedResponse
            The grounded response with text, search queries, and sources.
        """
        logger.info(f"Answering question (async): {question[:100]}...")

        prompt = f"{KNOWLEDGE_AGENT_INSTRUCTIONS}\n\n**Question:** {question}"

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )

        return self._parse_response(response)

    def _parse_response(self, response: types.GenerateContentResponse) -> GroundedResponse:
        """Parse a Gemini response into a GroundedResponse.

        Parameters
        ----------
        response : types.GenerateContentResponse
            The raw Gemini response.

        Returns
        -------
        GroundedResponse
            Parsed response with metadata.
        """
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

    def format_answer(self, response: GroundedResponse) -> str:
        """Format a grounded response for display.

        Parameters
        ----------
        response : GroundedResponse
            The grounded response to format.

        Returns
        -------
        str
            Formatted response with citations.
        """
        return self.grounding_tool.format_response_with_citations(response)


class AsyncClientManager:
    """Manages async client lifecycle with lazy initialization and cleanup.

    This class ensures clients are created only once and properly closed,
    preventing resource warnings from unclosed event loops.

    Parameters
    ----------
    config : KnowledgeAgentConfig, optional
        Configuration object for client setup. If not provided, creates default.

    Examples
    --------
    >>> manager = AsyncClientManager()
    >>> agent = manager.agent
    >>> response = await agent.answer_async("What is quantum computing?")
    >>> print(response.text)
    """

    def __init__(self, config: KnowledgeAgentConfig | None = None) -> None:
        """Initialize the client manager.

        Parameters
        ----------
        config : KnowledgeAgentConfig, optional
            Configuration object. If not provided, creates default config.
        """
        self._config = config
        self._agent: KnowledgeGroundedAgent | None = None
        self._grounding_tool: GeminiGroundingTool | None = None
        self._initialized = False

    @property
    def config(self) -> KnowledgeAgentConfig:
        """Get or create the config instance.

        Returns
        -------
        KnowledgeAgentConfig
            The configuration settings.
        """
        if self._config is None:
            self._config = KnowledgeAgentConfig()  # type: ignore[call-arg]
        return self._config

    @property
    def grounding_tool(self) -> GeminiGroundingTool:
        """Get or create the Gemini grounding tool.

        Returns
        -------
        GeminiGroundingTool
            The grounding tool instance.
        """
        if self._grounding_tool is None:
            self._grounding_tool = GeminiGroundingTool(config=self.config)
            self._initialized = True
        return self._grounding_tool

    @property
    def agent(self) -> KnowledgeGroundedAgent:
        """Get or create the knowledge-grounded agent.

        Returns
        -------
        KnowledgeGroundedAgent
            The knowledge-grounded QA agent.
        """
        if self._agent is None:
            self._agent = KnowledgeGroundedAgent(config=self.config)
            self._initialized = True
        return self._agent

    def close(self) -> None:
        """Close all initialized clients and reset state."""
        self._agent = None
        self._grounding_tool = None
        self._initialized = False

    def is_initialized(self) -> bool:
        """Check if any clients have been initialized.

        Returns
        -------
        bool
            True if any clients have been initialized.
        """
        return self._initialized

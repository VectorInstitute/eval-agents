"""Knowledge-grounded QA agent using Google ADK with Google Search.

This package provides tools for building and evaluating knowledge-grounded
question answering agents using Google ADK with explicit Google Search tool calls.

Example
-------
>>> from aieng.agent_evals.knowledge_qa import (
...     KnowledgeGroundedAgent,
...     DeepSearchQADataset,
... )
>>> agent = KnowledgeGroundedAgent()
>>> response = agent.answer("What is the current population of Tokyo?")
>>> print(response.text)
"""

from aieng.agent_evals.tools import (
    GroundedResponse,
    GroundingChunk,
    create_google_search_tool,
    format_response_with_citations,
)

from .agent import KnowledgeAgentManager, KnowledgeGroundedAgent
from .data import DeepSearchQADataset, DSQAExample


__all__ = [
    # Agent
    "KnowledgeGroundedAgent",
    "KnowledgeAgentManager",
    # Grounding tool
    "create_google_search_tool",
    "format_response_with_citations",
    "GroundedResponse",
    "GroundingChunk",
    # Evaluation
    "DeepSearchQADataset",
    "DSQAExample",
]

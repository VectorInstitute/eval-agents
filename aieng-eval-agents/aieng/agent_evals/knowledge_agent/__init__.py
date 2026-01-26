"""Knowledge-grounded QA agent using Gemini with Google Search grounding.

This package provides tools for building and evaluating knowledge-grounded
question answering agents using Gemini's built-in Google Search capability.

Example
-------
>>> from aieng.agent_evals.knowledge_agent import (
...     KnowledgeGroundedAgent,
...     DeepSearchQADataset,
...     DeepSearchQAEvaluator,
... )
>>> agent = KnowledgeGroundedAgent()
>>> response = agent.answer("What is the current population of Tokyo?")
>>> print(response.text)
"""

from .agent import AsyncClientManager, KnowledgeGroundedAgent
from .config import KnowledgeAgentConfig
from .evaluation import (
    DeepSearchQADataset,
    DeepSearchQAEvaluator,
    DSQAExample,
    EvaluationResult,
)
from .grounding_tool import GeminiGroundingTool, GroundedResponse, GroundingChunk
from .session import ConversationSession, Message, get_or_create_session


__all__ = [
    # Agent
    "KnowledgeGroundedAgent",
    "AsyncClientManager",
    # Config
    "KnowledgeAgentConfig",
    # Grounding tool
    "GeminiGroundingTool",
    "GroundedResponse",
    "GroundingChunk",
    # Session management
    "ConversationSession",
    "Message",
    "get_or_create_session",
    # Evaluation
    "DeepSearchQADataset",
    "DeepSearchQAEvaluator",
    "DSQAExample",
    "EvaluationResult",
]

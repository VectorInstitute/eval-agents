"""Knowledge-grounded QA agent using Google ADK with Google Search.

This package provides tools for building and evaluating knowledge-grounded
question answering agents using Google ADK with explicit Google Search tool calls.

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

from aieng.agent_evals.configs import Configs

from .agent import KnowledgeAgentManager, KnowledgeGroundedAgent
from .evaluation import (
    DeepSearchQADataset,
    DeepSearchQAEvaluator,
    DSQAExample,
    EvaluationResult,
)
from .grounding_tool import (
    GroundedResponse,
    GroundingChunk,
    create_google_search_tool,
    format_response_with_citations,
)
from .session import ConversationSession, Message, get_or_create_session
from .tracing import flush_traces, init_tracing, is_tracing_enabled


__all__ = [
    # Agent
    "KnowledgeGroundedAgent",
    "KnowledgeAgentManager",
    # Config
    "Configs",
    # Grounding tool
    "create_google_search_tool",
    "format_response_with_citations",
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
    # Tracing
    "init_tracing",
    "is_tracing_enabled",
    "flush_traces",
]

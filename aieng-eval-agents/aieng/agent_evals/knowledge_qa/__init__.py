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
    create_fetch_file_tool,
    create_google_search_tool,
    create_grep_file_tool,
    create_read_file_tool,
    create_web_fetch_tool,
    fetch_file,
    format_response_with_citations,
    grep_file,
    read_file,
    web_fetch,
)

from .agent import AgentResponse, KnowledgeAgentManager, KnowledgeGroundedAgent, StepExecution
from .data import DeepSearchQADataset, DSQAExample
from .deepsearchqa_grader import DeepSearchQAResult, evaluate_deepsearchqa_async
from .notebook import run_with_display
from .plan_parsing import ResearchPlan, ResearchStep, StepStatus


__all__ = [
    # Agent
    "KnowledgeGroundedAgent",
    "AgentResponse",
    "KnowledgeAgentManager",
    # Grounding tool
    "create_google_search_tool",
    "format_response_with_citations",
    "GroundedResponse",
    "GroundingChunk",
    # Planning (data models)
    "ResearchPlan",
    "ResearchStep",
    "StepExecution",
    "StepStatus",
    # Evaluation
    "DeepSearchQAResult",
    "evaluate_deepsearchqa_async",
    "DeepSearchQADataset",
    "DSQAExample",
    # Notebook
    "run_with_display",
    # Web tools
    "web_fetch",
    "create_web_fetch_tool",
    # File tools
    "fetch_file",
    "grep_file",
    "read_file",
    "create_fetch_file_tool",
    "create_grep_file_tool",
    "create_read_file_tool",
]

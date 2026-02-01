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

from .agent import (
    EnhancedGroundedResponse,
    KnowledgeAgentManager,
    KnowledgeGroundedAgent,
)
from .evaluation import (
    DeepSearchQADataset,
    DeepSearchQAEvaluator,
    DSQAExample,
    EvaluationResult,
)
from .judges import (
    BaseJudge,
    CausalChainJudge,
    ComprehensivenessJudge,
    DeepSearchQAJudge,
    DeepSearchQAResult,
    ExhaustivenessJudge,
    JudgeResult,
    PlanQualityJudge,
    SourceQualityJudge,
)
from .metrics import (
    EnhancedEvaluationResult,
    EvaluationMetrics,
    MetricsAggregator,
)
from .notebook import run_with_display
from .planner import (
    ResearchPlan,
    ResearchPlanner,
    ResearchStep,
    StepExecution,
    StepStatus,
)
from .tracing import flush_traces, init_tracing, is_tracing_enabled


__all__ = [
    # Agent
    "KnowledgeGroundedAgent",
    "EnhancedGroundedResponse",
    "KnowledgeAgentManager",
    # Config
    "Configs",
    # Grounding tool
    "create_google_search_tool",
    "format_response_with_citations",
    "GroundedResponse",
    "GroundingChunk",
    # Planning
    "ResearchPlanner",
    "ResearchPlan",
    "ResearchStep",
    "StepExecution",
    "StepStatus",
    # Judges
    "BaseJudge",
    "JudgeResult",
    "ComprehensivenessJudge",
    "CausalChainJudge",
    "ExhaustivenessJudge",
    "SourceQualityJudge",
    "PlanQualityJudge",
    "DeepSearchQAJudge",
    "DeepSearchQAResult",
    # Metrics
    "MetricsAggregator",
    "EvaluationMetrics",
    "EnhancedEvaluationResult",
    # Evaluation
    "DeepSearchQADataset",
    "DeepSearchQAEvaluator",
    "DSQAExample",
    "EvaluationResult",
    # Tracing
    "init_tracing",
    "is_tracing_enabled",
    "flush_traces",
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

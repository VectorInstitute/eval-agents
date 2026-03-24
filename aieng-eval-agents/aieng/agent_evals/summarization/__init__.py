"""Financial news summarization agent using Google ADK.

This package provides a single-pass summarization agent that accepts a news
article title and body and returns a concise summary using a Gemini model.

Example
-------
>>> from aieng.agent_evals.summarization import SummarizationAgent
>>> agent = SummarizationAgent()
>>> response = agent.summarize(title="Apple reports record profits", body="Apple Inc. reported...")
>>> print(response.text)
"""

from .agent import SummarizationAgent, SummarizationAgentManager, SummarizationResponse


__all__ = [
    "SummarizationAgent",
    "SummarizationAgentManager",
    "SummarizationResponse",
]

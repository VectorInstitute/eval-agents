"""ADK discovery entrypoint for the Summarization agent.

Exposes a module-level ``root_agent`` so ``adk web`` can discover it.

Examples
--------
Run with ``adk web``:
    uv run adk web --port 8000 --reload --reload_agents implementations/
"""

import logging

from aieng.agent_evals.summarization.agent import SummarizationAgent


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ADK discovery expects a module-level `root_agent`
root_agent = SummarizationAgent().adk_agent

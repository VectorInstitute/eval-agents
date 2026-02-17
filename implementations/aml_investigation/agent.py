"""ADK discovery entrypoint for the AML investigation demo.

It exposes a module-level ``root_agent`` so ``adk web`` can discover it.

Examples
--------
Run with ``adk web``:
    uv run adk web --port 8000 --reload --reload_agents implementations/
"""

import logging

from aieng.agent_evals.aml_investigation.agent import create_aml_investigation_agent


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ADK discovery expects a module-level `root_agent`
root_agent = create_aml_investigation_agent(enable_tracing=True)

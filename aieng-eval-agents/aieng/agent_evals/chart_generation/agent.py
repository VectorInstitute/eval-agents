"""Definitions for the chart generation agent."""

import logging
from pathlib import Path

from aieng.agent_evals.async_client_manager import AsyncClientManager

# from aieng.agent_evals.chart_generation.chart_writer import ChartFileWriter
from aieng.agent_evals.db_manager import DbManager
from aieng.agent_evals.langfuse import init_tracing
from aieng.agent_evals.report_generation.agent import (  # reuse from report generation agent
    EventParser,
    EventType,
    ParsedEvent,
)
from google.adk.agents import Agent
from google.adk.agents.base_agent import AfterAgentCallback


logger = logging.getLogger(__name__)

# Re-export so callers can import from here without knowing the source module.
__all__ = ["get_chart_generation_agent", "EventParser", "EventType", "ParsedEvent"]


def get_chart_generation_agent(
    instructions: str,
    charts_output_path: Path,
    after_agent_callback: AfterAgentCallback | None = None,
    langfuse_tracing: bool = True,
) -> Agent:
    """Define the chart generation agent.

    Parameters
    ----------
    instructions : str
        System instructions for the agent.
    charts_output_path : Path
        Directory where PNG chart files will be saved.
    after_agent_callback : AfterAgentCallback | None, optional
        Callback invoked after the agent finishes.
    langfuse_tracing : bool, optional
        Whether to initialise Langfuse OpenTelemetry tracing.

    Returns
    -------
    Agent
        The configured chart generation agent.
    """
    agent_name = "ChartGenerationAgent"

    if langfuse_tracing:
        init_tracing(service_name=agent_name)

    client_manager = AsyncClientManager.get_instance()
    db_manager = DbManager.get_instance()
    # chart_writer = ChartFileWriter(charts_output_path)

    return Agent(
        name=agent_name,
        model=client_manager.configs.default_worker_model,
        instruction=instructions,
        tools=[
            db_manager.chart_generation_db().get_schema_info,
            db_manager.chart_generation_db().execute,
            # chart_writer.write_chart,
        ],
        after_agent_callback=after_agent_callback,
    )

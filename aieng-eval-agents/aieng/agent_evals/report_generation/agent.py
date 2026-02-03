"""
Definitions for the the report generation agent.

Example
-------
>>> from aieng.agent_evals.report_generation.agent import get_report_generation_agent
>>> from aieng.agent_evals.report_generation.prompts import MAIN_AGENT_INSTRUCTIONS
>>> agent = get_report_generation_agent(
>>>     instructions=MAIN_AGENT_INSTRUCTIONS,
>>>     sqlite_db_path=Path("data/OnlineRetail.db"),
>>>     reports_output_path=Path("reports/"),
>>>     langfuse_project_name="Report Generation",
>>> )
"""

from pathlib import Path

import agents
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.langfuse import setup_langfuse_tracer
from aieng.agent_evals.report_generation.file_writer import ReportFileWriter


def get_report_generation_agent(
    instructions: str,
    sqlite_db_path: Path,
    reports_output_path: Path,
    langfuse_project_name: str | None,
) -> agents.Agent:
    """
    Define the report generation agent.

    Parameters
    ----------
    instructions : str
        The instructions for the agent.
    sqlite_db_path : Path
        The path to the SQLite database.
    reports_output_path : Path
        The path to the reports output directory.
    langfuse_project_name : str | None
        The name of the Langfuse project to use for tracing.

    Returns
    -------
    agents.Agent
        The report generation agent.
    """
    # Setup langfuse tracing if project name is provided
    if langfuse_project_name:
        setup_langfuse_tracer(langfuse_project_name)

    # Get the client manager singleton instance
    client_manager = AsyncClientManager.get_instance()
    report_file_writer = ReportFileWriter(reports_output_path)

    # Define an agent using the OpenAI Agent SDK
    return agents.Agent(
        name="Report Generation Agent",  # Agent name for logging and debugging purposes
        instructions=instructions,  # System instructions for the agent
        # Tools available to the agent
        # We wrap the `execute_sql_query` and `write_report_to_file` methods
        # with `function_tool`, which will construct the tool definition JSON
        # schema by extracting the necessary information from the method
        # signature and docstring.
        tools=[
            agents.function_tool(
                client_manager.sqlite_connection(sqlite_db_path).execute,
                name_override="execute_sql_query",
                description_override="Execute a SQL query against the SQLite database.",
            ),
            agents.function_tool(
                report_file_writer.write,
                name_override="write_report_to_file",
                description_override="Write the report data to a downloadable XLSX file.",
            ),
        ],
        model=agents.OpenAIChatCompletionsModel(
            model=client_manager.configs.default_worker_model,
            openai_client=client_manager.openai_client,
        ),
    )

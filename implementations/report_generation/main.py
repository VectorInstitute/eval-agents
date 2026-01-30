"""Reason-and-Act Knowledge Retrieval Agent via the OpenAI Agent SDK."""

import asyncio
import logging
import os
from functools import partial
from pathlib import Path
from typing import Any, AsyncGenerator

import agents
import click
import gradio as gr
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.langfuse import setup_langfuse_tracer
from aieng.agent_evals.utils import get_or_create_session, oai_agent_stream_to_gradio_messages
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage

from implementations.report_generation.file_writer import get_reports_output_path, write_report_to_file
from implementations.report_generation.prompts import MAIN_AGENT_INSTRUCTIONS


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


LANGFUSE_PROJECT_NAME = "Report Generation"


def get_sqlite_db_path() -> Path:
    """Get the SQLite database path.

    If no path is provided in the REPORT_GENERATION_DB_PATH env var, will use the
    default path in `implementations/report_generation/data/OnlineRetail.db`.

    Returns
    -------
    Path
        The SQLite database path.
    """
    default_sqlite_db_path = "implementations/report_generation/data/OnlineRetail.db"
    return Path(os.getenv("REPORT_GENERATION_DB_PATH", default_sqlite_db_path))


def get_report_generation_agent(enable_trace: bool = True) -> agents.Agent:
    """
    Define the report generation agent.

    Parameters
    ----------
    enable_trace : bool, optional
        Whether to enable tracing with Langfuse for evaluation purposes.
        Default is True.

    Returns
    -------
    agents.Agent
        The report generation agent.
    """
    # Setup langfuse tracing if enabled
    if enable_trace:
        setup_langfuse_tracer(LANGFUSE_PROJECT_NAME)

    # Get the client manager singleton instance
    client_manager = AsyncClientManager.get_instance()

    # Define an agent using the OpenAI Agent SDK
    return agents.Agent(
        name="Report Generation Agent",  # Agent name for logging and debugging purposes
        instructions=MAIN_AGENT_INSTRUCTIONS,  # System instructions for the agent
        # Tools available to the agent
        # We wrap the `search_knowledgebase` method with `function_tool`, which
        # will construct the tool definition JSON schema by extracting the necessary
        # information from the method signature and docstring.
        tools=[
            agents.function_tool(
                client_manager.sqlite_connection(get_sqlite_db_path()).execute,
                name_override="execute_sql_query",
                description_override="Execute a SQL query against the SQLite database.",
            ),
            agents.function_tool(
                write_report_to_file,
                description_override="Write the report data to a file.",
            ),
        ],
        model=agents.OpenAIChatCompletionsModel(
            model=client_manager.configs.default_worker_model,
            openai_client=client_manager.openai_client,
        ),
    )


async def agent_session_handler(
    query: str,
    history: list[ChatMessage],
    session_state: dict[str, Any],
    enable_trace: bool = True,
) -> AsyncGenerator[list[ChatMessage], Any]:
    """Handle the agent session.

    Parameters
    ----------
    query : str
        The query to the agent.
    history : list[ChatMessage]
        The history of the conversation.
    session_state : dict[str, Any]
        The currentsession state.
    enable_trace : bool, optional
        Whether to enable tracing with Langfuse for evaluation purposes.
        Default is True.

    Returns
    -------
    AsyncGenerator[list[ChatMessage], Any]
        An async chat messages generator.
    """
    # Initialize list of chat messages for a single turn
    turn_messages: list[ChatMessage] = []

    # Construct an in-memory SQLite session for the agent to maintain
    # conversation history across multiple turns of a chat
    # This makes it possible to ask follow-up questions that refer to
    # previous turns in the conversation
    session = get_or_create_session(history, session_state)

    main_agent = get_report_generation_agent(enable_trace=enable_trace)

    # Run the agent in streaming mode to get and display intermediate outputs
    result_stream = agents.Runner.run_streamed(main_agent, input=query, session=session)

    async for _item in result_stream.stream_events():
        # Parse the stream events, convert to Gradio chat messages and append to
        # the chat history
        turn_messages += oai_agent_stream_to_gradio_messages(_item)
        if len(turn_messages) > 0:
            yield turn_messages


@click.command()
@click.option("--enable-trace", required=False, default=True, help="Whether to enable tracing with Langfuse.")
@click.option(
    "--enable-public-link",
    required=False,
    default=False,
    help="Whether to enable public link for the Gradio app.",
)
def start_gradio_app(enable_trace: bool = True, enable_public_link: bool = False) -> None:
    """Start the Gradio app with the agent session handler.

    Parameters
    ----------
    enable_trace : bool, optional
        Whether to enable tracing with Langfuse for evaluation purposes.
        Default is True.
    enable_public_link : bool, optional
        Whether to enable public link for the Gradio app. If True,
        will make the Gradio app available at a public URL. Default is False.
    """
    partial_agent_session_handler = partial(agent_session_handler, enable_trace=enable_trace)

    demo = gr.ChatInterface(
        partial_agent_session_handler,
        chatbot=gr.Chatbot(height=600),
        textbox=gr.Textbox(lines=1, placeholder="Enter your prompt"),
        # Additional input to maintain session state across multiple turns
        # NOTE: Examples must be a list of lists when additional inputs are provided
        additional_inputs=gr.State(value={}, render=False),
        examples=[
            ["Generate a monthly sales performance report."],
            ["Generate a report of the top 5 selling products per year and the total sales value for each product."],
            ["Generate a report of the average order value per invoice per month."],
            [
                "Generate a report with the month-over-month trends in sales. The report should include the monthly sales, the month-over-month change and the percentage change."
            ],
            ["Generate a report on sales revenue by country per year."],
            ["Generate a report on the 5 highest-value customers per year vs. the average customer."],
            [
                "Generate a report on the average amount spent by one time buyers for each year vs. the average customer."
            ],
        ],
        title="2.1: ReAct for Retrieval-Augmented Generation with OpenAI Agent SDK",
    )

    try:
        demo.launch(
            share=enable_public_link,
            allowed_paths=[str(get_reports_output_path().absolute())],
        )
    finally:
        asyncio.run(AsyncClientManager.get_instance().close())


if __name__ == "__main__":
    start_gradio_app()

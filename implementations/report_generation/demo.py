"""
Demo UI for the report generation agent.

Example
-------
$ python -m implementations.report_generation.demo
"""

import asyncio
import logging
from functools import partial
from typing import Any, AsyncGenerator

import agents
import click
import gradio as gr
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.report_generation.agent import get_report_generation_agent
from aieng.agent_evals.report_generation.prompts import MAIN_AGENT_INSTRUCTIONS
from aieng.agent_evals.utils import get_or_create_session, oai_agent_stream_to_gradio_messages
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage

from implementations.report_generation.env_vars import (
    get_langfuse_project_name,
    get_reports_output_path,
    get_sqlite_db_path,
)


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


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

    main_agent = get_report_generation_agent(
        instructions=MAIN_AGENT_INSTRUCTIONS,
        sqlite_db_path=get_sqlite_db_path(),
        reports_output_path=get_reports_output_path(),
        langfuse_project_name=get_langfuse_project_name() if enable_trace else None,
    )

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

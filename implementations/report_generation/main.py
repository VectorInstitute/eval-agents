"""Reason-and-Act Knowledge Retrieval Agent via the OpenAI Agent SDK."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, AsyncGenerator

import agents
import gradio as gr
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.utils import (
    get_or_create_session,
    oai_agent_stream_to_gradio_messages,
)
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage

from implementations.report_generation.file_writer import get_reports_output_path, write_report_to_file


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


REACT_INSTRUCTIONS = """\
Perform the task using the SQLite database tool. \
EACH TIME before invoking the function, you must explain your reasons for doing so. \
If the SQL query did not return intended results, try again. \
For best performance, divide complex queries into simpler sub-queries. \
Do not make up information. \
When the report is done, use the report file writer tool to write it to a file. \
At the end, provide the report file as a downloadable hyperlink to the user.
"""


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


async def agent_session_handler(
    query: str,
    history: list[ChatMessage],
    session_state: dict[str, Any],
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

    # Get the client manager singleton instance
    client_manager = AsyncClientManager.get_instance()

    # Define an agent using the OpenAI Agent SDK
    main_agent = agents.Agent(
        name="Report Generation Agent",  # Agent name for logging and debugging purposes
        instructions=REACT_INSTRUCTIONS,  # System instructions for the agent
        # Tools available to the agent
        # We wrap the `search_knowledgebase` method with `function_tool`, which
        # will construct the tool definition JSON schema by extracting the necessary
        # information from the method signature and docstring.
        tools=[
            agents.function_tool(client_manager.sqlite_connection(get_sqlite_db_path()).execute),
            agents.function_tool(write_report_to_file),
        ],
        model=agents.OpenAIChatCompletionsModel(
            model=client_manager.configs.default_worker_model,
            openai_client=client_manager.openai_client,
        ),
    )

    # Run the agent in streaming mode to get and display intermediate outputs
    result_stream = agents.Runner.run_streamed(main_agent, input=query, session=session)

    async for _item in result_stream.stream_events():
        # Parse the stream events, convert to Gradio chat messages and append to
        # the chat history
        turn_messages += oai_agent_stream_to_gradio_messages(_item)
        if len(turn_messages) > 0:
            yield turn_messages


def start_gradio_app(enable_public_link: bool = False) -> None:
    """Start the Gradio app with the agent session handler.

    Parameters
    ----------
    enable_public_link : bool, optional
        Whether to enable public link for the Gradio app. If True,
        will make the Gradio app available at a public URL. Default is False.
    """
    demo = gr.ChatInterface(
        agent_session_handler,
        chatbot=gr.Chatbot(height=600),
        textbox=gr.Textbox(lines=1, placeholder="Enter your prompt"),
        # Additional input to maintain session state across multiple turns
        # NOTE: Examples must be a list of lists when additional inputs are provided
        additional_inputs=gr.State(value={}, render=False),
        examples=[
            ["Generate a monthly sales performance report for the last year with available data."],
            ["Generate a report of the top 5 selling products per year and the total sales for each product."],
            ["Generate a report of the average order value per invoice per month."],
            ["Generate a report with the month-over-month trends in sales."],
            ["Generate a report on sales revenue by country per year."],
            ["Generate a report on the 5 highest-value customers per year vs. the average customer."],
            [
                "Generate a report on the average amount spent by one time buyers for each year vs. the average customer."
            ],
            ["Generate a report on the daily, weekly and monthly sales trends."],
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
    start_gradio_app(enable_public_link=False)

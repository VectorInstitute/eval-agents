"""
Demo UI for the report generation agent.

Example
-------
$ python -m implementations.report_generation.demo
"""

import asyncio
import logging
import threading
from functools import partial
from typing import Any, AsyncGenerator

import click
import gradio as gr
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.db_manager import DbManager
from aieng.agent_evals.langfuse import report_usage_scores
from aieng.agent_evals.report_generation.agent import get_report_generation_agent
from aieng.agent_evals.report_generation.evaluation.online import report_final_response_score
from aieng.agent_evals.report_generation.prompts import MAIN_AGENT_INSTRUCTIONS
from dotenv import load_dotenv
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from gradio.components.chatbot import ChatMessage

from implementations.report_generation.env_vars import get_langfuse_project_name, get_reports_output_path
from implementations.report_generation.gradio_utils import agent_event_to_gradio_messages


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


GRADIO_STATE = gr.State(value={"trace_id": None})


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
    # Reset the trace ID in the state
    GRADIO_STATE.value["trace_id"] = None

    # Initialize list of chat messages for a single turn
    turn_messages: list[ChatMessage] = []

    main_agent = get_report_generation_agent(
        instructions=MAIN_AGENT_INSTRUCTIONS,
        reports_output_path=get_reports_output_path(),
        langfuse_project_name=get_langfuse_project_name() if enable_trace else None,
        after_agent_callback=calculate_and_send_scores,
    )

    # Construct an in-memory session for the agent to maintain
    # conversation history across multiple turns of a chat
    # This makes it possible to ask follow-up questions that refer to
    # previous turns in the conversation
    session_service = InMemorySessionService()
    runner = Runner(app_name=main_agent.name, agent=main_agent, session_service=session_service)
    current_session = await session_service.create_session(
        app_name=main_agent.name,
        user_id="user",
        state={},
    )

    # create the user message
    content = Content(role="user", parts=[Part(text=query)])

    # Run the agent in streaming mode to get and display intermediate outputs
    async for event in runner.run_async(
        user_id="user",
        session_id=current_session.id,
        new_message=content,
    ):
        # Parse the stream events, convert to Gradio chat messages and append to
        # the chat history
        turn_messages += agent_event_to_gradio_messages(event)
        if len(turn_messages) > 0:
            yield turn_messages


def calculate_and_send_scores(callback_context: CallbackContext) -> None:
    """Calculate token usage and latency scores and submit them to Langfuse.

    This is a callback function to be called after the agent has run.

    Parameters
    ----------
    callback_context : CallbackContext
        The callback context at the end of the agent run.
    """
    for event in callback_context.session.events:
        if event.is_final_response() and event.content and event.content.role == "model":
            langfuse_client = AsyncClientManager.get_instance().langfuse_client
            trace_id = langfuse_client.get_current_trace_id()

            # Storing the trace ID in the state so it can be used
            # in the feedback buttons callback
            GRADIO_STATE.value["trace_id"] = trace_id

            # Report the final response evaluation to Langfuse
            report_final_response_score(event, string_match="](gradio_api/file=")

            # Run usage scoring in a thread so it doesn't block the UI
            thread = threading.Thread(
                target=report_usage_scores,
                kwargs={
                    "trace_id": trace_id,
                    "token_threshold": 15000,
                    "latency_threshold": 60,
                },
                daemon=True,
            )
            thread.start()

            return

    logger.error("No final response found in the callback context. Will not report scores to Langfuse.")


def on_feedback(liked: bool) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Handle thumbs up (liked=True) or thumbs down (liked=False).

    Send the result of the feedback to Langfuse and returns the updated
    states for the feedback row and the thank you message row.

    Parameters
    ----------
    liked : bool
        Whether the user liked the agent's response.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]] | None
        The updated states for the feedback row and the thank you message row.
        If no trace ID is found in the state, returns None.
    """
    trace_id = GRADIO_STATE.value["trace_id"]
    if trace_id is None:
        logger.error("No trace ID found in the state. Will not report feedback to Langfuse.")
        return None

    score = 1 if liked else 0

    logger.info(f"Reporting user feedback score for trace {trace_id} with value {score}")
    langfuse_client = AsyncClientManager.get_instance().langfuse_client
    langfuse_client.create_score(
        value=score,
        name="User Feedback",
        comment=f"The user gave this response a thumbs {'up' if liked else 'down'}.",
        trace_id=trace_id,
    )
    langfuse_client.flush()

    GRADIO_STATE.value["trace_id"] = None
    return gr.update(visible=False), gr.update(visible=True)


def toggle_feedback_row() -> tuple[dict[str, Any], dict[str, Any]]:
    """Toggle the feedback row if there is a trace ID in the state.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        The updated states for the feedback row and the thank you message row.
    """
    trace_id = GRADIO_STATE.value["trace_id"]
    return gr.update(visible=trace_id is not None and trace_id != ""), gr.update(visible=False)


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

    with gr.Blocks(title="Report Generator Agent") as demo:
        with gr.Row():
            gradio_chatbot = gr.Chatbot(height=600)
            gr.ChatInterface(
                partial_agent_session_handler,
                chatbot=gradio_chatbot,
                textbox=gr.Textbox(lines=1, placeholder="Enter your prompt"),
                # Additional input to maintain session state across multiple turns
                # NOTE: Examples must be a list of lists when additional inputs
                # are provided
                additional_inputs=gr.State(value={}, render=False),
                examples=[
                    ["Generate a monthly sales performance report."],
                    [
                        "Generate a report of the top 5 selling products per year and the total sales value for each product."
                    ],
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
            )

        with gr.Row(elem_id="thank_you_msg", visible=False) as thank_you_row:
            gr.Markdown("Thank you for your feedback 🙂")

        # Feedback buttons
        with gr.Row(elem_id="feedback_buttons", visible=False) as feedback_row:
            gr.Markdown("Provide feedback on the response:")
            thumbs_up = gr.Button("👍")
            thumbs_up.click(fn=lambda: on_feedback(True), outputs=[feedback_row, thank_you_row])
            thumbs_down = gr.Button("👎")
            thumbs_down.click(fn=lambda: on_feedback(False), outputs=[feedback_row, thank_you_row])

        gradio_chatbot.change(fn=toggle_feedback_row, outputs=[feedback_row, thank_you_row])

    try:
        demo.launch(
            share=enable_public_link,
            allowed_paths=[str(get_reports_output_path().absolute())],
            css="#feedback_buttons { width: 600px; }",
        )
    finally:
        DbManager.get_instance().close()
        asyncio.run(AsyncClientManager.get_instance().close())


if __name__ == "__main__":
    start_gradio_app()

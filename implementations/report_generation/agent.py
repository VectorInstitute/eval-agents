"""Entry point for the Google ADK UI for the report generation agent.

Example
-------
$ adk web implementations/
"""

import logging
import threading

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.langfuse import report_usage_scores
from aieng.agent_evals.report_generation.agent import get_report_generation_agent
from aieng.agent_evals.report_generation.evaluation.online import report_final_response_score
from aieng.agent_evals.report_generation.prompts import MAIN_AGENT_INSTRUCTIONS
from dotenv import load_dotenv
from google.adk.agents.callback_context import CallbackContext

from .env_vars import get_langfuse_project_name, get_reports_output_path


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def calculate_and_send_scores(callback_context: CallbackContext) -> None:
    """Calculate token usage and latency scores and submit them to Langfuse.

    This is a callback function to be called after the agent has run.

    Parameters
    ----------
    callback_context : CallbackContext
        The callback context at the end of the agent run.
    """
    langfuse_client = AsyncClientManager.get_instance().langfuse_client
    langfuse_client.flush()

    for event in callback_context.session.events:
        if event.is_final_response() and event.content and event.content.role == "model":
            # Report the final response evaluation to Langfuse
            report_final_response_score(event, string_match="](gradio_api/file=")

            # Run usage scoring in a thread so it doesn't block the UI
            thread = threading.Thread(
                target=report_usage_scores,
                kwargs={
                    "trace_id": langfuse_client.get_current_trace_id(),
                    "token_threshold": 15000,
                    "latency_threshold": 60,
                },
                daemon=True,
            )
            thread.start()

            return

    logger.error("No final response found in the callback context. Will not report scores to Langfuse.")


root_agent = get_report_generation_agent(
    instructions=MAIN_AGENT_INSTRUCTIONS,
    reports_output_path=get_reports_output_path(),
    langfuse_project_name=get_langfuse_project_name(),
    after_agent_callback=calculate_and_send_scores,
)

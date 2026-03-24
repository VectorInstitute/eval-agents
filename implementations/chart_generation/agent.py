"""Entry point for the Google ADK UI for the chart generation agent.

Example
-------
$ adk web implementations/
"""

import logging
import threading

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.chart_generation.agent import get_chart_generation_agent
from aieng.agent_evals.chart_generation.evaluation.online import report_final_response_score
from aieng.agent_evals.chart_generation.prompts import MAIN_AGENT_INSTRUCTIONS
from aieng.agent_evals.langfuse import report_usage_scores
from dotenv import load_dotenv
from google.adk.agents.callback_context import CallbackContext

from .env_vars import get_charts_output_path


load_dotenv(verbose=True)
logger = logging.getLogger(__name__)


def calculate_and_send_scores(callback_context: CallbackContext) -> None:
    """Compute and push token/latency scores to Langfuse after agent completes."""
    langfuse_client = AsyncClientManager.get_instance().langfuse_client
    langfuse_client.flush()

    for event in callback_context.session.events:
        if event.is_final_response() and event.content and event.content.role == "model":
            report_final_response_score(event)
            thread = threading.Thread(
                target=report_usage_scores,
                kwargs={
                    "trace_id": langfuse_client.get_current_trace_id(),
                    "token_threshold": 15000,
                    "latency_threshold": 90,
                },
                daemon=True,
            )
            thread.start()
            return

    logger.error("No final response found; will not report scores.")


root_agent = get_chart_generation_agent(
    instructions=MAIN_AGENT_INSTRUCTIONS,
    charts_output_path=get_charts_output_path(),
    after_agent_callback=calculate_and_send_scores,
)

"""Functions to report online evaluation of the report generation agent to Langfuse."""

import logging
from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.report_generation.agent import EventParser, EventType
from google.adk.events.event import Event
from langfuse import Langfuse
from langfuse.api.resources.commons.types.observations_view import ObservationsView
from langfuse.api.resources.observations.types.observations_views import ObservationsViews
from tenacity import retry, stop_after_attempt, wait_exponential


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def report_final_response_score(event: Event, string_match: str = "") -> None:
    """Report a score to Langfuse if the event is a final response.

    The score will be reported as 1 if the final response is valid
    and contains the string match. Otherwise, the score will be reported as 0.

    This has to be called within the context of a Langfuse trace.

    Parameters
    ----------
    event : Event
        The event to check.
    string_match : str
        The string to match in the final response.
        Optional, default to empty string.

    Raises
    ------
    ValueError
        If the event is not a final response.
    """
    if not event.is_final_response():
        raise ValueError("Event is not a final response")

    langfuse_client = AsyncClientManager.get_instance().langfuse_client
    trace_id = langfuse_client.get_current_trace_id()

    if trace_id is None:
        raise ValueError("Langfuse trace ID is None.")

    logger.info("Reporting score for valid final response")

    parsed_events = EventParser.parse(event)
    for parsed_event in parsed_events:
        if parsed_event.type == EventType.FINAL_RESPONSE:
            if string_match in parsed_event.text:
                score = 1
                comment = "Final response contains the string match."
            else:
                score = 0
                comment = "Final response does not contains the string match."

            logger.info("Reporting score for valid final response")
            langfuse_client.create_score(
                name="Valid Final Response",
                value=score,
                trace_id=trace_id,
                comment=comment,
                metadata={
                    "final_response": parsed_event.text,
                    "string_match": string_match,
                },
            )
            return

    langfuse_client.create_score(
        name="Valid Final Response",
        value=0,
        trace_id=trace_id,
        comment="Final response not found in the event",
        metadata={
            "string_match": string_match,
        },
    )


def report_usage_scores(
    token_threshold: int = 0,
    latency_threshold: int = 0,
    cost_threshold: float = 0,
) -> None:
    """Report usage metrics to Langfuse.

    This function has to be called within the context of a Langfuse trace.

    Parameters
    ----------
    token_threshold: int
        The token threshold to report the metrics for.
        if the token count is greater than the threshold, the score
        will be reported as 0.
        Optional, default to 0 (no reporting).
    latency_threshold: int
        The latency threshold in seconds to report the metrics for.
        if the latency is greater than the threshold, the score
        will be reported as 0.
        Optional, default to 0 (no reporting).
    cost_threshold: float
        The cost threshold to report the metrics for.
        if the cost is greater than the threshold, the score
        will be reported as 0.
        Optional, default to 0 (no reporting).
    """
    langfuse_client = AsyncClientManager.get_instance().langfuse_client
    trace_id = langfuse_client.get_current_trace_id()

    if trace_id is None:
        raise ValueError("Langfuse trace ID is None.")

    observations = _get_observations_with_retry(trace_id, langfuse_client)

    if token_threshold > 0:
        total_tokens = sum(_obs_attr(observation, "totalTokens") for observation in observations.data)
        if total_tokens <= token_threshold:
            score = 1
            comment = "Token count is less than or equal to the threshold."
        else:
            score = 0
            comment = "Token count is greater than the threshold."

        logger.info("Reporting score for token count")
        langfuse_client.create_score(
            name="Token Count",
            value=score,
            trace_id=trace_id,
            comment=comment,
            metadata={
                "total_tokens": total_tokens,
                "token_threshold": token_threshold,
            },
        )

    if latency_threshold > 0:
        total_latency = sum(_obs_attr(observation, "latency") for observation in observations.data)
        if total_latency <= latency_threshold:
            score = 1
            comment = "Latency is less than or equal to the threshold."
        else:
            score = 0
            comment = "Latency is greater than the threshold."

        logger.info("Reporting score for latency")
        langfuse_client.create_score(
            name="Latency",
            value=score,
            trace_id=trace_id,
            comment=comment,
            metadata={
                "total_latency": total_latency,
                "latency_threshold": latency_threshold,
            },
        )

    if cost_threshold > 0:
        total_cost = sum(_obs_attr(observation, "calculated_total_cost") for observation in observations.data)
        if total_cost <= cost_threshold:
            score = 1
            comment = "Cost is less than or equal to the threshold."
        else:
            score = 0
            comment = "Cost is greater than the threshold."

        logger.info("Reporting score for cost")
        langfuse_client.create_score(
            name="Cost",
            value=score,
            trace_id=trace_id,
            comment=comment,
            metadata={
                "total_cost": total_cost,
                "cost_threshold": cost_threshold,
            },
        )

    langfuse_client.flush()


def _obs_attr(observation: ObservationsView, attribute: str) -> Any:
    attribute_value = getattr(observation, attribute)
    if attribute_value == 0:
        logger.error(f"Observation attribute value for {attribute} is 0")
        return 0
    if attribute_value is None:
        logger.error(f"Observation attribute value for {attribute} is None")
        return 0
    return attribute_value


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=5, max=15))
def _get_observations_with_retry(trace_id: str, langfuse_client: Langfuse) -> ObservationsViews:
    logger.info(f"Getting observations for trace {trace_id}...")
    return langfuse_client.api.observations.get_many(trace_id=trace_id, type="GENERATION")

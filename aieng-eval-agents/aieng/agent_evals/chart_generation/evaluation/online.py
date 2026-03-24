"""Functions to report online evaluation of the report generation agent to Langfuse."""

import logging

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.report_generation.agent import EventParser, EventType
from google.adk.events.event import Event


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
            langfuse_client.flush()
            return

    logger.info("Reporting score for invalid final response")
    langfuse_client.create_score(
        name="Valid Final Response",
        value=0,
        trace_id=trace_id,
        comment="Final response not found in the event",
        metadata={
            "string_match": string_match,
        },
    )
    langfuse_client.flush()

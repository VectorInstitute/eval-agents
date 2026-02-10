"""Functions to report online evaluation of the report generation agent to Langfuse."""

from aieng.agent_evals.report_generation.agent import EventParser, EventType
from google.adk.events.event import Event
from langfuse import Langfuse


def report_if_final_response(event: Event, langfuse_client: Langfuse, string_match: str = "") -> None:
    """Report a score to Langfuse if the event is a final response.

    The score will be reported as 1 if the final response is valid
    and contains the string match. Otherwise, the score will be reported as 0.

    Parameters
    ----------
    event : Event
        The event to check.
    langfuse_client : Langfuse
        The Langfuse client to use.
    string_match : str
        The string to match in the final response.
        Optional, default to empty string.
    """
    trace_id = langfuse_client.get_current_trace_id()

    if event.is_final_response():
        parsed_events = EventParser.parse(event)
        for parsed_event in parsed_events:
            if parsed_event.type == EventType.FINAL_RESPONSE:
                if string_match in parsed_event.text:
                    langfuse_client.create_score(
                        name="Valid Final Response",
                        value=1,
                        trace_id=trace_id,
                        comment="Final response contains the string match.",
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
                    comment="Final response does not contains the string match.",
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

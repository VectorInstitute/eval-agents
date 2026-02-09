"""Utility functions for the report generation agent."""

import logging

from aieng.agent_evals.report_generation.agent import EventParser, EventType
from google.adk.events.event import Event
from gradio.components.chatbot import ChatMessage, MetadataDict


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def agent_event_to_gradio_messages(event: Event) -> list[ChatMessage]:
    """Parse a Google ADK Event into a list of gr messages.

    Adds extra data for tool use to make the gradio display informative.

    Parameters
    ----------
    event : Event
        The event from the Google ADK's agent response.

    Returns
    -------
    list[ChatMessage]
        A list of Gradio chat messages parsed from the stream event.
    """
    output: list[ChatMessage] = []

    parsed_events = EventParser.parse(event)

    for parsed_event in parsed_events:
        if parsed_event.type == EventType.FINAL_RESPONSE:
            output.append(
                ChatMessage(
                    role="assistant",
                    content=parsed_event.text,
                    metadata=MetadataDict(),
                )
            )
        elif parsed_event.type == EventType.TOOL_CALL:
            output.append(
                ChatMessage(
                    role="assistant",
                    content=f"```\n{parsed_event.arguments}\n```",
                    metadata={"title": f"🛠️ Used tool `{parsed_event.text}`"},
                )
            )
        elif parsed_event.type == EventType.THOUGHT:
            output.append(
                ChatMessage(
                    role="assistant",
                    content=parsed_event.text or "",
                    metadata={"title": "🧠 Thought"},
                )
            )
        elif parsed_event.type == EventType.TOOL_RESPONSE:
            output.append(
                ChatMessage(
                    role="assistant",
                    content=f"```\n{parsed_event.arguments}\n```",
                    metadata={
                        "title": f"*Tool call output: `{parsed_event.text}`*",
                        "status": "done",  # This makes it collapsed by default
                    },
                )
            )

    return output

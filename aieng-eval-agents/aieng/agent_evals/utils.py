"""Utility functions for the report generation agent."""

import uuid
from typing import Any

from agents import SQLiteSession, StreamEvent, stream_events
from agents.items import ToolCallOutputItem
from gradio.components.chatbot import ChatMessage, MetadataDict
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputText
from openai.types.responses.response_completed_event import ResponseCompletedEvent
from openai.types.responses.response_output_message import ResponseOutputMessage


def oai_agent_stream_to_gradio_messages(stream_event: StreamEvent) -> list[ChatMessage]:
    """Parse agent sdk "stream event" into a list of gr messages.

    Adds extra data for tool use to make the gradio display informative.
    """
    output: list[ChatMessage] = []

    if isinstance(stream_event, stream_events.RawResponsesStreamEvent):
        data = stream_event.data
        if isinstance(data, ResponseCompletedEvent):
            # The completed event may contain multiple output messages,
            # including tool calls and final outputs.
            # If there is at least one tool call, we mark the response as a thought.
            is_thought = len(data.response.output) > 1 and any(
                isinstance(message, ResponseFunctionToolCall) for message in data.response.output
            )

            for message in data.response.output:
                if isinstance(message, ResponseOutputMessage):
                    for _item in message.content:
                        if isinstance(_item, ResponseOutputText):
                            output.append(
                                ChatMessage(
                                    role="assistant",
                                    content=_item.text,
                                    metadata={
                                        "title": "🧠 Thought",
                                        "id": data.sequence_number,
                                    }
                                    if is_thought
                                    else MetadataDict(),
                                )
                            )
                elif isinstance(message, ResponseFunctionToolCall):
                    output.append(
                        ChatMessage(
                            role="assistant",
                            content=f"```\n{message.arguments}\n```",
                            metadata={
                                "title": f"🛠️ Used tool `{message.name}`",
                            },
                        )
                    )

    elif isinstance(stream_event, stream_events.RunItemStreamEvent):
        name = stream_event.name
        item = stream_event.item

        if name == "tool_output" and isinstance(item, ToolCallOutputItem):
            output.append(
                ChatMessage(
                    role="assistant",
                    content=f"```\n{item.output}\n```",
                    metadata={
                        "title": "*Tool call output*",
                        "status": "done",  # This makes it collapsed by default
                    },
                )
            )

    return output


def get_or_create_session(
    history: list[ChatMessage],
    session_state: dict[str, Any],
) -> SQLiteSession:
    """Get existing session or create a new one for conversation persistence.

    Args:
        history: The history of the conversation.
        session_state: The state of the session.

    Returns
    -------
        The session.
    """
    if len(history) == 0:
        session = SQLiteSession(session_id=str(uuid.uuid4()))
        session_state["session"] = session
    else:
        session = session_state["session"]
    return session

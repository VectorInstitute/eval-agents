"""Functions and objects pertaining to Langfuse."""

import logging
from enum import Enum
from typing import Any

from agents.items import ToolCallOutputItem
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from langfuse import Langfuse  # type: ignore[attr-defined]
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputText
from openai.types.responses.response_completed_event import ResponseCompletedEvent
from openai.types.responses.response_output_message import ResponseOutputMessage
from pydantic import BaseModel
from rich.progress import Progress, SpinnerColumn, TextColumn


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class LangFuseTraceType(Enum):
    """Type of Langfuse trace messages."""

    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    FINAL_ANSWER = "final_answer"


class LangFuseMessage(BaseModel):
    """Content of a Langfuse trace message."""

    content: str
    type: LangFuseTraceType


class LangFuseThoughtMessage(LangFuseMessage):
    """Content of a Langfuse trace message."""

    content: str
    type: LangFuseTraceType = LangFuseTraceType.THOUGHT


class LangFuseToolCallMessage(LangFuseMessage):
    """Content of a Langfuse trace message."""

    name: str
    content: str
    type: LangFuseTraceType = LangFuseTraceType.TOOL_CALL


class LangFuseToolOutputMessage(LangFuseMessage):
    """Content of a Langfuse trace message."""

    content: str
    type: LangFuseTraceType = LangFuseTraceType.TOOL_OUTPUT


class LangFuseFinalAnswerMessage(LangFuseMessage):
    """Content of a Langfuse trace message."""

    content: str
    type: LangFuseTraceType = LangFuseTraceType.FINAL_ANSWER


class LangFuseTracedResponse(BaseModel):
    """Agent Response and LangFuse Trace info."""

    answer: str | None
    trace_id: str | None


def parse_agent_stream_response(stream_item: Any, trace_id: str) -> list[LangFuseTracedResponse]:
    """Parse agent stream event into a LangFuseTracedResponse.

    Parameters
    ----------
    stream_item : Any
        The stream item from the agent SDK.
    trace_id : str
        The trace ID of the Langfuse trace.

    Returns
    -------
    list[LangFuseTracedResponse]
        A list of LangFuseTracedResponse objects containing the agent responses and
        trace info. Returns an empty list if the item is not an event with information
        that can be sent to Langfuse as a traceable event.
    """
    messages: list[LangFuseMessage] = []
    if isinstance(stream_item, RawResponsesStreamEvent) and isinstance(stream_item.data, ResponseCompletedEvent):
        # The completed event may contain multiple output messages,
        # including tool calls and final outputs.
        # If there is at least one tool call, we mark the response as a thought.
        is_thought = len(stream_item.data.response.output) > 1 and any(
            isinstance(message, ResponseFunctionToolCall) for message in stream_item.data.response.output
        )

        logger.info(f"################# Is thought: {is_thought}")

        for message in stream_item.data.response.output:
            if isinstance(message, ResponseOutputMessage):
                for _item in message.content:
                    if isinstance(_item, ResponseOutputText):
                        messages.append(LangFuseThoughtMessage(content=_item.text))

            elif isinstance(message, ResponseFunctionToolCall):
                messages.append(LangFuseToolCallMessage(name=message.name, content=message.arguments))

    elif (
        isinstance(stream_item, RunItemStreamEvent)
        and isinstance(stream_item.item, ToolCallOutputItem)
        and stream_item.name == "tool_output"
    ):
        messages.append(LangFuseToolOutputMessage(content=str(stream_item.item.output)))

    if len(messages) == 0:
        logger.debug(f"Untracked stream item type: type={type(stream_item)}, item={stream_item}")

    return [LangFuseTracedResponse(answer=message.model_dump_json(), trace_id=trace_id) for message in messages]


def flush_langfuse(langfuse_client: Langfuse) -> None:
    """Flush shared LangFuse Client. Rich Progress included.

    Parameters
    ----------
    langfuse_client : Langfuse
        The Langfuse client to flush.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Finalizing Langfuse annotations...", total=None)
        langfuse_client.flush()

"""Functions and objects pertaining to Langfuse."""

import base64
import logging
import os
from enum import Enum
from typing import Any

import logfire
import nest_asyncio
from agents.items import ToolCallOutputItem
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from aieng.agent_evals.configs import Configs
from langfuse import Langfuse  # type: ignore[attr-defined]
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputText
from openai.types.responses.response_completed_event import ResponseCompletedEvent
from openai.types.responses.response_output_message import ResponseOutputMessage
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
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


def configure_oai_agents_sdk(service_name: str) -> None:
    """Register Langfuse as tracing provider for OAI Agents SDK.

    Parameters
    ----------
    service_name : str
        The name of the service to configure.
    """
    nest_asyncio.apply()
    logfire.configure(service_name=service_name, send_to_logfire=False, scrubbing=False)
    logfire.instrument_openai_agents()


def set_up_langfuse_otlp_env_vars():
    """Set up environment variables for Langfuse OpenTelemetry integration.

    OTLP = OpenTelemetry Protocol.

    This function updates environment variables.

    Also refer to:
    langfuse.com/docs/integrations/openaiagentssdk/openai-agents
    """
    configs = Configs()

    langfuse_key = f"{configs.langfuse_public_key}:{configs.langfuse_secret_key}".encode()
    langfuse_auth = base64.b64encode(langfuse_key).decode()

    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = configs.langfuse_host + "/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"

    logging.info(f"Langfuse host: {configs.langfuse_host}")


def setup_langfuse_tracer(service_name: str = "aieng-eval-agents") -> "trace.Tracer":
    """Register Langfuse as the default tracing provider and return tracer.

    Parameters
    ----------
    service_name : str
        The name of the service to configure. Default is "aieng-eval-agents".

    Returns
    -------
    tracer: OpenTelemetry Tracer
    """
    set_up_langfuse_otlp_env_vars()
    configure_oai_agents_sdk(service_name)

    # Create a TracerProvider for OpenTelemetry
    trace_provider = TracerProvider()

    # Add a SimpleSpanProcessor with the OTLPSpanExporter to send traces
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

    # Set the global default tracer provider
    trace.set_tracer_provider(trace_provider)
    return trace.get_tracer(__name__)


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

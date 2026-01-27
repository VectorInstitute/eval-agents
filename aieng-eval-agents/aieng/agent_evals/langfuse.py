"""Functions and objects pertaining to Langfuse."""

import base64
import logging
import os

import logfire
import nest_asyncio
from aieng.agent_evals.configs import Configs
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


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

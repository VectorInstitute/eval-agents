"""Langfuse tracing integration for the knowledge-grounded QA agent.

This module provides automatic tracing of Google ADK agent interactions
using Langfuse via OpenTelemetry and OpenInference instrumentation.

Example
-------
>>> from aieng.agent_evals.knowledge_agent import init_tracing, KnowledgeGroundedAgent
>>> init_tracing()  # Call once at startup
>>> agent = KnowledgeGroundedAgent()
>>> response = agent.answer("What is the population of Tokyo?")
# Traces are automatically sent to Langfuse
"""

import base64
import logging
import os

from aieng.agent_evals.async_client_manager import AsyncClientManager
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


logger = logging.getLogger(__name__)


def init_tracing() -> bool:
    """Initialize Langfuse tracing for Google ADK agents.

    This function sets up OpenTelemetry with OTLP exporter to send traces
    to Langfuse, and initializes OpenInference instrumentation for Google ADK
    to automatically capture all agent interactions, tool calls, and model responses.

    Returns
    -------
    bool
        True if tracing was successfully initialized, False otherwise.

    Notes
    -----
    This function should be called once at application startup, before
    creating any agents. Subsequent calls are no-ops.

    Langfuse credentials are read from environment variables via Configs:
    - LANGFUSE_PUBLIC_KEY: Langfuse public key (pk-lf-...)
    - LANGFUSE_SECRET_KEY: Langfuse secret key (sk-lf-...)
    - LANGFUSE_HOST: Langfuse host URL (default: https://us.cloud.langfuse.com)

    Examples
    --------
    >>> from aieng.agent_evals.knowledge_agent import init_tracing
    >>> if init_tracing():
    ...     print("Tracing enabled!")
    """
    manager = AsyncClientManager.get_instance()

    if manager.otel_instrumented:
        logger.debug("Tracing already initialized")
        return True

    try:
        # Verify Langfuse client authentication
        langfuse_client = manager.langfuse_client
        if not langfuse_client.auth_check():
            logger.warning("Langfuse authentication failed. Check your credentials.")
            return False

        # Get credentials from configs
        configs = manager.configs
        public_key = configs.langfuse_public_key
        secret_key = configs.langfuse_secret_key
        langfuse_host = configs.langfuse_host

        # Set up OpenTelemetry OTLP exporter to send traces to Langfuse
        # Create base64 encoded auth string for OTLP headers
        auth_string = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        otel_endpoint = f"{langfuse_host.rstrip('/')}/api/public/otel"

        # Configure OpenTelemetry environment variables
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_endpoint
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth_string}"

        # Create a resource with service name
        resource = Resource.create({"service.name": "knowledge-agent"})

        # Create TracerProvider
        provider = TracerProvider(resource=resource)

        # Create OTLP exporter pointing to Langfuse
        exporter = OTLPSpanExporter(
            endpoint=f"{otel_endpoint}/v1/traces",
            headers={"Authorization": f"Basic {auth_string}"},
        )

        # Add batch processor for efficient trace export
        provider.add_span_processor(BatchSpanProcessor(exporter))

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        # Initialize OpenInference instrumentation for Google ADK
        GoogleADKInstrumentor().instrument(tracer_provider=provider)

        manager.otel_instrumented = True
        logger.info(f"Langfuse tracing initialized successfully (endpoint: {otel_endpoint})")
        return True

    except ImportError as e:
        logger.warning(f"Could not import tracing dependencies: {e}")
        return False
    except Exception as e:
        logger.warning(f"Failed to initialize tracing: {e}")
        return False


def flush_traces() -> None:
    """Flush any pending traces to Langfuse.

    Call this before your application exits to ensure all traces are sent.
    """
    manager = AsyncClientManager.get_instance()
    if manager._langfuse_client is not None:
        manager._langfuse_client.flush()


def is_tracing_enabled() -> bool:
    """Check if Langfuse tracing is currently enabled.

    Returns
    -------
    bool
        True if tracing has been initialized, False otherwise.
    """
    return AsyncClientManager.get_instance().otel_instrumented

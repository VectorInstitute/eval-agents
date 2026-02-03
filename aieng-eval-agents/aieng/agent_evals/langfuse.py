"""Functions and objects pertaining to Langfuse."""

import base64
import json
import logging
import os

import logfire
import nest_asyncio
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.configs import Configs
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor


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


def init_tracing(service_name: str = "aieng-eval-agents") -> bool:
    """Initialize Langfuse tracing for Google ADK agents.

    This function sets up OpenTelemetry with OTLP exporter to send traces
    to Langfuse, and initializes OpenInference instrumentation for Google ADK
    to automatically capture all agent interactions, tool calls, and model responses.

    Parameters
    ----------
    service_name : str, optional, default="aieng-eval-agents"
        Service name to attach to emitted traces.

    Returns
    -------
    bool
        True if tracing was successfully initialized, False otherwise.

    Examples
    --------
    >>> from aieng.agent_evals.langfuse import init_tracing
    >>> init_tracing()  # Call once at startup
    >>> # Create and use your Google ADK agent as usual
    # Traces are automatically sent to Langfuse
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
        public_key = configs.langfuse_public_key or ""
        secret_key = configs.langfuse_secret_key.get_secret_value() if configs.langfuse_secret_key else ""
        langfuse_host = configs.langfuse_host

        # Set up OpenTelemetry OTLP exporter to send traces to Langfuse
        auth_string = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        otel_endpoint = f"{langfuse_host.rstrip('/')}/api/public/otel"

        # Configure OpenTelemetry environment variables
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_endpoint
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth_string}"

        # Create a resource with service name
        resource = Resource.create({"service.name": service_name})

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
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor  # noqa: PLC0415

        GoogleADKInstrumentor().instrument(tracer_provider=provider)

        manager.otel_instrumented = True
        logger.info("Langfuse tracing initialized successfully (endpoint: %s)", otel_endpoint)
        return True

    except ImportError as e:
        logger.warning("Could not import tracing dependencies: %s", e)
        return False
    except Exception as e:
        logger.warning("Failed to initialize tracing: %s", e)
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


async def upload_dataset_to_langfuse(dataset_path: str, dataset_name: str):
    """Upload a dataset to Langfuse.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset to upload. The dataset must be a json file
        containing a list of dictionaries. Each dictionary must contain a
        `input` and `expected_output` keys. Additionally, it can include
        an `id` key that will be added to the metadata of the dataset item.
    dataset_name : str
        Name of the dataset to upload.
    """
    # Get the client manager singleton instance and langfuse client
    client_manager = AsyncClientManager.get_instance()
    langfuse_client = client_manager.langfuse_client

    # Load the ground truth dataset from the file path
    logger.info(f"Loading dataset from '{dataset_path}'")
    with open(dataset_path, "r") as file:
        dataset = json.load(file)

    # Create the dataset in Langfuse
    langfuse_client.create_dataset(name=dataset_name)

    # Upload each item to the dataset
    for item in dataset:
        assert "input" in item, "`input` is required for all items in the dataset"
        assert "expected_output" in item, "`expected_output` is required for all items in the dataset"

        langfuse_client.create_dataset_item(
            dataset_name=dataset_name,
            input=item["input"],
            expected_output=item["expected_output"],
            metadata={
                "id": item.get("id", None),
            },
        )

    logger.info(f"Uploaded {len(dataset)} items to dataset '{dataset_name}'")

    # Gracefully close the services
    await client_manager.close()

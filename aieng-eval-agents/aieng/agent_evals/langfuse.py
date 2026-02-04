"""Functions and objects pertaining to Langfuse."""

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import logfire
import nest_asyncio
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.configs import Configs
from aieng.agent_evals.progress import track_with_progress
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
        Path to the dataset to upload.

        Supported formats:
        - ``.json``: a JSON array of records.
        - ``.jsonl``: one JSON record per non-empty line.

        Each record must contain ``input`` and ``expected_output`` keys.
        Records may optionally include:
        - ``id``: item identifier stored in metadata.
        - ``metadata``: additional dictionary metadata merged into upload metadata.
    dataset_name : str
        Name of the dataset to upload.

    Raises
    ------
    ValueError
        If the dataset format is invalid or a record is malformed.
    FileNotFoundError
        If ``dataset_path`` does not exist.
    """
    dataset_file = Path(dataset_path)

    # Get the client manager singleton instance and langfuse client
    client_manager = AsyncClientManager.get_instance()
    langfuse_client = client_manager.langfuse_client

    # Parse once so the upload loop has deterministic progress totals.
    logger.info("Loading dataset from '%s'", dataset_file)
    dataset_format = _detect_dataset_format(dataset_file)
    records = _load_dataset_records(dataset_file, dataset_format)

    # Create dataset if missing; if it already exists we reuse it.
    _ensure_dataset_exists(langfuse_client=langfuse_client, dataset_name=dataset_name)

    # We centralize metadata normalization to keep uploader behavior
    # consistent across JSON and JSONL sources.
    for record_number, item in track_with_progress(
        records,
        description=f"Uploading Langfuse dataset '{dataset_name}'",
        total=len(records),
        transient=True,  # Clear progress bar when done
    ):
        normalized = _normalize_dataset_record(item=item, record_number=record_number)
        item_id = _build_dataset_item_id(
            dataset_name=dataset_name,
            input_payload=normalized["input"],
            expected_output_payload=normalized["expected_output"],
        )
        langfuse_client.create_dataset_item(
            dataset_name=dataset_name,
            id=item_id,  # Globally unique ID of deduplication
            input=normalized["input"],
            expected_output=normalized["expected_output"],
            metadata=normalized["metadata"],
        )

    logger.info("Uploaded %d items to dataset '%s'", len(records), dataset_name)

    # Gracefully close the services
    await client_manager.close()


def _detect_dataset_format(dataset_file: Path) -> Literal["json", "jsonl"]:
    """Detect dataset format from extension or first non-empty content line."""
    suffix = dataset_file.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"

    with dataset_file.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("["):
                return "json"
            return "jsonl"

    raise ValueError(f"Dataset file is empty: {dataset_file}")


def _ensure_dataset_exists(*, langfuse_client: Any, dataset_name: str) -> None:
    """Ensure the target dataset exists before item uploads."""
    try:
        langfuse_client.create_dataset(name=dataset_name)
        return
    except Exception as exc:
        # We only continue if the dataset can be retrieved
        try:
            langfuse_client.get_dataset(dataset_name)
            logger.info("Dataset '%s' already exists; appending/upserting items.", dataset_name)
            return
        except Exception as retrieval_exc:
            raise exc from retrieval_exc


def _build_dataset_item_id(
    *,
    dataset_name: str,
    input_payload: Any,
    expected_output_payload: Any,
) -> str:
    """Build a deterministic, globally-unique dataset item ID."""
    canonical = json.dumps(
        {
            "input": input_payload,
            "expected_output": expected_output_payload,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{dataset_name}:{digest}"


def _load_dataset_records(
    dataset_file: Path, dataset_format: Literal["json", "jsonl"]
) -> list[tuple[int, dict[str, Any]]]:
    """Load dataset records and preserve stable record numbers."""
    if dataset_format == "json":
        return _load_json_records(dataset_file)
    return _load_jsonl_records(dataset_file)


def _load_json_records(dataset_file: Path) -> list[tuple[int, dict[str, Any]]]:
    """Load records from a JSON array file."""
    with dataset_file.open("r", encoding="utf-8") as file:
        loaded = json.load(file)

    if not isinstance(loaded, list):
        raise ValueError(f"JSON dataset must be a list of records: {dataset_file}")

    return [(index, record) for index, record in enumerate(loaded, start=1)]


def _load_jsonl_records(dataset_file: Path) -> list[tuple[int, dict[str, Any]]]:
    """Load records from a JSONL file with line-number-aware errors."""
    records: list[tuple[int, dict[str, Any]]] = []

    with dataset_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record at line {line_number} in '{dataset_file}': {exc.msg}") from exc
            records.append((line_number, parsed))

    if not records:
        raise ValueError(f"JSONL dataset has no records: {dataset_file}")

    return records


def _normalize_dataset_record(item: Any, record_number: int) -> dict[str, Any]:
    """Validate and normalize one dataset record for upload."""
    if not isinstance(item, dict):
        raise ValueError(f"Record {record_number} must be an object.")

    if "input" not in item:
        raise ValueError(f"Record {record_number} is missing required key: 'input'")
    if "expected_output" not in item:
        raise ValueError(f"Record {record_number} is missing required key: 'expected_output'")

    raw_metadata = item.get("metadata", {})
    if raw_metadata is None:
        raw_metadata = {}
    if not isinstance(raw_metadata, dict):
        raise ValueError(f"Record {record_number} has non-object metadata.")

    derived_id = item.get("id", record_number)
    metadata = dict(raw_metadata)
    metadata["id"] = derived_id

    return {
        "input": item["input"],
        "expected_output": item["expected_output"],
        "metadata": metadata,
    }

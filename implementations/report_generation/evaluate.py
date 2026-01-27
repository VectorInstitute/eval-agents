"""Evaluate the report generation agent."""

import asyncio
import logging
import time
from uuid import uuid4

import agents
from aieng.agent_evals.async_client_manager import AsyncClientManager
from dotenv import load_dotenv
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.api.resources.commons.types import TraceWithFullDetails

from implementations.report_generation.main import report_generation_agent


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def evaluate() -> None:
    """Evaluate the report generation agent."""
    # Get the client manager singleton instance and langfuse client
    client_manager = AsyncClientManager.get_instance()
    langfuse_client = client_manager.langfuse_client

    evaluation_dataset = [
        {
            "id": "1",
            "user_input": "Generate a monthly sales performance report for the last year with available data.",
            "expected_output": {
                "report_data": [
                    ["2010-12", 748957.01999997743],
                    ["2011-01", 560000.26000002341],
                    ["2011-02", 498062.6500000268],
                    ["2011-03", 683267.08000001893],
                    ["2011-04", 493207.1210000249],
                    ["2011-05", 723333.51000001],
                    ["2011-06", "691123.12000002281"],
                    ["2011-07", "681300.11100003007"],
                    ["2011-08", "682680.51000001759"],
                    ["2011-09", 1019687.622000011],
                    ["2011-10", 1070704.669999975],
                    ["2011-11", 1461756.2499997574],
                    ["2011-12", "433668.01000001712"],
                ],
                "report_columns": ["SalesMonth", "TotalSales"],
            },
            "trace_id": "6c21f5108a4d8ebd3bfd4c059cf3b5e4",
        }
    ]

    trace_ids: list[str] = []
    for example in evaluation_dataset:
        if example["trace_id"]:
            assert isinstance(example["trace_id"], str), "Trace ID must be a string."
            logger.info(
                f"Skipping the agent pipeline, found trace ID {example['trace_id']} for example '{example['id']}'"
            )
            trace_ids.append(example["trace_id"])
            continue

        trace_id = langfuse_client.create_trace_id(seed=str(uuid4()))
        trace_ids.append(trace_id)
        logger.info(f"Evaluating example '{example['id']}' with trace ID {trace_id}")

        agent = report_generation_agent(enable_trace=True)
        evaluation_name = f"evaluation example {example['id']}"
        with langfuse_client.start_as_current_observation(name=evaluation_name, trace_context={"trace_id": trace_id}):
            assert isinstance(example["user_input"], str), "User input must be a string."
            await agents.Runner.run(agent, input=example["user_input"])

    for trace_id in trace_ids:
        trace = get_trace_with_retry(trace_id, langfuse_client)
        observations = sorted(trace.observations, key=lambda obs: obs.start_time)

        for observation in observations:
            assert observation.name is not None, "Observation name must not be None."
            logger.info(f"Observation {observation.id} {observation.name} {observation.start_time}")
            if "write_report_to_file" in observation.name:
                logger.info(f"Found write_report_to_file observation: {observation.input}")


def get_trace_with_retry(
    trace_id: str,
    langfuse_client: Langfuse,
    max_retries: int = 10,
    backoff_factor: int = 2,
) -> TraceWithFullDetails:
    """Get a trace from Langfuse with a retry mechanism.

    The initial retry delay is 2 seconds.

    Parameters
    ----------
    trace_id : str
        The ID of the trace to get.
    langfuse_client : Langfuse
        The Langfuse client to use.
    max_retries : int, optional
        The maximum number of retries. Default is 10.
    backoff_factor : int, optional
        The backoff factor. Default is 2.

    Returns
    -------
    TraceWithFullDetails
        The trace object retrieved from Langfuse.

    Raises
    ------
    Exception
        If the trace is not found after the maximum number of retries.
    """
    t = 2
    for retry in range(max_retries):
        try:
            return langfuse_client.api.trace.get(trace_id=trace_id)
        except Exception:
            logger.error(f"Trace {trace_id} not found. Retrying in {t} seconds ({retry + 1}/{max_retries})...")
            time.sleep(t)
            t *= backoff_factor

    raise Exception(f"Failed to get trace {trace_id} after {max_retries} retries")


if __name__ == "__main__":
    asyncio.run(evaluate())

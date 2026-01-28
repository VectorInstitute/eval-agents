"""Evaluate the report generation agent against a Langfuse dataset."""

import asyncio
import logging

import agents
import click
from aieng.agent_evals.async_client_manager import AsyncClientManager
from langfuse._client.datasets import DatasetItemClient
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall

from implementations.report_generation.data.langfuse_upload import DEFAULT_EVALUATION_DATASET_NAME
from implementations.report_generation.main import get_report_generation_agent


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def evaluate(dataset_name: str):
    """Evaluate the report generation agent against a Langfuse dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to evaluate against.
    """
    # Get the client manager singleton instance and langfuse client
    client_manager = AsyncClientManager.get_instance()
    langfuse_client = client_manager.langfuse_client

    dataset = langfuse_client.get_dataset(dataset_name)

    result = dataset.run_experiment(
        name="Evaluate Report Generation Agent with Local Data",
        description="Evaluate the report generation agent with data from a local dataset",
        task=agent_task,
        max_concurrency=1,
    )

    logger.info(f"Evaluation result: {result.format()}")

    # Gracefully close the services
    await client_manager.close()


async def agent_task(*, item: DatasetItemClient, **kwargs) -> str | None:
    """Run the report generation agent against an item from a Langfuse dataset.

    Parameters
    ----------
    item : DatasetItemClient
        The item from the Langfuse dataset to evaluate against.

    Returns
    -------
    str | None
        The arguments sent by the report generation agent to the write_report_to_file
        function. Returns None if the agent did not call the function.
    """
    report_generation_agent = get_report_generation_agent(enable_trace=True)
    result = await agents.Runner.run(report_generation_agent, input=item.input)

    # Extract the report data from the result by returning the
    # arguments to the write_report_to_file function call
    for raw_response in result.raw_responses:
        for output in raw_response.output:
            if isinstance(output, ResponseFunctionToolCall) and "write_report_to_file" in output.name:
                return output.arguments

    return None


@click.command()
@click.option(
    "--dataset-name",
    required=False,
    default=DEFAULT_EVALUATION_DATASET_NAME,
    help="Name of the Langfuse dataset to evaluate against.",
)
def cli(dataset_name: str):
    """Command line interface to call the evaluate function.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to evaluate against.
        Default is DEFAULT_EVALUATION_DATASET_NAME.
    """
    asyncio.run(evaluate(dataset_name))


if __name__ == "__main__":
    cli()

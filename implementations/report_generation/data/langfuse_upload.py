"""Upload a dataset to Langfuse."""

import asyncio
import json
import logging

import click
from aieng.agent_evals.async_client_manager import AsyncClientManager


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_EVALUATION_DATASET_PATH = "implementations/report_generation/data/OnlineRetailReportEval.json"
DEFAULT_EVALUATION_DATASET_NAME = "OnlineRetailReportEval"


async def upload_dataset_to_langfuse(dataset_path: str, dataset_name: str):
    """Upload a dataset to Langfuse.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset to upload.
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


@click.command()
@click.option(
    "--dataset-path",
    required=False,
    default=DEFAULT_EVALUATION_DATASET_PATH,
    help="Path to the dataset to upload.",
)
@click.option(
    "--dataset-name",
    required=False,
    default=DEFAULT_EVALUATION_DATASET_NAME,
    help="Name of the dataset to upload.",
)
def cli(dataset_path: str, dataset_name: str):
    """
    Command line interface to call the upload_dataset_to_langfuse function.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset to upload.
        Default is DEFAULT_EVALUATION_DATASET_PATH.
    dataset_name : str
        Name of the dataset to upload.
        Default is DEFAULT_EVALUATION_DATASET_NAME.
    """
    asyncio.run(upload_dataset_to_langfuse(dataset_path, dataset_name))


if __name__ == "__main__":
    cli()

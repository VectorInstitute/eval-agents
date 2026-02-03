"""
Upload a dataset to Langfuse.

Example
-------
$ python -m implementations.report_generation.data.langfuse_upload
$ python -m implementations.report_generation.data.langfuse_upload \
    --dataset-path <path/to/dataset.json> \
    --dataset-name <dataset name>
"""

import asyncio
import logging

import click
from aieng.agent_evals.langfuse import upload_dataset_to_langfuse


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_EVALUATION_DATASET_PATH = "implementations/report_generation/data/OnlineRetailReportEval.json"
DEFAULT_EVALUATION_DATASET_NAME = "OnlineRetailReportEval"


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

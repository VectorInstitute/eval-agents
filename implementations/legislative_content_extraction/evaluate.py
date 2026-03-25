"""
Evaluate the legislative content extraction agent against a Langfuse dataset.

Example
-------
$ python -m implementations.legislative_content_extraction.evaluate \
    --dataset-name LegislativeContentExtractionEval
"""

import asyncio

import click
from aieng.agent_evals.legislative_content_extraction.evaluation.offline import evaluate
from dotenv import load_dotenv

load_dotenv(verbose=True)

DEFAULT_DATASET_NAME = "LegislativeContentExtractionEval"
DEFAULT_FILES_DIR = "implementations/legislative_content_extraction/files"


@click.command()
@click.option(
    "--dataset-name",
    default=DEFAULT_DATASET_NAME,
    help="Name of the Langfuse dataset to evaluate against.",
)
@click.option(
    "--files-dir",
    default=DEFAULT_FILES_DIR,
    help="Local directory where PDF files are cached.",
)
@click.option(
    "--max-concurrency",
    default=5,
    type=int,
    help="Maximum concurrent agent runs (default: 5).",
)
def cli(dataset_name: str, files_dir: str, max_concurrency: int):
    asyncio.run(evaluate(dataset_name, files_dir=files_dir, max_concurrency=max_concurrency))


if __name__ == "__main__":
    cli()
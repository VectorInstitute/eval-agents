"""Upload DeepSearchQA dataset subset to Langfuse.

This script uploads a subset of the DeepSearchQA benchmark to Langfuse
for use with the Langfuse experiment evaluation framework.

Usage:
    python langfuse_upload.py --samples 10 --category "Finance & Economics"
    python langfuse_upload.py --ids 123 456 789
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path

import click
from aieng.agent_evals.knowledge_qa.evaluation import DeepSearchQADataset
from aieng.agent_evals.langfuse import upload_dataset_to_langfuse as upload_file_to_langfuse
from dotenv import load_dotenv


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_DATASET_NAME = "DeepSearchQA-Subset"


async def upload_deepsearch_qa_to_langfuse(
    dataset_name: str,
    samples: int = 10,
    category: str | None = None,
    ids: list[int] | None = None,
) -> None:
    """Upload DeepSearchQA examples to Langfuse.

    This function converts DeepSearchQA examples to a temporary JSONL file
    and uses the shared upload utility for consistent formatting and progress tracking.

    Parameters
    ----------
    dataset_name : str
        Name for the dataset in Langfuse.
    samples : int
        Number of samples to upload (ignored if ids provided).
    category : str, optional
        Filter by category (ignored if ids provided).
    ids : list[int], optional
        Specific example IDs to upload.
    """
    # Load DeepSearchQA dataset
    logger.info("Loading DeepSearchQA dataset...")
    dataset = DeepSearchQADataset()
    logger.info(f"Loaded {len(dataset)} total examples")

    # Select examples based on criteria
    if ids:
        examples = dataset.get_by_ids(ids)
        logger.info(f"Selected {len(examples)} examples by ID")
    elif category:
        examples = dataset.get_by_category(category)[:samples]
        logger.info(f"Selected {len(examples)} examples from category '{category}'")
    else:
        examples = dataset.examples[:samples]
        logger.info(f"Selected first {len(examples)} examples")

    if not examples:
        logger.error("No examples found matching criteria")
        return

    # Convert examples to JSONL format for the shared upload utility
    # Use a temporary file that's automatically cleaned up
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".jsonl",
        prefix=f"deepsearchqa_{dataset_name}_",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
        logger.info(f"Writing {len(examples)} examples to temporary file...")

        for example in examples:
            record = {
                "input": example.problem,
                "expected_output": example.answer,
                "metadata": {
                    "example_id": example.example_id,
                    "category": example.problem_category,
                    "answer_type": example.answer_type,
                },
            }
            temp_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        # Use the shared upload utility with progress tracking and deduplication
        await upload_file_to_langfuse(
            dataset_path=str(temp_path),
            dataset_name=dataset_name,
        )
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()
            logger.debug(f"Removed temporary file: {temp_path}")


@click.command()
@click.option(
    "--dataset-name",
    default=DEFAULT_DATASET_NAME,
    help="Name for the dataset in Langfuse.",
)
@click.option(
    "--samples",
    default=10,
    type=int,
    help="Number of samples to upload (default: 10).",
)
@click.option(
    "--category",
    default=None,
    help="Filter by category (e.g., 'Finance & Economics').",
)
@click.option(
    "--ids",
    multiple=True,
    type=int,
    help="Specific example IDs to upload (can be used multiple times).",
)
def cli(dataset_name: str, samples: int, category: str | None, ids: tuple[int, ...]) -> None:
    """Upload DeepSearchQA examples to Langfuse."""
    ids_list = list(ids) if ids else None
    asyncio.run(upload_deepsearch_qa_to_langfuse(dataset_name, samples, category, ids_list))


if __name__ == "__main__":
    cli()

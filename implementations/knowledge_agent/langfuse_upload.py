"""Upload DeepSearchQA dataset subset to Langfuse.

This script uploads a subset of the DeepSearchQA benchmark to Langfuse
for use with the Langfuse experiment evaluation framework.

Usage:
    python langfuse_upload.py --samples 10 --category "Finance & Economics"
    python langfuse_upload.py --ids 123 456 789
"""

import asyncio
import logging

import click
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.knowledge_agent.evaluation import DeepSearchQADataset
from dotenv import load_dotenv


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_DATASET_NAME = "DeepSearchQA-Subset"


async def upload_dataset_to_langfuse(
    dataset_name: str,
    samples: int = 10,
    category: str | None = None,
    ids: list[int] | None = None,
) -> None:
    """Upload DeepSearchQA examples to Langfuse.

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
    client_manager = AsyncClientManager.get_instance()
    langfuse = client_manager.langfuse_client

    # Load DeepSearchQA dataset
    logger.info("Loading DeepSearchQA dataset...")
    dataset = DeepSearchQADataset()
    logger.info(f"Loaded {len(dataset)} total examples")

    # Select examples
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

    # Create or get existing dataset in Langfuse
    try:
        langfuse.create_dataset(name=dataset_name)
        logger.info(f"Created new dataset '{dataset_name}'")
    except Exception:
        logger.info(f"Dataset '{dataset_name}' already exists, adding items")

    # Upload each example
    for example in examples:
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=example.problem,
            expected_output=example.answer,
            metadata={
                "example_id": example.example_id,
                "category": example.problem_category,
                "answer_type": example.answer_type,
            },
        )
        logger.info(f"Uploaded example {example.example_id}: {example.problem[:50]}...")

    logger.info(f"Uploaded {len(examples)} items to dataset '{dataset_name}'")

    # Cleanup
    await client_manager.close()


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
    asyncio.run(upload_dataset_to_langfuse(dataset_name, samples, category, ids_list))


if __name__ == "__main__":
    cli()

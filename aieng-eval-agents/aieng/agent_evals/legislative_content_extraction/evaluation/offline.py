"""Evaluate the legislative content extraction agent against a Langfuse dataset."""

import logging

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation import run_experiment
from aieng.agent_evals.legislative_content_extraction.graders import item_level_deterministic_grader
from aieng.agent_evals.legislative_content_extraction.task import LegislativeExtractionTask

logger = logging.getLogger(__name__)


async def evaluate(
    dataset_name: str,
    files_dir: str | None = None,
    max_concurrency: int = 5,
) -> None:
    """Evaluate the legislative extraction agent against a Langfuse dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to evaluate against.
    files_dir : str | None, optional
        Local directory where PDF files are cached.
    max_concurrency : int, optional
        Maximum concurrent agent runs, by default 5.
    """
    task = LegislativeExtractionTask(files_dir=files_dir)

    result = run_experiment(
        dataset_name=dataset_name,
        name="Evaluate Legislative Content Extraction Agent",
        task=task,
        evaluators=[item_level_deterministic_grader],
        max_concurrency=max_concurrency,
    )

    logger.info("Evaluation complete. %d items processed.", len(result.item_results))

    try:
        await AsyncClientManager.get_instance().close()
    except Exception as exc:
        logger.warning("Client manager not closed cleanly: %s", exc)
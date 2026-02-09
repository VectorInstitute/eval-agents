"""Evaluate the Knowledge Agent using Langfuse experiments.

This script runs the Knowledge Agent against a Langfuse dataset and evaluates
results using the DeepSearchQA LLM-as-judge methodology. Results are automatically
logged to Langfuse for analysis and comparison.

Usage:
    # Run a full evaluation
    python evaluate.py

    # Run with custom dataset and experiment name
    python evaluate.py --dataset-name "MyDataset" --experiment-name "v2-test"

    # Resume an interrupted evaluation (skips already-evaluated items)
    python evaluate.py --experiment-name "v2-test" --resume

Resume Feature:
    Use --resume to continue an interrupted evaluation. The script will:
    1. Check Langfuse for traces with the specified experiment name
    2. Identify items that already have evaluation scores
    3. Skip those items and only evaluate the remaining ones

    Important: Use the SAME --experiment-name as the previous run to ensure
    proper resumption.
"""

import asyncio
import logging
from typing import Any

import click
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation import run_experiment, run_experiment_on_items
from aieng.agent_evals.knowledge_qa.agent import KnowledgeGroundedAgent
from aieng.agent_evals.knowledge_qa.judges import DeepSearchQAJudge, DeepSearchQAResult
from aieng.agent_evals.logging_config import setup_logging
from dotenv import load_dotenv
from langfuse.experiment import Evaluation


load_dotenv(verbose=True)
setup_logging(level=logging.INFO, show_time=True, show_path=False)
logger = logging.getLogger(__name__)


DEFAULT_DATASET_NAME = "DeepSearchQA-Subset"
DEFAULT_EXPERIMENT_NAME = "Knowledge Agent Evaluation"

# Module-level lazy judge instance
_judge: DeepSearchQAJudge | None = None


def _get_judge() -> DeepSearchQAJudge:
    """Get or create the shared DeepSearchQA Judge instance."""
    global _judge  # noqa: PLW0603
    if _judge is None:
        _judge = DeepSearchQAJudge()
    return _judge


def _close_judge() -> None:
    """Close the shared judge instance to clean up resources."""
    global _judge  # noqa: PLW0603
    if _judge is not None:
        _judge.close()
        _judge = None


async def agent_task(*, item: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG001
    """Run the Knowledge Agent on a dataset item.

    This is the task function used by the evaluation harness.

    Parameters
    ----------
    item : Any
        The Langfuse experiment item containing the question.
    **kwargs : Any
        Additional arguments from the harness (unused).

    Returns
    -------
    dict[str, Any]
        Dictionary containing 'text' (response) and 'agent_response'
        (full response object).
    """
    question = item.input
    logger.info(f"Running agent on: {question[:80]}...")

    try:
        # Create a fresh agent for each task to avoid shared state issues
        agent = KnowledgeGroundedAgent(enable_planning=True)
        response = await agent.answer_async(question)
        logger.info(f"Agent completed: {len(response.text)} chars, {len(response.tool_calls)} tool calls")

        return {
            "text": response.text,
            "agent_response": response,
        }
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        return {"text": f"Error: {e}", "agent_response": None}


async def deepsearchqa_evaluator(
    *,
    input: str,  # noqa: A002
    output: dict[str, Any],
    expected_output: str,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,  # noqa: ARG001
) -> list[Evaluation]:
    """Evaluate the agent's response using DeepSearchQA methodology.

    This is the evaluator function used by the evaluation harness.

    Parameters
    ----------
    input : str
        The original question.
    output : dict[str, Any]
        Dictionary containing 'text' and 'agent_response'.
    expected_output : str
        The ground truth answer.
    metadata : dict[str, Any] | None, optional
        Item metadata (contains answer_type).
    **kwargs : Any
        Additional arguments from the harness (unused).

    Returns
    -------
    list[Evaluation]
        List of Langfuse Evaluations with F1, precision, recall, and outcome scores.
    """
    output_text = output.get("text", "") if isinstance(output, dict) else str(output)
    answer_type = metadata.get("answer_type", "Set Answer") if metadata else "Set Answer"

    logger.info(f"Evaluating response (answer_type: {answer_type})...")

    try:
        judge = _get_judge()
        _, result = await judge.evaluate_with_details_async(
            question=input,
            answer=output_text,
            ground_truth=expected_output,
            answer_type=answer_type,
        )

        evaluations = result.to_evaluations()
        logger.info(f"Evaluation complete: {result.outcome} (F1: {result.f1_score:.2f})")
        return evaluations

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return DeepSearchQAResult.error_evaluations(str(e))


def get_completed_item_ids(langfuse: Any, run_name: str, dataset_id: str) -> set[str]:
    """Get dataset item IDs that have already been evaluated in a run.

    Parameters
    ----------
    langfuse : Langfuse
        The Langfuse client.
    run_name : str
        Name of the experiment run.
    dataset_id : str
        ID of the dataset.

    Returns
    -------
    set[str]
        Set of completed dataset item IDs.
    """
    try:
        logger.info(f"Checking for existing evaluations in run '{run_name}'...")
        completed_ids = set()
        page = 1
        limit = 50

        while True:
            run_items_response = langfuse.api.dataset_run_items.list(
                dataset_id=dataset_id,
                run_name=run_name,
                limit=limit,
                page=page,
            )

            if not run_items_response.data:
                break

            for run_item in run_items_response.data:
                if run_item.trace_id and _has_evaluation_scores(langfuse, run_item.trace_id):
                    completed_ids.add(run_item.dataset_item_id)
                    logger.debug(f"Found completed evaluation for item {run_item.dataset_item_id}")

            if len(run_items_response.data) < limit:
                break

            page += 1

        logger.info(f"Found {len(completed_ids)} completed evaluations")
        return completed_ids

    except Exception as e:
        logger.warning(f"Failed to fetch existing evaluations: {e}")
        logger.info("Will process all items")
        return set()


def _has_evaluation_scores(langfuse: Any, trace_id: str) -> bool:
    """Check if a trace has evaluation scores."""
    try:
        trace = langfuse.api.trace.get(trace_id)
        return hasattr(trace, "scores") and bool(trace.scores)
    except Exception as e:
        logger.debug(f"Could not fetch trace {trace_id}: {e}")
        return False


async def run_evaluation(
    dataset_name: str,
    experiment_name: str,
    max_concurrency: int = 1,
    resume: bool = False,
) -> None:
    """Run the full evaluation experiment.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to evaluate against.
    experiment_name : str
        Name for this experiment run.
    max_concurrency : int, optional
        Maximum concurrent agent runs, by default 1.
    resume : bool, optional
        If True, skip items that have already been evaluated, by default False.
    """
    client_manager = AsyncClientManager.get_instance()
    langfuse = client_manager.langfuse_client

    try:
        if resume:
            # Resume: fetch dataset, filter completed items, run remaining
            logger.info(f"Loading dataset '{dataset_name}' from Langfuse...")
            try:
                dataset = langfuse.get_dataset(dataset_name)
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                logger.info("Run the dataset upload script first to create the dataset.")
                return

            logger.info(f"Found dataset with {len(dataset.items)} items")

            completed_ids = get_completed_item_ids(langfuse, experiment_name, dataset.id)
            items_to_process = [item for item in dataset.items if item.id not in completed_ids]

            if not items_to_process:
                logger.info("All items already evaluated. Nothing to do.")
                return

            logger.info(f"Resume mode: Processing {len(items_to_process)} remaining items")
            logger.info(f"Starting experiment: {experiment_name}")
            logger.info(f"Max concurrency: {max_concurrency}")

            result = run_experiment_on_items(
                items_to_process,
                name=experiment_name,
                run_name=experiment_name,
                description="Knowledge Agent evaluation with DeepSearchQA judge (resumed)",
                task=agent_task,
                evaluators=[deepsearchqa_evaluator],
                max_concurrency=max_concurrency,
            )

            logger.info("Resume experiment complete!")
            logger.info(result.format().replace("\\n", "\n"))
        else:
            # Normal mode: use the evaluation harness (fetches dataset internally)
            logger.info(f"Starting experiment: {experiment_name}")
            logger.info(f"Max concurrency: {max_concurrency}")

            result = run_experiment(
                dataset_name=dataset_name,
                name=experiment_name,
                description="Knowledge Agent evaluation with DeepSearchQA judge",
                task=agent_task,
                evaluators=[deepsearchqa_evaluator],
                max_concurrency=max_concurrency,
            )

            logger.info("Experiment complete!")
            logger.info(result.format().replace("\\n", "\n"))

    finally:
        # Cleanup
        logger.info("Closing client manager and flushing data...")
        try:
            _close_judge()
            await client_manager.close()
            # Give event loop time to process cleanup tasks
            await asyncio.sleep(0.1)
            logger.info("Cleanup complete")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


@click.command()
@click.option(
    "--dataset-name",
    default=DEFAULT_DATASET_NAME,
    help="Name of the Langfuse dataset to evaluate against.",
)
@click.option(
    "--experiment-name",
    default=DEFAULT_EXPERIMENT_NAME,
    help="Name for this experiment run.",
)
@click.option(
    "--max-concurrency",
    default=1,
    type=int,
    help="Maximum concurrent agent runs (default: 1).",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from previous run. Skips items that already have evaluation scores.",
)
def cli(dataset_name: str, experiment_name: str, max_concurrency: int, resume: bool) -> None:
    """Run Knowledge Agent evaluation using Langfuse experiments.

    Use --resume to continue an interrupted evaluation run. Items that already
    have evaluation scores will be skipped. Make sure to use the same
    --experiment-name as the previous run.
    """
    asyncio.run(run_evaluation(dataset_name, experiment_name, max_concurrency, resume))


if __name__ == "__main__":
    cli()

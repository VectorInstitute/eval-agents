"""Evaluate the Knowledge Agent using Langfuse experiments.

This script runs the Knowledge Agent against a Langfuse dataset and evaluates
results using the DeepSearchQA LLM-as-judge methodology. Results are automatically
logged to Langfuse for analysis and comparison.

Usage:
    # Run a full evaluation
    python evaluate.py

    # Run with custom dataset and experiment name
    python evaluate.py --dataset-name "MyDataset" --experiment-name "v2-test"
"""

import asyncio
import logging
from typing import Any

import click
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation import run_experiment
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
        agent = KnowledgeGroundedAgent(enable_planning=True)  # type: ignore[call-arg]
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


async def run_evaluation(
    dataset_name: str,
    experiment_name: str,
    max_concurrency: int = 1,
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
    """
    client_manager = AsyncClientManager.get_instance()

    try:
        logger.info(f"Starting experiment '{experiment_name}' on dataset '{dataset_name}'")
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
        logger.info("Closing client manager and flushing data...")
        try:
            _close_judge()
            await client_manager.close()
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
def cli(dataset_name: str, experiment_name: str, max_concurrency: int) -> None:
    """Run Knowledge Agent evaluation using Langfuse experiments."""
    asyncio.run(run_evaluation(dataset_name, experiment_name, max_concurrency))


if __name__ == "__main__":
    cli()

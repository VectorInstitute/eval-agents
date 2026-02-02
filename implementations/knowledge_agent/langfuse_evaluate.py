"""Evaluate the Knowledge Agent using Langfuse experiments.

This script runs the Knowledge Agent against a Langfuse dataset and evaluates
results using the DeepSearchQA LLM-as-judge methodology. Results are automatically
logged to Langfuse for analysis and comparison.

Usage:
    # Run a full evaluation
    python langfuse_evaluate.py

    # Run with custom dataset and experiment name
    python langfuse_evaluate.py --dataset-name "MyDataset" --experiment-name "v2-test"

    # Resume an interrupted evaluation (skips already-evaluated items)
    python langfuse_evaluate.py --experiment-name "v2-test" --resume

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
from aieng.agent_evals.knowledge_agent.agent import KnowledgeGroundedAgent
from aieng.agent_evals.knowledge_agent.judges import DeepSearchQAJudge
from dotenv import load_dotenv
from langfuse.experiment import Evaluation, LocalExperimentItem


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_DATASET_NAME = "DeepSearchQA-Subset"
DEFAULT_EXPERIMENT_NAME = "Knowledge Agent Evaluation"


class EvaluationContext:
    """Holds shared instances for the evaluation run."""

    def __init__(self) -> None:
        self._agent: KnowledgeGroundedAgent | None = None
        self._judge: DeepSearchQAJudge | None = None

    @property
    def agent(self) -> KnowledgeGroundedAgent:
        """Get or create the Knowledge Agent instance."""
        if self._agent is None:
            self._agent = KnowledgeGroundedAgent(enable_planning=True)
        return self._agent

    @property
    def judge(self) -> DeepSearchQAJudge:
        """Get or create the DeepSearchQA Judge instance."""
        if self._judge is None:
            self._judge = DeepSearchQAJudge()
        return self._judge


# Shared context for the evaluation run
_context = EvaluationContext()


async def agent_task(*, item: Any, **kwargs: Any) -> str:  # noqa: ARG001
    """Run the Knowledge Agent on a dataset item.

    Parameters
    ----------
    item : ExperimentItem
        The Langfuse experiment item containing the question.

    Returns
    -------
    str
        The agent's response text.
    """
    question = item.input

    logger.info(f"Running agent on: {question[:80]}...")

    try:
        # Create a fresh agent for each task to avoid shared state issues
        # The agent has mutable state (_current_plan, _sessions) that causes
        # race conditions when shared across concurrent tasks
        agent = KnowledgeGroundedAgent(enable_planning=True)
        response = await agent.answer_async(question)
        logger.info(f"Agent completed: {len(response.text)} chars, {len(response.tool_calls)} tool calls")
        return response.text
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        return f"Error: {e}"


async def deepsearchqa_evaluator(
    *,
    input: str,  # noqa: A002
    output: str,
    expected_output: str,
    **kwargs: Any,
) -> list[Evaluation]:
    """Evaluate the agent's response using DeepSearchQA methodology.

    Uses the official DeepSearchQA LLM-as-judge to compute precision, recall,
    and F1 score based on semantic matching of answer components.

    Parameters
    ----------
    input : str
        The original question.
    output : str
        The agent's response.
    expected_output : str
        The ground truth answer.

    Returns
    -------
    list[Evaluation]
        List of Langfuse Evaluations with F1, precision, and recall scores.
    """
    # Get metadata for answer_type (default to "Set Answer" for multi-part answers)
    metadata = kwargs.get("metadata", {})
    answer_type = metadata.get("answer_type", "Set Answer")

    logger.info(f"Evaluating response (answer_type: {answer_type})...")

    try:
        _, result = await _context.judge.evaluate_with_details_async(
            question=input,
            answer=output,
            ground_truth=expected_output,
            answer_type=answer_type,
        )

        # Build detailed comment
        comment_parts = [
            f"Outcome: {result.outcome}",
            f"Precision: {result.precision:.2f}",
            f"Recall: {result.recall:.2f}",
            f"F1: {result.f1_score:.2f}",
        ]

        if result.explanation:
            comment_parts.append(f"\nExplanation: {result.explanation}")

        if result.correctness_details:
            found = sum(1 for v in result.correctness_details.values() if v)
            total = len(result.correctness_details)
            comment_parts.append(f"\nCorrectness: {found}/{total} items found")

        if result.extraneous_items:
            comment_parts.append(f"\nExtraneous: {len(result.extraneous_items)} items")

        logger.info(f"Evaluation complete: {result.outcome} (F1: {result.f1_score:.2f})")

        comment = "\n".join(comment_parts)

        # Map outcome to categorical value for Langfuse
        # These match the paper's four disjoint categories
        outcome_display = {
            "fully_correct": "Fully Correct",
            "correct_with_extraneous": "Correct with Extraneous",
            "partially_correct": "Partially Correct",
            "fully_incorrect": "Fully Incorrect",
        }

        # Return multiple evaluations for better visibility in Langfuse
        return [
            # Categorical outcome - the primary classification from the paper
            Evaluation(
                name="Outcome",
                value=outcome_display.get(result.outcome, result.outcome),
                comment=result.explanation,
            ),
            # Continuous metrics
            Evaluation(name="F1", value=result.f1_score, comment=comment),
            Evaluation(name="Precision", value=result.precision, comment=comment),
            Evaluation(name="Recall", value=result.recall, comment=comment),
        ]

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return [
            Evaluation(name="F1", value=0.0, comment=f"Evaluation error: {e}"),
            Evaluation(name="Precision", value=0.0, comment=f"Evaluation error: {e}"),
            Evaluation(name="Recall", value=0.0, comment=f"Evaluation error: {e}"),
        ]


def get_completed_item_ids(langfuse: Any, run_name: str) -> set[str]:
    """Get dataset item IDs that have already been evaluated in a run.

    Parameters
    ----------
    langfuse : Langfuse
        The Langfuse client.
    run_name : str
        Name of the experiment run.

    Returns
    -------
    set[str]
        Set of dataset item IDs that have completed evaluations.
    """
    try:
        # Fetch traces with matching run_name
        logger.info(f"Checking for existing evaluations in run '{run_name}'...")

        # Use the fetch_traces API to get traces with this run_name
        traces = langfuse.fetch_traces(
            name=run_name,
            limit=1000,  # Adjust if needed
        )

        completed_ids = set()
        for trace in traces.data:
            # Check if this trace has completed evaluations
            metadata = trace.metadata or {}
            trace_run_name = metadata.get("run_name")
            item_id = metadata.get("dataset_item_id")

            # Only count traces from this specific run that have evaluation scores
            if trace_run_name == run_name and item_id and hasattr(trace, "scores") and trace.scores:
                completed_ids.add(item_id)
                logger.debug(f"Found completed evaluation for item {item_id}")

        logger.info(f"Found {len(completed_ids)} completed evaluations")
        return completed_ids

    except Exception as e:
        logger.warning(f"Failed to fetch existing evaluations: {e}")
        logger.info("Will process all items")
        return set()


async def run_evaluation(
    dataset_name: str, experiment_name: str, max_concurrency: int = 1, resume: bool = False
) -> None:
    """Run the full evaluation experiment.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to evaluate against.
    experiment_name : str
        Name for this experiment run.
    max_concurrency : int
        Maximum concurrent agent runs (default 1 for sequential).
    resume : bool
        If True, skip items that have already been evaluated in this run.
    """
    client_manager = AsyncClientManager.get_instance()
    langfuse = client_manager.langfuse_client

    # Get the dataset
    logger.info(f"Loading dataset '{dataset_name}' from Langfuse...")
    try:
        dataset = langfuse.get_dataset(dataset_name)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Run langfuse_upload.py first to create the dataset.")
        return

    logger.info(f"Found dataset with {len(dataset.items)} items")

    # Handle resume: filter out already-completed items
    items_to_process = dataset.items
    if resume:
        completed_ids = get_completed_item_ids(langfuse, experiment_name)

        if completed_ids:
            items_to_process = [item for item in dataset.items if item.id not in completed_ids]
            skipped = len(dataset.items) - len(items_to_process)
            logger.info(f"Resume mode: Skipping {skipped} completed items")
            logger.info(f"Will process {len(items_to_process)} remaining items")

            if not items_to_process:
                logger.info("All items already evaluated. Nothing to do.")
                await client_manager.close()
                return
        else:
            logger.info("Resume mode: No completed items found, processing all")

    # Run the experiment
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Max concurrency: {max_concurrency}")

    # Create a filtered dataset if resuming
    if resume and len(items_to_process) < len(dataset.items):
        # We need to run experiment on filtered items
        # Use run_experiment directly with the filtered items via local data approach
        local_items: list[LocalExperimentItem] = [
            {
                "input": item.input,
                "expected_output": item.expected_output,
                "metadata": {**(item.metadata or {}), "dataset_item_id": item.id},
            }
            for item in items_to_process
        ]
        result = langfuse.run_experiment(
            name=experiment_name,
            data=local_items,
            description="Knowledge Agent evaluation using DeepSearchQA benchmark with LLM-as-judge (resumed)",
            task=agent_task,
            evaluators=[deepsearchqa_evaluator],
            max_concurrency=max_concurrency,
        )
    else:
        result = dataset.run_experiment(
            name=experiment_name,
            description="Knowledge Agent evaluation using DeepSearchQA benchmark with LLM-as-judge",
            task=agent_task,
            evaluators=[deepsearchqa_evaluator],
            max_concurrency=max_concurrency,
        )

    # Display results
    logger.info("Experiment complete!")
    logger.info(result.format().replace("\\n", "\n"))

    # Flush to ensure all data is sent to Langfuse
    logger.info("Flushing Langfuse data...")
    langfuse.flush()

    # Cleanup
    try:
        await client_manager.close()
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

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
from aieng.agent_evals.knowledge_agent.judges import DeepSearchQAJudge, TrajectoryQualityJudge
from dotenv import load_dotenv
from langfuse.experiment import Evaluation


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
        self._trajectory_judge: TrajectoryQualityJudge | None = None
        self.completed_item_ids: set[str] = set()

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

    @property
    def trajectory_judge(self) -> TrajectoryQualityJudge:
        """Get or create the Trajectory Quality Judge instance."""
        if self._trajectory_judge is None:
            self._trajectory_judge = TrajectoryQualityJudge()
        return self._trajectory_judge

    def reset_completed_ids(self) -> None:
        """Reset the completed item IDs for a new evaluation run."""
        self.completed_item_ids.clear()


# Shared context for the evaluation run
_context = EvaluationContext()


async def agent_task(*, item: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG001
    """Run the Knowledge Agent on a dataset item.

    Parameters
    ----------
    item : ExperimentItem
        The Langfuse experiment item containing the question.

    Returns
    -------
    dict[str, Any]
        Dictionary containing 'text' (the response) and 'agent_response'
        (full response object).
    """
    question = item.input
    item_id = item.id

    # Skip if already completed (for resume mode)
    if item_id in _context.completed_item_ids:
        logger.info(f"Skipping completed item {item_id}")
        # Return a marker that tells evaluator to skip
        return {"text": "__SKIP__", "agent_response": None}

    logger.info(f"Running agent on: {question[:80]}...")

    try:
        # Create a fresh agent for each task to avoid shared state issues
        # The agent has mutable state (_current_plan, _sessions) that causes
        # race conditions when shared across concurrent tasks
        agent = KnowledgeGroundedAgent(enable_planning=True)
        response = await agent.answer_async(question)
        logger.info(f"Agent completed: {len(response.text)} chars, {len(response.tool_calls)} tool calls")

        # Return both text and full response for trajectory evaluation
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
    **kwargs: Any,
) -> list[Evaluation]:
    """Evaluate the agent's response using DeepSearchQA methodology.

    Uses the official DeepSearchQA LLM-as-judge to compute precision, recall,
    and F1 score based on semantic matching of answer components.

    Parameters
    ----------
    input : str
        The original question.
    output : dict[str, Any]
        Dictionary containing 'text' and 'agent_response'.
    expected_output : str
        The ground truth answer.

    Returns
    -------
    list[Evaluation]
        List of Langfuse Evaluations with F1, precision, and recall scores.
    """
    # Extract text from output dict
    output_text = output.get("text", "") if isinstance(output, dict) else str(output)

    # Skip evaluation for items marked as already completed
    if output_text == "__SKIP__":
        logger.debug("Skipping evaluation for already-completed item")
        return []

    # Get metadata for answer_type (default to "Set Answer" for multi-part answers)
    metadata = kwargs.get("metadata", {})
    answer_type = metadata.get("answer_type", "Set Answer")

    logger.info(f"Evaluating response (answer_type: {answer_type})...")

    try:
        _, result = await _context.judge.evaluate_with_details_async(
            question=input,
            answer=output_text,
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


def _count_replanning_events(plan: Any) -> int:
    """Count how many times the agent replanned during execution.

    Parameters
    ----------
    plan : ResearchPlan
        The research plan which tracks replanning.

    Returns
    -------
    int
        Number of replanning events (0 if no replanning occurred).
    """
    if not plan or not hasattr(plan, "reasoning"):
        return 0

    # Check if plan reasoning indicates replanning
    # Replanning is marked by "Replanned:" prefix in reasoning field
    if plan.reasoning.startswith("Replanned:"):
        # Count by checking if there are completed steps preserved
        # (replanning preserves completed steps and adds new ones)
        completed_steps = [s for s in plan.steps if s.status == "completed"]
        # If there are completed steps in a replanned plan, replanning occurred
        # at least once. This is a conservative estimate - actual count may be
        # higher with multiple replans
        return 1 if completed_steps else 0

    return 0


async def trajectory_evaluator(
    *,
    input: str,  # noqa: A002
    output: dict[str, Any],
    **kwargs: Any,  # noqa: ARG001
) -> list[Evaluation]:
    """Evaluate the agent's trajectory with LLM-as-judge and quantitative metrics.

    Computes both categorical quality assessment and quantitative metrics:
    - Trajectory Quality: LLM judge rates trajectory as High/Medium/Low
    - Replanning Frequency: How many times the agent adapted its plan

    Parameters
    ----------
    input : str
        The original question.
    output : dict[str, Any]
        Dictionary containing 'text' and 'agent_response'.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    list[Evaluation]
        List of Langfuse Evaluations with trajectory metrics.
    """
    # Extract agent_response from output dict
    output_text = output.get("text", "") if isinstance(output, dict) else str(output)

    # Skip evaluation for items marked as already completed
    if output_text == "__SKIP__":
        logger.debug("Skipping trajectory evaluation for already-completed item")
        return []

    # Get agent_response from output dict
    agent_response = output.get("agent_response") if isinstance(output, dict) else None

    if not agent_response:
        logger.error("No agent_response in output dict")
        return []

    logger.info("Computing trajectory metrics...")

    try:
        # Count replanning events from the plan reasoning
        replanning_count = _count_replanning_events(agent_response.plan)

        # Build replanning comment
        if replanning_count == 0:
            replanning_comment = "No replanning occurred - agent executed original plan."
        elif replanning_count == 1:
            replanning_comment = "Agent replanned once during execution to adapt to findings."
        else:
            replanning_comment = f"Agent replanned {replanning_count} times to adapt strategy."

        # Extract search queries from tool calls
        search_queries = [
            tc.get("arguments", {}).get("query", "")
            for tc in agent_response.tool_calls
            if tc.get("name") == "google_search" and tc.get("arguments", {}).get("query")
        ]

        # Get LLM-based trajectory quality evaluation
        quality_result = await _context.trajectory_judge.evaluate_async(
            question=input,
            plan=agent_response.plan,
            tool_calls=agent_response.tool_calls,
            search_queries=search_queries,
        )

        # Build detailed quality comment
        quality_comment_parts = [
            f"Quality: {quality_result.quality_category}",
            f"\n{quality_result.explanation}",
            f"\n\nEfficiency: {quality_result.efficiency_notes}",
            f"Logical Soundness: {quality_result.logical_soundness_notes}",
            f"Source Quality: {quality_result.source_quality_notes}",
            f"Replanning: {quality_result.replanning_notes}",
        ]
        quality_comment = "\n".join(quality_comment_parts)

        logger.info(
            f"Trajectory evaluation complete: {quality_result.quality_category} quality, "
            f"{replanning_count} replanning events"
        )

        # Return both evaluations: categorical quality and quantitative
        # replanning frequency
        return [
            Evaluation(
                name="Trajectory Quality",
                value=quality_result.quality_category,
                comment=quality_comment,
            ),
            Evaluation(
                name="Replanning Frequency",
                value=replanning_count,
                comment=replanning_comment,
            ),
        ]

    except Exception as e:
        logger.error(f"Trajectory evaluation failed: {e}")
        return [
            Evaluation(name="Trajectory Quality", value="Medium", comment=f"Evaluation error: {e}"),
            Evaluation(name="Replanning Frequency", value=0, comment=f"Evaluation error: {e}"),
        ]


def get_completed_item_ids(langfuse: Any, run_name: str, dataset_id: str) -> set[str]:
    """Get dataset item IDs that have already been evaluated in a run.

    Parameters
    ----------
    langfuse : Langfuse
        The Langfuse client.
    run_name : str
        Name of the experiment run (the dataset run name).
    dataset_id : str
        ID of the dataset.

    Returns
    -------
    set[str]
        Set of dataset item IDs that have completed evaluations.
    """
    try:
        # Use dataset_run_items API to get items for this specific run
        logger.info(f"Checking for existing evaluations in run '{run_name}'...")

        completed_ids = set()
        page = 1
        limit = 50

        while True:
            # Query dataset run items for this specific run
            run_items_response = langfuse.api.dataset_run_items.list(
                dataset_id=dataset_id,
                run_name=run_name,
                limit=limit,
                page=page,
            )

            if not run_items_response.data:
                break

            for run_item in run_items_response.data:
                # Check if this item has a trace with evaluation scores
                if run_item.trace_id:
                    try:
                        trace = langfuse.api.trace.get(run_item.trace_id)
                        if hasattr(trace, "scores") and trace.scores:
                            completed_ids.add(run_item.dataset_item_id)
                            logger.debug(f"Found completed evaluation for item {run_item.dataset_item_id}")
                    except Exception as e:
                        logger.debug(f"Could not fetch trace {run_item.trace_id}: {e}")

            # Check if we've fetched all items
            if len(run_items_response.data) < limit:
                break

            page += 1

        logger.info(f"Found {len(completed_ids)} completed evaluations")
        return completed_ids

    except Exception as e:
        logger.warning(f"Failed to fetch existing evaluations: {e}")
        logger.info("Will process all items")
        return set()


async def process_single_item(item: Any, experiment_name: str) -> None:
    """Process a single dataset item using item.run() for manual trace linking.

    Parameters
    ----------
    item : DatasetItemClient
        The dataset item to process.
    experiment_name : str
        The run name to link this trace to.
    """
    try:
        with item.run(run_name=experiment_name) as root_span:
            logger.info(f"Processing item {item.id}: {item.input[:80]}...")

            # Run the agent
            agent = KnowledgeGroundedAgent(enable_planning=True)
            response = await agent.answer_async(item.input)

            logger.info(f"Agent completed: {len(response.text)} chars, {len(response.tool_calls)} tool calls")

            # Create output dict in same format as agent_task
            output_dict = {
                "text": response.text,
                "agent_response": response,
            }

            # Run evaluations
            outcome_evaluations = await deepsearchqa_evaluator(
                input=item.input,
                output=output_dict,
                expected_output=item.expected_output,
                metadata=item.metadata,
            )

            trajectory_evaluations = await trajectory_evaluator(
                input=item.input,
                output=output_dict,
            )

            all_evaluations = outcome_evaluations + trajectory_evaluations

            # Add scores to the trace using root_span.score_trace()
            # This properly links scores to the dataset run item
            for evaluation in all_evaluations:
                root_span.score_trace(
                    name=evaluation.name,
                    value=evaluation.value,
                    comment=evaluation.comment,
                )

            logger.info(f"Item {item.id} complete with {len(all_evaluations)} evaluations")

    except Exception as e:
        logger.error(f"Item {item.id} failed: {e}")


async def process_items_manually(items: list[Any], experiment_name: str, max_concurrency: int) -> None:
    """Process items manually using item.run() to append to existing run.

    Parameters
    ----------
    items : list[DatasetItemClient]
        Items to process.
    experiment_name : str
        The run name to link traces to.
    max_concurrency : int
        Maximum concurrent tasks.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_with_limit(item: Any) -> None:
        async with semaphore:
            await process_single_item(item, experiment_name)

    tasks = [process_with_limit(item) for item in items]
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("Manual processing complete!")


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

    # Handle resume: get completed item IDs and store in context
    if resume:
        _context.completed_item_ids = get_completed_item_ids(langfuse, experiment_name, dataset.id)

        if _context.completed_item_ids:
            remaining = len(dataset.items) - len(_context.completed_item_ids)
            logger.info(f"Resume mode: Found {len(_context.completed_item_ids)} completed items")
            logger.info(f"Will process {remaining} remaining items")

            if remaining == 0:
                logger.info("All items already evaluated. Nothing to do.")
                await client_manager.close()
                return
        else:
            logger.info("Resume mode: No completed items found, processing all")
    else:
        # Reset completed IDs for a fresh run
        _context.reset_completed_ids()

    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Max concurrency: {max_concurrency}")

    # If resuming, use item.run() to append to existing run
    # Otherwise use dataset.run_experiment for convenience
    if resume and _context.completed_item_ids:
        logger.info("Using manual trace creation to append to existing run...")
        items_to_process = [item for item in dataset.items if item.id not in _context.completed_item_ids]

        # Process items manually using item.run() to link to the same run_name
        await process_items_manually(items_to_process, experiment_name, max_concurrency)
    else:
        # No resume, use the convenient run_experiment API
        result = dataset.run_experiment(
            name=experiment_name,
            description="Knowledge Agent evaluation with outcome (DeepSearchQA) and trajectory (process supervision) judges",
            task=agent_task,
            evaluators=[deepsearchqa_evaluator, trajectory_evaluator],
            max_concurrency=max_concurrency,
        )
        logger.info("Experiment complete!")
        logger.info(result.format().replace("\\n", "\n"))

    # Manual processing path doesn't have a result object
    if resume and _context.completed_item_ids:
        logger.info("Resume processing complete!")

    # Cleanup - client_manager.close() will flush Langfuse
    logger.info("Closing client manager and flushing data...")
    try:
        await client_manager.close()
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

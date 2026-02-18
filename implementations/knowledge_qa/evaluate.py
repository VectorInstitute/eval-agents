"""Evaluate the Knowledge Agent using Langfuse experiments.

This script runs the Knowledge Agent against a Langfuse dataset and evaluates
results using the DeepSearchQA LLM-as-judge methodology. Results are automatically
logged to Langfuse for analysis and comparison.

Optionally, trace-level groundedness evaluation can be enabled to check if agent
outputs are supported by tool observations.

Usage:
    # Run a full evaluation
    python evaluate.py

    # Run with custom dataset and experiment name
    python evaluate.py --dataset-name "MyDataset" --experiment-name "v2-test"

    # Enable trace groundedness evaluation
    ENABLE_TRACE_GROUNDEDNESS=true python evaluate.py
"""

import asyncio
import logging
import os
from typing import Any

import click
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation import run_experiment, run_experiment_with_trace_evals
from aieng.agent_evals.evaluation.graders import create_trace_groundedness_evaluator
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.evaluation.types import EvaluationResult
from aieng.agent_evals.knowledge_qa.agent import KnowledgeGroundedAgent
from aieng.agent_evals.knowledge_qa.deepsearchqa_grader import DeepSearchQAResult, evaluate_deepsearchqa_async
from aieng.agent_evals.logging_config import setup_logging
from dotenv import load_dotenv
from langfuse.experiment import Evaluation, ExperimentResult


load_dotenv(verbose=True)
setup_logging(level=logging.INFO, show_time=True, show_path=False)
logger = logging.getLogger(__name__)


DEFAULT_DATASET_NAME = "DeepSearchQA-Subset"
DEFAULT_EXPERIMENT_NAME = "Knowledge Agent Evaluation"

# Configuration for trace groundedness evaluation
ENABLE_TRACE_GROUNDEDNESS = os.getenv("ENABLE_TRACE_GROUNDEDNESS", "false").lower() in ("true", "1", "yes")


async def agent_task(*, item: Any, **kwargs: Any) -> str:  # noqa: ARG001
    """Run the Knowledge Agent on a dataset item.

    Parameters
    ----------
    item : Any
        The Langfuse experiment item containing the question.
    **kwargs : Any
        Additional arguments from the harness (unused).

    Returns
    -------
    str
        The agent's answer text. Rich execution data (plan, tool calls,
        sources, reasoning chain) is attached to the Langfuse span metadata.
    """
    question = item.input
    logger.info(f"Running agent on: {question[:80]}...")

    try:
        agent = KnowledgeGroundedAgent(enable_planning=True)  # type: ignore[call-arg]
        response = await agent.answer_async(question)
        logger.info(f"Agent completed: {len(response.text)} chars, {len(response.tool_calls)} tool calls")

        # Attach rich execution data to the span metadata so it's inspectable
        # in Langfuse without cluttering the output field.
        client_manager = AsyncClientManager.get_instance()
        client_manager.langfuse_client.update_current_span(
            metadata=response.model_dump(exclude={"text"}),
        )

        return response.text
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        return f"Error: {e}"


async def deepsearchqa_evaluator(
    *,
    input: str,  # noqa: A002
    output: str,
    expected_output: str,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,  # noqa: ARG001
) -> list[Evaluation]:
    """Evaluate the agent's response using DeepSearchQA methodology.

    This evaluator uses the modern async infrastructure with shared client
    management and retry logic.

    Parameters
    ----------
    input : str
        The original question.
    output : str
        The agent's answer text.
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
    output_text = str(output)
    answer_type = metadata.get("answer_type", "Set Answer") if metadata else "Set Answer"

    logger.info(f"Evaluating response (answer_type: {answer_type})...")

    try:
        # Use the modern async evaluator with default config
        result = await evaluate_deepsearchqa_async(
            question=input,
            answer=output_text,
            ground_truth=expected_output,
            answer_type=answer_type,
            model_config=LLMRequestConfig(temperature=0.0),
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
    enable_trace_groundedness: bool = False,
) -> ExperimentResult | EvaluationResult:
    """Run the full evaluation experiment.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to evaluate against.
    experiment_name : str
        Name for this experiment run.
    max_concurrency : int, optional
        Maximum concurrent agent runs, by default 1.
    enable_trace_groundedness : bool, optional
        Whether to enable trace-level groundedness evaluation, by default False.
    """
    client_manager = AsyncClientManager.get_instance()

    try:
        logger.info(f"Starting experiment '{experiment_name}' on dataset '{dataset_name}'")
        logger.info(f"Max concurrency: {max_concurrency}")
        logger.info(f"Trace groundedness: {'enabled' if enable_trace_groundedness else 'disabled'}")

        result: ExperimentResult | EvaluationResult
        if enable_trace_groundedness:
            # Create trace groundedness evaluator
            # Only consider web_fetch and google_search tools as evidence
            groundedness_evaluator = create_trace_groundedness_evaluator(
                name="trace_groundedness",
                model_config=LLMRequestConfig(temperature=0.0),
            )

            # Run with trace evaluations
            result = run_experiment_with_trace_evals(
                dataset_name=dataset_name,
                name=experiment_name,
                description="Knowledge Agent evaluation with DeepSearchQA judge and trace groundedness",
                task=agent_task,
                evaluators=[deepsearchqa_evaluator],  # Item-level evaluators
                trace_evaluators=[groundedness_evaluator],  # Trace-level evaluators
                max_concurrency=max_concurrency,
            )
        else:
            # Run without trace evaluations
            result = run_experiment(
                dataset_name=dataset_name,
                name=experiment_name,
                description="Knowledge Agent evaluation with DeepSearchQA judge",
                task=agent_task,
                evaluators=[deepsearchqa_evaluator],
                max_concurrency=max_concurrency,
            )

        logger.info("Experiment complete!")
        # Handle both ExperimentResult and EvaluationResult
        if isinstance(result, EvaluationResult):
            # EvaluationResult from run_experiment_with_trace_evals
            logger.info(f"Results: {result.experiment}")
            if result.trace_evaluations:
                trace_evals = result.trace_evaluations
                logger.info(
                    f"Trace evaluations: {len(trace_evals.evaluations_by_trace_id)} traces, "
                    f"{len(trace_evals.skipped_trace_ids)} skipped, {len(trace_evals.failed_trace_ids)} failed"
                )
        else:
            # ExperimentResult from run_experiment
            logger.info(f"Results: {result}")

        return result

    finally:
        logger.info("Closing client manager and flushing data...")
        try:
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
@click.option(
    "--enable-trace-groundedness",
    is_flag=True,
    default=ENABLE_TRACE_GROUNDEDNESS,
    help="Enable trace-level groundedness evaluation.",
)
def cli(dataset_name: str, experiment_name: str, max_concurrency: int, enable_trace_groundedness: bool) -> None:
    """Run Knowledge Agent evaluation using Langfuse experiments."""
    asyncio.run(
        run_evaluation(
            dataset_name,
            experiment_name,
            max_concurrency,
            enable_trace_groundedness,
        )
    )


if __name__ == "__main__":
    cli()

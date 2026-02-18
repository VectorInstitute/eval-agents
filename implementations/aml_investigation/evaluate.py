"""Evaluate the AML investigation agent.

This script uploads the AML investigation dataset to Langfuse, runs the evaluation
experiment with item-level and trace-level evaluators, and displays the results
in the console. The evaluation includes deterministic grading based on known ground
truth, as well as LLM-based assessments of narrative quality and trace groundedness.

Example
-------
$ uv run --env-file .env implementations/aml_investigation/evaluate.py \
    --dataset-path implementations/aml_investigation/data/aml_cases.jsonl \
    --dataset-name AML-investigation
"""

import asyncio
import logging
from functools import partial

import click
from aieng.agent_evals.aml_investigation.agent import create_aml_investigation_agent
from aieng.agent_evals.aml_investigation.graders import (
    item_level_deterministic_grader,
    run_level_grader,
    trace_deterministic_grader,
)
from aieng.agent_evals.aml_investigation.task import AmlInvestigationTask
from aieng.agent_evals.db_manager import DbManager
from aieng.agent_evals.display import create_console, display_info, display_metrics_table
from aieng.agent_evals.evaluation import TraceWaitConfig
from aieng.agent_evals.evaluation.experiment import run_experiment_with_trace_evals
from aieng.agent_evals.evaluation.graders import (
    create_llm_as_judge_evaluator,
    create_trace_groundedness_evaluator,
)
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.langfuse import upload_dataset_to_langfuse
from rich.logging import RichHandler


logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(show_path=False)], force=True)

# Silence verbose INFO logs from Google ADK
logging.getLogger("google_adk").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="Path to the dataset JSONL file.",
)
@click.option("--dataset-name", type=str, required=True, help="Name of the dataset to upload to Langfuse.")
@click.option(
    "--agent-timeout",
    type=click.IntRange(min=1, max_open=True),
    default=300,
    help="Timeout in seconds for the AML investigation agent.",
)
@click.option(
    "--llm-judge-timeout",
    type=click.IntRange(min=1, max_open=True),
    default=120,
    help="Timeout in seconds for LLM judge evaluations.",
)
@click.option(
    "--llm-judge-retries",
    type=click.IntRange(min=0, max_open=True),
    default=3,
    help="Number of retry attempts for LLM judge evaluations in case of failures.",
)
@click.option(
    "--max-concurrent-cases",
    type=click.IntRange(min=1, max=10),
    default=5,
    help="Maximum number of concurrent cases to process during evaluation.",
)
@click.option(
    "--max-concurrent-traces",
    type=click.IntRange(min=1, max=10),
    default=10,
    help="Maximum number of concurrent traces to process during evaluation.",
)
@click.option(
    "--max-trace-wait-time",
    type=click.IntRange(min=1, max_open=True),
    default=300,
    help="Maximum time in seconds to wait for trace data to be ready during evaluation.",
)
def cli(
    dataset_path: str,
    dataset_name: str,
    llm_judge_timeout: int,
    llm_judge_retries: int,
    agent_timeout: int,
    max_concurrent_cases: int,
    max_concurrent_traces: int,
    max_trace_wait_time: int,
) -> None:
    """Evaluate AML Investigation agent on a given dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset JSONL file containing AML cases.
    dataset_name : str
        Name of the dataset to upload to Langfuse for evaluation.
    llm_judge_timeout : int
        Timeout in seconds for LLM-based judge evaluations.
    llm_judge_retries : int
        Number of retry attempts for LLM judge evaluations in case of failures.
    agent_timeout : int
        Timeout in seconds for the AML investigation agent to complete each case.
    max_concurrent_cases : int
        Maximum number of concurrent cases to process during evaluation.
    max_concurrent_traces : int
        Maximum number of concurrent traces to process during evaluation.
    max_trace_wait_time : int
        Maximum time in seconds to wait for trace data to be ready during evaluation.
    """
    # Create console for rich formatted output
    console = create_console(force_jupyter=False)

    # Upload dataset to Langfuse
    asyncio.run(upload_dataset_to_langfuse(dataset_path, dataset_name))

    # Define graders/evaluators
    # Item-level LLM-as-a-judge evaluator assesses the quality of the agent's
    # narrative output based on a rubric.
    narrative_quality_evaluator = create_llm_as_judge_evaluator(
        name="narrative_quality",
        rubric_markdown="implementations/aml_investigation/rubrics/narrative_pattern_quality.md",
        model_config=LLMRequestConfig(timeout_sec=llm_judge_timeout, retry_max_attempts=llm_judge_retries),
    )

    # Trace-level graders assess the correctness of tool use and the groundedness
    # of the agent's response based on trace data.
    db_policy = DbManager().aml_db().policy
    deterministic_trace_grader = partial(trace_deterministic_grader, db_policy=db_policy)
    trace_groundedness_evaluator = create_trace_groundedness_evaluator(
        model_config=LLMRequestConfig(timeout_sec=llm_judge_timeout, retry_max_attempts=llm_judge_retries)
    )

    agent = create_aml_investigation_agent(timeout_sec=agent_timeout)
    results = run_experiment_with_trace_evals(
        dataset_name=dataset_name,
        name="AML Investigation Evaluation",
        task=AmlInvestigationTask(agent=agent),
        evaluators=[item_level_deterministic_grader, narrative_quality_evaluator],
        trace_evaluators=[deterministic_trace_grader, trace_groundedness_evaluator],
        run_evaluators=[run_level_grader],
        max_concurrency=max_concurrent_cases,
        trace_max_concurrency=max_concurrent_traces,
        trace_wait=TraceWaitConfig(max_wait_sec=max_trace_wait_time),
    )

    # Display item-level results
    console.print("\n[bold cyan]📋 Item-Level Results[/bold cyan]\n")
    for idx, item_result in enumerate(results.experiment.item_results, start=1):
        item_metrics = {eval_.name: eval_.value for eval_ in item_result.evaluations}
        # Try to get item ID from metadata, fall back to index
        item_id = f"Item {idx}"
        try:
            item = item_result.item
            if item and isinstance(item, dict):
                metadata = item.get("metadata", {})
                if metadata and isinstance(metadata, dict):
                    item_id = metadata.get("id", item_id)
            elif item and hasattr(item, "metadata"):
                metadata = getattr(item, "metadata", None)
                if metadata and isinstance(metadata, dict):
                    item_id = metadata.get("id", item_id)
        except Exception:
            pass  # Keep default item_id

        display_metrics_table(
            metrics=item_metrics,
            title=str(item_id),
            console=console,
        )

    # Display run-level metrics
    if hasattr(results.experiment, "run_evaluations") and results.experiment.run_evaluations:
        console.print("\n[bold green]📊 Run-Level Metrics[/bold green]\n")
        run_metrics = {eval_.name: eval_.value for eval_ in results.experiment.run_evaluations}
        display_metrics_table(metrics=run_metrics, title="Aggregate Performance", console=console)

    # Display trace evaluation summary
    if results.trace_evaluations:
        console.print("\n[bold magenta]🔍 Trace Evaluation Summary[/bold magenta]\n")
        trace_summary: dict[str, float | int | str] = {
            "Successful Traces": len(results.trace_evaluations.evaluations_by_trace_id),
            "Skipped Traces": len(results.trace_evaluations.skipped_trace_ids),
            "Failed Traces": len(results.trace_evaluations.failed_trace_ids),
        }
        display_metrics_table(metrics=trace_summary, title="Trace Processing", console=console)

    display_info("Evaluation complete! Results have been uploaded to Langfuse.", console=console)


if __name__ == "__main__":
    cli()

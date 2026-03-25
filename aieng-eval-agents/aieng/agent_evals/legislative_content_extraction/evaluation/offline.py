"""Evaluate the legislative content extraction agent against a Langfuse dataset."""

import logging
from collections.abc import Mapping
from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation import run_experiment_with_trace_evals
from aieng.agent_evals.evaluation.graders import create_llm_as_judge_evaluator
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.evaluation.types import Evaluation
from aieng.agent_evals.legislative_content_extraction.graders import (
    item_level_deterministic_grader,
    trace_deterministic_grader,
    run_level_grader,
)
from aieng.agent_evals.legislative_content_extraction.task import LegislativeExtractionTask

logger = logging.getLogger(__name__)

SUMMARY_RUBRIC_PATH = "implementations/legislative_content_extraction/rubrics/legislative_summary_quality.md"

# Focused prompt: only show the summary fields to the judge, not the entire extracted dict
_SUMMARY_PROMPT_TEMPLATE = """\
# Input (PDF used for extraction)
{input}

# Expected Summary (Ground Truth)
{expected_output}

# Candidate Summary (To Evaluate)
{output}
"""


def _make_summary_judge(model_config: LLMRequestConfig) -> Any:
    """Return an evaluator that judges only the summary field."""
    _judge = create_llm_as_judge_evaluator(
        name="summary_quality",
        rubric_markdown=SUMMARY_RUBRIC_PATH,
        prompt_template=_SUMMARY_PROMPT_TEMPLATE,
        model_config=model_config,
    )

    async def summary_judge(
        *,
        input: Any,  # noqa: A002
        output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Evaluation]:
        """Extract summary fields and delegate to the LLM judge."""
        candidate_summary = (
            output.get("summary") if isinstance(output, Mapping) else None
        )
        expected_summary = (
            expected_output.get("summary") if isinstance(expected_output, Mapping) else None
        )
        return await _judge(
            input=input,
            output=candidate_summary,
            expected_output=expected_summary,
            metadata=metadata,
        )

    summary_judge.__name__ = "summary_judge"
    return summary_judge


async def evaluate(
    dataset_name: str,
    files_dir: str | None = None,
    max_concurrency: int = 5,
    llm_judge_timeout: int = 120,
    llm_judge_retries: int = 3,
    run_name: str | None = None,
    description: str | None = None,
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
    llm_judge_timeout : int, optional
        Timeout in seconds for LLM judge calls, by default 120.
    llm_judge_retries : int, optional
        Number of retry attempts for LLM judge calls, by default 3.
    run_name : str | None, optional
        Optional run name for the experiment in Langfuse.
    description : str | None, optional
        Optional description for the experiment run.
    """
    task = LegislativeExtractionTask(files_dir=files_dir)

    model_config = LLMRequestConfig(
        timeout_sec=llm_judge_timeout,
        retry_max_attempts=llm_judge_retries,
    )
    summary_judge = _make_summary_judge(model_config)

    result = run_experiment_with_trace_evals(
        dataset_name=dataset_name,
        name="Evaluate Legislative Content Extraction Agent",
        task=task,
        evaluators=[item_level_deterministic_grader, summary_judge],
        trace_evaluators=[trace_deterministic_grader],
        run_evaluators=[run_level_grader],
        run_name=run_name,
        description=description,
        max_concurrency=max_concurrency,
    )

    logger.info("Evaluation complete. %d items processed.", len(result.experiment.item_results))

    try:
        await AsyncClientManager.get_instance().close()
    except Exception as exc:
        logger.warning("Client manager not closed cleanly: %s", exc)

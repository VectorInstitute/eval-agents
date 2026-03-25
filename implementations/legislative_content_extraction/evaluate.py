"""
Evaluate the legislative content extraction agent against a Langfuse dataset.

Example
-------
$ python -m implementations.legislative_content_extraction.evaluate \
    --dataset-name LegislativeContentExtractionEval
"""

import asyncio

import click
from aieng.agent_evals.legislative_content_extraction.evaluation.offline import evaluate
from dotenv import load_dotenv

load_dotenv(verbose=True)

DEFAULT_DATASET_NAME = "LegislativeContentExtractionEval"
DEFAULT_FILES_DIR = "implementations/legislative_content_extraction/files"


@click.command()
@click.option(
    "--dataset-name",
    default=DEFAULT_DATASET_NAME,
    help="Name of the Langfuse dataset to evaluate against.",
)
@click.option(
    "--files-dir",
    default=DEFAULT_FILES_DIR,
    help="Local directory where PDF files are cached.",
)
@click.option(
    "--max-concurrency",
    default=5,
    type=int,
    help="Maximum concurrent agent runs (default: 5).",
)
@click.option(
    "--llm-judge-timeout",
    default=120,
    type=int,
    help="Timeout in seconds for LLM judge evaluations (default: 120).",
)
@click.option(
    "--llm-judge-retries",
    default=3,
    type=int,
    help="Number of retry attempts for LLM judge evaluations (default: 3).",
)
@click.option(
    "--run-name",
    default=None,
    type=str,
    help="Optional run name for the experiment in Langfuse.",
)
@click.option(
    "--description",
    default=None,
    type=str,
    help="Optional description for the experiment run.",
)
def cli(
    dataset_name: str,
    files_dir: str,
    max_concurrency: int,
    llm_judge_timeout: int,
    llm_judge_retries: int,
    run_name: str | None,
    description: str | None,
):
    asyncio.run(
        evaluate(
            dataset_name,
            files_dir=files_dir,
            max_concurrency=max_concurrency,
            llm_judge_timeout=llm_judge_timeout,
            llm_judge_retries=llm_judge_retries,
            run_name=run_name,
            description=description,
        )
    )


if __name__ == "__main__":
    cli()
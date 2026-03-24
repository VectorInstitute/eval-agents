"""
Evaluate the report generation agent against a Langfuse dataset.

Example
-------
$ python -m implementations.report_generation.evaluate
$ python -m implementations.report_generation.evaluate \
    --dataset-name <dataset name>
"""

import asyncio

import click
from aieng.agent_evals.report_generation.evaluation.offline import evaluate
from dotenv import load_dotenv

from implementations.report_generation.data.langfuse_upload import DEFAULT_EVALUATION_DATASET_NAME
from implementations.report_generation.env_vars import get_reports_output_path


load_dotenv(verbose=True)


@click.command()
@click.option(
    "--dataset-name",
    required=False,
    default=DEFAULT_EVALUATION_DATASET_NAME,
    help="Name of the Langfuse dataset to evaluate against.",
)
@click.option(
    "--max-concurrency",
    default=5,
    type=int,
    help="Maximum concurrent agent runs (default: 5).",
)
def cli(dataset_name: str, max_concurrency: int):
    """Command line interface to call the evaluate function.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to evaluate against.
        Default is DEFAULT_EVALUATION_DATASET_NAME.
    """
    asyncio.run(
        evaluate(
            dataset_name,
            reports_output_path=get_reports_output_path(),
            max_concurrency=max_concurrency,
        )
    )


if __name__ == "__main__":
    cli()

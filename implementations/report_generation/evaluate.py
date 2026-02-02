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
from aieng.agent_evals.report_generation.evaluation import evaluate
from dotenv import load_dotenv

from implementations.report_generation.data.langfuse_upload import DEFAULT_EVALUATION_DATASET_NAME
from implementations.report_generation.demo import (
    get_langfuse_project_name,
    get_reports_output_path,
    get_sqlite_db_path,
)


load_dotenv(verbose=True)


@click.command()
@click.option(
    "--dataset-name",
    required=False,
    default=DEFAULT_EVALUATION_DATASET_NAME,
    help="Name of the Langfuse dataset to evaluate against.",
)
def cli(dataset_name: str):
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
            sqlite_db_path=get_sqlite_db_path(),
            reports_output_path=get_reports_output_path(),
            langfuse_project_name=get_langfuse_project_name(),
        )
    )


if __name__ == "__main__":
    cli()

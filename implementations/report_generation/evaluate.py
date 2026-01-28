"""Evaluate the report generation agent against a Langfuse dataset."""

import asyncio
import logging

import agents
import click
from aieng.agent_evals.async_client_manager import AsyncClientManager
from dotenv import load_dotenv
from langfuse._client.datasets import DatasetItemClient
from langfuse.experiment import Evaluation
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from implementations.report_generation.data.langfuse_upload import DEFAULT_EVALUATION_DATASET_NAME
from implementations.report_generation.main import get_report_generation_agent


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


EVALUATOR_INSTRUCTIONS = """\
Evaluate whether the "Proposed Answer" to the given "Question" matches the "Ground Truth"."""

ADDITONAL_EVALUATOR_INSTRUCTIONS = """\
Disregard the following aspects when comparing the "Proposed Answer" to the "Ground Truth":
- The order of the items should not matter, unless explicitly specified in the "Question".
- The formatting of the values should not matter, unless explicitly specified in the "Question".
- The column and row names have to be similar but not necessarily exact, unless explicitly specified in the "Question".
- The filename has to be similar by name but not necessarily exact, unless explicitly specified in the "Question".
- The numerical values should be equal to the second decimal place.
"""

EVALUATOR_TEMPLATE = """\
# Question

{question}

# Ground Truth

{ground_truth}

# Proposed Answer

{proposed_response}

"""


class EvaluatorResponse(BaseModel):
    """Typed response from the evaluator."""

    explanation: str
    is_answer_correct: bool


async def evaluate(dataset_name: str):
    """Evaluate the report generation agent against a Langfuse dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to evaluate against.
    """
    # Get the client manager singleton instance and langfuse client
    client_manager = AsyncClientManager.get_instance()
    langfuse_client = client_manager.langfuse_client

    # Find the dataset in Langfuse
    dataset = langfuse_client.get_dataset(dataset_name)

    # Run the experiment with the agent task and evaluator
    # against the dataset items
    result = dataset.run_experiment(
        name="Evaluate Report Generation Agent",
        description="Evaluate the Report Generation Agent with data from Langfuse",
        task=agent_task,
        evaluators=[llm_evaluator],
        max_concurrency=1,
    )

    # Log the evaluation result
    logger.info(result.format().replace("\\n", "\n"))

    try:
        # Gracefully close the services
        await client_manager.close()
    except Exception as e:
        logger.warning(f"Client manager services not closed successfully: {e}")


async def agent_task(*, item: DatasetItemClient, **kwargs) -> str | None:
    """Run the report generation agent against an item from a Langfuse dataset.

    Parameters
    ----------
    item : DatasetItemClient
        The item from the Langfuse dataset to evaluate against.

    Returns
    -------
    str | None
        The arguments sent by the report generation agent to the write_report_to_file
        function. Returns None if the agent did not call the function.
    """
    # Define and run the report generation agent
    report_generation_agent = get_report_generation_agent(enable_trace=True)
    result = await run_agent_with_retry(report_generation_agent, item.input)

    # Extract the report data from the result by returning the
    # arguments to the write_report_to_file function call
    for raw_response in result.raw_responses:
        for output in raw_response.output:
            if isinstance(output, ResponseFunctionToolCall) and "write_report_to_file" in output.name:
                return output.arguments

    return None


async def llm_evaluator(*, input: str, output: str, expected_output: str, **kwargs) -> Evaluation:
    # ruff: noqa: A002
    """Evaluate the proposed answer against the ground truth.

    Uses LLM-as-a-judge and returns the reasoning behind the answer.

    Parameters
    ----------
    input : str
        The input to the report generation agent.
    output : str
        The output of the report generation agent.
    expected_output : str
        The expected output of the report generation agent.
    kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    Evaluation
        The evaluation result, including the reasoning behind the answer.
    """
    # Define the evaluator agent
    client_manager = AsyncClientManager.get_instance()
    evaluator_agent = agents.Agent(
        name="Evaluator Agent",
        instructions=EVALUATOR_INSTRUCTIONS + ADDITONAL_EVALUATOR_INSTRUCTIONS,
        output_type=EvaluatorResponse,
        model=agents.OpenAIChatCompletionsModel(
            model=client_manager.configs.default_planner_model,
            openai_client=client_manager.openai_client,
        ),
    )
    # Format the input for the evaluator agent
    evaluator_input = EVALUATOR_TEMPLATE.format(
        question=input,
        ground_truth=expected_output,
        proposed_response=output,
    )
    # Run the evaluator agent with retry
    result = await run_agent_with_retry(evaluator_agent, evaluator_input)
    evaluation_response = result.final_output_as(EvaluatorResponse)

    # Return the evaluation result
    return Evaluation(
        name="LLM-as-a-judge",
        value=evaluation_response.is_answer_correct,
        comment=evaluation_response.explanation,
    )


@retry(stop=stop_after_attempt(5), wait=wait_exponential())
async def run_agent_with_retry(agent: agents.Agent, agent_input: str) -> agents.RunResult:
    """Run an agent with Tenacity's retry mechanism.

    Parameters
    ----------
    agent : agents.Agent
        The agent to run.
    agent_input : str
        The input to the agent.

    Returns
    -------
    agents.RunnerResult
        The result of the agent run.
    """
    logger.info(f"Running agent {agent.name}...")
    return await agents.Runner.run(agent, input=agent_input)


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
    asyncio.run(evaluate(dataset_name))


if __name__ == "__main__":
    cli()

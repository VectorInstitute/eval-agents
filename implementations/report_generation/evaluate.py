"""Evaluate the report generation agent against a Langfuse dataset."""

import asyncio
import logging
from typing import Any

import agents
import click
from aieng.agent_evals.async_client_manager import AsyncClientManager
from dotenv import load_dotenv
from langfuse._client.datasets import DatasetItemClient
from langfuse.experiment import Evaluation
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_refusal import ResponseOutputRefusal
from openai.types.responses.response_output_text import ResponseOutputText
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from implementations.report_generation.data.langfuse_upload import DEFAULT_EVALUATION_DATASET_NAME
from implementations.report_generation.main import get_report_generation_agent
from implementations.report_generation.prompts import (
    RESULT_EVALUATOR_INSTRUCTIONS,
    RESULT_EVALUATOR_TEMPLATE,
    TRAJECTORY_EVALUATOR_INSTRUCTIONS,
    TRAJECTORY_EVALUATOR_TEMPLATE,
)


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# Will have the structure:
# {
#     "final_report": dict[str, Any] | None,
#     "trajectory": {
#         "actions": list[str],
#         "parameters": list[str],
#     },
# }
EvaluationOutput = dict[str, None | Any]


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
        evaluators=[final_result_evaluator, trajectory_evaluator],
        max_concurrency=1,
    )

    # Log the evaluation result
    logger.info(result.format().replace("\\n", "\n"))

    try:
        # Gracefully close the services
        await client_manager.close()
    except Exception as e:
        logger.warning(f"Client manager services not closed successfully: {e}")


async def agent_task(*, item: DatasetItemClient, **kwargs) -> EvaluationOutput:
    """Run the report generation agent against an item from a Langfuse dataset.

    Parameters
    ----------
    item : DatasetItemClient
        The item from the Langfuse dataset to evaluate against.

    Returns
    -------
    EvaluationOutput
        The output of the report generation agent with the values it should
        be evaluated against.
    """
    # Define and run the report generation agent
    report_generation_agent = get_report_generation_agent(enable_trace=True)
    result = await run_agent_with_retry(report_generation_agent, item.input)

    # Extract the report data and trajectory from the agent's response
    actions = []
    parameters = []
    final_report = None
    for raw_response in result.raw_responses:
        for output in raw_response.output:
            # The trajectory will be the list of actions and the
            # parameters passed to each one of them
            if isinstance(output, ResponseFunctionToolCall):
                actions.append(output.name)
                parameters.append(output.arguments)

                # The final report will be the arguments sent by the
                # write_report_to_file function call
                # If there is more than one call to the write_report_to_file function,
                # the last one will be used because the previous calls were likely
                # failed calls
                if isinstance(output, ResponseFunctionToolCall) and "write_report_to_file" in output.name:
                    final_report = output.arguments

            if isinstance(output, ResponseOutputMessage):
                for content in output.content:
                    actions.append(content.type)
                    if isinstance(content, ResponseOutputText):
                        parameters.append(content.text)
                    elif isinstance(content, ResponseOutputRefusal):
                        parameters.append(content.refusal)

    if final_report is None:
        logger.warning("No call to write_report_to_file function found in the agent's response")

    return {
        "final_report": final_report,
        "trajectory": {
            "actions": actions,
            "parameters": parameters,
        },
    }


async def final_result_evaluator(
    *,
    input: str,
    output: EvaluationOutput,
    expected_output: EvaluationOutput,
    **kwargs,
) -> Evaluation:
    # ruff: noqa: A002
    """Evaluate the proposed final answer against the ground truth.

    Uses LLM-as-a-judge and returns the reasoning behind the answer.

    Parameters
    ----------
    input : str
        The input to the report generation agent.
    output : EvaluationOutput
        The output of the report generation agent with the values it should be
        evaluated against.
    expected_output : EvaluationOutput
        The evaluation output the report generation agent should have.
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
        name="Final Result Evaluator Agent",
        instructions=RESULT_EVALUATOR_INSTRUCTIONS,
        output_type=EvaluatorResponse,
        model=agents.OpenAIChatCompletionsModel(
            model=client_manager.configs.default_planner_model,
            openai_client=client_manager.openai_client,
        ),
    )
    # Format the input for the evaluator agent
    evaluator_input = RESULT_EVALUATOR_TEMPLATE.format(
        question=input,
        ground_truth=expected_output["final_report"],
        proposed_response=output["final_report"],
    )
    # Run the evaluator agent with retry
    result = await run_agent_with_retry(evaluator_agent, evaluator_input)
    evaluation_response = result.final_output_as(EvaluatorResponse)

    # Return the evaluation result
    return Evaluation(
        name="Final Result",
        value=evaluation_response.is_answer_correct,
        comment=evaluation_response.explanation,
    )


async def trajectory_evaluator(
    *,
    input: str,
    output: EvaluationOutput,
    expected_output: EvaluationOutput,
    **kwargs,
) -> Evaluation:
    # ruff: noqa: A002
    """Evaluate the agent's trajectory against the ground truth.

    Uses LLM-as-a-judge and returns the reasoning behind the answer.

    Parameters
    ----------
    input : str
        The input to the report generation agent.
    output : EvaluationOutput
        The output of the report generation agent with the values it should be
        evaluated against.
    expected_output : EvaluationOutput
        The evaluation output the report generation agent should have.
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
        name="Trajectory Evaluator Agent",
        instructions=TRAJECTORY_EVALUATOR_INSTRUCTIONS,
        output_type=EvaluatorResponse,
        model=agents.OpenAIChatCompletionsModel(
            model=client_manager.configs.default_planner_model,
            openai_client=client_manager.openai_client,
        ),
    )

    assert isinstance(expected_output["trajectory"], dict), "Expected trajectory must be a dictionary"
    assert isinstance(output["trajectory"], dict), "Actual trajectory must be a dictionary"

    # Format the input for the evaluator agent
    evaluator_input = TRAJECTORY_EVALUATOR_TEMPLATE.format(
        question=input,
        expected_actions=expected_output["trajectory"]["actions"],
        expected_descriptions=expected_output["trajectory"]["description"],
        actual_actions=output["trajectory"]["actions"],
        actual_parameters=output["trajectory"]["parameters"],
    )
    # Run the evaluator agent with retry
    result = await run_agent_with_retry(evaluator_agent, evaluator_input)
    evaluation_response = result.final_output_as(EvaluatorResponse)

    # Return the evaluation result
    return Evaluation(
        name="Trajectory",
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
    logger.info(f"Running agent {agent.name} with input '{agent_input[:100]}...'")
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

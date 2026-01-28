"""Evaluate the report generation agent."""

import asyncio
import json
import logging
import time
from pathlib import Path
from uuid import uuid4

import agents
import click
from aieng.agent_evals.async_client_manager import AsyncClientManager
from dotenv import load_dotenv
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.api.resources.commons.types import TraceWithFullDetails
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

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

DEFAULT_EVALUATION_DATASET_PATH = "implementations/report_generation/data/OnlineRetailReportEval.json"
DEFAULT_EVALUATION_OUTPUT_PATH = "implementations/report_generation/reports/"


class EvaluatorQuery(BaseModel):
    """Query to the evaluator agent."""

    question: str
    ground_truth: str
    proposed_response: str

    def get_query(self) -> str:
        """Obtain query string to the evaluator agent."""
        return EVALUATOR_TEMPLATE.format(**self.model_dump())


class EvaluatorResponse(BaseModel):
    """Typed response from the evaluator."""

    explanation: str
    is_answer_correct: bool


async def evaluate(
    dataset_path: str = DEFAULT_EVALUATION_DATASET_PATH,
    output_path: str = DEFAULT_EVALUATION_OUTPUT_PATH,
) -> None:
    """Evaluate the report generation agent against a given dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the evaluation dataset. Default is DEFAULT_EVALUATION_DATASET_PATH.
    output_path : str
        Path to the evaluation output reports. Default is
        DEFAULT_EVALUATION_OUTPUT_PATH.
    """
    # Get the client manager singleton instance and langfuse client
    client_manager = AsyncClientManager.get_instance()
    langfuse_client = client_manager.langfuse_client

    logger.info(f"Loading evaluation dataset from '{dataset_path}'")
    with open(dataset_path, "r") as file:
        evaluation_dataset = json.load(file)
    logger.info(f"Loaded {len(evaluation_dataset)} examples from '{dataset_path}'")

    # Get a report generation agent to run the examples
    report_generation_agent = get_report_generation_agent(enable_trace=True)

    # Run the examples in the dataset
    trace_id_by_example_id = {}
    for example in evaluation_dataset:
        trace_id = await run_example(example, report_generation_agent, langfuse_client)
        trace_id_by_example_id[example["id"]] = trace_id

    # Get an evaluator agent to evaluate the results of the examples
    # stored in Langfuse traces against the ground truth
    evaluator_agent = get_evaluator_agent(EVALUATOR_INSTRUCTIONS + ADDITONAL_EVALUATOR_INSTRUCTIONS)

    # Run the evaluations
    evaluation_results = {}
    correct_count = 0
    total_count = 0
    for example in evaluation_dataset:
        trace_id = trace_id_by_example_id[example["id"]]
        evaluation_response = await evaluate_example(example, trace_id, langfuse_client, evaluator_agent)

        # Record the evaluation results
        evaluation_results[example["id"]] = evaluation_response.model_dump()
        evaluation_results[example["id"]]["trace_id"] = trace_id

        # Update the metrics
        total_count += 1
        if evaluation_response.is_answer_correct:
            correct_count += 1
        else:
            logger.error(f"Example '{example['id']}' is incorrect. Explanation: {evaluation_response.explanation}")

    # Print the evaluation results
    logger.info("Evaluation Finished.")
    logger.info(f"Accuracy: {correct_count / total_count} ({correct_count}/{total_count})")

    data_file_name = Path(dataset_path).name
    output_file_name = data_file_name.replace(".json", f"_{int(time.time())}.json")
    output_file_path = Path(output_path) / output_file_name
    with open(output_file_path, "w") as file:
        json.dump(evaluation_results, file, indent=4)
        logger.info(f"Evaluation results saved to '{output_file_path}'")

    # Gracefully close the services
    await client_manager.close()


async def run_example(example: dict, agent: agents.Agent, langfuse_client: Langfuse) -> str:
    """Run an example and return the langfusetrace ID.

    Parameters
    ----------
    example : dict
        The example to run.
    agent : agents.Agent
        The agent to run the example with.
    langfuse_client : Langfuse
        The Langfuse client to record the agent's trace.

    Returns
    -------
    str
        The Langfuse trace ID.
    """
    # If a trace ID is provided, skip running and use it instead
    if "trace_id" in example and example["trace_id"]:
        assert isinstance(example["trace_id"], str), "Trace ID must be a string."
        logger.info(f"Skipping the inference, found trace ID {example['trace_id']} for example '{example['id']}'")
        return example["trace_id"]

    # Create a new trace ID for this run and record it so we can
    # recover the details later
    trace_id = langfuse_client.create_trace_id(seed=str(uuid4()))
    logger.info(f"Running example '{example['id']}' with trace ID {trace_id}")

    # Open a new Langfuse observation and run the example
    evaluation_name = f"evaluation example {example['id']}"
    with langfuse_client.start_as_current_observation(name=evaluation_name, trace_context={"trace_id": trace_id}):
        assert isinstance(example["user_input"], str), "User input must be a string."
        await call_agent_with_retry(agent, example["user_input"])

    return trace_id


async def evaluate_example(
    example: dict,
    trace_id: str,
    langfuse_client: Langfuse,
    agent: agents.Agent,
) -> EvaluatorResponse:
    """Evaluate an example and return the evaluation response.

    Parameters
    ----------
    example : dict
        The example to evaluate.
    trace_id : str
        The trace ID of the example.
    langfuse_client : Langfuse
        The Langfuse client to use.
    agent : agents.Agent
        The agent to use to evaluate the example.

    Returns
    -------
    EvaluatorResponse
        The evaluation response.
    """
    # Get the Langfuse trace for the example given its trace ID
    logger.info(f"Getting trace for example '{example['id']}' with trace ID {trace_id}")
    trace = call_trace_with_retry(trace_id, langfuse_client)

    for observation in trace.observations:
        assert observation.name is not None, "Observation name must not be None."
        # Find the observation that contains the report data, i.e. the one
        # that contains the call to the report write_report_to_file function
        if "write_report_to_file" in observation.name:
            assert isinstance(example["user_input"], str), "User input must be a string."

            # Evaluate the example against the ground truth using the evaluator
            logger.info(f"Evaluating example '{example['id']}'")
            evaluator_query = EvaluatorQuery(
                question=example["user_input"],
                ground_truth=json.dumps(example["expected_output"]),
                proposed_response=json.dumps(observation.input),
            )

            logger.info(f"Calling evaluator agent for example '{example['id']}'")
            result = await call_agent_with_retry(agent, evaluator_query.get_query())
            return result.final_output_as(EvaluatorResponse)

    # If no call to the write_report_to_file function was found
    # record a failed evaluation
    logger.error(f"No report found for example '{example['id']}'")
    return EvaluatorResponse(explanation="No report found", is_answer_correct=False)


@retry(stop=stop_after_attempt(5), wait=wait_exponential())
async def call_agent_with_retry(agent: agents.Agent, agent_input: str) -> agents.RunResult:
    """
    Call an agent using Tenacity's retry mechanism.

    Parameters
    ----------
    agent : agents.Agent
        The agent to call.
    agent_input : str
        The input to the agent.

    Returns
    -------
    agents.RunResult
        The result of the agent call.
    """
    logger.info(f"Calling agent '{agent.name}'...")
    return await agents.Runner.run(agent, input=agent_input)


@retry(stop=stop_after_attempt(5), wait=wait_exponential())
def call_trace_with_retry(trace_id: str, langfuse_client: Langfuse) -> TraceWithFullDetails:
    """
    Call a trace using Tenacity's retry mechanism.

    Parameters
    ----------
    trace_id : str
        The trace ID to call.
    langfuse_client : Langfuse
        The Langfuse client to use.

    Returns
    -------
    TraceWithFullDetails
        The trace as returned by langfuse.
    """
    logger.info(f"Calling langfuse with trace ID {trace_id}...")
    return langfuse_client.api.trace.get(trace_id=trace_id)


def get_evaluator_agent(evaluator_instructions: str) -> agents.Agent:
    """Get an evaluator agent instance.

    Returns
    -------
    agents.Agent
        The evaluator agent.
    """
    # Disabling tracing in case it has been enabled
    agents.set_tracing_disabled(disabled=True)

    client_manager = AsyncClientManager.get_instance()

    return agents.Agent(
        name="Evaluator Agent",
        instructions=evaluator_instructions,
        output_type=EvaluatorResponse,
        model=agents.OpenAIChatCompletionsModel(
            model=client_manager.configs.default_planner_model,
            openai_client=client_manager.openai_client,
        ),
    )


@click.command()
@click.option(
    "--dataset-path",
    required=False,
    default=DEFAULT_EVALUATION_DATASET_PATH,
    help="Path to the evaluation dataset.",
)
@click.option(
    "--output-path",
    required=False,
    default=DEFAULT_EVALUATION_OUTPUT_PATH,
    help="Path to the evaluation output reports.",
)
def cli(dataset_path: str, output_path: str) -> None:
    """Evaluate the report generation agent against a given dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the evaluation dataset.
    output_path : str
        Path to the evaluation output reports.
    """
    asyncio.run(evaluate(dataset_path, output_path))


if __name__ == "__main__":
    cli()

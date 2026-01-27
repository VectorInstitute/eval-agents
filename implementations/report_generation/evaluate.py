"""Evaluate the report generation agent."""

import asyncio
import json
import logging
import time
from uuid import uuid4

import agents
from aieng.agent_evals.async_client_manager import AsyncClientManager
from dotenv import load_dotenv
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.api.resources.commons.types import TraceWithFullDetails
from pydantic import BaseModel

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
- The filename should not matter, unless explicitly specified in the "Question".
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


async def evaluate() -> None:
    """Evaluate the report generation agent."""
    # Get the client manager singleton instance and langfuse client
    client_manager = AsyncClientManager.get_instance()
    langfuse_client = client_manager.langfuse_client

    evaluation_dataset = [
        {
            "id": "1",
            "user_input": "Generate a report of the top 5 selling products per year and the total sales for each product.",
            "expected_output": {
                "report_data": [
                    ["2010", "REGENCY CAKESTAND 3 TIER", 26897.360000000022],
                    ["2010", "DOTCOM POSTAGE", 24671.189999999995],
                    ["2010", "WHITE HANGING HEART T-LIGHT HOLDER", 9877.8200000000052],
                    ["2010", "RED WOOLLY HOTTIE WHITE HEART.", 9291.7299999999959],
                    ["2010", "PAPER CHAIN KIT 50'S CHRISTMAS ", 9205.1499999999942],
                    ["2011", "DOTCOM POSTAGE", 181574.29000000004],
                    ["2011", "REGENCY CAKESTAND 3 TIER", 137864.82999999981],
                    ["2011", "PARTY BUNTING", 97095.240000000456],
                    ["2011", "WHITE HANGING HEART T-LIGHT HOLDER", 89790.649999999092],
                    ["2011", "JUMBO BAG RED RETROSPOT", 88383.680000001812],
                ],
                "report_columns": ["SaleYear", "Description", "TotalSales"],
                "filename": "top_selling_products_report.xlsx",
            },
            # "trace_id": "34990d826a68971d76cbb439a613010e",
        }
    ]

    report_generation_agent = get_report_generation_agent(enable_trace=True)

    trace_id_by_example_id = {}
    for example in evaluation_dataset:
        if "trace_id" in example and example["trace_id"]:
            assert isinstance(example["trace_id"], str), "Trace ID must be a string."
            logger.info(f"Skipping the inference, found trace ID {example['trace_id']} for example '{example['id']}'")
            trace_id_by_example_id[example["id"]] = example["trace_id"]
            continue

        trace_id = langfuse_client.create_trace_id(seed=str(uuid4()))
        trace_id_by_example_id[example["id"]] = trace_id
        logger.info(f"Running example '{example['id']}' with trace ID {trace_id}")

        evaluation_name = f"evaluation example {example['id']}"
        with langfuse_client.start_as_current_observation(name=evaluation_name, trace_context={"trace_id": trace_id}):
            assert isinstance(example["user_input"], str), "User input must be a string."

            await agents.Runner.run(report_generation_agent, input=example["user_input"])

    evaluator_agent = get_evaluator_agent(EVALUATOR_INSTRUCTIONS + ADDITONAL_EVALUATOR_INSTRUCTIONS)

    evaluation_results = {}
    correct_count = 0
    total_count = 0
    for example in evaluation_dataset:
        trace_id = trace_id_by_example_id[example["id"]]
        trace = get_trace_with_retry(trace_id, langfuse_client)
        observations = sorted(trace.observations, key=lambda obs: obs.start_time)

        found_report = False
        for observation in observations:
            assert observation.name is not None, "Observation name must not be None."
            if "write_report_to_file" in observation.name:
                found_report = True

                logger.info(f"Evaluating example '{example['id']}'")

                assert isinstance(example["user_input"], str), "User input must be a string."
                evaluator_query = EvaluatorQuery(
                    question=example["user_input"],
                    ground_truth=json.dumps(example["expected_output"]),
                    proposed_response=json.dumps(observation.input),
                )
                result = await agents.Runner.run(evaluator_agent, input=evaluator_query.get_query())
                evaluation_response = result.final_output_as(EvaluatorResponse)

                evaluation_results[example["id"]] = evaluation_response.model_dump()
                evaluation_results[example["id"]]["trace_id"] = trace_id

                correct_count += 1 if evaluation_response.is_answer_correct else 0
                total_count += 1

                break

        if not found_report:
            logger.error(f"No report found for example '{example['id']}'")
            evaluation_response = EvaluatorResponse(explanation="No report found", is_answer_correct=False)
            evaluation_results[example["id"]] = evaluation_response.model_dump()
            evaluation_results[example["id"]]["trace_id"] = trace_id
            total_count += 1

    logger.info("Evaluation Finished.")
    logger.info(f"Accuracy: {correct_count / total_count} ({correct_count}/{total_count})")
    logger.info("Evaluation Results:")
    logger.info(f"{evaluation_results}")

    await client_manager.close()


def get_trace_with_retry(
    trace_id: str,
    langfuse_client: Langfuse,
    max_retries: int = 10,
    backoff_factor: int = 2,
) -> TraceWithFullDetails:
    """Get a trace from Langfuse with a retry mechanism.

    The initial retry delay is 2 seconds.

    Parameters
    ----------
    trace_id : str
        The ID of the trace to get.
    langfuse_client : Langfuse
        The Langfuse client to use.
    max_retries : int, optional
        The maximum number of retries. Default is 10.
    backoff_factor : int, optional
        The backoff factor. Default is 2.

    Returns
    -------
    TraceWithFullDetails
        The trace object retrieved from Langfuse.

    Raises
    ------
    Exception
        If the trace is not found after the maximum number of retries.
    """
    t = 2
    for retry in range(max_retries):
        try:
            return langfuse_client.api.trace.get(trace_id=trace_id)
        except Exception:
            logger.error(f"Trace {trace_id} not found. Retrying in {t} seconds ({retry + 1}/{max_retries})...")
            time.sleep(t)
            t *= backoff_factor

    raise Exception(f"Failed to get trace {trace_id} after {max_retries} retries")


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


if __name__ == "__main__":
    asyncio.run(evaluate())

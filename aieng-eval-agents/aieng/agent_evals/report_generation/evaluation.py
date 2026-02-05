"""
Evaluate the report generation agent against a Langfuse dataset.

Example
-------
>>> from aieng.agent_evals.report_generation.evaluation import evaluate
>>> evaluate(
>>>     dataset_name="OnlineRetailReportEval",
>>>     sqlite_db_path=Path("data/OnlineRetail.db"),
>>>     reports_output_path=Path("reports/"),
>>>     langfuse_project_name="Report Generation",
>>> )
"""

import json
import logging
from pathlib import Path
from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.report_generation.agent import EventParser, EventType, get_report_generation_agent
from aieng.agent_evals.report_generation.prompts import (
    MAIN_AGENT_INSTRUCTIONS,
    RESULT_EVALUATOR_INSTRUCTIONS,
    RESULT_EVALUATOR_TEMPLATE,
    TRAJECTORY_EVALUATOR_INSTRUCTIONS,
    TRAJECTORY_EVALUATOR_TEMPLATE,
)
from google.adk.agents import Agent
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from langfuse._client.datasets import DatasetItemClient
from langfuse.experiment import Evaluation, LocalExperimentItem
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Will have the structure:
# {
#     "final_report": str | None,
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


async def evaluate(
    dataset_name: str,
    sqlite_db_path: Path,
    reports_output_path: Path,
    langfuse_project_name: str,
):
    """Evaluate the report generation agent against a Langfuse dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to evaluate against.
    sqlite_db_path : Path
        The path to the SQLite database.
    reports_output_path : Path
        The path to the reports output directory.
    langfuse_project_name : str
        The name of the Langfuse project to use for tracing.
    """
    # Get the client manager singleton instance and langfuse client
    client_manager = AsyncClientManager.get_instance()
    langfuse_client = client_manager.langfuse_client

    # Find the dataset in Langfuse
    dataset = langfuse_client.get_dataset(dataset_name)

    # Initialize the task for the report generation agent evaluation
    # We need this task so we can pass parameters to the agent, since
    # the agent has to be instantiated inside the task function
    report_generation_task = ReportGenerationTask(
        sqlite_db_path=sqlite_db_path,
        reports_output_path=reports_output_path,
        langfuse_project_name=langfuse_project_name,
    )

    # Run the experiment with the agent task and evaluator
    # against the dataset items
    result = dataset.run_experiment(
        name="Evaluate Report Generation Agent",
        description="Evaluate the Report Generation Agent with data from Langfuse",
        task=report_generation_task.run,
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


class ReportGenerationTask:
    """Define a task for the the report generation agent."""

    def __init__(
        self,
        sqlite_db_path: Path,
        reports_output_path: Path,
        langfuse_project_name: str,
    ):
        """Initialize the task for an report generation agent evaluation.

        Parameters
        ----------
        sqlite_db_path : Path
            The path to the SQLite database.
        reports_output_path : Path
            The path to the reports output directory.
        langfuse_project_name : str
            The name of the Langfuse project to use for tracing.
        """
        self.sqlite_db_path = sqlite_db_path
        self.reports_output_path = reports_output_path
        self.langfuse_project_name = langfuse_project_name

    async def run(self, *, item: LocalExperimentItem | DatasetItemClient, **kwargs: dict[str, Any]) -> EvaluationOutput:
        """Run the report generation agent against an item from a Langfuse dataset.

        Parameters
        ----------
        item : LocalExperimentItem | DatasetItemClient
            The item from the Langfuse dataset to evaluate against.

        Returns
        -------
        EvaluationOutput
            The output of the report generation agent with the values it should
            be evaluated against.
        """
        # Run the report generation agent
        report_generation_agent = get_report_generation_agent(
            instructions=MAIN_AGENT_INSTRUCTIONS,
            sqlite_db_path=self.sqlite_db_path,
            reports_output_path=self.reports_output_path,
            langfuse_project_name=self.langfuse_project_name,
        )
        # Handle both TypedDict and class access patterns
        item_input = item["input"] if isinstance(item, dict) else item.input
        events = await run_agent_with_retry(report_generation_agent, item_input)

        # Extract the report data and trajectory from the agent's response
        actions: list[str] = []
        parameters: list[Any | None] = []
        final_report: str | None = None

        # The trajectory will be the list of actions and the
        # parameters passed to each one of them
        for event in events:
            parsed_events = EventParser.parse(event)

            for parsed_event in parsed_events:
                if parsed_event.type == EventType.FINAL_RESPONSE:
                    # Picking up the final message displayed to the user
                    actions.append("final_response")
                    parameters.append(parsed_event.text)

                if parsed_event.type == EventType.TOOL_CALL:
                    # Picking up tool calls and their arguments
                    actions.append(parsed_event.text)
                    parameters.append(parsed_event.arguments)

                    # The final report will be the arguments sent by the
                    # write tool call
                    # If there is more than one call to the write tool call,
                    # the last one will be used because the previous
                    # calls are likely failed calls
                    if parsed_event.text == "write":
                        final_report = parsed_event.arguments

                # Not tracking EventType.THOUGHT or EventType.TOOL_RESPONSE

        if final_report is None:
            logger.warning("No call to `write` function found in the agent's response")

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
    additional_instructions = _get_additional_instructions(expected_output, "final_report")

    # Define the evaluator agent
    client_manager = AsyncClientManager.get_instance()
    evaluator_agent = Agent(
        name="FinalResultEvaluatorAgent",
        instruction=RESULT_EVALUATOR_INSTRUCTIONS + additional_instructions,
        model=client_manager.configs.default_worker_model,
        output_schema=EvaluatorResponse,
    )
    # Format the input for the evaluator agent
    evaluator_input = RESULT_EVALUATOR_TEMPLATE.format(
        question=input,
        ground_truth=expected_output["final_report"],
        proposed_response=output["final_report"],
    )

    # Run the evaluator agent with retry
    events = await run_agent_with_retry(evaluator_agent, evaluator_input)

    # Return the evaluation result
    evaluator_response = get_evaluator_reponse(events)
    return Evaluation(
        name="Final Result",
        value=evaluator_response.is_answer_correct,
        comment=evaluator_response.explanation,
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
    additional_instructions = _get_additional_instructions(expected_output, "trajectory")

    # Define the evaluator agent
    client_manager = AsyncClientManager.get_instance()
    evaluator_agent = Agent(
        name="TrajectoryEvaluatorAgent",
        instruction=TRAJECTORY_EVALUATOR_INSTRUCTIONS + additional_instructions,
        model=client_manager.configs.default_planner_model,
        output_schema=EvaluatorResponse,
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
    events = await run_agent_with_retry(evaluator_agent, evaluator_input)

    # Return the evaluation result
    evaluator_response = get_evaluator_reponse(events)
    return Evaluation(
        name="Trajectory",
        value=evaluator_response.is_answer_correct,
        comment=evaluator_response.explanation,
    )


def get_evaluator_reponse(events: list[Event]) -> EvaluatorResponse:
    """Get the evaluator response from a list of events.

    It must be a list of events from an evaluator that has
    EvaluatorResponse as output schema

    Parameters
    ----------
    events : list[Event]
        The list of events to get the evaluator response from.

    Returns
    -------
    EvaluatorResponse
        The evaluator response.
    """
    for event in events:
        if event.is_final_response():
            assert event.content, "Event content must be present"
            assert event.content.parts, "Event content parts must be present"
            assert len(event.content.parts) > 0, "Event content parts must not be empty"
            assert isinstance(event.content.parts[0].text, str), "Event content parts text must be a string"
            return EvaluatorResponse(**json.loads(event.content.parts[0].text))

    raise Exception("No final response found in the events")


def _get_additional_instructions(expected_output: EvaluationOutput, key: str) -> str:
    additional_instructions_dict = expected_output.get("additional_instructions", {})
    if additional_instructions_dict:
        return additional_instructions_dict.get(key, "")

    return ""


@retry(stop=stop_after_attempt(5), wait=wait_exponential())
async def run_agent_with_retry(agent: Agent, agent_input: str) -> list[Event]:
    """Run an agent with Tenacity's retry mechanism.

    Parameters
    ----------
    agent : agents.Agent
        The agent to run.
    agent_input : str
        The input to the agent.

    Returns
    -------
    list[Event]
        The events from the agent run.
    """
    logger.info(f"Running agent {agent.name} with input '{agent_input[:100]}...'")

    # Create session and runner
    session_service = InMemorySessionService()
    runner = Runner(app_name=agent.name, agent=agent, session_service=session_service)
    current_session = await session_service.create_session(
        app_name=agent.name,
        user_id="user",
        state={},
    )

    # create the user message and run the agent
    content = Content(role="user", parts=[Part(text=agent_input)])
    events = []
    async for event in runner.run_async(
        user_id="user",
        session_id=current_session.id,
        new_message=content,
    ):
        events.append(event)

    return events

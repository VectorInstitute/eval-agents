"""
Definitions for the the report generation agent.

Example
-------
>>> from aieng.agent_evals.report_generation.agent import get_report_generation_agent
>>> from aieng.agent_evals.report_generation.prompts import MAIN_AGENT_INSTRUCTIONS
>>> agent = get_report_generation_agent(
>>>     instructions=MAIN_AGENT_INSTRUCTIONS,
>>>     sqlite_db_path=Path("data/OnlineRetail.db"),
>>>     reports_output_path=Path("reports/"),
>>>     langfuse_project_name="Report Generation",
>>> )
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.langfuse import setup_langfuse_tracer
from aieng.agent_evals.report_generation.file_writer import ReportFileWriter
from google.adk.agents import Agent
from google.adk.events.event import Event
from pydantic import BaseModel


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def get_report_generation_agent(
    instructions: str,
    sqlite_db_path: Path,
    reports_output_path: Path,
    langfuse_project_name: str | None,
) -> Agent:
    """
    Define the report generation agent.

    Parameters
    ----------
    instructions : str
        The instructions for the agent.
    sqlite_db_path : Path
        The path to the SQLite database.
    reports_output_path : Path
        The path to the reports output directory.
    langfuse_project_name : str | None
        The name of the Langfuse project to use for tracing.

    Returns
    -------
    agents.Agent
        The report generation agent.
    """
    # Setup langfuse tracing if project name is provided
    if langfuse_project_name:
        setup_langfuse_tracer(langfuse_project_name)

    # Get the client manager singleton instance
    client_manager = AsyncClientManager.get_instance()
    report_file_writer = ReportFileWriter(reports_output_path)

    # Define an agent using Google ADK
    return Agent(
        name="ReportGenerationAgent",
        model=client_manager.configs.default_worker_model,
        instruction=instructions,
        tools=[
            client_manager.sqlite_connection(sqlite_db_path).execute,
            report_file_writer.write,
        ],
    )


class EventType(Enum):
    """Types of events from agents."""

    FINAL_RESPONSE = "final_response"
    TOOL_CALL = "tool_call"
    THOUGHT = "thought"
    TOOL_RESPONSE = "tool_response"


class ParsedEvent(BaseModel):
    """Parsed event from an agent."""

    type: EventType
    text: str
    arguments: Any | None = None


class EventParser:
    """Parser for agent events."""

    @classmethod
    def parse(cls, event: Event) -> list[ParsedEvent]:
        """Parse an agent event into a list of parsed events.

        The event can be a final response, a thought, a tool call,
        or a tool response.

        Parameters
        ----------
        event : Event
            The event to parse.

        Returns
        -------
        list[ParsedEvent]
            A list of parsed events.
        """
        parsed_events = []

        if event.is_final_response():
            parsed_events.extend(cls._parse_final_response(event))

        elif event.content:
            if event.content.role == "model":
                parsed_events.extend(cls._parse_model_response(event))

            elif event.content.role == "user":
                parsed_events.extend(cls._parse_user_response(event))

        else:
            logger.warning(f"Unknown stream event: {event}")

        return parsed_events

    @classmethod
    def _parse_final_response(cls, event: Event) -> list[ParsedEvent]:
        if (
            not event.content
            or not event.content.parts
            or len(event.content.parts) == 0
            or not event.content.parts[0].text
        ):
            logger.warning(f"Final response's content is not valid: {event}")
            return []

        return [
            ParsedEvent(
                type=EventType.FINAL_RESPONSE,
                text=event.content.parts[0].text,
            )
        ]

    @classmethod
    def _parse_model_response(cls, event: Event) -> list[ParsedEvent]:
        if not event.content or not event.content.parts:
            logger.warning(f"Model response's content is not valid: {event}")
            return []

        parsed_events = []

        for part in event.content.parts:
            # Parsing tool calls and their arguments
            if part.function_call:
                if not part.function_call.name:
                    logger.warning(f"No name in function call: {part}")
                    continue

                parsed_events.append(
                    ParsedEvent(
                        type=EventType.TOOL_CALL,
                        text=part.function_call.name,
                        arguments=part.function_call.args,
                    )
                )

            # Parsing the agent's thoughts
            elif part.thought_signature or (part.text and not part.thought_signature):
                if not part.text:
                    logger.warning(f"No text in part: {part}")
                    continue

                parsed_events.append(
                    ParsedEvent(
                        type=EventType.THOUGHT,
                        text=part.text,
                    )
                )

            else:
                logger.warning(f"Unknown part type: {part}")

        return parsed_events

    @classmethod
    def _parse_user_response(cls, event: Event) -> list[ParsedEvent]:
        if not event.content or not event.content.parts:
            logger.warning(f"Model response's content is not valid: {event}")
            return []

        parsed_events = []

        for part in event.content.parts:
            if part.function_response:
                if not part.function_response.name:
                    logger.warning(f"No name in function response: {part}")
                    continue

                parsed_events.append(
                    ParsedEvent(
                        type=EventType.TOOL_RESPONSE,
                        text=part.function_response.name,
                        arguments=part.function_response.response,
                    )
                )
            else:
                logger.warning(f"Unknown part type: {part}")

        return parsed_events

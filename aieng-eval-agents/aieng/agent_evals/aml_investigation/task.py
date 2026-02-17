"""Task function for AML investigation experiment execution.

This module provides a Langfuse-compatible task callable that executes the AML
investigation agent on one dataset item and returns a structured analyst output.

The task is designed for use with the evaluation harness and Langfuse
``run_experiment`` APIs. It handles:

- Input normalization for both dict and dataset item objects.
- Running the ADK agent through a shared ``Runner``.
- Extracting and validating final model output.
- Returning consistent ``dict`` results for evaluator consumption.

Examples
--------
>>> import asyncio
>>> from aieng.agent_evals.aml_investigation.task import AmlInvestigationTask
>>> task = AmlInvestigationTask()
>>> # Run one AML case in an async context
>>> sample_item = {
...     "input": {
...         "case_id": "case-001",
...         "seed_transaction_id": "txn-001",
...         "seed_timestamp": "2022-09-01T12:00:00",
...         "window_start": "2022-09-01T00:00:00",
...         "trigger_label": "RANDOM_REVIEW",
...     }
... }
>>> _ = asyncio.run(task(item=sample_item))
>>> # Use the task in an experiment
>>> from aieng.agent_evals.evaluation.experiment import run_experiment
>>> result = run_experiment(
...     dataset_name="aml_eval_dataset",
...     name="AML Investigation Evaluation",
...     task=AmlInvestigationTask(),
...     evaluators=[...],
... )
"""

import getpass
import json
import logging
import uuid
from typing import Any

from aieng.agent_evals.aml_investigation.agent import create_aml_investigation_agent
from aieng.agent_evals.aml_investigation.data import AnalystOutput
from aieng.agent_evals.db_manager import DbManager
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from langfuse.experiment import ExperimentItem


logger = logging.getLogger(__name__)


class AmlInvestigationTask:
    """Langfuse-compatible task wrapper for AML case investigations.

    This class implements the ``TaskFunction`` callable protocol expected by
    Langfuse experiments: ``__call__(*, item, **kwargs)``.

    A single task instance owns:

    - One AML investigation agent.
    - One ADK runner used to execute agent calls.

    Parameters
    ----------
    agent : LlmAgent | None, optional
        Pre-configured AML investigation agent to use. If ``None``, the default
        factory ``create_aml_investigation_agent()`` is used.

    Examples
    --------
    >>> # Create a task with the default agent:
    >>> task = AmlInvestigationTask()
    >>> isinstance(task, AmlInvestigationTask)
    True
    >>> # Create a task with a custom agent:
    >>> from aieng.agent_evals.aml_investigation import create_aml_investigation_agent
    >>> custom_agent = create_aml_investigation_agent(name="aml_custom")
    >>> task = AmlInvestigationTask(agent=custom_agent)
    """

    def __init__(self, *, agent: LlmAgent | None = None) -> None:
        """Initialize the AML task with an agent and runner."""
        self._agent = agent or create_aml_investigation_agent()
        self._runner = Runner(
            app_name="aml_investigation",
            agent=self._agent,
            session_service=InMemorySessionService(),
            auto_create_session=True,
        )

    async def __call__(self, *, item: ExperimentItem, **kwargs: Any) -> dict[str, Any] | None:
        """Run one AML investigation case and return structured output.

        Parameters
        ----------
        item : ExperimentItem
            One Langfuse experiment item. This can be:

            - A dict-like local item with an ``"input"`` key.
            - A Langfuse dataset item object with an ``input`` attribute.

            The input payload is serialized to JSON and passed as the user
            message to the agent.
        **kwargs : Any
            Additional keyword arguments forwarded by Langfuse. They are
            accepted for protocol compatibility and ignored by this task.

        Returns
        -------
        dict[str, Any] | None
            Parsed analyst output as a dictionary if a valid final response was
            produced, otherwise ``None``.

        Notes
        -----
        The method first attempts strict schema parsing with
        ``AnalystOutput.model_validate_json``. If that fails, it falls back to a
        direct ``json.loads`` parse and validates the resulting object.
        """
        item_input = item.get("input") if isinstance(item, dict) else item.input
        serialized_input = json.dumps(item_input, ensure_ascii=False, indent=2)
        message = types.Content(parts=[types.Part(text=serialized_input)], role="user")

        final_text: str | None = None
        async for event in self._runner.run_async(
            session_id=str(uuid.uuid4()), user_id=getpass.getuser(), new_message=message
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(part.text or "" for part in event.content.parts if part.text)

        if not final_text:
            metadata = item.get("metadata", {}) if isinstance(item, dict) else item.metadata
            case_id = metadata.get("id") if metadata else "unknown"
            logger.warning("No analyst output produced for case_id=%s", case_id)
            return None

        # Prefer strict schema parse first if output_schema is respected.
        try:
            return AnalystOutput.model_validate_json(final_text.strip()).model_dump()
        except Exception:
            # fallback: extract JSON substring if needed
            return AnalystOutput.model_validate(json.loads(final_text)).model_dump()

    async def close(self) -> None:
        """Close runner and database connections used by this task instance.

        Notes
        -----
        This method should be called when the task instance is no longer needed,
        especially in long-running processes or repeated evaluation runs.
        """
        await self._runner.close()
        DbManager().aml_db().close()

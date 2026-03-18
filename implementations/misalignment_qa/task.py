from __future__ import annotations

import getpass
import logging
import uuid
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from langfuse.experiment import ExperimentItem

logger = logging.getLogger(__name__)


class MisalignmentTask:
    """
    Langfuse-compatible task wrapper that:
    - runs the configured ADK agent on `item["input"]`
    - returns the final assistant output text as a string

    Multi-turn “partway through conversation” is handled upstream by prompt-embedding
    (the agent itself still executes a single “next user message” call).
    """

    def __init__(self, *, agent: Any, max_output_chars: int) -> None:
        self._agent = agent
        self._max_output_chars = max_output_chars
        self._runner = Runner(
            app_name=getattr(agent, "name", "misalignment_qa"),
            agent=agent,
            session_service=InMemorySessionService(),
            auto_create_session=True,
        )

    async def __call__(self, *, item: ExperimentItem, **kwargs: Any) -> str | None:
        del kwargs  # accepted for protocol compatibility

        # item can be a local dict-like item or a Langfuse experiment item object.
        raw_input = item.get("input") if isinstance(item, dict) else item.input

        # If the dataset contains a separate, agent-focused prompt payload, prefer it.
        metadata: Any = item.get("metadata", {}) if isinstance(item, dict) else getattr(item, "metadata", {})  # noqa: ANN401
        agent_input = None
        if isinstance(metadata, dict):
            agent_input = metadata.get("agent_input")

        effective_input = agent_input if agent_input is not None else raw_input

        if effective_input is None:
            logger.warning("Task received item without input: %r", item)
            return None

        message = types.Content(role="user", parts=[types.Part(text=str(effective_input))])

        final_text: str | None = None
        async for event in self._runner.run_async(
            session_id=str(uuid.uuid4()),
            user_id=getpass.getuser(),
            new_message=message,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(part.text or "" for part in event.content.parts if part.text)

        if final_text is None:
            # Keep this deterministic-ish so evaluators see a string.
            metadata = item.get("metadata", {}) if isinstance(item, dict) else item.metadata
            task_id = metadata.get("task_id") if isinstance(metadata, dict) else None
            logger.warning("No final response produced (task_id=%s)", task_id)
            return ""

        final_text = final_text.strip()

        # Keep judge prompts under token limits.
        # The Langfuse LLM-judge evaluator includes `input + expected_output + output`,
        # so large outputs can leave no room for JSON generation.
        if len(final_text) > self._max_output_chars:
            final_text = final_text[: self._max_output_chars] + "\n...[truncated for evaluator]"

        return final_text


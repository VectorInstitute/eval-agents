"""Langfuse-compatible task wrapper that runs an ADK agent and returns its output."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from langfuse.experiment import ExperimentItem


logger = logging.getLogger(__name__)


class MisalignmentTask:
    """Langfuse-compatible task wrapper that runs an ADK agent on dataset items.

    Runs the configured ADK agent on ``item["input"]`` and returns the final
    assistant output text as a string.

    Multi-turn contexts are seeded into the ADK session as prior chat history.
    The task-specific turns come from the dataset item metadata, and optional
    shared example turns are supplied per variant when the task is created.

    When ``user_context_preamble`` is set, examples are injected as plain text
    prepended to the user's message rather than as API-level conversation turns.
    This simulates the ``user_context`` inject mode where any end-user (not just
    a developer with API access) can embed adversarial examples in a prompt.
    """

    def __init__(
        self,
        *,
        agent: LlmAgent,
        shared_turns: Sequence[dict[str, Any]] | None = None,
        user_context_preamble: str | None = None,
    ) -> None:
        self._agent = agent
        self._shared_turns = shared_turns or []
        self._user_context_preamble = user_context_preamble
        # Keep a dedicated session service so we can seed per-item history.
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            app_name=getattr(agent, "name", "misalignment_qa"),
            agent=agent,
            session_service=self._session_service,
            auto_create_session=False,
        )

    async def __call__(self, *, item: ExperimentItem, **kwargs: Any) -> str | None:
        """Run the agent on one dataset item and return the final response text."""
        del kwargs  # accepted for protocol compatibility

        # item can be a local dict-like item or a Langfuse experiment item object.
        raw_input = item.get("input") if isinstance(item, dict) else item.input

        # If the dataset contains task-local chat turns, combine them with any
        # shared example turns configured for this variant.
        metadata: Any = item.get("metadata", {}) if isinstance(item, dict) else getattr(item, "metadata", {})  # noqa: ANN401
        task_turns: list[dict[str, Any]] | None = None
        if isinstance(metadata, dict):
            turns_raw = metadata.get("task_turns")
            if isinstance(turns_raw, list):
                task_turns = [t for t in turns_raw if isinstance(t, dict)]

        agent_turns = [*self._shared_turns, *(task_turns or [])]

        if raw_input is None and not agent_turns:
            logger.warning("Task received item without input: %r", item)
            return None

        # Prepend user-context examples to the raw input when configured.
        if self._user_context_preamble and raw_input is not None:
            raw_input = f"{self._user_context_preamble}\n\n{raw_input}"

        user_id = "user"

        if agent_turns:
            final_text = await self._run_with_seeded_history(user_id=user_id, agent_turns=agent_turns)
        else:
            final_text = await self._run_single_turn(user_id=user_id, raw_input=str(raw_input))

        if final_text is None:
            # Keep this deterministic-ish so evaluators see a string.
            metadata = item.get("metadata", {}) if isinstance(item, dict) else item.metadata
            task_id = metadata.get("task_id") if isinstance(metadata, dict) else None
            logger.warning("No final response produced (task_id=%s)", task_id)
            return ""

        return final_text.strip()

    async def _run_with_seeded_history(self, *, user_id: str, agent_turns: list[dict[str, Any]]) -> str | None:
        """Seed transcript history into the ADK session, then run one turn."""
        session = await self._session_service.create_session(
            app_name=getattr(self._agent, "name", "misalignment_qa"),
            user_id=user_id,
            state={},
        )

        history_turns = agent_turns[:-1]
        latest_turn = agent_turns[-1]

        for turn in history_turns:
            role = (turn.get("role") or "user").lower()
            content_role = "model" if role == "assistant" else "user"
            author_role = getattr(self._agent, "name", "assistant") if role == "assistant" else "user"
            content_text = str(turn.get("content", ""))
            if not content_text:
                continue

            await self._session_service.append_event(
                session=session,
                event=Event(
                    author=author_role,
                    content=types.Content(
                        role=content_role,
                        parts=[types.Part(text=content_text)],
                    ),
                ),
            )

        latest_content = str(latest_turn.get("content", ""))
        if not latest_content:
            logger.warning("Latest turn for agent_turns has empty content: %r", latest_turn)
            return None

        new_message = types.Content(role="user", parts=[types.Part(text=latest_content)])
        return await self._collect_final_text(
            session_id=session.id,
            user_id=user_id,
            new_message=new_message,
        )

    async def _run_single_turn(self, *, user_id: str, raw_input: str) -> str | None:
        """Run a simple one-turn task without any seeded chat history."""
        session = await self._session_service.create_session(
            app_name=getattr(self._agent, "name", "misalignment_qa"),
            user_id=user_id,
            state={},
        )
        message = types.Content(role="user", parts=[types.Part(text=raw_input)])
        return await self._collect_final_text(
            session_id=session.id,
            user_id=user_id,
            new_message=message,
        )

    async def _collect_final_text(
        self,
        *,
        session_id: str,
        user_id: str,
        new_message: types.Content,
    ) -> str | None:
        final_text: str | None = None
        async for event in self._runner.run_async(
            session_id=session_id,
            user_id=user_id,
            new_message=new_message,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                # Exclude thinking parts (part.thought = True) from the returned text.
                # Thinking tokens remain visible in the raw Langfuse trace observation
                # via ADK's model-call logging — nothing is lost for debugging.
                # Including them in the task output would pollute the judge's input with
                # internal reasoning that isn't part of the actual response.
                final_text = "".join(
                    part.text or "" for part in event.content.parts if part.text and not getattr(part, "thought", False)
                )
        return final_text

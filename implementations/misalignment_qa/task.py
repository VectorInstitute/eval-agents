from __future__ import annotations

import getpass
import logging
import uuid
from typing import Any

from google.adk.events import Event
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
        # Keep a dedicated session service so we can seed per-item history.
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            app_name=getattr(agent, "name", "misalignment_qa"),
            agent=agent,
            session_service=self._session_service,
            auto_create_session=False,
        )

    async def __call__(self, *, item: ExperimentItem, **kwargs: Any) -> str | None:
        del kwargs  # accepted for protocol compatibility

        # item can be a local dict-like item or a Langfuse experiment item object.
        raw_input = item.get("input") if isinstance(item, dict) else item.input

        # If the dataset contains a separate, agent-focused prompt payload, prefer it.
        metadata: Any = item.get("metadata", {}) if isinstance(item, dict) else getattr(item, "metadata", {})  # noqa: ANN401
        agent_input = None
        agent_turns: list[dict[str, Any]] | None = None
        if isinstance(metadata, dict):
            agent_input = metadata.get("agent_input")
            turns_raw = metadata.get("agent_turns")
            if isinstance(turns_raw, list):
                agent_turns = [t for t in turns_raw if isinstance(t, dict)]

        effective_input = agent_input if agent_input is not None else raw_input

        if effective_input is None and not agent_turns:
            logger.warning("Task received item without input: %r", item)
            return None

        user_id = getpass.getuser()

        final_text: str | None = None

        if agent_turns:
            # Use structured turns to seed a true multi-turn chat history in the ADK session.
            session = await self._session_service.create_session(
                app_name=getattr(self._agent, "name", "misalignment_qa"),
                user_id=user_id,
                state={},
            )

            # All but the last turn are treated as prior history.
            history_turns = agent_turns[:-1]
            latest_turn = agent_turns[-1]

            for t in history_turns:
                role = (t.get("role") or "user").lower()
                # Map assistant transcript turns to the ADK/model author role.
                author_role = "model" if role == "assistant" else "user"
                content_text = str(t.get("content", ""))
                if not content_text:
                    continue

                await self._session_service.append_event(
                    session=session,
                    event=Event(
                        author=author_role,
                        content=types.Content(
                            role=author_role,
                            parts=[types.Part(text=content_text)],
                        ),
                    ),
                )

            # The latest user message becomes the new_message for this invocation.
            latest_content = str(latest_turn.get("content", ""))
            if not latest_content:
                logger.warning("Latest turn for agent_turns has empty content: %r", latest_turn)
                return None

            new_message = types.Content(
                role="user",
                parts=[types.Part(text=latest_content)],
            )

            async for event in self._runner.run_async(
                session_id=session.id,
                user_id=user_id,
                new_message=new_message,
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    final_text = "".join(part.text or "" for part in event.content.parts if part.text)
        else:
            # Fallback: single-turn input flattened into a user message.
            message = types.Content(role="user", parts=[types.Part(text=str(effective_input))])

            async for event in self._runner.run_async(
                session_id=str(uuid.uuid4()),
                user_id=user_id,
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


"""Helpers for preparing experiment datasets, variant runs, and agent configs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from implementations.misalignment_qa.config_types import (
    AgentSpec,
    ExamplePairSpec,
    ExperimentConfig,
    MessageSpec,
    TaskItemSpec,
    VariantSpec,
)


_USER_CONTEXT_HEADER = "Here are some example interactions to guide your responses:"
_USER_CONTEXT_SEPARATOR = "---\nNow please respond to the following:"


@dataclass(frozen=True)
class PreparedTaskItem:
    """A task item ready for upload to Langfuse, with all fields resolved."""

    task_id: str
    task_fingerprint: str
    upload_input: str
    expected_output: str
    task_turns: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_upload_item(self) -> dict[str, Any]:
        """Serialize to the dict format expected by upload_dataset_to_langfuse."""
        return {
            "input": self.upload_input,
            "expected_output": self.expected_output,
            "metadata": {
                "task_id": self.task_id,
                "task_fingerprint": self.task_fingerprint,
                "task_turns": self.task_turns,
                **self.metadata,
            },
        }


@dataclass(frozen=True)
class PreparedVariantRun:
    """A variant fully resolved and ready to run against the Langfuse dataset."""

    variant_id: str
    display_label: str
    description: str | None
    run_instance_id: str
    run_started_at: str
    run_name: str
    run_display_name: str
    run_metadata: dict[str, Any]
    agent_spec: AgentSpec
    shared_turns: list[dict[str, Any]]
    user_context_preamble: str | None = None


def get_tasks_subset(config: ExperimentConfig) -> list[TaskItemSpec]:
    """Return the configured task subset, or all tasks if no subset is set."""
    return config.tasks[: config.dataset_upload_subset] if config.dataset_upload_subset else config.tasks


def build_task_turns(task: TaskItemSpec) -> list[MessageSpec]:
    """Convert a task spec into MessageSpec turns for seeding into the agent session."""
    if task.input is not None:
        return [MessageSpec(role="user", content=task.input)]

    assert task.transcript is not None
    assert task.current_user_message is not None

    turns = list(task.transcript)
    turns.append(MessageSpec(role="user", content=task.current_user_message))
    return turns


def build_judge_input(task: TaskItemSpec, *, max_chars: int = 1000) -> str:
    """Extract the user-facing input text to pass to the LLM judge."""
    if task.current_user_message is not None:
        text = f"Latest user message: {task.current_user_message}"
    elif task.input is not None:
        text = task.input
    elif task.transcript:
        text = f"Conversation context: {task.transcript[-1].content}"
    else:
        text = ""
    return text[:max_chars]


def example_pair_to_messages(example: ExamplePairSpec) -> list[MessageSpec]:
    return [
        MessageSpec(role="user", content=example.user),
        MessageSpec(role="assistant", content=example.assistant),
    ]


def format_examples_as_user_context(examples: list[ExamplePairSpec]) -> str:
    """Format example pairs as a plain-text block to prepend to the user message.

    This simulates a realistic attack surface where examples are embedded directly
    in a user prompt rather than injected at the LLM API message level — something
    any end-user could do, not just developers with API access.
    """
    lines: list[str] = [_USER_CONTEXT_HEADER, ""]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"User: {ex.user}")
        lines.append(f"Assistant: {ex.assistant}")
        lines.append("")
    lines.append(_USER_CONTEXT_SEPARATOR)
    return "\n".join(lines)


def build_shared_turns(config: ExperimentConfig, variant: VariantSpec) -> list[MessageSpec]:
    """Build the list of shared example turns to seed into the agent session."""
    example_pairs = variant.examples if variant.examples is not None else config.examples
    shared_turns: list[MessageSpec] = []
    for example in example_pairs:
        shared_turns.extend(example_pair_to_messages(example))
    return shared_turns


def resolve_agent_spec(config: ExperimentConfig, variant: VariantSpec) -> AgentSpec:
    """Merge base_agent and variant agent overrides into a fully-resolved AgentSpec.

    Uses model_fields_set to distinguish 'not mentioned' (inherit base) from
    'explicitly set to null' (intentionally clear the base value).
    """
    # Base provides defaults; the variant overrides only fields it explicitly declares.
    # We must distinguish "field not mentioned" (inherit base) from "field set to null"
    # (intentionally clear the base value, e.g. temperature: null for models that
    # have deprecated it). model_fields_set contains only fields the variant author
    # actually wrote, so we include those even when their value is None.
    base = config.base_agent.model_dump(exclude_none=True)
    variant_explicit = {k: v for k, v in variant.agent.model_dump().items() if k in variant.agent.model_fields_set}
    merged = {**base, **variant_explicit}

    if not merged.get("system_prompt") or not merged.get("model"):
        raise ValueError(
            f"Variant '{variant.id}' does not resolve to a complete agent config; "
            "make sure system_prompt and model are set across base_agent + variant.agent."
        )

    if merged.get("provider") == "litellm":
        merged["thinking_include_thoughts"] = False
        merged["thinking_budget"] = None

    return AgentSpec(
        system_prompt=merged["system_prompt"],
        model=merged["model"],
        provider=merged.get("provider", "google"),
        api_base=merged.get("api_base"),
        api_key_env=merged.get("api_key_env"),
        temperature=merged.get("temperature"),  # None = let provider use its default
        max_output_tokens=merged.get("max_output_tokens"),
        tools=merged.get("tools", []),
        thinking_include_thoughts=merged.get("thinking_include_thoughts", False),
        thinking_budget=merged.get("thinking_budget"),
        timeout_sec=merged.get("timeout_sec"),
    )


def effective_variant_label(variant: VariantSpec) -> str:
    """Return the variant display label, falling back to the variant ID."""
    return variant.display_label or variant.id


@dataclass(frozen=True)
class ExecutionIdentity:
    """Unique identity for one experiment launch (timestamp + random suffix)."""

    run_instance_id: str
    run_started_at: str


def create_execution_identity(*, now: datetime | None = None) -> ExecutionIdentity:
    """Create a fresh execution identity with a timestamped, unique run_instance_id."""
    timestamp = (now or datetime.now(tz=UTC)).astimezone(UTC)
    timestamp_slug = timestamp.strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return ExecutionIdentity(
        run_instance_id=f"{timestamp_slug}_{suffix}",
        run_started_at=timestamp.isoformat(),
    )


def build_run_name(config: ExperimentConfig, variant: VariantSpec, *, execution: ExecutionIdentity) -> str:
    return f"{config.id}__{execution.run_instance_id}__{variant.id}"


def build_run_display_name(config: ExperimentConfig, variant: VariantSpec, *, execution: ExecutionIdentity) -> str:
    return f"{config.display_label} / {effective_variant_label(variant)} / {execution.run_instance_id}"


def build_run_metadata(
    config: ExperimentConfig,
    variant: VariantSpec,
    *,
    execution: ExecutionIdentity,
    resolved_model: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "exp_id": config.id,
        "variant_id": variant.id,
        "model": resolved_model,
        "run_instance_id": execution.run_instance_id,
        "run_started_at": execution.run_started_at,
        "run_family": f"{config.id}__{variant.id}",
    }
    for key, value in variant.condition_metadata.items():
        metadata[f"condition_{key}"] = value
    return metadata


def build_task_fingerprint(task: TaskItemSpec) -> str:
    payload = task.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def build_dataset_input(task: TaskItemSpec, *, task_fingerprint: str) -> str:
    judge_input = build_judge_input(task)
    return "\n".join(
        [
            f"Task ID: {task.id}",
            f"Task fingerprint: {task_fingerprint}",
            judge_input,
        ]
    ).strip()


def prepare_task_item(task: TaskItemSpec) -> PreparedTaskItem:
    """Resolve a single task spec into a PreparedTaskItem ready for Langfuse upload."""
    task_fingerprint = build_task_fingerprint(task)
    return PreparedTaskItem(
        task_id=task.id,
        task_fingerprint=task_fingerprint,
        upload_input=build_dataset_input(task, task_fingerprint=task_fingerprint),
        expected_output=task.expected_output,
        task_turns=[message.model_dump() for message in build_task_turns(task)],
        metadata=dict(task.metadata),
    )


def prepare_dataset_items(config: ExperimentConfig) -> list[PreparedTaskItem]:
    """Prepare all task items (respecting dataset_upload_subset) for upload."""
    return [prepare_task_item(task) for task in get_tasks_subset(config)]


def prepare_variant_runs(
    config: ExperimentConfig,
    *,
    execution: ExecutionIdentity | None = None,
) -> list[PreparedVariantRun]:
    """Resolve all variant specs into PreparedVariantRun objects for the runner."""
    resolved_execution = execution or create_execution_identity()
    prepared_runs: list[PreparedVariantRun] = []
    for variant in config.variants:
        resolved_agent = resolve_agent_spec(config, variant)
        example_pairs = variant.examples if variant.examples is not None else config.examples

        if variant.examples_inject_mode == "user_context":
            shared_turns: list[dict[str, Any]] = []
            user_context_preamble: str | None = (
                format_examples_as_user_context(example_pairs) if example_pairs else None
            )
        else:
            shared_turns = [message.model_dump() for message in build_shared_turns(config, variant)]
            user_context_preamble = None

        prepared_runs.append(
            PreparedVariantRun(
                variant_id=variant.id,
                display_label=effective_variant_label(variant),
                description=variant.description or config.description,
                run_instance_id=resolved_execution.run_instance_id,
                run_started_at=resolved_execution.run_started_at,
                run_name=build_run_name(config, variant, execution=resolved_execution),
                run_display_name=build_run_display_name(config, variant, execution=resolved_execution),
                run_metadata=build_run_metadata(
                    config,
                    variant,
                    execution=resolved_execution,
                    resolved_model=resolved_agent.model,
                ),
                agent_spec=resolved_agent,
                shared_turns=shared_turns,
                user_context_preamble=user_context_preamble,
            )
        )
    return prepared_runs


__all__ = [
    "ExecutionIdentity",
    "PreparedTaskItem",
    "PreparedVariantRun",
    "build_judge_input",
    "build_task_turns",
    "create_execution_identity",
    "format_examples_as_user_context",
    "get_tasks_subset",
    "prepare_dataset_items",
    "prepare_task_item",
    "prepare_variant_runs",
    "resolve_agent_spec",
]

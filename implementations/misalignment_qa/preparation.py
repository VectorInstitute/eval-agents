from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from implementations.misalignment_qa.config_types import (
    AgentSpec,
    ExamplePairSpec,
    ExperimentConfig,
    MessageSpec,
    TaskItemSpec,
    VariantSpec,
)


@dataclass(frozen=True)
class PreparedTaskItem:
    task_id: str
    task_fingerprint: str
    upload_input: str
    judge_input: str
    expected_output: str
    task_turns: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_upload_item(self) -> dict[str, Any]:
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
    variant_id: str
    display_label: str
    description: str
    run_name: str
    run_display_name: str
    run_metadata: dict[str, Any]
    agent_spec: AgentSpec
    shared_turns: list[dict[str, Any]]


def get_tasks_subset(config: ExperimentConfig) -> list[TaskItemSpec]:
    return config.tasks[: config.dataset_upload_subset] if config.dataset_upload_subset else config.tasks


def build_task_turns(task: TaskItemSpec) -> list[MessageSpec]:
    if task.input is not None:
        return [MessageSpec(role="user", content=task.input)]

    assert task.transcript is not None
    assert task.current_user_message is not None

    turns = list(task.transcript)
    turns.append(MessageSpec(role="user", content=task.current_user_message))
    return turns


def build_judge_input(task: TaskItemSpec, *, max_chars: int = 1000) -> str:
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


def build_shared_turns(config: ExperimentConfig, variant: VariantSpec) -> list[MessageSpec]:
    example_pairs = variant.examples if variant.examples is not None else config.examples
    shared_turns: list[MessageSpec] = []
    for example in example_pairs:
        shared_turns.extend(example_pair_to_messages(example))
    return shared_turns


def resolve_agent_spec(config: ExperimentConfig, variant: VariantSpec) -> AgentSpec:
    base_agent = config.base_agent
    variant_agent = variant.agent

    system_prompt = variant_agent.system_prompt if variant_agent.system_prompt is not None else base_agent.system_prompt
    model = variant_agent.model if variant_agent.model is not None else base_agent.model

    if system_prompt is None or model is None:
        raise ValueError(
            f"Variant '{variant.id}' does not resolve to a complete agent config; "
            "make sure system_prompt and model are set across base_agent + variant.agent."
        )

    temperature = variant_agent.temperature if variant_agent.temperature is not None else base_agent.temperature
    thinking_include_thoughts = (
        variant_agent.thinking_include_thoughts
        if variant_agent.thinking_include_thoughts is not None
        else base_agent.thinking_include_thoughts
    )

    return AgentSpec(
        system_prompt=system_prompt,
        model=model,
        temperature=temperature if temperature is not None else 0.7,
        max_output_tokens=(
            variant_agent.max_output_tokens
            if variant_agent.max_output_tokens is not None
            else base_agent.max_output_tokens
        ),
        tools=variant_agent.tools if variant_agent.tools is not None else (base_agent.tools or []),
        thinking_include_thoughts=thinking_include_thoughts if thinking_include_thoughts is not None else False,
        timeout_sec=variant_agent.timeout_sec if variant_agent.timeout_sec is not None else base_agent.timeout_sec,
    )


def effective_variant_label(variant: VariantSpec) -> str:
    return variant.display_label or variant.id


def build_run_name(config: ExperimentConfig, variant: VariantSpec) -> str:
    return f"{config.id}__{variant.id}"


def build_run_display_name(config: ExperimentConfig, variant: VariantSpec) -> str:
    return f"{config.display_label} / {effective_variant_label(variant)}"


def build_run_metadata(config: ExperimentConfig, variant: VariantSpec, *, resolved_model: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "exp_id": config.id,
        "variant_id": variant.id,
        "model": resolved_model,
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
    task_fingerprint = build_task_fingerprint(task)
    return PreparedTaskItem(
        task_id=task.id,
        task_fingerprint=task_fingerprint,
        upload_input=build_dataset_input(task, task_fingerprint=task_fingerprint),
        judge_input=build_judge_input(task),
        expected_output=task.expected_output,
        task_turns=[message.model_dump() for message in build_task_turns(task)],
        metadata=dict(task.metadata),
    )


def prepare_dataset_items(config: ExperimentConfig) -> list[PreparedTaskItem]:
    return [prepare_task_item(task) for task in get_tasks_subset(config)]


def prepare_variant_runs(config: ExperimentConfig) -> list[PreparedVariantRun]:
    prepared_runs: list[PreparedVariantRun] = []
    for variant in config.variants:
        resolved_agent = resolve_agent_spec(config, variant)
        prepared_runs.append(
            PreparedVariantRun(
                variant_id=variant.id,
                display_label=effective_variant_label(variant),
                description=variant.description or config.description,
                run_name=build_run_name(config, variant),
                run_display_name=build_run_display_name(config, variant),
                run_metadata=build_run_metadata(config, variant, resolved_model=resolved_agent.model),
                agent_spec=resolved_agent,
                shared_turns=[message.model_dump() for message in build_shared_turns(config, variant)],
            )
        )
    return prepared_runs


__all__ = [
    "PreparedTaskItem",
    "PreparedVariantRun",
    "build_judge_input",
    "build_task_turns",
    "get_tasks_subset",
    "prepare_dataset_items",
    "prepare_task_item",
    "prepare_variant_runs",
    "resolve_agent_spec",
]

from __future__ import annotations

from typing import Any, Literal

from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from pydantic import BaseModel, Field, model_validator

ExamplesInjectMode = Literal["llm_messages", "user_context"]


class AgentToolSpec(BaseModel):
    name: str = Field(description="Tool name; e.g. google_search, web_fetch, read_file.")
    enabled: bool = True


class AgentSpec(BaseModel):
    system_prompt: str = Field(description="System prompt/instructions for the agent.")
    model: str = Field(description="ADK model name (e.g. gemini-2.5-flash).")
    provider: Literal["google", "litellm"] = Field(
        default="google",
        description="Model backend provider. Use 'google' for native ADK/Google models and 'litellm' for providers like Anthropic/OpenAI/xAI.",
    )
    api_base: str | None = Field(
        default=None,
        description="Optional LiteLLM API base URL for OpenAI-compatible custom endpoints.",
    )
    api_key_env: str | None = Field(
        default=None,
        description="Optional environment variable name containing the API key for this LiteLLM endpoint.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature. Set to None to let the provider use its default — "
            "recommended for newer Anthropic models where temperature is deprecated."
        ),
    )
    max_output_tokens: int | None = Field(default=None, ge=1)
    tools: list[AgentToolSpec] = Field(default_factory=list)
    thinking_include_thoughts: bool = Field(default=False)
    thinking_budget: int | None = Field(default=None, ge=-1)
    timeout_sec: int | None = Field(default=None, ge=1)


class AgentOverrideSpec(BaseModel):
    system_prompt: str | None = Field(default=None, description="Optional override for the agent system prompt.")
    model: str | None = Field(default=None, description="Optional override for the ADK model name.")
    provider: Literal["google", "litellm"] | None = Field(
        default=None,
        description="Optional override for the model backend provider.",
    )
    api_base: str | None = Field(
        default=None,
        description="Optional LiteLLM API base URL for OpenAI-compatible custom endpoints.",
    )
    api_key_env: str | None = Field(
        default=None,
        description="Optional environment variable name containing the API key for this LiteLLM endpoint.",
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    tools: list[AgentToolSpec] | None = None
    thinking_include_thoughts: bool | None = None
    thinking_budget: int | None = Field(default=None, ge=-1)
    timeout_sec: int | None = Field(default=None, ge=1)


class MessageSpec(BaseModel):
    role: str
    content: str


class ExamplePairSpec(BaseModel):
    user: str
    assistant: str


class TaskItemSpec(BaseModel):
    id: str
    expected_output: str = Field(
        description="Aligned behavior statement or example output (acts as desired behavior when no firm ground truth exists)."
    )

    # Either provide `input`, or provide a transcript + current user message.
    input: str | None = None
    transcript: list[MessageSpec] | None = None
    current_user_message: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_prompt_fields(self) -> "TaskItemSpec":
        has_direct_input = self.input is not None
        has_transcript = self.transcript is not None
        has_current = self.current_user_message is not None
        if has_direct_input:
            # If input is set, transcript fields must be empty/ignored for clarity.
            if has_transcript or has_current:
                raise ValueError("Provide either `input` OR (`transcript` + `current_user_message`), not both.")
            return self

        if has_transcript and has_current:
            return self

        raise ValueError("TaskItemSpec must have either `input` or both `transcript` and `current_user_message`.")


class LLMJudgeSpec(BaseModel):
    rubric_markdown: str = Field(
        description="Rubric markdown that instructs the judge to emit structured JSON metrics."
    )
    judge_model_config: LLMRequestConfig = Field(default_factory=LLMRequestConfig)
    max_output_chars: int = Field(default=1000, ge=1)


class TraceUsageMetricsSpec(BaseModel):
    tool_call_count: bool = True
    turn_count: bool = True
    observation_count: bool = False
    latency_sec: bool = False
    total_input_tokens: bool = False
    total_output_tokens: bool = False
    total_cost: bool = False


class EvalSpec(BaseModel):
    llm_judge: LLMJudgeSpec
    trace_usage_metrics: TraceUsageMetricsSpec = Field(default_factory=TraceUsageMetricsSpec)
    max_concurrency: int = Field(default=1, ge=1)
    trace_max_concurrency: int = Field(default=10, ge=1)
    trace_wait_max_sec: float = Field(default=180.0, gt=0.0)


class VariantSpec(BaseModel):
    id: str
    display_label: str | None = None
    description: str | None = None
    agent: AgentOverrideSpec = Field(default_factory=AgentOverrideSpec)
    examples: list[ExamplePairSpec] | None = None
    examples_inject_mode: ExamplesInjectMode = Field(
        default="llm_messages",
        description=(
            "How to inject examples into the agent context. "
            "'llm_messages' (default) seeds examples as prior conversation turns via the LLM API. "
            "'user_context' formats them as a text block prepended to the user's message — "
            "this mimics a realistic attack surface where examples are embedded in the prompt "
            "rather than injected at the API level."
        ),
    )
    condition_metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    id: str
    display_label: str
    langfuse_dataset_name: str
    description: str | None = None

    base_agent: AgentOverrideSpec = Field(default_factory=AgentOverrideSpec)
    examples: list[ExamplePairSpec] = Field(default_factory=list)
    variants: list[VariantSpec]
    tasks: list[TaskItemSpec]
    evaluation: EvalSpec

    dataset_upload_subset: int | None = Field(
        default=None,
        description="Optional limit for how many task items to upload/evaluate (useful for quick end-to-end smoke tests).",
        ge=1,
    )

    @model_validator(mode="after")
    def _validate_experiment(self) -> "ExperimentConfig":
        task_ids = [task.id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Task IDs must be unique within an experiment config.")

        if not self.variants:
            raise ValueError("ExperimentConfig must define at least one variant.")

        variant_ids = [variant.id for variant in self.variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("Variant IDs must be unique within an experiment config.")

        for variant in self.variants:
            has_system_prompt = self.base_agent.system_prompt is not None or variant.agent.system_prompt is not None
            has_model = self.base_agent.model is not None or variant.agent.model is not None
            if not has_system_prompt or not has_model:
                raise ValueError(
                    f"Variant '{variant.id}' does not resolve to a complete agent config; "
                    "make sure system_prompt and model are set across base_agent + variant.agent."
                )

        return self

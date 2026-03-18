from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig


class AgentToolSpec(BaseModel):
    name: str = Field(description="Tool name; e.g. google_search, web_fetch, read_file.")
    enabled: bool = True


class AgentSpec(BaseModel):
    system_prompt: str = Field(description="System prompt/instructions for the agent.")
    model: str = Field(description="ADK model name (e.g. gemini-2.5-flash).")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    tools: list[AgentToolSpec] = Field(default_factory=list)
    thinking_include_thoughts: bool = Field(default=False)
    timeout_sec: int | None = Field(default=None, ge=1)


class MessageSpec(BaseModel):
    role: str
    content: str


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

    def build_agent_input(self) -> str:
        """Build the full prompt that the agent should answer."""
        if self.input is not None:
            return self.input

        assert self.transcript is not None
        assert self.current_user_message is not None

        lines: list[str] = ["Conversation so far:"]
        for msg in self.transcript:
            role = msg.role.strip()
            label = role.capitalize() if role.lower() in ("user", "assistant") else role
            lines.append(f"{label}: {msg.content}")
        lines.extend(["", "Now respond to the latest user message.", f"User: {self.current_user_message}"])
        return "\n".join(lines)

    def build_judge_input(self, *, max_chars: int = 1000) -> str:
        """Build a compact task description for the LLM judge."""
        if self.current_user_message is not None:
            text = f"Latest user message: {self.current_user_message}"
        elif self.input is not None:
            text = self.input
        elif self.transcript:
            text = f"Conversation context: {self.transcript[-1].content}"
        else:
            text = ""
        return text[:max_chars]


class LLMJudgeSpec(BaseModel):
    rubric_markdown: str = Field(description="Rubric markdown that instructs the judge to emit structured JSON metrics.")
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


class ExperimentConfig(BaseModel):
    langfuse_dataset_name: str
    experiment_name: str
    description: str = "Misalignment experiment"

    agent: AgentSpec
    tasks: list[TaskItemSpec]
    evaluation: EvalSpec

    dataset_upload_subset: int | None = Field(
        default=None,
        description="Optional limit for how many task items to upload/evaluate (useful for quick end-to-end smoke tests).",
        ge=1,
    )


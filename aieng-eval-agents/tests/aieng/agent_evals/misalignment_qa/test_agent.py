"""Tests for misalignment_qa agent builder and agent spec resolution."""

from typing import Any, cast

from aieng.agent_evals.misalignment_qa.agent import build_misalignment_agent
from aieng.agent_evals.misalignment_qa.config_types import (
    AgentOverrideSpec,
    AgentSpec,
    EvalSpec,
    ExperimentConfig,
    LLMJudgeSpec,
    VariantSpec,
)
from aieng.agent_evals.misalignment_qa.preparation import resolve_agent_spec
from google.adk.models.lite_llm import LiteLlm
from pytest import MonkeyPatch


def test_build_misalignment_agent_uses_litellm_for_litellm_provider(monkeypatch: MonkeyPatch) -> None:
    """Agent built with provider='litellm' should use a LiteLlm model backend."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-google-key")

    agent = build_misalignment_agent(
        AgentSpec(
            system_prompt="Be helpful",
            provider="litellm",
            model="anthropic/claude-sonnet-4-6",
            temperature=0.2,
            max_output_tokens=1024,
        )
    )

    assert isinstance(agent.model, LiteLlm)
    assert agent.model.model == "anthropic/claude-sonnet-4-6"


def test_build_misalignment_agent_passes_custom_litellm_endpoint(monkeypatch: MonkeyPatch) -> None:
    """api_base and api_key_env should be forwarded to the LiteLlm model."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-google-key")
    monkeypatch.setenv("VECTOR_INFERENCE_API_KEY", "test-key")

    agent = build_misalignment_agent(
        AgentSpec(
            system_prompt="Be helpful",
            provider="litellm",
            model="openai/gpt-oss-120b",
            api_base="https://proxy.vectorinstitute.ai/v1",
            api_key_env="VECTOR_INFERENCE_API_KEY",
            temperature=0.2,
            max_output_tokens=1024,
        )
    )

    assert isinstance(agent.model, LiteLlm)
    additional_args = cast(dict[str, Any], agent.model._additional_args)
    assert additional_args["api_base"] == "https://proxy.vectorinstitute.ai/v1"
    assert additional_args["api_key"] == "test-key"


def test_resolve_agent_spec_clears_gemini_thinking_for_litellm_variants() -> None:
    """Thinking budget/thoughts should be cleared when resolving a LiteLLM variant."""
    config = ExperimentConfig(
        id="demo",
        display_label="Demo",
        langfuse_dataset_name="demo-dataset",
        description="demo",
        base_agent=AgentOverrideSpec(
            system_prompt="Be helpful",
            provider="google",
            model="gemini-2.5-flash",
            thinking_budget=-1,
            thinking_include_thoughts=True,
        ),
        examples=[],
        variants=[
            VariantSpec(
                id="claude",
                agent=AgentOverrideSpec(
                    provider="litellm",
                    model="anthropic/claude-opus-4-6",
                ),
            )
        ],
        tasks=[],
        evaluation=EvalSpec(llm_judge=LLMJudgeSpec(rubric_markdown="Return JSON only.")),
    )

    resolved = resolve_agent_spec(config, config.variants[0])

    assert resolved.provider == "litellm"
    assert resolved.model == "anthropic/claude-opus-4-6"
    assert resolved.thinking_budget is None
    assert resolved.thinking_include_thoughts is False


def test_resolve_agent_spec_preserves_custom_litellm_endpoint() -> None:
    """api_base and api_key_env from base_agent should pass through to resolved spec."""
    config = ExperimentConfig(
        id="demo",
        display_label="Demo",
        langfuse_dataset_name="demo-dataset",
        description="demo",
        base_agent=AgentOverrideSpec(
            system_prompt="Be helpful",
            provider="litellm",
            model="openai/gpt-oss-120b",
            api_base="https://proxy.vectorinstitute.ai/v1",
            api_key_env="VECTOR_INFERENCE_API_KEY",
        ),
        examples=[],
        variants=[
            VariantSpec(
                id="vector",
                agent=AgentOverrideSpec(),
            )
        ],
        tasks=[],
        evaluation=EvalSpec(llm_judge=LLMJudgeSpec(rubric_markdown="Return JSON only.")),
    )

    resolved = resolve_agent_spec(config, config.variants[0])

    assert resolved.provider == "litellm"
    assert resolved.model == "openai/gpt-oss-120b"
    assert resolved.api_base == "https://proxy.vectorinstitute.ai/v1"
    assert resolved.api_key_env == "VECTOR_INFERENCE_API_KEY"

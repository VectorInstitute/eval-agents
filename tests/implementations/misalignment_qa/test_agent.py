from pathlib import Path
import sys

from google.adk.models.lite_llm import LiteLlm

sys.path.append(str(Path(__file__).resolve().parents[3]))

from implementations.misalignment_qa.agent import build_misalignment_agent
from implementations.misalignment_qa.config_types import AgentOverrideSpec, AgentSpec, EvalSpec, ExperimentConfig, LLMJudgeSpec, VariantSpec
from implementations.misalignment_qa.preparation import resolve_agent_spec


def test_build_misalignment_agent_uses_litellm_for_litellm_provider() -> None:
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


def test_resolve_agent_spec_clears_gemini_thinking_for_litellm_variants() -> None:
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

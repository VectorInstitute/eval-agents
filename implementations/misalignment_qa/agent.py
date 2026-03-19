from __future__ import annotations

import logging
from typing import Any

from google.genai.types import GenerateContentConfig, HttpOptions, ThinkingConfig

from aieng.agent_evals.configs import Configs
from aieng.agent_evals.tools import (
    create_fetch_file_tool,
    create_grep_file_tool,
    create_google_search_tool,
    create_read_file_tool,
    create_web_fetch_tool,
)

from implementations.misalignment_qa.config_types import AgentSpec, AgentToolSpec

from google.adk.agents import LlmAgent


logger = logging.getLogger(__name__)


TOOL_FACTORIES: dict[str, Any] = {
    "google_search": lambda configs: create_google_search_tool(config=configs),
    "web_fetch": lambda _configs: create_web_fetch_tool(),
    "fetch_file": lambda _configs: create_fetch_file_tool(),
    "grep_file": lambda _configs: create_grep_file_tool(),
    "read_file": lambda _configs: create_read_file_tool(),
}
SUPPORTED_TOOL_NAMES: tuple[str, ...] = tuple(TOOL_FACTORIES.keys())


def _build_tools(configs: Configs, tools: list[AgentToolSpec]) -> list[Any]:
    enabled = [t for t in tools if t.enabled]
    if not enabled:
        return []

    out: list[Any] = []
    for spec in enabled:
        factory = TOOL_FACTORIES.get(spec.name)
        if not factory:
            raise ValueError(f"Unsupported tool: {spec.name}")
        out.append(factory(configs))

    return out


def build_misalignment_agent(spec: AgentSpec, *, name: str = "misalignment_qa_agent") -> LlmAgent:
    """
    Build a configurable ADK LlmAgent.

    This is intentionally minimal: it focuses on prompt/system-instruction configurability
    and tool selection so the evaluator/test harness can remain the main “experiment driver”.
    """

    configs = Configs()  # reads env/.env via pydantic-settings

    tool_list = _build_tools(configs=configs, tools=spec.tools)

    generate_cfg = GenerateContentConfig(
        http_options=HttpOptions(timeout=spec.timeout_sec * 1000) if spec.timeout_sec is not None else None,
        temperature=spec.temperature,
        max_output_tokens=spec.max_output_tokens,
        thinking_config=ThinkingConfig(include_thoughts=spec.thinking_include_thoughts, thinking_budget=spec.thinking_budget),
    )

    # NOTE: We intentionally do not force a planner; for misalignment probing you often want
    # the agent to produce the next completion directly (tools may or may not be enabled).
    return LlmAgent(
        name=name,
        description="Configurable misalignment probe agent (ADK).",
        instruction=spec.system_prompt,
        tools=tool_list,
        model=spec.model,
        generate_content_config=generate_cfg,
    )


__all__ = ["SUPPORTED_TOOL_NAMES", "build_misalignment_agent"]


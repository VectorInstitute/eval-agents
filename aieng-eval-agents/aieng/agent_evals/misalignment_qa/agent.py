"""ADK agent builder for misalignment QA experiments."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from aieng.agent_evals.configs import Configs
from aieng.agent_evals.misalignment_qa.config_types import AgentSpec, AgentToolSpec
from aieng.agent_evals.tools import (
    create_fetch_file_tool,
    create_google_search_tool,
    create_grep_file_tool,
    create_read_file_tool,
    create_web_fetch_tool,
)
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai.types import GenerateContentConfig, HttpOptions, ThinkingConfig


logger = logging.getLogger(__name__)


TOOL_FACTORIES: dict[str, Callable[[Configs], Any]] = {
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


def _build_generate_content_config(spec: AgentSpec) -> GenerateContentConfig:
    if spec.provider == "litellm":
        # Pass temperature when it is set; None causes ADK to omit the field
        # entirely (provider uses its default). Set temperature: null in the
        # variant's agent config for models that have deprecated it
        # (e.g. claude-opus-4-7).
        return GenerateContentConfig(
            temperature=spec.temperature,
            max_output_tokens=spec.max_output_tokens,
        )

    return GenerateContentConfig(
        http_options=HttpOptions(timeout=spec.timeout_sec * 1000) if spec.timeout_sec is not None else None,
        temperature=spec.temperature,
        max_output_tokens=spec.max_output_tokens,
        thinking_config=ThinkingConfig(
            include_thoughts=spec.thinking_include_thoughts,
            thinking_budget=spec.thinking_budget,
        ),
    )


def _resolve_api_key(configs: Configs, api_key_env: str) -> str | None:
    """Return the API key for *api_key_env*, preferring Configs SecretStr fields.

    Configs fields carry ``SecretStr`` protection, which prevents values from
    appearing in logs or exception tracebacks. For env vars not mirrored in
    ``Configs`` we fall back to ``os.getenv``.
    """
    _config_secrets = {
        "ANTHROPIC_API_KEY": configs.anthropic_api_key,
        "VECTOR_INFERENCE_API_KEY": configs.vector_inference_api_key,
    }
    secret = _config_secrets.get(api_key_env)
    if secret is not None:
        return secret.get_secret_value()
    return os.getenv(api_key_env)


def _build_model(spec: AgentSpec, configs: Configs) -> str | LiteLlm:
    if spec.provider == "litellm":
        if spec.thinking_budget is not None or spec.thinking_include_thoughts:
            logger.warning(
                "Ignoring thinking settings for LiteLLM-backed model '%s'; those settings are Gemini-specific.",
                spec.model,
            )
        kwargs: dict[str, Any] = {"drop_params": True}
        if spec.timeout_sec is not None:
            kwargs["timeout"] = spec.timeout_sec
        if spec.api_base is not None:
            kwargs["api_base"] = spec.api_base
        if spec.api_key_env is not None:
            api_key = _resolve_api_key(configs, spec.api_key_env)
            if not api_key:
                raise ValueError(
                    f"Environment variable '{spec.api_key_env}' is required for LiteLLM model '{spec.model}'."
                )
            kwargs["api_key"] = api_key
        return LiteLlm(model=spec.model, **kwargs)

    return spec.model


def build_misalignment_agent(spec: AgentSpec, *, name: str = "assistant") -> LlmAgent:
    """Build a configurable ADK ``LlmAgent`` for misalignment QA experiments.

    Intentionally minimal: focuses on prompt/system-instruction configurability
    and tool selection so the test harness remains the main experiment driver.

    Parameters
    ----------
    spec : AgentSpec
        Resolved agent specification (provider, model, prompt, tools, etc.).
    name : str, optional
        Name assigned to the underlying ``LlmAgent``. Defaults to ``"assistant"``.

    Returns
    -------
    LlmAgent
        A configured ADK agent ready to be invoked by the experiment runner.

    Raises
    ------
    ValueError
        If ``spec.tools`` contains an unsupported tool name, or if
        ``spec.api_key_env`` is set but the corresponding environment
        variable is empty.
    """
    configs = Configs()  # type: ignore[call-arg]  # fields populated from env vars

    tool_list = _build_tools(configs=configs, tools=spec.tools)
    generate_cfg = _build_generate_content_config(spec)
    model = _build_model(spec, configs)

    # No planner forced — for misalignment probing we want the agent to produce
    # the next completion directly (tools may or may not be enabled).
    return LlmAgent(
        name=name,
        description="",
        instruction=spec.system_prompt,
        tools=tool_list,
        model=model,
        generate_content_config=generate_cfg,
    )


__all__ = ["SUPPORTED_TOOL_NAMES", "build_misalignment_agent"]

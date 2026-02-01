"""Configuration settings for agent evaluations.

This module provides centralized configuration management using Pydantic settings,
supporting environment variables and .env file loading.
"""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Configs(BaseSettings):
    """Central configuration for all agent evaluations.

    This class automatically loads configuration values from environment variables
    and a .env file. Service-specific fields are optional - agents validate
    required fields at initialization.

    Examples
    --------
    >>> from aieng.agent_evals.configs import Configs
    >>> config = Configs()
    >>> print(config.default_worker_model)
    'gemini-2.5-flash'
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", env_ignore_empty=True)

    # === Core LLM Settings ===
    openai_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        description="Base URL for OpenAI-compatible API (defaults to Gemini endpoint).",
    )
    openai_api_key: str = Field(
        validation_alias=AliasChoices("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
        description="API key for OpenAI-compatible API (accepts OPENAI_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY).",
    )
    default_planner_model: str = Field(
        default="gemini-2.5-pro",
        description="Model name for planning/complex reasoning tasks.",
    )
    default_worker_model: str = Field(
        default="gemini-2.5-flash",
        description="Model name for worker/simple tasks.",
    )
    default_evaluator_model: str = Field(
        default="gemini-2.5-pro",
        description="Model name for LLM-as-judge evaluation tasks.",
    )
    default_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Default temperature for LLM generation. Lower values (0.0-0.3) produce more consistent outputs.",
    )
    default_evaluator_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM-as-judge evaluations. Default 0.0 for deterministic judging.",
    )

    # === Tracing (Langfuse) ===
    langfuse_public_key: str | None = Field(
        default=None,
        pattern=r"^pk-lf-.*$",
        description="Langfuse public key for tracing (must start with 'pk-lf-').",
    )
    langfuse_secret_key: str | None = Field(
        default=None,
        pattern=r"^sk-lf-.*$",
        description="Langfuse secret key for tracing (must start with 'sk-lf-').",
    )
    langfuse_host: str = Field(
        default="https://us.cloud.langfuse.com",
        description="Langfuse host URL.",
    )

    # === Embedding Service ===
    embedding_base_url: str | None = Field(
        default=None,
        description="Base URL for embedding API service.",
    )
    embedding_api_key: str | None = Field(
        default=None,
        description="API key for embedding service.",
    )
    embedding_model_name: str = Field(
        default="@cf/baai/bge-m3",
        description="Name of the embedding model.",
    )

    # === E2B Code Interpreter ===
    e2b_api_key: str | None = Field(
        default=None,
        pattern=r"^e2b_.*$",
        description="E2B.dev API key for code interpreter (must start with 'e2b_').",
    )
    default_code_interpreter_template: str | None = Field(
        default="9p6favrrqijhasgkq1tv",
        description="Default template name or ID for E2B.dev code interpreter.",
    )

    # === Web Search ===
    web_search_base_url: str | None = Field(
        default=None,
        description="Base URL for web search service.",
    )
    web_search_api_key: str | None = Field(
        default=None,
        description="API key for web search service.",
    )

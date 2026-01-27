"""Configuration settings for the knowledge-grounded QA agent.

This module provides centralized configuration management using Pydantic settings,
supporting environment variables and .env file loading.
"""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class KnowledgeAgentConfig(BaseSettings):
    """Configuration settings for the knowledge-grounded QA agent.

    This class automatically loads configuration values from environment variables
    and a .env file, providing type-safe access to all settings.

    Attributes
    ----------
    openai_base_url : str
        Base URL for OpenAI-compatible API (defaults to Gemini endpoint).
    openai_api_key : str
        API key for OpenAI-compatible API (accepts OPENAI_API_KEY, GEMINI_API_KEY,
        or GOOGLE_API_KEY).
    default_planner_model : str
        Model name for planning/complex reasoning tasks.
    default_worker_model : str
        Model name for worker/simple tasks.
    langfuse_public_key : str or None
        Langfuse public key for tracing (optional).
    langfuse_secret_key : str or None
        Langfuse secret key for tracing (optional).
    langfuse_host : str
        Langfuse host URL.

    Examples
    --------
    >>> from aieng.agent_evals.knowledge_agent.config import KnowledgeAgentConfig
    >>> config = KnowledgeAgentConfig()
    >>> print(config.default_worker_model)
    'gemini-2.5-flash'
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", env_ignore_empty=True)

    # OpenAI-compatible API settings (works with Gemini via OpenAI endpoint)
    openai_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    openai_api_key: str = Field(validation_alias=AliasChoices("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"))

    # Model selection (using stable Gemini 2.5 models)
    # See https://ai.google.dev/gemini-api/docs/models for latest models
    default_planner_model: str = "gemini-2.5-pro"
    default_worker_model: str = "gemini-2.5-flash"

    # Langfuse configuration (optional)
    langfuse_public_key: str | None = Field(default=None, pattern=r"^pk-lf-.*$")
    langfuse_secret_key: str | None = Field(default=None, pattern=r"^sk-lf-.*$")
    langfuse_host: str = "https://us.cloud.langfuse.com"

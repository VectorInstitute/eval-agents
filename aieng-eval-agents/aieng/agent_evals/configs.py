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

    # === Weaviate Vector Database ===
    weaviate_collection_name: str = Field(
        default="enwiki_20250520",
        description="Name of the Weaviate collection to use.",
    )
    weaviate_api_key: str | None = Field(
        default=None,
        description="API key for Weaviate cloud instance.",
    )
    weaviate_http_host: str | None = Field(
        default=None,
        pattern=r"^.*\.weaviate\.cloud$|^localhost$",
        description="Weaviate HTTP host (must end with .weaviate.cloud or be 'localhost').",
    )
    weaviate_grpc_host: str | None = Field(
        default=None,
        pattern=r"^grpc-.*\.weaviate\.cloud$|^localhost$",
        description="Weaviate gRPC host (must start with 'grpc-' and end with .weaviate.cloud, or be 'localhost').",
    )
    weaviate_http_port: int = Field(
        default=443,
        description="Port for Weaviate HTTP connections.",
    )
    weaviate_grpc_port: int = Field(
        default=443,
        description="Port for Weaviate gRPC connections.",
    )
    weaviate_http_secure: bool = Field(
        default=True,
        description="Use secure HTTP connection for Weaviate.",
    )
    weaviate_grpc_secure: bool = Field(
        default=True,
        description="Use secure gRPC connection for Weaviate.",
    )

    # === Vertex AI / Google Cloud ===
    vertex_ai_project: str | None = Field(
        default=None,
        validation_alias="VERTEX_AI_PROJECT",
        description="Google Cloud project ID for Vertex AI.",
    )
    vertex_ai_location: str = Field(
        default="us-central1",
        validation_alias="VERTEX_AI_LOCATION",
        description="Google Cloud region for Vertex AI.",
    )
    vector_search_index_endpoint: str | None = Field(
        default=None,
        validation_alias="VECTOR_SEARCH_INDEX_ENDPOINT",
        description="Vertex AI Vector Search index endpoint.",
    )
    vector_search_index_id: str | None = Field(
        default=None,
        validation_alias="VECTOR_SEARCH_INDEX_ID",
        description="Vertex AI Vector Search index ID.",
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

"""Configuration settings for agent evaluations.

This module provides centralized configuration management using Pydantic settings,
supporting environment variables and .env file loading.
"""

from typing import Any

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine.url import URL


class DatabaseConfig(BaseModel):
    """Database connection configuration."""

    driver: str = Field(
        ...,
        description="SQLAlchemy dialect (e.g., 'sqlite', 'postgresql', 'mysql+pymysql').",
    )
    username: str | None = Field(
        default=None, description="Database username. For SQLite or integrated authentication, this can be None."
    )
    host: str | None = Field(default=None, description="Database host address or file path for SQLite.")
    password: SecretStr | None = Field(
        default=None, description="Database password. For SQLite or integrated authentication, this can be None."
    )
    port: int | None = Field(default=None, description="Database port number.")
    database: str | None = Field(default=None, description="Database name or file path for SQLite.")
    query: dict[str, Any] = Field(
        default_factory=dict, description="URL query parameters (e.g. {'mode': 'ro'} for read-only SQLite)."
    )

    def build_uri(self) -> str:
        """Construct the SQLAlchemy connection URI safely using the official URL object.

        This handles special character escaping in passwords automatically.

        Returns
        -------
        str
            The full database connection URI.
        """
        return URL.create(
            drivername=self.driver,
            username=self.username,
            password=self.password.get_secret_value() if self.password else None,
            host=self.host,
            port=self.port,
            database=self.database,
            query=self.query,
        ).render_as_string(hide_password=False)


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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        env_nested_delimiter="__",
    )

    aml_db: DatabaseConfig | None = Field(
        default=None,
        description="Anti-Money Laundering database configuration. Used by the Fraud Investigation Agent.",
    )

    # === Core LLM Settings ===
    openai_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        description="Base URL for OpenAI-compatible API (defaults to Gemini endpoint).",
    )
    openai_api_key: SecretStr = Field(
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
    langfuse_secret_key: SecretStr | None = Field(
        default=None,
        description="Langfuse secret key for tracing (must start with 'sk-lf-').",
    )
    langfuse_host: str = Field(
        default="https://us.cloud.langfuse.com", validation_alias="LANGFUSE_BASE_URL", description="Langfuse base URL."
    )

    # === Embedding Service ===
    embedding_base_url: str | None = Field(default=None, description="Base URL for embedding API service.")
    embedding_api_key: SecretStr | None = Field(default=None, description="API key for embedding service.")
    embedding_model_name: str = Field(default="@cf/baai/bge-m3", description="Name of the embedding model.")

    # === Weaviate Vector Database ===
    weaviate_collection_name: str = Field(
        default="enwiki_20250520", description="Name of the Weaviate collection to use."
    )
    weaviate_api_key: SecretStr | None = Field(default=None, description="API key for Weaviate cloud instance.")
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
    weaviate_http_port: int = Field(default=443, description="Port for Weaviate HTTP connections.")
    weaviate_grpc_port: int = Field(default=443, description="Port for Weaviate gRPC connections.")
    weaviate_http_secure: bool = Field(default=True, description="Use secure HTTP connection for Weaviate.")
    weaviate_grpc_secure: bool = Field(default=True, description="Use secure gRPC connection for Weaviate.")

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
    e2b_api_key: SecretStr | None = Field(
        default=None,
        description="E2B.dev API key for code interpreter (must start with 'e2b_').",
    )
    default_code_interpreter_template: str | None = Field(
        default="9p6favrrqijhasgkq1tv",
        description="Default template name or ID for E2B.dev code interpreter.",
    )

    # === Web Search ===
    web_search_base_url: str | None = Field(default=None, description="Base URL for web search service.")
    web_search_api_key: SecretStr | None = Field(default=None, description="API key for web search service.")

    # Validators for the SecretStr fields
    @field_validator("langfuse_secret_key")
    @classmethod
    def validate_langfuse_secret(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate that the Langfuse secret key starts with 'sk-lf-'."""
        if v is not None and not v.get_secret_value().startswith("sk-lf-"):
            raise ValueError("Langfuse secret key must start with 'sk-lf-'")
        return v

    @field_validator("e2b_api_key")
    @classmethod
    def validate_e2b_key(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate that the E2B API key starts with 'e2b_' if provided."""
        if v is not None and not v.get_secret_value().startswith("e2b_"):
            raise ValueError("E2B API key must start with 'e2b_'")
        return v

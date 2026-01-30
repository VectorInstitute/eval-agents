"""Configuration settings for the agent evals."""

from typing import Any

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine.url import URL


class DatabaseConfig(BaseModel):
    """Database connection configuration."""

    driver: str = Field(..., description="SQLAlchemy dialect (e.g., 'sqlite', 'postgresql', 'mysql+pymysql').")
    """SQLAlchemy dialect (e.g., 'sqlite', 'postgresql', 'mysql+pymysql')."""
    username: str | None = None
    """Database username. For SQLite or integrated authentication, this can be None."""
    host: str | None = None
    """Database host address or file path for SQLite."""
    password: SecretStr | None = None
    """Database password. For SQLite or integrated authentication, this can be None."""
    port: int | None = None
    """Database port number."""
    database: str | None = Field(None, description="Database name or file path for SQLite.")
    """Database name or file path for SQLite."""
    query: dict[str, Any] = Field(default_factory=dict, description="URL query parameters (e.g. {'mode': 'ro'})")
    """URL query parameters (e.g. {'mode': 'ro'} for read-only SQLite)."""

    def build_uri(self) -> str:
        """Construct the SQLAlchemy connection URI safely using the official URL object.

        This handles special character escaping in passwords automatically.
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
    """Configuration settings loaded from environment variables.

    This class automatically loads configuration values from environment variables
    and a .env file, and provides type-safe access to all settings. It validates
    environment variables on instantiation.

    Examples
    --------
    >>> from src.utils.env_vars import Configs
    >>> config = Configs()
    >>> print(config.default_planner_model)
    'gemini-2.5-pro'

    Notes
    -----
    Create a .env file in your project root with the required environment
    variables. The class will automatically load and validate them.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_ignore_empty=True, env_nested_delimiter="__"
    )

    aml_db: DatabaseConfig | None = None
    """Anti-Money Laundering database configuration. Used by the Fraud Investigation \
    Agent.
    """

    openai_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    """Base URL for OpenAI-compatible API (defaults to Gemini endpoint)."""
    openai_api_key: SecretStr = Field(
        validation_alias=AliasChoices("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    """API key for OpenAI-compatible API (accepts OPENAI_API_KEY, GEMINI_API_KEY, \
    or GOOGLE_API_KEY)."""

    default_planner_model: str = "gemini-2.5-pro"
    """Model name for planning tasks. This is typically a more capable and expensive \
    model."""
    default_worker_model: str = "gemini-2.5-flash"
    """Model name for worker tasks. This is typically a less expensive model."""

    embedding_base_url: str
    """Base URL for embedding API service."""
    embedding_api_key: SecretStr
    """API key for embedding service."""
    embedding_model_name: str = "@cf/baai/bge-m3"
    """Name of the embedding model."""

    weaviate_collection_name: str = "enwiki_20250520"
    """Name of the Weaviate collection to use."""
    weaviate_api_key: SecretStr | None = None
    """API key for Weaviate cloud instance."""
    weaviate_http_host: str = Field(pattern=r"^.*\.weaviate\.cloud$|localhost")
    """HTTP host for Weaviate cloud instance. Must end with .weaviate.cloud or be \
    'localhost'."""
    weaviate_grpc_host: str = Field(pattern=r"^grpc-.*\.weaviate\.cloud$|localhost")
    """gRPC host for Weaviate cloud instance. Must start with grpc- and end with \
    .weaviate.cloud or be 'localhost'."""
    weaviate_http_port: int = 443
    """Port for Weaviate HTTP connections."""
    weaviate_grpc_port: int = 443
    """Port for Weaviate gRPC connections."""
    weaviate_http_secure: bool = True
    """Use secure HTTP connection."""
    weaviate_grpc_secure: bool = True
    """Use secure gRPC connection."""

    langfuse_public_key: str = Field(pattern=r"^pk-lf-.*$")
    """Langfuse public key (must start with pk-lf-)."""
    langfuse_secret_key: SecretStr
    """Langfuse secret key (must start with sk-lf-)."""
    langfuse_host: str = "https://us.cloud.langfuse.com"
    """Langfuse host URL."""

    e2b_api_key: SecretStr | None = None
    """Optional E2B.dev API key for code interpreter (must start with e2b_)."""
    default_code_interpreter_template: str | None = "9p6favrrqijhasgkq1tv"
    """Optional default template name or ID for E2B.dev code interpreter."""

    # Optional configs for web search tool
    web_search_base_url: str | None = None
    """Optional base URL for web search service."""
    web_search_api_key: SecretStr | None = None
    """Optional API key for web search service."""

    # Add Validators for the SecretStr fields
    @field_validator("langfuse_secret_key")
    @classmethod
    def validate_langfuse_secret(cls, v: SecretStr) -> SecretStr:
        """Validate that the Langfuse secret key starts with 'sk-lf-'."""
        if not v.get_secret_value().startswith("sk-lf-"):
            raise ValueError("Langfuse secret key must start with 'sk-lf-'")
        return v

    @field_validator("e2b_api_key")
    @classmethod
    def validate_e2b_key(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate that the E2B API key starts with 'e2b_' if provided."""
        if v is not None and not v.get_secret_value().startswith("e2b_"):
            raise ValueError("E2B API key must start with 'e2b_'")
        return v

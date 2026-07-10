"""Interface for storing and accessing config env vars."""

from os import environ

import pydantic


class Configs(pydantic.BaseModel):
    """Type-friendly collection of env var configs."""

    # Embeddings
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None

    # Vertex AI Search (custom knowledge base)
    # Full resource name, e.g.:
    # projects/{project}/locations/global/collections/default_collection/dataStores/{id}
    vertex_datastore_id: str | None = None
    google_cloud_location: str = "us-central1"

    # Langfuse
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str = "https://us.cloud.langfuse.com"

    def _check_langfuse(self):
        """Ensure that Langfuse pk and sk are in the right place."""
        if not self.langfuse_public_key.startswith("pk-lf-"):
            raise ValueError("LANGFUSE_PUBLIC_KEY should start with pk-lf-")

        if not self.langfuse_secret_key.startswith("sk-lf-"):
            raise ValueError("LANGFUSE_SECRET_KEY should start with sk-lf-")

    @staticmethod
    def from_env_var() -> "Configs":
        """Initialize from env vars."""
        # Add only config line items defined in Configs.
        data: dict[str, str] = {}
        for k, v in environ.items():
            _key = k.lower()
            data[_key] = v

        try:
            config = Configs(**data)
            config._check_langfuse()
            return config

        except pydantic.ValidationError as e:
            raise ValueError(
                "Some ENV VARs are missing. See above for details. "
                "Try to load your .env file as follows: \n"
                "```\nuv run --env-file .env -m ...\n```"
            ) from e

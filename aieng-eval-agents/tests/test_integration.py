"""Integration tests for Langfuse and Gemini API key validation."""

import pytest
from aieng.agent_evals.configs import Configs
from aieng.agent_evals.tools.search import google_search
from dotenv import load_dotenv
from langfuse import Langfuse


load_dotenv(verbose=True)


@pytest.fixture()
def configs() -> Configs:
    """Load env var configs for testing."""
    return Configs()  # type: ignore[call-arg]


@pytest.mark.integration_test
def test_langfuse_auth(configs: Configs) -> None:
    """Test that Langfuse API keys are valid and authentication succeeds."""
    if not configs.langfuse_public_key or not configs.langfuse_secret_key:
        pytest.skip("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY not set in env vars")

    langfuse_client = Langfuse(
        public_key=configs.langfuse_public_key,
        secret_key=configs.langfuse_secret_key.get_secret_value(),
        host=configs.langfuse_host,
    )

    assert langfuse_client.auth_check(), "Langfuse authentication failed. Check your API keys."


@pytest.mark.integration_test
@pytest.mark.asyncio
async def test_gemini_google_search(configs: Configs) -> None:
    """Test that the Gemini API key is valid by performing a Google Search."""
    result = await google_search("What is the capital of France?", model=configs.default_worker_model)

    assert result["status"] == "success", f"Gemini search failed: {result.get('error')}"
    assert result["summary"], "Gemini search returned an empty summary"
    assert result["source_count"] > 0, "Gemini search returned no sources"

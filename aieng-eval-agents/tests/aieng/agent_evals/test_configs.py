"""Tests for Configs and DatabaseConfig configuration models."""

import os

import pytest
from aieng.agent_evals.configs import Configs, DatabaseConfig
from pydantic import SecretStr, ValidationError


def make_configs() -> Configs:
    """Create Configs without loading any .env file.

    Wraps ``Configs(_env_file=None)`` to avoid a Pyright false-positive:
    pydantic-settings accepts ``_env_file`` as a special init override but it
    is absent from the generated type stubs.
    """
    return Configs(_env_file=None)  # type: ignore[call-arg]


class TestDatabaseConfig:
    """Tests for DatabaseConfig and its build_uri() method."""

    def test_build_uri_sqlite(self):
        """SQLite URI with only driver and database is valid."""
        config = DatabaseConfig(driver="sqlite", database="/tmp/test.db")
        assert config.build_uri() == "sqlite:////tmp/test.db"

    def test_build_uri_postgresql_with_credentials(self):
        """PostgreSQL URI includes host, port, username, and password."""
        config = DatabaseConfig(
            driver="postgresql",
            username="user",
            password=SecretStr("secret"),
            host="localhost",
            port=5432,
            database="mydb",
        )
        assert config.build_uri() == "postgresql://user:secret@localhost:5432/mydb"

    def test_build_uri_includes_query_params(self):
        """Query parameters appear in the rendered URI."""
        config = DatabaseConfig(driver="sqlite", database="/tmp/test.db", query={"mode": "ro"})
        assert "mode=ro" in config.build_uri()

    def test_build_uri_escapes_special_password_chars(self):
        """Special characters in the password are URL-encoded, not exposed verbatim."""
        config = DatabaseConfig(
            driver="postgresql",
            username="user",
            password=SecretStr("p@ss/word"),
            host="localhost",
            port=5432,
            database="db",
        )
        uri = config.build_uri()
        assert "p@ss/word" not in uri  # must be percent-encoded
        assert "user" in uri

    def test_optional_fields_default_to_none(self):
        """username, host, password, port, and database all default to None."""
        config = DatabaseConfig(driver="sqlite")
        assert config.username is None
        assert config.host is None
        assert config.password is None
        assert config.port is None
        assert config.database is None

    def test_query_defaults_to_empty_dict(self):
        """Query field defaults to an empty dict."""
        assert DatabaseConfig(driver="sqlite").query == {}


class TestConfigsDefaults:
    """Tests for default field values in Configs."""

    @pytest.fixture(autouse=True)
    def _required_env(self, monkeypatch):
        """Run with a fully isolated environment containing only required fields."""
        monkeypatch.setattr(os, "environ", {"OPENAI_API_KEY": "test-openai-key", "GEMINI_API_KEY": "test-google-key"})

    def test_default_worker_model(self):
        """default_worker_model is gemini-2.5-flash."""
        assert make_configs().default_worker_model == "gemini-2.5-flash"

    def test_default_planner_model(self):
        """default_planner_model is gemini-2.5-pro."""
        assert make_configs().default_planner_model == "gemini-2.5-pro"

    def test_default_evaluator_model(self):
        """default_evaluator_model is gemini-2.5-pro."""
        assert make_configs().default_evaluator_model == "gemini-2.5-pro"

    def test_default_temperature(self):
        """default_temperature is 1.0."""
        assert make_configs().default_temperature == 1.0

    def test_default_evaluator_temperature(self):
        """default_evaluator_temperature is 0.0."""
        assert make_configs().default_evaluator_temperature == 0.0

    def test_default_openai_base_url(self):
        """openai_base_url defaults to the Gemini googleapis endpoint."""
        assert "googleapis.com" in make_configs().openai_base_url

    def test_optional_fields_default_none(self):
        """All optional service fields default to None."""
        config = make_configs()
        assert config.aml_db is None
        assert config.report_generation_db is None
        assert config.langfuse_public_key is None
        assert config.langfuse_secret_key is None
        assert config.e2b_api_key is None


class TestGoogleApiKey:
    """Tests for the google_api_key field and its env var aliases."""

    @pytest.fixture(autouse=True)
    def _required_env(self, monkeypatch):
        """Run with a clean environment: only OPENAI_API_KEY set, no Google keys."""
        monkeypatch.setattr(os, "environ", {"OPENAI_API_KEY": "test-openai-key"})

    def test_loaded_from_gemini_api_key(self, monkeypatch):
        """google_api_key is populated from GEMINI_API_KEY."""
        monkeypatch.setenv("GEMINI_API_KEY", "my-gemini-key")
        assert make_configs().google_api_key.get_secret_value() == "my-gemini-key"

    def test_loaded_from_google_api_key(self, monkeypatch):
        """google_api_key is populated from GOOGLE_API_KEY."""
        monkeypatch.setenv("GOOGLE_API_KEY", "my-google-key")
        assert make_configs().google_api_key.get_secret_value() == "my-google-key"

    def test_gemini_api_key_takes_priority_over_google_api_key(self, monkeypatch):
        """GEMINI_API_KEY takes priority over GOOGLE_API_KEY when both are set."""
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        config = make_configs()
        assert config.google_api_key.get_secret_value() == "gemini-key"

    def test_secret_value_not_exposed_in_repr(self, monkeypatch):
        """SecretStr does not leak the raw key in repr or str."""
        monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key")
        key = make_configs().google_api_key
        assert "super-secret-key" not in repr(key)
        assert "super-secret-key" not in str(key)


class TestOpenAiApiKeyAliases:
    """Tests for openai_api_key env var aliases."""

    @pytest.fixture(autouse=True)
    def _clear_google_env(self, monkeypatch):
        monkeypatch.setattr(os, "environ", {})

    def test_loaded_from_openai_api_key(self, monkeypatch):
        """openai_api_key is loaded from OPENAI_API_KEY when it is set."""
        monkeypatch.setenv("OPENAI_API_KEY", "my-openai-key")
        monkeypatch.setenv("GEMINI_API_KEY", "test-google-key")
        config = make_configs()
        assert config.openai_api_key.get_secret_value() == "my-openai-key"

    def test_loaded_from_gemini_api_key(self, monkeypatch):
        """openai_api_key falls back to GEMINI_API_KEY when OPENAI_API_KEY is absent."""
        monkeypatch.setenv("GEMINI_API_KEY", "my-gemini-key")
        config = make_configs()
        assert config.openai_api_key.get_secret_value() == "my-gemini-key"

    def test_loaded_from_google_api_key(self, monkeypatch):
        """openai_api_key falls back to GOOGLE_API_KEY as the last alias."""
        monkeypatch.setenv("GOOGLE_API_KEY", "my-google-key")
        config = make_configs()
        assert config.openai_api_key.get_secret_value() == "my-google-key"

    def test_missing_raises_validation_error(self):
        """Configs raises ValidationError when no API key env var is set."""
        with pytest.raises(ValidationError):
            make_configs()


class TestConfigsValidators:
    """Tests for Configs field validators."""

    @pytest.fixture(autouse=True)
    def _required_env(self, monkeypatch):
        monkeypatch.setattr(os, "environ", {"OPENAI_API_KEY": "test-openai-key", "GEMINI_API_KEY": "test-google-key"})

    def test_langfuse_secret_key_valid(self, monkeypatch):
        """langfuse_secret_key accepts values starting with 'sk-lf-'."""
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-valid-secret")
        config = make_configs()
        assert config.langfuse_secret_key is not None
        assert config.langfuse_secret_key.get_secret_value() == "sk-lf-valid-secret"

    def test_langfuse_secret_key_invalid_prefix_raises(self, monkeypatch):
        """langfuse_secret_key rejects values not starting with 'sk-lf-'."""
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "invalid-secret")
        with pytest.raises(ValidationError, match="sk-lf-"):
            make_configs()

    def test_langfuse_secret_key_none_is_allowed(self):
        """langfuse_secret_key accepts None (key not configured)."""
        assert make_configs().langfuse_secret_key is None

    def test_e2b_api_key_valid(self, monkeypatch):
        """e2b_api_key accepts values starting with 'e2b_'."""
        monkeypatch.setenv("E2B_API_KEY", "e2b_valid_key")
        config = make_configs()
        assert config.e2b_api_key is not None
        assert config.e2b_api_key.get_secret_value() == "e2b_valid_key"

    def test_e2b_api_key_invalid_prefix_raises(self, monkeypatch):
        """e2b_api_key rejects values not starting with 'e2b_'."""
        monkeypatch.setenv("E2B_API_KEY", "invalid_key")
        with pytest.raises(ValidationError, match="e2b_"):
            make_configs()

    def test_e2b_api_key_none_is_allowed(self):
        """e2b_api_key accepts None (key not configured)."""
        assert make_configs().e2b_api_key is None

    def test_langfuse_public_key_valid_pattern(self, monkeypatch):
        """langfuse_public_key accepts values matching 'pk-lf-*'."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-abc123")
        config = make_configs()
        assert config.langfuse_public_key == "pk-lf-abc123"

    def test_langfuse_public_key_invalid_pattern_raises(self, monkeypatch):
        """langfuse_public_key rejects values not matching 'pk-lf-*'."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "invalid-key")
        with pytest.raises(ValidationError):
            make_configs()

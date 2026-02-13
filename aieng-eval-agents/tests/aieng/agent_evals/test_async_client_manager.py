"""Tests for AsyncClientManager singleton and client lifecycle."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aieng.agent_evals.async_client_manager import AsyncClientManager


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset AsyncClientManager singleton before and after each test."""
    AsyncClientManager._singleton_instance = None
    yield
    AsyncClientManager._singleton_instance = None


class TestGetInstance:
    """Tests for the get_instance() class method."""

    def test_returns_same_instance(self):
        """get_instance() always returns the same object."""
        first = AsyncClientManager.get_instance()
        second = AsyncClientManager.get_instance()
        assert first is second

    def test_constructor_creates_separate_instance(self):
        """Direct constructor creates a different object than get_instance()."""
        singleton = AsyncClientManager.get_instance()
        separate = AsyncClientManager()
        assert singleton is not separate


class TestConfigs:
    """Tests for lazy config creation."""

    def test_lazy_config_creation(self):
        """Accessing .configs creates a Configs instance when none was provided."""
        manager = AsyncClientManager()
        assert manager._configs is None
        with patch("aieng.agent_evals.async_client_manager.Configs") as mock_configs_cls:
            mock_instance = MagicMock()
            mock_configs_cls.return_value = mock_instance
            result = manager.configs
            assert result is mock_instance


class TestClose:
    """Tests for close() method."""

    @pytest.mark.asyncio
    async def test_closes_openai_client(self):
        """close() closes the OpenAI client."""
        manager = AsyncClientManager()
        mock_client = AsyncMock()
        manager._openai_client = mock_client
        manager._initialized = True

        await manager.close()

        mock_client.close.assert_awaited_once()
        assert manager._openai_client is None

    @pytest.mark.asyncio
    async def test_flushes_langfuse(self):
        """close() flushes and clears the Langfuse client."""
        manager = AsyncClientManager()
        mock_langfuse = MagicMock()
        manager._langfuse_client = mock_langfuse
        manager._initialized = True

        await manager.close()

        mock_langfuse.flush.assert_called_once()
        assert manager._langfuse_client is None

    @pytest.mark.asyncio
    async def test_resets_initialized(self):
        """close() sets _initialized to False."""
        manager = AsyncClientManager()
        manager._initialized = True

        await manager.close()

        assert manager._initialized is False

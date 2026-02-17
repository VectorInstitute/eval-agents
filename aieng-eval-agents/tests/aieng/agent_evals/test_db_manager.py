"""Tests for DbManager singleton and database connection management."""

from unittest.mock import MagicMock, patch

import pytest
from aieng.agent_evals.db_manager import DbManager


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset DbManager singleton before and after each test."""
    DbManager._singleton_instance = None
    yield
    DbManager._singleton_instance = None


class TestGetInstance:
    """Tests for the get_instance() class method."""

    def test_returns_same_instance(self):
        """get_instance() always returns the same object."""
        assert DbManager.get_instance() is DbManager.get_instance()

    def test_constructor_creates_separate_instance(self):
        """Direct constructor creates a different object than get_instance()."""
        singleton = DbManager.get_instance()
        separate = DbManager()
        assert singleton is not separate


class TestConfigHandling:
    """Tests for lazy config creation and setter."""

    def test_lazy_config_creation(self):
        """Accessing .configs creates a Configs instance when none was provided."""
        manager = DbManager()
        assert manager._configs is None
        with patch("aieng.agent_evals.db_manager.Configs") as mock_configs_cls:
            mock_instance = MagicMock()
            mock_configs_cls.return_value = mock_instance
            result = manager.configs
            assert result is mock_instance

    def test_configs_setter(self):
        """Setting .configs stores the value."""
        manager = DbManager()
        mock_configs = MagicMock()
        manager.configs = mock_configs
        assert manager.configs is mock_configs


class TestAmlDb:
    """Tests for aml_db() method."""

    def test_raises_when_config_missing(self):
        """aml_db() raises ValueError when aml_db config is None."""
        mock_configs = MagicMock()
        mock_configs.aml_db = None
        manager = DbManager(configs=mock_configs)
        with pytest.raises(ValueError, match="AML database configuration is missing"):
            manager.aml_db()

    @patch("aieng.agent_evals.db_manager.ReadOnlySqlDatabase")
    def test_creates_correct_connection(self, mock_db_cls):
        """aml_db() creates a ReadOnlySqlDatabase with the right URI."""
        mock_configs = MagicMock()
        mock_configs.aml_db.build_uri.return_value = "sqlite:///test.db"
        manager = DbManager(configs=mock_configs)

        result = manager.aml_db()

        mock_db_cls.assert_called_once_with(
            connection_uri="sqlite:///test.db",
            agent_name="FraudInvestigationAnalyst",
        )
        assert result is mock_db_cls.return_value

    @patch("aieng.agent_evals.db_manager.ReadOnlySqlDatabase")
    def test_returns_cached_instance(self, mock_db_cls):
        """Repeated calls return the same instance without re-creating."""
        mock_configs = MagicMock()
        mock_configs.aml_db.build_uri.return_value = "sqlite:///test.db"
        manager = DbManager(configs=mock_configs)

        first = manager.aml_db()
        second = manager.aml_db()

        assert first is second
        assert mock_db_cls.call_count == 1


class TestReportGenerationDb:
    """Tests for report_generation_db() method."""

    def test_raises_when_config_missing(self):
        """report_generation_db() raises ValueError when config is None."""
        mock_configs = MagicMock()
        mock_configs.report_generation_db = None
        manager = DbManager(configs=mock_configs)
        with pytest.raises(ValueError, match="Report Generation database configuration is missing"):
            manager.report_generation_db()

    @patch("aieng.agent_evals.db_manager.ReadOnlySqlDatabase")
    def test_creates_correct_connection(self, mock_db_cls):
        """report_generation_db() creates a ReadOnlySqlDatabase with the right URI."""
        mock_configs = MagicMock()
        mock_configs.report_generation_db.build_uri.return_value = "sqlite:///reports.db"
        manager = DbManager(configs=mock_configs)

        result = manager.report_generation_db()

        mock_db_cls.assert_called_once_with(
            connection_uri="sqlite:///reports.db",
            agent_name="ReportGenerationAgent",
        )
        assert result is mock_db_cls.return_value

    @patch("aieng.agent_evals.db_manager.ReadOnlySqlDatabase")
    def test_returns_cached_instance(self, mock_db_cls):
        """Repeated calls return the same instance without re-creating."""
        mock_configs = MagicMock()
        mock_configs.report_generation_db.build_uri.return_value = "sqlite:///reports.db"
        manager = DbManager(configs=mock_configs)

        first = manager.report_generation_db()
        second = manager.report_generation_db()

        assert first is second
        assert mock_db_cls.call_count == 1


class TestClose:
    """Tests for close() method."""

    @patch("aieng.agent_evals.db_manager.ReadOnlySqlDatabase")
    def test_disposes_both_connections(self, mock_db_cls):
        """close() disposes both DB connections and sets them to None."""
        mock_aml = MagicMock()
        mock_report = MagicMock()
        mock_db_cls.side_effect = [mock_aml, mock_report]

        mock_configs = MagicMock()
        mock_configs.aml_db.build_uri.return_value = "sqlite:///aml.db"
        mock_configs.report_generation_db.build_uri.return_value = "sqlite:///reports.db"
        manager = DbManager(configs=mock_configs)

        manager.aml_db()
        manager.report_generation_db()

        manager.close()

        mock_aml.close.assert_called_once()
        mock_report.close.assert_called_once()
        assert manager._aml_db is None
        assert manager._report_generation_db is None

    def test_idempotent_when_no_connections(self):
        """close() is a no-op when no connections have been created."""
        mock_configs = MagicMock()
        manager = DbManager(configs=mock_configs)
        manager.close()  # Should not raise

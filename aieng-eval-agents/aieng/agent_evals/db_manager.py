"""Database connection manager with thread-safe singleton pattern.

Provides centralized DB lifecycle management independent of async client handling,
avoiding circular imports with the tools package.
"""

import logging
import threading

from aieng.agent_evals.configs import Configs
from aieng.agent_evals.tools.sql_database import ReadOnlySqlDatabase


logger = logging.getLogger(__name__)


class SingletonMeta(type):
    """Thread-safe metaclass-based singleton.

    Uses double-checked locking to ensure only one instance is created,
    even under concurrent access.
    """

    _instances: dict[type, object] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """Return the singleton instance, creating it on first call."""
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

    def reset_instance(cls) -> None:
        """Remove the singleton instance, allowing a fresh one to be created.

        Intended for test teardown only.
        """
        with cls._lock:
            cls._instances.pop(cls, None)


class DbManager(metaclass=SingletonMeta):
    """Manages database connections with lazy initialization.

    Parameters
    ----------
    configs : Configs | None, optional
        Configuration object. If ``None``, created lazily on first access.
    """

    def __init__(self, configs: Configs | None = None) -> None:
        self._configs: Configs | None = configs
        self._aml_db: ReadOnlySqlDatabase | None = None
        self._report_generation_db: ReadOnlySqlDatabase | None = None

    @property
    def configs(self) -> Configs:
        """Get or create configs instance.

        Returns
        -------
        Configs
            The configuration instance.
        """
        if self._configs is None:
            self._configs = Configs()  # type: ignore[call-arg]
        return self._configs

    @configs.setter
    def configs(self, value: Configs) -> None:
        """Set the configs instance.

        Parameters
        ----------
        value : Configs
            The configuration instance to set.
        """
        self._configs = value

    def aml_db(self, agent_name: str = "FraudInvestigationAnalyst") -> ReadOnlySqlDatabase:
        """Get or create the AML database connection.

        Parameters
        ----------
        agent_name : str, optional
            Name of the agent using this connection,
            by default ``"FraudInvestigationAnalyst"``.

        Returns
        -------
        ReadOnlySqlDatabase
            The AML database connection instance.

        Raises
        ------
        ValueError
            If AML database configuration is missing.
        """
        if self._aml_db is None:
            if self.configs.aml_db is None:
                raise ValueError("AML database configuration is missing.")

            self._aml_db = ReadOnlySqlDatabase(
                connection_uri=self.configs.aml_db.build_uri(),
                agent_name=agent_name,
            )

        return self._aml_db

    def report_generation_db(self, agent_name: str = "ReportGenerationAgent") -> ReadOnlySqlDatabase:
        """Get or create the Report Generation database connection.

        Parameters
        ----------
        agent_name : str, optional
            Name of the agent using this connection,
            by default ``"ReportGenerationAgent"``.

        Returns
        -------
        ReadOnlySqlDatabase
            The Report Generation database connection instance.

        Raises
        ------
        ValueError
            If Report Generation database configuration is missing.
        """
        if self._report_generation_db is None:
            if self.configs.report_generation_db is None:
                raise ValueError("Report Generation database configuration is missing.")

            self._report_generation_db = ReadOnlySqlDatabase(
                connection_uri=self.configs.report_generation_db.build_uri(),
                agent_name=agent_name,
            )

        return self._report_generation_db

    def close(self) -> None:
        """Dispose of all database connections."""
        if self._aml_db is not None:
            self._aml_db.close()
            self._aml_db = None

        if self._report_generation_db is not None:
            self._report_generation_db.close()
            self._report_generation_db = None

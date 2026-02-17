"""Database connection manager for Gradio applications.

Provides centralized DB lifecycle management independent of async client handling,
avoiding circular imports with the tools package.
"""

import logging

from aieng.agent_evals.configs import Configs
from aieng.agent_evals.tools.sql_database import ReadOnlySqlDatabase


logger = logging.getLogger(__name__)


class DbManager:
    """Manages database connections with lazy initialization.

    Parameters
    ----------
    configs : Configs | None, optional
        Configuration object. If ``None``, created lazily on first access.
    """

    _singleton_instance: "DbManager | None" = None

    @classmethod
    def get_instance(cls) -> "DbManager":
        """Get the singleton instance of the DB manager.

        Returns
        -------
        DbManager
            The singleton instance of the DB manager.
        """
        if cls._singleton_instance is None:
            cls._singleton_instance = DbManager()
        return cls._singleton_instance

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

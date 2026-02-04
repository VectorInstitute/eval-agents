"""Async client lifecycle manager for Gradio applications.

Provides idempotent initialization and proper cleanup of async clients
like OpenAI to prevent event loop conflicts during Gradio's hot-reload process.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any

from aieng.agent_evals.configs import Configs
from langfuse import Langfuse


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class SQLiteConnection:
    """SQLite connection."""

    def __init__(self, db_path: Path) -> None:
        """Initialize the SQLite connection.

        Parameters
        ----------
        db_path : Path
            The path to the SQLite database.
        """
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)

    def execute(self, query: str) -> list[Any] | str:
        """Execute a SQLite query.

        Parameters
        ----------
        query : str
            The SQLite query to execute.

        Returns
        -------
        list[Any] | str
            The result of the query. Will return the result of
            `execute(query).fetchall()`.
            Returns a string with an error message if the query fails.
        """
        try:
            return self.connection.execute(query).fetchall()
        except Exception as e:
            logger.exception(f"Error executing query: {e}")
            return [str(e)]

    def close(self) -> None:
        """Close the SQLite connection."""
        self.connection.close()


class AsyncClientManager:
    """Manages async client lifecycle with lazy initialization and cleanup.

    This class ensures clients are created only once and properly closed,
    preventing ResourceWarning errors from unclosed event loops.

    Parameters
    ----------
    configs: Configs | None, optional, default=None
        Configuration object for client setup. If None, a new ``Configs()`` is created.

    Examples
    --------
    >>> manager = AsyncClientManager()
    >>> # Access clients (created on first access)
    >>> sqlite_connection = manager.sqlite_connection("my_sqlite.db")
    >>> langfuse = manager.langfuse_client
    >>> # In finally block or cleanup
    >>> await manager.close()
    """

    _singleton_instance: "AsyncClientManager | None" = None

    @classmethod
    def get_instance(cls) -> "AsyncClientManager":
        """Get the singleton instance of the client manager.

        Returns
        -------
        AsyncClientManager
            The singleton instance of the client manager.
        """
        if cls._singleton_instance is None:
            cls._singleton_instance = AsyncClientManager()
        return cls._singleton_instance

    def __init__(self, configs: Configs | None = None) -> None:
        """Initialize manager with optional configs.

        Parameters
        ----------
        configs : Configs | None, optional
            Configuration object for client setup. If None, a new ``Configs()``
            is created.
        """
        self._configs: Configs | None = configs
        self._sqlite_connection: SQLiteConnection | None = None
        self._langfuse_client: Langfuse | None = None
        self._otel_instrumented: bool = False
        self._initialized: bool = False

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

    def sqlite_connection(self, db_path: Path) -> SQLiteConnection:
        """Get or create SQLite session.

        Parameters
        ----------
        db_path : Path
            The path to the SQLite database.

        Returns
        -------
        SQLiteConnection
            The SQLite connection instance.
        """
        if self._sqlite_connection is None or self._sqlite_connection.db_path != db_path:
            self._sqlite_connection = SQLiteConnection(db_path)
            self._initialized = True
        return self._sqlite_connection

    @property
    def langfuse_client(self) -> Langfuse:
        """Get or create Langfuse client.

        Returns
        -------
        Langfuse
            The Langfuse client instance.
        """
        if self._langfuse_client is None:
            self._langfuse_client = Langfuse(
                public_key=self.configs.langfuse_public_key,
                secret_key=self.configs.langfuse_secret_key,
                host=self.configs.langfuse_host,
            )
            self._initialized = True
        return self._langfuse_client

    @property
    def otel_instrumented(self) -> bool:
        """Check if OpenTelemetry instrumentation has been set up.

        Returns
        -------
        bool
            True if OTEL instrumentation is active, False otherwise.
        """
        return self._otel_instrumented

    @otel_instrumented.setter
    def otel_instrumented(self, value: bool) -> None:
        """Set the OpenTelemetry instrumentation state.

        Parameters
        ----------
        value : bool
            The new instrumentation state.
        """
        self._otel_instrumented = value

    async def close(self) -> None:
        """Close all initialized async clients.

        This method closes the SQLite connection, and Langfuse
        client if they have been initialized.
        """
        if self._sqlite_connection is not None:
            self._sqlite_connection.close()
            self._sqlite_connection = None

        if self._langfuse_client is not None:
            self._langfuse_client.flush()
            self._langfuse_client = None

        self._initialized = False

    def is_initialized(self) -> bool:
        """Check if any clients have been initialized.

        Returns
        -------
        bool
            True if any clients have been initialized, False otherwise.
        """
        return self._initialized

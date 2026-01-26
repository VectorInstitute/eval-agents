"""Async client lifecycle manager for Gradio applications.

Provides idempotent initialization and proper cleanup of async clients
like Weaviate and OpenAI to prevent event loop conflicts during Gradio's
hot-reload process.
"""

import os
import sqlite3
from pathlib import Path
from typing import Any

from aieng.agent_evals.configs import Configs
from openai import AsyncOpenAI
from weaviate.client import WeaviateAsyncClient


# Will use this as default if no path is provided in the
# REPORT_GENERATION_DB_PATH env var
DEFAULT_SQLITE_DB_PATH = Path("aieng-eval-agents/aieng/agent_evals/impl/report_generation/data/OnlineRetail.db")


class SQLiteConnection:
    """SQLite connection."""

    def __init__(self) -> None:
        db_path = os.getenv("REPORT_GENERATION_DB_PATH", DEFAULT_SQLITE_DB_PATH)
        self._connection = sqlite3.connect(db_path)

    def execute(self, query: str) -> list[Any]:
        """Execute a SQLite query.

        Parameters
        ----------
        query : str
            The SQLite query to execute.

        Returns
        -------
        list[Any]
            The result of the query. Will return the result of
            `execute(query).fetchall()`.
        """
        return self._connection.execute(query).fetchall()

    def close(self) -> None:
        """Close the SQLite connection."""
        self._connection.close()


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
    >>> weaviate = manager.weaviate_client
    >>> kb = manager.knowledgebase
    >>> openai = manager.openai_client
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
        self._weaviate_client: WeaviateAsyncClient | None = None
        self._openai_client: AsyncOpenAI | None = None
        self._sqlite_connection: SQLiteConnection | None = None
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
            self._configs = Configs()  # pyright: ignore[reportCallIssue]
        return self._configs

    @property
    def openai_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client.

        Returns
        -------
        AsyncOpenAI
            The OpenAI async client instance.
        """
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI()
            self._initialized = True
        return self._openai_client

    @property
    def sqlite_connection(self) -> SQLiteConnection:
        """Get or create SQLite session.

        Returns
        -------
        SQLiteConnection
            The SQLite connection instance.
        """
        if self._sqlite_connection is None:
            self._sqlite_connection = SQLiteConnection()
            self._initialized = True
        return self._sqlite_connection

    async def close(self) -> None:
        """Close all initialized async clients.

        This method closes the OpenAI client and SQLite connection if they
        have been initialized.
        """
        if self._openai_client is not None:
            await self._openai_client.close()
            self._openai_client = None

        if self._sqlite_connection is not None:
            self._sqlite_connection.close()
            self._sqlite_connection = None

        self._initialized = False

    def is_initialized(self) -> bool:
        """Check if any clients have been initialized.

        Returns
        -------
        bool
            True if any clients have been initialized, False otherwise.
        """
        return self._initialized

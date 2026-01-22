"""Async client lifecycle manager for Gradio applications.

Provides idempotent initialization and proper cleanup of async clients
like Weaviate and OpenAI to prevent event loop conflicts during Gradio's
hot-reload process.
"""

import sqlite3
import urllib.parse
from pathlib import Path
from typing import Any

import pandas as pd
from aieng.agent_evals.report_generation.utils import Configs
from openai import AsyncOpenAI
from weaviate.client import WeaviateAsyncClient


SQLITE_DB_PATH = Path("aieng-eval-agents/aieng/agent_evals/report_generation/data/OnlineRetail.db")
REPORTS_OUTPUT_PATH = Path("aieng-eval-agents/aieng/agent_evals/report_generation/reports/")


class SQLiteConnection:
    """SQLite connection."""

    def __init__(self) -> None:
        self._connection = sqlite3.connect(SQLITE_DB_PATH)

    def execute(self, query: str) -> list[Any]:
        """Execute a SQLite query.

        Args:
            query: The SQLite query to execute.

        Returns
        -------
            The result of the query. Will return the result of
            `execute(query).fetchall()`.
        """
        return self._connection.execute(query).fetchall()

    def close(self) -> None:
        """Close the SQLite connection."""
        self._connection.close()


class ReportFileWriter:
    """Write reports to a file."""

    def write_report_to_file(
        self, report_data: list[Any], report_columns: list[str], filename: str = "report.xlsx"
    ) -> str:
        """Write a report to a XLSX file.

        Args:
            report_data: The data of the report
            report_columns: The columns of the report
            filename: The name of the file to create. Default is "report.xlsx".

        Returns
        -------
            The URL link to the report file.
        """
        # Create reports directory if it doesn't exist
        REPORTS_OUTPUT_PATH.mkdir(exist_ok=True)
        filepath = REPORTS_OUTPUT_PATH / filename

        report_df = pd.DataFrame(report_data, columns=report_columns)
        report_df.to_excel(filepath, index=False)

        return _make_gradio_file_link(filepath)


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
            The singleton instance of the client manager.
        """
        if cls._singleton_instance is None:
            cls._singleton_instance = AsyncClientManager()
        return cls._singleton_instance

    def __init__(self, configs: Configs | None = None) -> None:
        """Initialize manager with optional configs."""
        self._configs: Configs | None = configs
        self._weaviate_client: WeaviateAsyncClient | None = None
        self._openai_client: AsyncOpenAI | None = None
        self._sqlite_connection: SQLiteConnection | None = None
        self._report_file_writer: ReportFileWriter | None = None
        self._initialized: bool = False

    @property
    def configs(self) -> Configs:
        """Get or create configs instance."""
        if self._configs is None:
            self._configs = Configs()  # pyright: ignore[reportCallIssue]
        return self._configs

    @property
    def openai_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client."""
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI()
            self._initialized = True
        return self._openai_client

    @property
    def sqlite_connection(self) -> SQLiteConnection:
        """Get or create SQLite session."""
        if self._sqlite_connection is None:
            self._sqlite_connection = SQLiteConnection()
            self._initialized = True
        return self._sqlite_connection

    @property
    def report_file_writer(self) -> ReportFileWriter:
        """Get or create ReportFileWriter."""
        if self._report_file_writer is None:
            self._report_file_writer = ReportFileWriter()
            self._initialized = True
        return self._report_file_writer

    async def close(self) -> None:
        """Close all initialized async clients."""
        if self._openai_client is not None:
            await self._openai_client.close()
            self._openai_client = None

        if self._sqlite_connection is not None:
            self._sqlite_connection.close()
            self._sqlite_connection = None

        self._initialized = False

    def is_initialized(self) -> bool:
        """Check if any clients have been initialized."""
        return self._initialized


def _make_gradio_file_link(filepath: Path) -> str:
    """Make a Gradio file link from a filepath.

    Args:
        filepath: The path to the file.

    Returns
    -------
        A Gradio file link.
    """
    return f"gradio_api/file={urllib.parse.quote(str(filepath), safe='')}"

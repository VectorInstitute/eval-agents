"""Async client lifecycle manager for Gradio applications.

Provides idempotent initialization and proper cleanup of async clients
like Weaviate and OpenAI to prevent event loop conflicts during Gradio's
hot-reload process.
"""

import os
import sqlite3
import urllib.parse
from pathlib import Path
from typing import Any

import pandas as pd
from aieng.agent_evals.configs import Configs
from openai import AsyncOpenAI
from weaviate.client import WeaviateAsyncClient


# Will use these as default if no path is provided in the
# REPORT_GENERATION_DB_PATH and REPORTS_OUTPUT_PATH env vars
DEFAULT_SQLITE_DB_PATH = Path("aieng-eval-agents/aieng/agent_evals/report_generation/data/OnlineRetail.db")
DEFAULT_REPORTS_OUTPUT_PATH = Path("aieng-eval-agents/aieng/agent_evals/report_generation/reports/")


class SQLiteConnection:
    """SQLite connection."""

    def __init__(self) -> None:
        db_path = os.getenv("REPORT_GENERATION_DB_PATH", DEFAULT_SQLITE_DB_PATH)
        self._connection = sqlite3.connect(db_path)

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
        self,
        report_data: list[Any],
        report_columns: list[str],
        filename: str = "report.xlsx",
        gradio_link: bool = True,
    ) -> str:
        """Write a report to a XLSX file.

        Args:
            report_data: The data of the report
            report_columns: The columns of the report
            filename: The name of the file to create. Default is "report.xlsx".
            gradio_link: Whether to return a file link that works with Gradio UI.
                Default is True.

        Returns
        -------
            The path to the report file. If `gradio_link` is True, will return
                a URL link that allows Gradio UI to donwload the file.
        """
        # Create reports directory if it doesn't exist
        reports_output_path = self.get_reports_output_path()
        reports_output_path.mkdir(exist_ok=True)
        filepath = reports_output_path / filename

        report_df = pd.DataFrame(report_data, columns=report_columns)
        report_df.to_excel(filepath, index=False)

        file_uri = str(filepath)
        if gradio_link:
            file_uri = f"gradio_api/file={urllib.parse.quote(str(file_uri), safe='')}"

        return file_uri

    @staticmethod
    def get_reports_output_path() -> Path:
        """Get the reports output path.

        If no path is provided in the REPORTS_OUTPUT_PATH env var, will use the
        default path in DEFAULT_REPORTS_OUTPUT_PATH.

        Returns
        -------
            The reports output path.
        """
        return Path(os.getenv("REPORTS_OUTPUT_PATH", DEFAULT_REPORTS_OUTPUT_PATH))


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

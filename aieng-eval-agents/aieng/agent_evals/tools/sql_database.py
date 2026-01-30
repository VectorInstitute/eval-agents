"""SQL Database Tool with Read-Only Enforcement for Agents."""

import logging
from datetime import datetime
from typing import Optional

import sqlglot
from sqlalchemy import create_engine, inspect, text
from sqlglot import exp


logger = logging.getLogger(__name__)


class ReadOnlySqlDatabase:
    """A SQL database query tool for Agents.

    Features:
    - AST-based Read-Only Enforcement (SQLGlot)
    - Row Limits & Timeouts
    - Schema Introspection
    - Compliance Audit Logging

    Parameters
    ----------
    connection_uri : str
        SQLAlchemy connection string (e.g., 'sqlite:///data/prod.db?mode=ro'
        or 'postgresql://reader:pass@host/db').
    max_rows : int, default=100
        Hard limit on number of rows returned to the agent.
    query_timeout_sec : int, default=60
        Maximum execution time for queries in seconds.
    agent_name : str, default="UnknownAgent"
        Name of the agent using this tool (for audit logs).
    **kwargs : Any
        Additional keyword arguments passed to SQLAlchemy's ``create_engine`` function.
    """

    def __init__(
        self,
        connection_uri: str,
        max_rows: int = 100,
        query_timeout_sec: int = 60,
        agent_name: str = "UnknownAgent",
        **kwargs,
    ) -> None:
        """Initialize the database tool."""
        self.engine = create_engine(connection_uri, **kwargs)
        self.agent_name = agent_name
        self.max_rows = max_rows
        self.timeout = query_timeout_sec

    def _is_safe_readonly_query(self, query: str) -> bool:
        """Verify that query is semantically read-only using a SQL Parser (SQLGlot).

        Blocks: DELETE, UPDATE, INSERT, DROP, ALTER, etc.
        Allows: SELECT, WITH (if read-only), EXPLAIN.
        """
        try:
            # Parse the query into an AST (Abstract Syntax Tree)
            expressions = sqlglot.parse(query)

            if not expressions:
                # Assume unsafe if we can't parse anything
                logger.warning("Empty parse result - blocking query")
                return False

            for expression in expressions:
                # Check Root Expression Type
                if not isinstance(expression, (exp.Select, exp.Union, exp.Paren)):
                    logger.warning("Blocked Unsafe Query Type: %s", type(expression))
                    return False

                # Deep Search for Forbidden Nodes anywhere in the AST
                # Catches hidden writes inside CTEs or Subqueries
                if expression.find(exp.Delete, exp.Update, exp.Insert, exp.Drop, exp.Alter, exp.TruncateTable):
                    logger.warning("Blocked Query containing Write operation in AST")
                    return False

            return True
        except Exception as e:
            logger.error("SQL Parsing Error: %s", e)
            # If we can't parse it, we don't run it.
            return False

    def get_schema_info(self, table_names: Optional[list[str]] = None) -> str:
        """Return schema for specific tables/views or all if None.

        Parameters
        ----------
        table_names : Optional[list[str]], default=None
            List of table or view names to retrieve schema for. If ``None``,
            retrieves all tables and views.

        Returns
        -------
        str
            Formatted schema information.
        """
        inspector = inspect(self.engine)
        all_tables = inspector.get_table_names()
        try:
            all_views = inspector.get_view_names()
        except Exception:
            all_views = []

        # Normalize (name, kind) so we can report both tables and views.
        all_relations: list[tuple[str, str]] = [(t, "table") for t in all_tables] + [(v, "view") for v in all_views]

        # Filter logic (case-insensitive)
        if table_names:
            targets = {name.lower() for name in table_names}
            relations_to_scan = [(name, kind) for name, kind in all_relations if name.lower() in targets]
        else:
            relations_to_scan = all_relations

        schema_text = []
        for relation_name, relation_kind in relations_to_scan:
            label = "View" if relation_kind == "view" else "Table"
            try:
                columns = inspector.get_columns(relation_name)
                # Compact Format for LLM: "TableName (col1: type, col2: type)"
                col_strs = [f"{c['name']}: {str(c['type'])}" for c in columns]
                schema_text.append(f"{label}: {relation_name}\n  Columns: {', '.join(col_strs)}")
            except Exception:
                if self.engine.dialect.name == "sqlite":
                    try:
                        # SQLite supports PRAGMA table_info for both tables and views.
                        safe_relation_name = relation_name.replace('"', '""')
                        with self.engine.connect() as conn:
                            pragma = conn.execute(text(f'PRAGMA table_info("{safe_relation_name}")'))
                            pragma_rows = pragma.fetchall()
                        col_names = [row[1] for row in pragma_rows]
                        if col_names:
                            schema_text.append(f"{label}: {relation_name}\n  Columns: {', '.join(col_names)}")
                            continue
                    except Exception:
                        pass

                schema_text.append(f"{label}: {relation_name} (Error reading schema)")

        return "\n".join(schema_text)

    def execute(self, query: str) -> str:
        """Execute a SQL query against the database with read-only enforcement.

        Parameters
        ----------
        query : str
            The SQL query string to execute.

        Returns
        -------
        list[Any]
            List of result rows (as tuples or dicts depending on config).

        Raises
        ------
        PermissionError
            If the query attempts to perform a write operation.
        Exception
            For any database execution errors.

        Notes
        -----
        This method enforces a limit on the number of rows returned and logs
        all query attempts for compliance auditing.

        """
        start_time = datetime.now()
        status = "FAILED"
        error_msg = None
        row_count = 0

        try:
            # AST Safety Check
            if not self._is_safe_readonly_query(query):
                raise PermissionError("Security Violation: Query contains prohibited WRITE operations.")

            # Connection & Execution
            with self.engine.connect() as conn:
                # Apply Timeout (Database specific options)
                # Note: 'max_execution_time' syntax varies by DB (MySQL vs Postgres).
                # This generic approach relies on the driver or SQLAlchemy 1.4+
                # execution_options
                if self.engine.dialect.name == "sqlite":
                    # SQLite does not support `execution_options` for timeout directly
                    conn.execute(text(f"PRAGMA busy_timeout = {self.timeout * 1000}"))

                execution_options = {"timeout": self.timeout}
                result = conn.execute(text(query).execution_options(**execution_options))

                # Header Extraction
                keys = list(result.keys())

                # Fetch with Row Limit
                # This protects against excessive memory use
                rows = result.fetchmany(self.max_rows)
                row_count = len(rows)

                # Formatting for LLM (String Table)
                output = [f"| {' | '.join(keys)} |"]
                output.append(f"| {'---' * len(keys)} |")
                for row in rows:
                    output.append(f"| {' | '.join(map(str, row))} |")

                if row_count == self.max_rows:
                    output.append(f"\n... (Truncated at {self.max_rows} rows) ...")

                status = "SUCCESS"
                return "\n".join(output)

        except Exception as e:
            error_msg = str(e)
            return f"Query Error: {error_msg}"

        finally:
            # Compliance Audit Log
            duration = (datetime.now() - start_time).total_seconds()
            log_entry = {
                "timestamp": start_time.isoformat(),
                "agent": self.agent_name,
                "query": query,
                "status": status,
                "rows_returned": row_count,
                "duration_sec": duration,
                "error": error_msg,
            }
            logger.info("AUDIT: %s", log_entry)

    def close(self) -> None:
        """Dispose of the connection pool."""
        self.engine.dispose(close=True)

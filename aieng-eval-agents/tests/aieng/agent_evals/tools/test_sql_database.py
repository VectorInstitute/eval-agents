"""Tests for the ReadOnlySqlDatabase tool."""

from __future__ import annotations

import pytest
from aieng.agent_evals.tools import ReadOnlySqlDatabase, ReadOnlySqlPolicy, sql_database
from sqlalchemy import create_engine


@pytest.fixture()
def sqlite_db_uri(tmp_path) -> str:
    """Create a tiny SQLite database on disk and return its SQLAlchemy URI."""
    db_path = tmp_path / "test_readonly_sql_database.db"
    engine = create_engine(f"sqlite:///{db_path}")

    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                """
                CREATE TABLE customers (
                    customer_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    risk_score INTEGER NOT NULL
                )
                """
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE transactions (
                    transaction_id INTEGER PRIMARY KEY,
                    customer_id INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
                )
                """
            )
            conn.exec_driver_sql(
                """
                INSERT INTO customers (customer_id, name, risk_score) VALUES
                    (1, 'Alice', 10),
                    (2, 'Bob', 80)
                """
            )
            conn.exec_driver_sql(
                """
                INSERT INTO transactions (transaction_id, customer_id, amount, created_at) VALUES
                    (100, 1, 12.34, '2024-01-01T00:00:00Z'),
                    (101, 1, 2500.00, '2024-01-02T00:00:00Z'),
                    (102, 2, 99.99, '2024-01-03T00:00:00Z')
                """
            )
            conn.exec_driver_sql(
                """
                CREATE VIEW customer_totals AS
                SELECT
                    c.customer_id,
                    c.name,
                    SUM(t.amount) AS total_amount
                FROM customers c
                JOIN transactions t ON t.customer_id = c.customer_id
                GROUP BY c.customer_id, c.name
                """
            )

        return f"sqlite:///{db_path}"
    finally:
        engine.dispose()


@pytest.fixture()
def default_db(sqlite_db_uri: str) -> ReadOnlySqlDatabase:
    """Return a ReadOnlySqlDatabase instance using default settings."""
    return ReadOnlySqlDatabase(connection_uri=sqlite_db_uri)


def test_readonly_sql_database_initializes_with_defaults(sqlite_db_uri: str) -> None:
    """Initialize with defaults and store settings."""
    db = ReadOnlySqlDatabase(connection_uri=sqlite_db_uri)
    assert db.engine.dialect.name == "sqlite"
    assert db.agent_name == "UnknownAgent"
    assert db.max_rows == 100
    assert db.timeout == 60
    assert isinstance(db.policy, ReadOnlySqlPolicy)


def test_readonly_sql_database_initializes_with_custom_policy(sqlite_db_uri: str) -> None:
    """Initialize with a custom policy and custom settings."""
    policy = ReadOnlySqlPolicy(allowed_roots=("select", "with", "union", "paren"))
    db = ReadOnlySqlDatabase(connection_uri=sqlite_db_uri, policy=policy, agent_name="test-agent", max_rows=5)
    assert db.agent_name == "test-agent"
    assert db.max_rows == 5
    assert db.policy.allowed_roots == ("select", "with", "union", "paren")


@pytest.mark.parametrize(
    ("kwargs", "expected_error"),
    [
        ({"connection_uri": ""}, ValueError),
        ({"connection_uri": "   "}, ValueError),
        ({"connection_uri": "sqlite:///:memory:", "max_rows": 0}, ValueError),
        ({"connection_uri": "sqlite:///:memory:", "query_timeout_sec": 0}, ValueError),
        ({"connection_uri": "sqlite:///:memory:", "agent_name": ""}, ValueError),
        ({"connection_uri": "sqlite:///:memory:", "policy": object()}, TypeError),
    ],
)
def test_readonly_sql_database_rejects_bad_init_args(kwargs: dict, expected_error: type[Exception]) -> None:
    """Reject invalid initialization arguments."""
    with pytest.raises(expected_error):
        ReadOnlySqlDatabase(**kwargs)


def test_readonly_sql_database_rejects_empty_allowed_roots(sqlite_db_uri: str) -> None:
    """Reject policies with an empty allowlist."""
    policy = ReadOnlySqlPolicy(allowed_roots=())
    with pytest.raises(ValueError):
        ReadOnlySqlDatabase(connection_uri=sqlite_db_uri, policy=policy)


def test_readonly_sql_database_rejects_unknown_sqlglot_expression_types(sqlite_db_uri: str) -> None:
    """Reject policies that reference unknown sqlglot expression names."""
    policy = ReadOnlySqlPolicy(allowed_roots=("select", "no_such_expression"))
    with pytest.raises(ValueError):
        ReadOnlySqlDatabase(connection_uri=sqlite_db_uri, policy=policy)


def test_is_safe_readonly_query_allows_select_by_default(default_db: ReadOnlySqlDatabase) -> None:
    """Allow a simple SELECT by default."""
    assert default_db._is_safe_readonly_query("SELECT customer_id, name FROM customers") is True


def test_is_safe_readonly_query_blocks_with_by_default(default_db: ReadOnlySqlDatabase) -> None:
    """Block CTE usage unless explicitly allowed by policy."""
    assert default_db._is_safe_readonly_query("WITH x AS (SELECT 1) SELECT * FROM x") is False


@pytest.mark.parametrize("query", ["", "   ", "PRAGMA table_info(customers)"])
def test_is_safe_readonly_query_blocks_unsafe_or_empty_roots(default_db: ReadOnlySqlDatabase, query: str) -> None:
    """Block empty queries and unsafe root statement types."""
    assert default_db._is_safe_readonly_query(query) is False


def test_is_safe_readonly_query_blocks_multiple_statements_when_disabled(sqlite_db_uri: str) -> None:
    """Block multi-statement queries when policy forbids them."""
    policy = ReadOnlySqlPolicy(allowed_roots=("select",), allow_multiple_statements=False)
    db = ReadOnlySqlDatabase(connection_uri=sqlite_db_uri, policy=policy)
    assert db._is_safe_readonly_query("SELECT 1; SELECT 2") is False


def test_is_safe_readonly_query_blocks_forbidden_nodes_even_if_root_is_allowed(sqlite_db_uri: str) -> None:
    """Block forbidden nodes even if their root type is allowed."""
    policy = ReadOnlySqlPolicy(allowed_roots=("select", "delete"), forbidden_nodes=("delete",))
    db = ReadOnlySqlDatabase(connection_uri=sqlite_db_uri, policy=policy)
    assert db._is_safe_readonly_query("DELETE FROM customers") is False


def test_is_safe_readonly_query_returns_false_on_parse_errors(
    sqlite_db_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return False (fail closed) if parsing raises."""
    db = ReadOnlySqlDatabase(connection_uri=sqlite_db_uri)
    monkeypatch.setattr("aieng.agent_evals.tools.sql_database.sqlglot.parse", lambda _query: 1 / 0)
    assert db._is_safe_readonly_query("SELECT 1") is False


def test_get_schema_info_includes_tables_and_views(default_db: ReadOnlySqlDatabase) -> None:
    """Include tables/views and at least one expected column name."""
    schema = default_db.get_schema_info()
    assert "Table: customers" in schema
    assert "Table: transactions" in schema
    assert "View: customer_totals" in schema
    assert "customer_id" in schema


def test_get_schema_info_filters_relations_case_insensitively(default_db: ReadOnlySqlDatabase) -> None:
    """Filter requested relations in a case-insensitive way."""
    schema = default_db.get_schema_info(["CUSTOMERS"])
    assert "Table: customers" in schema
    assert "Table: transactions" not in schema


def test_get_schema_info_uses_sqlite_pragma_fallback(sqlite_db_uri: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fall back to PRAGMA table_info when SQLAlchemy inspection fails (SQLite)."""
    db = ReadOnlySqlDatabase(connection_uri=sqlite_db_uri)

    real_inspector = sql_database.inspect(db.engine)

    class FailingColumnsInspector:
        def get_table_names(self):
            """Return table names from the real inspector."""
            return real_inspector.get_table_names()

        def get_view_names(self):
            """Return view names from the real inspector."""
            return real_inspector.get_view_names()

        def get_columns(self, _name):
            """Raise to force the PRAGMA fallback path."""
            raise RuntimeError("boom")

    monkeypatch.setattr(sql_database, "inspect", lambda _engine: FailingColumnsInspector())
    schema = db.get_schema_info()
    assert "Table: customers" in schema
    assert "View: customer_totals" in schema
    assert "Columns: customer_id" in schema


def test_execute_formats_select_results_as_markdown(default_db: ReadOnlySqlDatabase) -> None:
    """Format SELECT results as a markdown table."""
    output = default_db.execute("SELECT customer_id, name FROM customers ORDER BY customer_id")
    assert output.splitlines()[0] == "| customer_id | name |"
    assert "| 1 | Alice |" in output
    assert "| 2 | Bob |" in output


def test_execute_truncates_to_max_rows(sqlite_db_uri: str) -> None:
    """Truncate output when max_rows is reached."""
    db = ReadOnlySqlDatabase(connection_uri=sqlite_db_uri, max_rows=2)
    output = db.execute("SELECT transaction_id FROM transactions ORDER BY transaction_id")
    assert "| transaction_id |" in output
    assert "| 100 |" in output
    assert "| 101 |" in output
    assert "| 102 |" not in output
    assert "... (Truncated at 2 rows) ..." in output


def test_execute_blocks_write_queries(default_db: ReadOnlySqlDatabase) -> None:
    """Return an error when a write query is attempted."""
    output = default_db.execute("INSERT INTO customers (customer_id, name, risk_score) VALUES (3, 'Mallory', 0)")
    assert output.startswith("Query Error:")
    assert "Security Violation" in output


def test_execute_returns_query_error_on_sql_execution_failure(default_db: ReadOnlySqlDatabase) -> None:
    """Return a Query Error string when SQL execution fails."""
    output = default_db.execute("SELECT definitely_not_a_column FROM customers")
    assert output.startswith("Query Error:")

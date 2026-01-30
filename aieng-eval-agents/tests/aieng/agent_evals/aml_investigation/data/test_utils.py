"""Tests for AML dataset utilities."""

from __future__ import annotations

import pandas as pd
import pytest
from aieng.agent_evals.aml_investigation.data import utils


def test_download_dataset_file_validates_inputs() -> None:
    """Reject unsupported download option values."""
    with pytest.raises(ValueError):
        utils.download_dataset_file(illicit_ratio="NOPE")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        utils.download_dataset_file(transactions_size="Huge")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        utils.download_dataset_file(filename="Nope.csv")  # type: ignore[arg-type]


def test_download_dataset_file_calls_kagglehub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Call kagglehub with the expected handle/path and return its result."""
    calls: list[tuple[str, str]] = []

    def fake_download(*, handle: str, path: str) -> str:
        calls.append((handle, path))
        return "/tmp/fake"

    monkeypatch.setattr(utils.kagglehub, "dataset_download", fake_download)
    result = utils.download_dataset_file(illicit_ratio="HI", transactions_size="Small", filename="Trans.csv")

    assert result == "/tmp/fake"
    assert calls == [("ealtman2019/ibm-transactions-for-anti-money-laundering-aml", "HI-Small_Trans.csv")]


def test_normalize_transactions_data_requires_expected_columns() -> None:
    """Raise when required Kaggle columns are missing."""
    with pytest.raises(ValueError):
        utils.normalize_transactions_data(pd.DataFrame([{"Timestamp": "2022/01/01 00:00"}]))


def test_normalize_transactions_data_normalizes_and_dedupes() -> None:
    """Normalize columns, compute transaction_id, drop duplicates, sort by timestamp."""
    df = pd.DataFrame(
        [
            {
                "Timestamp": "2022/08/01 09:00",
                "From Bank": 1,
                "Account": " acct-A ",
                "To Bank": 2,
                "Account.1": " acct-B ",
                "Amount Received": "10.00",
                "Receiving Currency": " USD ",
                "Amount Paid": "10.00",
                "Payment Currency": " USD ",
                "Payment Format": " ACH ",
                "Is Laundering": 0,
            },
            # Earlier timestamp to confirm sorting.
            {
                "Timestamp": "2022/08/01 08:00",
                "From Bank": 1,
                "Account": "acct-A",
                "To Bank": 2,
                "Account.1": "acct-B",
                "Amount Received": "10",
                "Receiving Currency": "USD",
                "Amount Paid": "10",
                "Payment Currency": "USD",
                "Payment Format": "ACH",
                "Is Laundering": 0,
            },
            # Duplicate of the first row to confirm dedupe on transaction_id.
            {
                "Timestamp": "2022/08/01 09:00",
                "From Bank": 1,
                "Account": "acct-A",
                "To Bank": 2,
                "Account.1": "acct-B",
                "Amount Received": "10.00",
                "Receiving Currency": "USD",
                "Amount Paid": "10.00",
                "Payment Currency": "USD",
                "Payment Format": "ACH",
                "Is Laundering": 0,
            },
        ]
    )

    normalized = utils.normalize_transactions_data(df)
    assert {"timestamp", "transaction_id", "from_account", "to_account", "is_laundering"}.issubset(normalized.columns)
    assert normalized["timestamp"].tolist() == ["2022-08-01T08:00:00", "2022-08-01T09:00:00"]
    assert normalized["from_account"].tolist() == ["acct-A", "acct-A"]
    assert normalized["to_account"].tolist() == ["acct-B", "acct-B"]
    assert normalized["transaction_id"].notna().all()
    assert normalized["transaction_id"].nunique() == len(normalized)


@pytest.mark.parametrize(
    ("base_timestamp", "lookback_days", "min_timestamp", "expected"),
    [
        ("2022/08/03 12:00", 0, None, "2022-08-03T12:00:00"),
        ("2022-08-03 12:00:00", 1, None, "2022-08-02T12:00:00"),
        ("2022-08-03T12:00:00", 10, "2022-08-01T00:00:00", "2022-08-01T00:00:00"),
        ("2022-08-03T12:00:00", 1, "2022-08-01T00:00:00", "2022-08-02T12:00:00"),
    ],
)
def test_apply_lookback_window_computes_expected_start(
    base_timestamp: str, lookback_days: int, min_timestamp: str | None, expected: str
) -> None:
    """Compute window start and apply optional minimum timestamp clamp."""
    assert utils.apply_lookback_window(base_timestamp, lookback_days, min_timestamp=min_timestamp) == expected


def test_apply_lookback_window_validates_inputs() -> None:
    """Reject invalid lookback window inputs."""
    with pytest.raises(ValueError):
        utils.apply_lookback_window("2022-08-01T00:00:00", -1)
    with pytest.raises(ValueError):
        utils.apply_lookback_window("", 0)

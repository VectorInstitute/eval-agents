"""Tests for AML case generation/parsing utilities."""

from pathlib import Path

import pandas as pd
import pytest
from aieng.agent_evals.aml_investigation.data.cases import LaunderingPattern, build_cases, parse_patterns_file


@pytest.fixture()
def patterns_file(tmp_path: Path) -> Path:
    """Write a minimal Patterns.txt fixture and return its path."""
    patterns_path = tmp_path / "Patterns.txt"
    patterns_path.write_text(
        "\n".join(
            [
                "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 5 hops",
                "2022/08/01 08:35,011,800D7AE80,008777,8034B4510,101883.33,Yuan,101883.33,Yuan,ACH,1",
                "2022/08/02 12:33,008777,8034B4510,021514,801089BD0,12575.99,Euro,12575.99,Euro,ACH,1",
                "2022/08/22 16:07,021514,801089BD0,001148,801AC0540,11763.23,Euro,11763.23,Euro,ACH,1",
                "2022/08/26 08:24,001148,801AC0540,011081,8056E3D30,13436.63,US Dollar,13436.63,US Dollar,ACH,1",
                "2022/08/31 18:45,011081,8056E3D30,011,800D7AE80,12442.10,US Dollar,12442.10,US Dollar,ACH,1",
                "END LAUNDERING ATTEMPT - CYCLE",
                "",
                "BEGIN LAUNDERING ATTEMPT - STACK",
                "2022/08/09 12:30,005,80C1F0980,023,811AF7CE0,362288.79,Yen,362288.79,Yen,ACH,1",
                "2022/08/13 23:55,023,811AF7CE0,005,80824CF90,353026.47,Yen,353026.47,Yen,ACH,1",
                "2022/08/03 18:12,005,80824CF90,013,80008C270,257267.97,Yen,257267.97,Yen,ACH,1",
                "2022/08/16 02:12,013,80008C270,023,811AF7C40,263730.95,Yen,263730.95,Yen,ACH,1",
                "2022/08/13 18:01,023,811AF7C40,013,80824FD40,142445.81,Yen,142445.81,Yen,ACH,1",
                "2022/08/15 10:53,013,80824FD40,005,80C1F0980,141030.04,Yen,141030.04,Yen,ACH,1",
                "2022/08/05 13:32,0022177,80824F9D0,005,80008D800,682100.98,Yen,682100.98,Yen,ACH,1",
                "2022/08/30 16:15,005,80008D800,0022177,80824F9D0,660159.30,Yen,660159.30,Yen,ACH,1",
                "END LAUNDERING ATTEMPT - STACK",
                "",
                "BEGIN LAUNDERING ATTEMPT - GATHER-SCATTER:  Max 2-degree Fan-In",
                "2022/08/02 00:04,0074695,81C54C0B0,0282,81C54C100,79123.59,Ruble,79123.59,Ruble,ACH,1",
                "2022/08/13 00:19,0173410,81C54C150,0282,81C54C100,748241.32,Ruble,748241.32,Ruble,ACH,1",
                "2022/08/13 13:38,0282,81C54C100,0282,81C54C100,714784.82,Ruble,714784.82,Ruble,ACH,1",
                "2022/09/11 13:00,0282,81C54C100,0074695,81C54C0B0,1503619.96,Ruble,1503619.96,Ruble,ACH,1",
                "END LAUNDERING ATTEMPT - GATHER-SCATTER",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return patterns_path


@pytest.fixture()
def transactions_df() -> pd.DataFrame:
    """Return a tiny transactions DataFrame suitable for build_cases()."""
    return pd.DataFrame(
        [
            {
                "timestamp": "2022-08-01T08:00:00",
                "transaction_id": "txn-001",
                "is_laundering": 0,
                "from_account": "acct-A",
            },
            {
                "timestamp": "2022-08-01T09:00:00",
                "transaction_id": "txn-002",
                "is_laundering": 0,
                "from_account": "acct-A",
            },
            {
                "timestamp": "2022-08-02T10:00:00",
                "transaction_id": "txn-003",
                "is_laundering": 1,
                "from_account": "acct-B",
            },
            {
                "timestamp": "2022-08-03T11:00:00",
                "transaction_id": "txn-004",
                "is_laundering": 0,
                "from_account": "acct-C",
            },
            {
                "timestamp": "2022-08-03T12:00:00",
                "transaction_id": "txn-005",
                "is_laundering": 0,
                "from_account": "acct-D",
            },
        ]
    )


def test_parse_patterns_file_parses_attempts_and_sets_seed_and_window(patterns_file: Path) -> None:
    """Parse multiple attempts and build CaseRecord objects."""
    min_timestamp = "2022-08-01T00:00:00"
    cases = parse_patterns_file(patterns_file, lookback_days=10, min_timestamp=min_timestamp)
    assert len(cases) == 3

    expected_by_pattern = {
        LaunderingPattern.CYCLE: ("2022-08-31T18:45:00", 5),
        LaunderingPattern.STACK: ("2022-08-30T16:15:00", 8),
        LaunderingPattern.GATHER_SCATTER: ("2022-09-11T13:00:00", 4),
    }

    for record in cases:
        assert record.expected_output.is_laundering is True
        assert record.input.seed_transaction_id
        assert record.input.case_id
        assert record.input.window_start == min_timestamp

        seed_timestamp, expected_count = expected_by_pattern[record.expected_output.pattern_type]
        assert record.input.seed_timestamp == seed_timestamp

        attempt_ids = [
            item.strip() for item in record.expected_output.attempt_transaction_ids.split(",") if item.strip()
        ]
        assert len(attempt_ids) == expected_count
        assert attempt_ids[-1] == record.input.seed_transaction_id


def test_parse_patterns_file_rejects_negative_lookback(patterns_file: Path) -> None:
    """Reject negative lookback windows."""
    with pytest.raises(ValueError):
        parse_patterns_file(patterns_file, lookback_days=-1)


def test_parse_patterns_file_raises_for_missing_file(tmp_path: Path) -> None:
    """Raise FileNotFoundError when the patterns file does not exist."""
    with pytest.raises(FileNotFoundError):
        parse_patterns_file(tmp_path / "missing.txt")


@pytest.mark.parametrize(
    "contents",
    [
        # Unterminated block.
        "\n".join(
            [
                "BEGIN LAUNDERING ATTEMPT - CYCLE",
                "2022/08/01 08:35,011,800D7AE80,008777,8034B4510,101883.33,Yuan,101883.33,Yuan,ACH,1",
            ]
        ),
        # Malformed transaction row (<11 columns).
        "\n".join(
            [
                "BEGIN LAUNDERING ATTEMPT - CYCLE",
                "2022/08/01 08:35,011",
                "END LAUNDERING ATTEMPT - CYCLE",
            ]
        ),
    ],
)
def test_parse_patterns_file_rejects_malformed_patterns(tmp_path: Path, contents: str) -> None:
    """Reject malformed patterns files."""
    patterns_path = tmp_path / "Patterns.txt"
    patterns_path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        parse_patterns_file(patterns_path)


def test_parse_patterns_file_rejects_invalid_min_timestamp(patterns_file: Path) -> None:
    """Reject invalid min_timestamp values."""
    with pytest.raises(ValueError):
        parse_patterns_file(patterns_file, min_timestamp="not-a-timestamp")


def test_build_cases_builds_each_case_type(patterns_file: Path, transactions_df: pd.DataFrame) -> None:
    """Build a mix of laundering/false-negative/false-positive/normal cases."""
    cases = build_cases(
        patterns_filepath=patterns_file,
        transactions=transactions_df,
        num_laundering_cases=1,
        num_false_positive_cases=1,
        num_false_negative_cases=1,
        num_normal_cases=1,
        lookback_days=10,
    )
    assert len(cases) == 4

    low_signal_labels = {"QA_SAMPLE", "RANDOM_REVIEW", "RETROSPECTIVE_REVIEW", "MODEL_MONITORING_SAMPLE"}
    laundering = [
        case
        for case in cases
        if case.expected_output.is_laundering and case.input.trigger_label == case.expected_output.pattern_type.value
    ]
    false_negatives = [
        case for case in cases if case.expected_output.is_laundering and case.input.trigger_label in low_signal_labels
    ]
    false_positives = [
        case
        for case in cases
        if (not case.expected_output.is_laundering) and case.input.trigger_label not in low_signal_labels
    ]
    normals = [
        case
        for case in cases
        if (not case.expected_output.is_laundering) and case.input.trigger_label in low_signal_labels
    ]

    assert len(laundering) == 1
    assert laundering[0].expected_output.pattern_type in {
        LaunderingPattern.CYCLE,
        LaunderingPattern.STACK,
        LaunderingPattern.GATHER_SCATTER,
    }

    assert len(false_negatives) == 1
    assert false_negatives[0].expected_output.is_laundering is True
    assert false_negatives[0].expected_output.pattern_type != LaunderingPattern.NONE

    assert len(false_positives) == 1
    assert false_positives[0].expected_output.pattern_type == LaunderingPattern.NONE

    assert len(normals) == 1
    assert normals[0].expected_output.pattern_type == LaunderingPattern.NONE


@pytest.mark.parametrize(
    ("transactions", "kwargs", "expected_error"),
    [
        (object(), {"lookback_days": 0}, TypeError),
        (pd.DataFrame(), {"lookback_days": -1}, ValueError),
        (pd.DataFrame(), {"lookback_days": 0, "num_laundering_cases": -1}, ValueError),
        (pd.DataFrame(), {"lookback_days": 0, "num_false_positive_cases": -1}, ValueError),
        (pd.DataFrame(), {"lookback_days": 0, "num_false_negative_cases": -1}, ValueError),
        (pd.DataFrame(), {"lookback_days": 0, "num_normal_cases": -1}, ValueError),
        (
            pd.DataFrame([{"timestamp": "2022-08-01T08:00:00", "transaction_id": "x", "is_laundering": 0}]),
            {"lookback_days": 0},
            ValueError,
        ),
    ],
)
def test_build_cases_validates_inputs(
    patterns_file: Path, transactions: object, kwargs: dict, expected_error: type[Exception]
) -> None:
    """Validate input types/values and required transaction columns."""
    base_kwargs = {
        "patterns_filepath": patterns_file,
        "transactions": transactions,
        "num_laundering_cases": 0,
        "num_false_positive_cases": 0,
        "num_false_negative_cases": 0,
        "num_normal_cases": 0,
        "lookback_days": 0,
    }
    base_kwargs.update(kwargs)
    with pytest.raises(expected_error):
        build_cases(**base_kwargs)


def test_build_cases_raises_for_missing_patterns_file(transactions_df: pd.DataFrame, tmp_path: Path) -> None:
    """Propagate FileNotFoundError when patterns_filepath does not exist."""
    with pytest.raises(FileNotFoundError):
        build_cases(
            patterns_filepath=tmp_path / "missing.txt",
            transactions=transactions_df,
            num_laundering_cases=0,
            num_false_positive_cases=0,
            num_false_negative_cases=0,
            num_normal_cases=0,
            lookback_days=0,
        )

"""Tests for Langfuse helper utilities."""

import json
from pathlib import Path
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from aieng.agent_evals import langfuse as langfuse_module
from aieng.agent_evals.progress import create_progress, track_with_progress


class _TrackRecorder:
    """Track helper stand-in for uploader tests."""

    def __init__(self) -> None:
        self.descriptions: list[str] = []
        self.totals: list[int | float | None] = []
        self.item_count = 0

    def wrap(
        self, iterable, *, description: str, total: int | float | None = None, transient: bool = False
    ) -> Generator[Any, Any, None]:
        """Yield items while recording invocations."""
        del transient
        self.descriptions.append(description)
        self.totals.append(total)
        for item in iterable:
            self.item_count += 1
            yield item


@pytest.fixture
def mock_client_manager(monkeypatch) -> MagicMock:
    """Patch AsyncClientManager singleton with a local mock."""
    manager = MagicMock()
    manager.langfuse_client = MagicMock()
    manager.close = AsyncMock()
    manager.langfuse_client.create_dataset = MagicMock()
    manager.langfuse_client.create_dataset_item = MagicMock()

    monkeypatch.setattr(langfuse_module.AsyncClientManager, "get_instance", lambda: manager)
    return manager


@pytest.fixture
def patch_progress(monkeypatch) -> _TrackRecorder:
    """Patch track helper so tests stay quiet and deterministic."""
    recorder = _TrackRecorder()
    monkeypatch.setattr(langfuse_module, "track_with_progress", recorder.wrap)
    return recorder


@pytest.mark.asyncio
async def test_upload_dataset_to_langfuse_json(tmp_path: Path, mock_client_manager, patch_progress) -> None:
    """Upload records from JSON and map metadata consistently."""
    dataset_file = tmp_path / "dataset.json"
    records = [
        {
            "id": "case-1",
            "input": {"q": "A"},
            "expected_output": {"a": 1},
            "metadata": {"split": "train"},
        },
        {
            "input": {"q": "B"},
            "expected_output": {"a": 2},
            "metadata": {"split": "test"},
        },
    ]
    dataset_file.write_text(json.dumps(records), encoding="utf-8")

    await langfuse_module.upload_dataset_to_langfuse(str(dataset_file), "json-dataset")

    mock_client_manager.langfuse_client.create_dataset.assert_called_once_with(name="json-dataset")
    assert mock_client_manager.langfuse_client.create_dataset_item.call_count == 2
    assert patch_progress.item_count == 2

    first_call = mock_client_manager.langfuse_client.create_dataset_item.call_args_list[0].kwargs
    second_call = mock_client_manager.langfuse_client.create_dataset_item.call_args_list[1].kwargs

    assert first_call["metadata"] == {"split": "train", "id": "case-1"}
    assert second_call["metadata"] == {"split": "test", "id": 2}
    mock_client_manager.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_upload_dataset_to_langfuse_jsonl(tmp_path: Path, mock_client_manager, patch_progress) -> None:
    """Upload records from JSONL line-by-line with stable line-based IDs."""
    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text(
        "\n".join(
            [
                json.dumps({"input": {"q": "A"}, "expected_output": {"a": 1}}),
                "",
                json.dumps({"input": {"q": "B"}, "expected_output": {"a": 2}, "id": "line-3"}),
            ]
        ),
        encoding="utf-8",
    )

    await langfuse_module.upload_dataset_to_langfuse(str(dataset_file), "jsonl-dataset")

    assert mock_client_manager.langfuse_client.create_dataset_item.call_count == 2
    first_call = mock_client_manager.langfuse_client.create_dataset_item.call_args_list[0].kwargs
    second_call = mock_client_manager.langfuse_client.create_dataset_item.call_args_list[1].kwargs

    assert first_call["metadata"]["id"] == 1
    assert second_call["metadata"]["id"] == "line-3"
    assert patch_progress.item_count == 2


@pytest.mark.asyncio
async def test_upload_dataset_to_langfuse_invalid_jsonl_reports_line(
    tmp_path: Path, mock_client_manager, patch_progress
) -> None:
    """Raise a helpful error when JSONL contains a malformed line."""
    dataset_file = tmp_path / "bad.jsonl"
    dataset_file.write_text(
        "\n".join(
            [
                json.dumps({"input": "ok", "expected_output": "ok"}),
                '{"input": "broken", "expected_output": ',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        await langfuse_module.upload_dataset_to_langfuse(str(dataset_file), "bad-jsonl")


@pytest.mark.asyncio
async def test_upload_dataset_to_langfuse_missing_required_key(
    tmp_path: Path, mock_client_manager, patch_progress
) -> None:
    """Raise a clear error when required fields are missing."""
    dataset_file = tmp_path / "missing.json"
    dataset_file.write_text(
        json.dumps([{"input": "only-input"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected_output"):
        await langfuse_module.upload_dataset_to_langfuse(str(dataset_file), "missing-key")


def test_progress_helpers_smoke() -> None:
    """Create and use shared progress helpers without errors."""
    with create_progress(transient=True) as progress:
        task_id = progress.add_task("Smoke", total=2)
        progress.update(task_id, advance=1)
        progress.update(task_id, advance=1)

        task = progress.tasks[0]
        assert task.completed == 2

    items = list(track_with_progress([1, 2], description="Smoke Track", transient=True))
    assert items == [1, 2]

"""Reusable Rich progress utilities.

This module provides a consistent progress bar style for long-running
workflows across the repository.
"""

from collections.abc import Iterable, Iterator
from typing import TypeVar

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn


T = TypeVar("T")


def create_progress(*, transient: bool = False) -> Progress:
    """Create a standardized Rich ``Progress`` instance.

    Parameters
    ----------
    transient : bool, optional, default=False
        Whether to clear the progress display after completion.
        Defaults to ``False`` so users can inspect completion state.

    Returns
    -------
    Progress
        Configured progress renderer with consistent columns.

    Examples
    --------
    >>> from aieng.agent_evals.progress import create_progress
    >>> progress = create_progress()
    >>> with progress:
    ...     task_id = progress.add_task("Uploading", total=10)
    ...     progress.update(task_id, advance=10)
    """
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        transient=transient,
    )


def track_with_progress(
    iterable: Iterable[T],
    *,
    description: str,
    total: int | float | None = None,
    transient: bool = False,
) -> Iterator[T]:
    """Iterate items while displaying a progress bar.

    This is a ``tqdm``-style helper for common loops. Pass an iterable and iterate
    as usual.

    Parameters
    ----------
    iterable : Iterable[T]
        Iterable of items to process.
    description : str
        Human-readable task description displayed in the progress bar.
    total : int or float or None, optional, default=None
        Expected number of units of work. If omitted, this function attempts
        to infer the total from ``len(iterable)`` when available.
    transient : bool, optional, default=False
        Whether to clear the progress display after completion.

    Examples
    --------
    >>> from aieng.agent_evals.progress import track_with_progress
    >>> for item in track_with_progress([1, 2, 3], description="Uploading"):
    ...     _ = item
    """
    resolved_total = total if total is not None else _infer_total(iterable)

    with create_progress(transient=transient) as progress:
        task_id = progress.add_task(description, total=resolved_total)
        for item in iterable:
            yield item
            progress.update(task_id, advance=1)


def _infer_total(iterable: Iterable[T]) -> int | None:
    """Infer iterable size when ``len`` is available."""
    try:
        return len(iterable)  # type: ignore[arg-type]
    except TypeError:
        return None


__all__ = ["create_progress", "track_with_progress"]

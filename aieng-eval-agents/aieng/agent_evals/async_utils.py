"""Utils for async workflows."""

import asyncio
import types
from typing import Any, Awaitable, Callable, Coroutine, Sequence, TypeVar

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn


T = TypeVar("T")

__all__ = ["rate_limited", "gather_with_progress"]


async def rate_limited(_fn: Callable[[], Awaitable[T]], semaphore: asyncio.Semaphore) -> T:
    """Run ``_fn`` with semaphore rate limit.

    Parameters
    ----------
    _fn : Callable[[], Awaitable[T]]
        The async function to run.
    semaphore : asyncio.Semaphore
        The semaphore to use for rate limiting.

    Returns
    -------
    T
        The result of the async function.
    """
    async with semaphore:
        return await _fn()


async def gather_with_progress(
    coros: "list[types.CoroutineType[Any, Any, T]]",
    description: str = "Running tasks",
) -> Sequence[T]:
    """Run a list of coroutines concurrently and display a rich.Progress bar.

    Returns the results in the same order as the input list.

    Parameters
    ----------
    coros : list[CoroutineType[Any, Any, T]]
        List of coroutines to run.
    description : str, optional
        Description to show in the progress bar, by default "Running tasks".

    Returns
    -------
    Sequence[T]
        List of results, ordered to match the input coroutines.
    """
    # Wrap each coroutine in a Task and remember its original index
    tasks = [asyncio.create_task(_indexed(index=index, coro=coro)) for index, coro in enumerate(coros)]

    # Pre‐allocate a results list; we'll fill in each slot as its Task completes
    results: list[T | None] = [None] * len(tasks)

    # Create and start a Progress bar with a total equal to the number of tasks
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ) as progress:
        progress_task = progress.add_task(description, total=len(tasks))

        # as_completed yields each Task as soon as it finishes
        for finished in asyncio.as_completed(tasks):
            index, result = await finished
            results[index] = result
            progress.update(progress_task, advance=1)

    # At this point, every slot in `results` is guaranteed to be non‐None
    # so we can safely cast it back to List[T]
    return results  # type: ignore


async def _indexed(index: int, coro: Coroutine[None, None, T]) -> tuple[int, T]:
    """Return (index, await coro).

    Parameters
    ----------
    index : int
        The index to pair with the coroutine result.
    coro : Coroutine[None, None, T]
        The coroutine to await.

    Returns
    -------
    tuple[int, T]
        A tuple of the index and the result of the coroutine.
    """
    return index, (await coro)

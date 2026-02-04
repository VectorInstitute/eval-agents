"""Utils for async workflows."""

import asyncio
import contextvars
import threading
from concurrent.futures import Future
from typing import Any, Awaitable, Callable, Coroutine, ParamSpec, Sequence, TypeVar

from aieng.agent_evals.progress import create_progress


T = TypeVar("T")
P = ParamSpec("P")

__all__ = ["rate_limited", "gather_with_progress", "run_coroutine_sync"]


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
    coros: Sequence[Coroutine[Any, Any, T]],
    description: str = "Running tasks",
) -> Sequence[T]:
    """Run a list of coroutines concurrently and display a rich.Progress bar.

    Returns the results in the same order as the input list.

    Parameters
    ----------
    coros : Sequence[Coroutine[Any, Any, T]]
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

    # Use the shared progress style with a total equal to the number of tasks.
    with create_progress() as progress:
        progress_task = progress.add_task(description, total=len(tasks))

        # as_completed yields each Task as soon as it finishes
        for finished in asyncio.as_completed(tasks):
            index, result = await finished
            results[index] = result
            progress.update(progress_task, advance=1)

    # At this point, every slot in `results` is guaranteed to be non‐None
    # so we can safely cast it back to List[T]
    return results  # type: ignore


def run_coroutine_sync(
    coro_fn: Callable[P, Coroutine[Any, Any, T]],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run an async callable from synchronous code.

    This helper supports both normal Python scripts (no active event loop)
    and environments that already have a running loop (for example notebooks).

    Parameters
    ----------
    coro_fn : Callable[P, Coroutine[Any, Any, T]]
        Async callable to execute.
    *args : P.args
        Positional arguments forwarded to ``coro_fn``.
    **kwargs : P.kwargs
        Keyword arguments forwarded to ``coro_fn``.

    Returns
    -------
    T
        Value returned by the awaited coroutine.

    Raises
    ------
    BaseException
        Re-raises any exception raised by ``coro_fn``.

    Notes
    -----
    If a loop is already running, the coroutine is executed in a dedicated
    worker thread with its own event loop.
    """
    # Check whether this thread already has a running event loop.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop: asyncio.run is the simplest and safest path.
        return asyncio.run(coro_fn(*args, **kwargs))

    # Create a thread-safe container for either a return value or an exception.
    result: Future[T] = Future()
    # Copy ContextVar values so request-local state survives the thread hop.
    context = contextvars.copy_context()

    def _runner() -> None:
        """Execute coroutine on a dedicated event loop."""
        try:
            # Run the coroutine in the copied context on a new loop in worker thread.
            result.set_result(context.run(asyncio.run, coro_fn(*args, **kwargs)))
        except BaseException as exc:  # pragma: no cover - surfaced to caller
            # Capture failure so the caller re-raises the original exception.
            result.set_exception(exc)

    # Use a separate thread to avoid calling asyncio.run inside an active loop.
    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join()

    # Return the value, or re-raise any exception stored in the Future.
    return result.result()


async def _indexed(index: int, coro: Coroutine[Any, Any, T]) -> tuple[int, T]:
    """Return (index, await coro).

    Parameters
    ----------
    index : int
        The index to pair with the coroutine result.
    coro : Coroutine[Any, Any, T]
        The coroutine to await.

    Returns
    -------
    tuple[int, T]
        A tuple of the index and the result of the coroutine.
    """
    return index, (await coro)

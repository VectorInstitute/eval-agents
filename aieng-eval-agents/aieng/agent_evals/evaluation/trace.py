"""Trace-based evaluation helpers for the harness."""

import asyncio
import functools
import inspect
import logging
from typing import Any, Awaitable, Literal

import httpx
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.async_utils import gather_with_progress, rate_limited, run_coroutine_sync
from aieng.agent_evals.evaluation.types import (
    Evaluation,
    ExperimentItemResult,
    ExperimentResult,
    TraceEvalResult,
    TraceEvalStatus,
    TraceEvaluatorFunction,
    TraceMetrics,
    TraceObservationPredicate,
    TraceWaitConfig,
)
from aieng.agent_evals.langfuse import flush_traces
from langfuse import Langfuse
from langfuse.api import ObservationsView
from langfuse.api.core import ApiError
from langfuse.api.resources import NotFoundError
from langfuse.api.resources.commons.types.trace_with_full_details import TraceWithFullDetails
from tenacity import AsyncRetrying, RetryError, retry_if_exception, stop_after_delay, wait_exponential


logger = logging.getLogger(__name__)


def run_trace_evaluations(
    experiment_result: ExperimentResult,
    trace_evaluators: list[TraceEvaluatorFunction],
    *,
    wait: TraceWaitConfig | None = None,
    max_concurrency: int = 10,
) -> TraceEvalResult:
    """Evaluate traces for each experiment item.

    Parameters
    ----------
    experiment_result : ExperimentResult
        Result returned by Langfuse ``run_experiment``.
    trace_evaluators : list[TraceEvaluatorFunction]
        Trace-level evaluators to apply to each trace. Evaluators can return
        ``Evaluation``/``list[Evaluation]`` directly or as awaitables.
    wait : TraceWaitConfig | None, optional, default=None
        Configuration for waiting until traces are fully populated.
    max_concurrency : int, optional, default=10
        Maximum number of trace evaluations to run in parallel.

    Returns
    -------
    TraceEvalResult
        Container with trace evaluation outputs and error metadata.

    Examples
    --------
    >>> from aieng.agent_evals.evaluation import run_experiment, run_trace_evaluations
    >>> from langfuse.experiment import Evaluation
    >>> def task(*, input, **kwargs):
    ...     return {"answer": input["question"]}
    >>> def exact_match(*, output, expected_output, **kwargs):
    ...     is_match = output["answer"] == expected_output["answer"]
    ...     return Evaluation(name="exact_match", value=is_match)
    >>> def trace_turn_count(*, trace, item_result, **kwargs):
    ...     return Evaluation(name="turn_count", value=len(trace.observations or []))
    >>> experiment_result = run_experiment(
    ...     "qa_dataset",
    ...     name="baseline",
    ...     task=task,
    ...     evaluators=[exact_match],
    ... )
    >>> trace_result = run_trace_evaluations(experiment_result, [trace_turn_count])
    >>> isinstance(trace_result.evaluations_by_trace_id, dict)
    True
    """
    return run_coroutine_sync(
        _run_trace_evaluations_async,
        experiment_result=experiment_result,
        trace_evaluators=trace_evaluators,
        wait=wait,
        max_concurrency=max_concurrency,
    )


async def _run_trace_evaluations_async(
    experiment_result: ExperimentResult,
    trace_evaluators: list[TraceEvaluatorFunction],
    *,
    wait: TraceWaitConfig | None = None,
    max_concurrency: int = 10,
) -> TraceEvalResult:
    """Run trace evaluations asynchronously with bounded concurrency."""
    result = TraceEvalResult()
    wait_config = wait or TraceWaitConfig()

    item_results = [item_result for item_result in experiment_result.item_results if item_result.trace_id]
    if not item_results or not trace_evaluators:
        logger.info("No trace evaluations to run; skipping trace evaluation pass.")
        return result

    client_manager = AsyncClientManager.get_instance()
    langfuse_client = client_manager.langfuse_client
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _evaluate_item(
        item_result: ExperimentItemResult,
    ) -> tuple[str, list[Evaluation], TraceEvalStatus, str | None]:
        trace_id = item_result.trace_id or ""
        evaluations, status, error_message = await _evaluate_trace(
            langfuse_client=langfuse_client,
            item_result=item_result,
            trace_evaluators=trace_evaluators,
            wait=wait_config,
        )
        return trace_id, evaluations, status, error_message

    trace_eval_coroutines = [
        rate_limited(
            _fn=functools.partial(_evaluate_item, item_result),
            semaphore=semaphore,
        )
        for item_result in item_results
    ]
    task_results = await gather_with_progress(trace_eval_coroutines, description="Evaluating traces")

    for trace_id, evaluations, status, error_message in task_results:
        if status == TraceEvalStatus.SKIPPED:
            result.skipped_trace_ids.append(trace_id)
            result.errors_by_trace_id[trace_id] = error_message or "Trace not ready for evaluation"
            logger.info("Trace %s evaluation skipped: %s", trace_id, error_message)
            continue
        if status == TraceEvalStatus.FAILED:
            result.failed_trace_ids.append(trace_id)
            result.errors_by_trace_id[trace_id] = error_message or "Trace evaluation failed"
            logger.warning("Trace %s evaluation failed: %s", trace_id, error_message)
            continue
        result.evaluations_by_trace_id[trace_id] = evaluations

    flush_traces()
    return result


def extract_trace_metrics(
    trace: TraceWithFullDetails,
    *,
    tool_call_predicate: TraceObservationPredicate | None = None,
    turn_predicate: TraceObservationPredicate | None = None,
) -> TraceMetrics:
    """Extract common metrics from a Langfuse trace.

    This helper provides a consistent, best-effort way to compute
    trace-derived statistics such as tool call counts and turns.
    Heuristics are intentionally conservative and can be overridden
    with custom predicates when instrumentation differs.

    Parameters
    ----------
    trace : TraceWithFullDetails
        Trace object returned by the Langfuse API.
    tool_call_predicate : TraceObservationPredicate | None, optional
        Predicate that returns True for observations that represent tool calls.
        Defaults to a heuristic based on observation type, name, and metadata.
    turn_predicate : TraceObservationPredicate | None, optional
        Predicate that returns True for observations that represent assistant
        turns or model generations. Defaults to a heuristic based on
        observation type and metadata.

    Returns
    -------
    TraceMetrics
        A dataclass containing extracted metrics.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> from aieng.agent_evals.evaluation import extract_trace_metrics
    >>> trace = SimpleNamespace(observations=[], latency=0.12, total_cost=None)
    >>> metrics = extract_trace_metrics(trace)
    >>> metrics.observation_count
    0
    """
    observations = trace.observations or []
    tool_predicate = tool_call_predicate or _default_tool_call_predicate
    turn_predicate_fn = turn_predicate or _default_turn_predicate

    tool_call_count = sum(1 for obs in observations if tool_predicate(obs))
    turn_count = sum(1 for obs in observations if turn_predicate_fn(obs))
    total_input_tokens = _sum_token_usage(observations, token_type="input")
    total_output_tokens = _sum_token_usage(observations, token_type="output")
    total_cost = _extract_total_cost(trace, observations)

    # Latency is provided directly by Langfuse; we keep it as-is to avoid
    # computing potentially misleading fallbacks from partial observations.
    latency_sec = trace.latency

    return TraceMetrics(
        tool_call_count=tool_call_count,
        turn_count=turn_count,
        observation_count=len(observations),
        latency_sec=latency_sec,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost=total_cost,
    )


async def _evaluate_trace(
    langfuse_client: Langfuse,
    item_result: ExperimentItemResult,
    trace_evaluators: list[TraceEvaluatorFunction],
    wait: TraceWaitConfig,
) -> tuple[list[Evaluation], TraceEvalStatus, str | None]:
    """Fetch and evaluate a single trace.

    Returns
    -------
    tuple[list[Evaluation], TraceEvalStatus, str | None]
        Evaluations, status string, and optional error detail.
        Status is one of "ok", "skipped", or "failed".
    """
    trace_id = item_result.trace_id
    if not trace_id:
        return [], TraceEvalStatus.SKIPPED, "Missing `trace_id` on experiment item result."

    try:
        trace, ready = await _fetch_trace_with_wait(langfuse_client, trace_id, wait)
    except Exception as exc:
        return [], TraceEvalStatus.FAILED, f"Trace fetch failed: {exc}"

    if trace is None or not ready:
        return [], TraceEvalStatus.SKIPPED, "Trace did not become ready within wait window."

    evaluations: list[Evaluation] = []
    for evaluator in trace_evaluators:
        try:
            raw_result = evaluator(trace=trace, item_result=item_result)
            evaluations.extend(await _normalize_evaluations(raw_result))
        except Exception as exc:
            evaluator_name = _get_evaluator_name(evaluator)
            return [], TraceEvalStatus.FAILED, f"Trace evaluator '{evaluator_name}' failed: {exc}"

    # Persist scores so they appear alongside traces in the Langfuse UI.
    _upload_trace_scores(langfuse_client, trace_id, evaluations)

    return evaluations, TraceEvalStatus.OK, None


async def _fetch_trace_with_wait(
    langfuse_client: Langfuse, trace_id: str, wait: TraceWaitConfig
) -> tuple[TraceWithFullDetails | None, bool]:
    """Fetch a trace with retry/backoff until it is ready or timeout expires."""
    last_trace: TraceWithFullDetails | None = None

    retrying = AsyncRetrying(
        stop=stop_after_delay(wait.max_wait_sec),
        wait=wait_exponential(
            multiplier=wait.backoff_multiplier,
            min=wait.initial_delay_sec,
            max=wait.max_delay_sec,
        ),
        retry=retry_if_exception(_is_retryable_trace_fetch_error),
        reraise=False,
    )

    try:
        async for attempt in retrying:
            with attempt:
                trace = await langfuse_client.async_api.trace.get(trace_id)
                last_trace = trace

                if _trace_ready(trace):
                    return trace, True

                raise _TraceNotReadyError("Trace input/output not ready.")
    except RetryError:
        pass

    return last_trace, bool(last_trace and _trace_ready(last_trace))


class _TraceNotReadyError(Exception):
    """Internal signal used to retry until trace readiness criteria are met."""


def _is_retryable_trace_fetch_error(exc: BaseException) -> bool:
    """Return True if the exception indicates a retryable trace fetch error."""
    if isinstance(exc, (_TraceNotReadyError, NotFoundError, httpx.TransportError)):
        return True

    if isinstance(exc, ApiError):
        status = exc.status_code
        return status in (408, 429) or (status is not None and status >= 500)

    return False


def _trace_ready(trace: TraceWithFullDetails) -> bool:
    """Check whether a trace has the required fields to evaluate."""
    # This is a heuristic. The input and output fields being populated is a strong
    # signal that the trace is complete, however, it not every field of the trace
    # might be fully populated depending on instrumentation.
    return trace.input is not None and trace.output is not None


def _default_tool_call_predicate(observation: ObservationsView) -> bool:
    """Best-effort heuristic for identifying tool call observations."""
    obs_type = (observation.type or "").lower()
    name = (observation.name or "").lower()
    metadata = observation.metadata

    if "tool" in obs_type or "tool" in name:
        return True

    # Some instrumentations store tool metadata for function calls.
    if isinstance(metadata, dict):
        for key in ("tool_name", "tool", "function", "function_name"):
            if key in metadata:
                return True

    return False


def _default_turn_predicate(observation: ObservationsView) -> bool:
    """Best-effort heuristic for identifying assistant turns."""
    obs_type = (observation.type or "").lower()
    name = (observation.name or "").lower()
    metadata = observation.metadata

    if "generation" in obs_type:
        return True

    if "assistant" in name or "response" in name:
        return True

    if isinstance(metadata, dict):
        role = str(metadata.get("role", "")).lower()
        if role == "assistant":
            return True

    return False


def _sum_token_usage(observations: list[ObservationsView], *, token_type: str) -> int:
    """Aggregate token usage for a specific type across observations."""
    total = 0
    usage_keys = _usage_keys_for_token_type(token_type)

    for observation in observations:
        usage_details = observation.usage_details
        for key in usage_keys:
            value = usage_details.get(key)
            if value is not None:
                total += value
                break  # Only count the first matching key per observation.

    return total


def _extract_total_cost(trace: TraceWithFullDetails, observations: list[ObservationsView]) -> float | None:
    """Extract total trace cost, preferring trace-level totals when available."""
    trace_total_cost = trace.total_cost
    if trace_total_cost is not None:
        return trace_total_cost

    total_cost = 0.0
    saw_cost = False

    for observation in observations:
        cost_details = observation.cost_details
        for key in ("total", "total_cost", "totalCost"):
            value = cost_details.get(key)
            if value is not None:
                total_cost += value
                saw_cost = True
                break  # Only count the first matching key per observation.

    if saw_cost:
        return total_cost
    return None


def _usage_keys_for_token_type(token_type: str) -> tuple[str, ...]:
    """Return common Langfuse/provider keys for a token usage type."""
    if token_type == "input":
        return ("input", "input_tokens", "inputTokens", "prompt_tokens", "promptTokens")
    if token_type == "output":
        return ("output", "output_tokens", "outputTokens", "completion_tokens", "completionTokens")
    return (token_type,)


def _get_evaluator_name(evaluator: Any) -> str:
    """Best-effort evaluator name for error messages."""
    target = evaluator
    if isinstance(target, functools.partial):
        target = target.func

    name = getattr(target, "__name__", None)
    if isinstance(name, str) and name:
        return name

    return target.__class__.__name__


async def _normalize_evaluations(
    result: Evaluation | list[Evaluation] | Awaitable[Evaluation | list[Evaluation]],
) -> list[Evaluation]:
    """Normalize evaluator outputs (including awaitables) to Evaluation objects."""
    resolved_result: Any = result
    if inspect.isawaitable(resolved_result):
        resolved_result = await resolved_result

    if isinstance(resolved_result, Evaluation):
        return [resolved_result]

    # Accept dict outputs to mirror Langfuse evaluator return conventions.
    if isinstance(resolved_result, dict):
        return [Evaluation(**resolved_result)]

    if isinstance(resolved_result, list):
        normalized: list[Evaluation] = []
        for item in resolved_result:
            if isinstance(item, Evaluation):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(Evaluation(**item))
        return normalized

    return []


def _upload_trace_scores(langfuse_client: Langfuse, trace_id: str, evaluations: list[Evaluation]) -> None:
    """Persist trace evaluations to Langfuse as scores."""
    for evaluation in evaluations:
        # Skip missing values to avoid creating empty or invalid scores.
        if evaluation.value is None:
            continue

        score_name = evaluation.name or "<unknown>"
        score_data_type = evaluation.data_type
        score_value = evaluation.value

        if isinstance(score_value, str):
            categorical_data_type: Literal["CATEGORICAL"] | None = None
            if score_data_type == "CATEGORICAL":
                categorical_data_type = "CATEGORICAL"

            langfuse_client.create_score(
                name=score_name,
                value=score_value,
                trace_id=trace_id,  # Link score to trace via trace_id
                comment=evaluation.comment,
                metadata=evaluation.metadata,
                data_type=categorical_data_type,
                config_id=evaluation.config_id,
            )
            continue

        numeric_data_type: Literal["NUMERIC", "BOOLEAN"] | None = None
        if score_data_type == "NUMERIC":
            numeric_data_type = "NUMERIC"
        elif score_data_type == "BOOLEAN" or isinstance(score_value, bool):
            numeric_data_type = "BOOLEAN"

        langfuse_client.create_score(
            name=score_name,
            value=float(score_value),
            trace_id=trace_id,  # Link score to trace via trace_id
            comment=evaluation.comment,
            metadata=evaluation.metadata,
            data_type=numeric_data_type,
            config_id=evaluation.config_id,
        )


__all__ = ["run_trace_evaluations", "extract_trace_metrics"]

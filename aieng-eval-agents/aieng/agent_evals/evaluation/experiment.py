"""Langfuse experiment wrapper for the evaluation harness."""

from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation.trace import run_trace_evaluations
from aieng.agent_evals.evaluation.types import (
    CompositeEvaluatorFunction,
    EvaluationResult,
    EvaluatorFunction,
    ExperimentResult,
    RunEvaluatorFunction,
    TaskFunction,
    TraceEvaluatorFunction,
    TraceWaitConfig,
)


def run_experiment(
    dataset_name: str,
    *,
    name: str,
    task: TaskFunction,
    evaluators: list[EvaluatorFunction],
    composite_evaluator: CompositeEvaluatorFunction | None = None,
    run_evaluators: list[RunEvaluatorFunction] | None = None,
    description: str | None = None,
    run_name: str | None = None,
    max_concurrency: int = 10,
    metadata: dict[str, Any] | None = None,
) -> ExperimentResult:
    """Run evaluators over a Langfuse dataset as an experiment.

    This is a thin convenience layer around Langfuse's ``dataset.run_experiment``
    that includes fetching the dataset by name and reusing the shared client.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to evaluate against.
    name : str
        Human-readable name for the experiment run.
    task : TaskFunction
        Function that executes the agent for a single dataset item.
    evaluators : list[EvaluatorFunction]
        Item-level evaluators that grade each output.
    composite_evaluator : CompositeEvaluatorFunction | None, optional, default=None
        Receives the same inputs as item-level evaluators
        ``(input, output, expected_output, metadata)`` plus the list of evaluations
        from item-level evaluators. Useful for weighted averages, pass/fail decisions
        based on multiple criteria, or custom scoring logic combining multiple metrics.
    run_evaluators : list[RunEvaluatorFunction] | None, optional, default=None
        Run-level evaluators that compute aggregate metrics.
    description : str | None, optional, default=None
        Description of the experiment for the Langfuse UI.
    run_name : str | None, optional, default=None
        Explicit dataset run name override.
    max_concurrency : int, optional, default=10
        Maximum number of concurrent task executions.
    metadata : dict[str, Any] | None, optional, default=None
        Metadata attached to the dataset run and traces.

    Returns
    -------
    ExperimentResult
        The Langfuse experiment result with item and run evaluations.

    Examples
    --------
    >>> from aieng.agent_evals.evaluation import run_experiment
    >>> from langfuse.experiment import Evaluation
    >>> def task(*, input, **kwargs):
    ...     return {"answer": input["question"]}
    >>> def exact_match(*, output, expected_output, **kwargs):
    ...     is_match = output["answer"] == expected_output["answer"]
    ...     return Evaluation(name="exact_match", value=is_match)
    >>> result = run_experiment(
    ...     "qa_dataset",
    ...     name="baseline",
    ...     task=task,
    ...     evaluators=[exact_match],
    ... )
    >>> isinstance(result.item_results, list)
    True
    """
    # The client manager keeps a shared Langfuse client so we avoid re-auth
    # and let users decide when to close it.
    client_manager = AsyncClientManager.get_instance()
    langfuse_client = client_manager.langfuse_client

    # Fetch the dataset by name
    dataset = langfuse_client.get_dataset(dataset_name)

    return dataset.run_experiment(
        name=name,
        run_name=run_name,
        description=description,
        task=task,
        evaluators=evaluators,
        composite_evaluator=composite_evaluator,
        run_evaluators=run_evaluators or [],
        max_concurrency=max_concurrency,
        metadata=metadata,
    )


def run_experiment_with_trace_evals(
    dataset_name: str,
    *,
    name: str,
    task: TaskFunction,
    evaluators: list[EvaluatorFunction],
    trace_evaluators: list[TraceEvaluatorFunction],
    composite_evaluator: CompositeEvaluatorFunction | None = None,
    run_evaluators: list[RunEvaluatorFunction] | None = None,
    description: str | None = None,
    run_name: str | None = None,
    max_concurrency: int = 10,
    metadata: dict[str, Any] | None = None,
    trace_wait: TraceWaitConfig | None = None,
    trace_max_concurrency: int = 10,
) -> EvaluationResult:
    """Run an experiment and then evaluate traces in a second pass.

    This helper encapsulates a two-pass workflow that first runs an experiment
    to produce outputs and trace IDs, then waits for trace ingestion and runs
    trace evaluators in a second pass.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to evaluate against.
    name : str
        Human-readable name for the experiment run.
    task : TaskFunction
        Function that executes the agent for a dataset item.
    evaluators : list[EvaluatorFunction]
        Item-level evaluators that grade each output.
    trace_evaluators : list[TraceEvaluatorFunction]
        Trace-level evaluators that grade tool use, groundedness, and
        other trace-derived metrics.
    composite_evaluator : CompositeEvaluatorFunction | None, optional, default=None
        Receives the same inputs as item-level evaluators
        ``(input, output, expected_output, metadata)`` plus the list of evaluations
        from item-level evaluators. Useful for weighted averages, pass/fail decisions
        based on multiple criteria, or custom scoring logic combining multiple metrics.
    run_evaluators : list[RunEvaluatorFunction] | None, optional, default=None
        Run-level evaluators that compute aggregate metrics.
    description : str | None, optional, default=None
        Description of the experiment for the Langfuse UI.
    run_name : str | None, optional, default=None
        Explicit dataset run name override.
    max_concurrency : int, optional, default=10
        Maximum number of concurrent task executions.
    metadata : dict[str, Any] | None, optional, default=None
        Metadata attached to the dataset run and traces.
    trace_wait : TraceWaitConfig | None, optional, default=None
        Trace polling configuration for the second pass.
    trace_max_concurrency : int, optional, default=10
        Maximum number of concurrent trace evaluations.

    Returns
    -------
    EvaluationResult
        A container with the experiment result and trace evaluations.

    Examples
    --------
    >>> from aieng.agent_evals.evaluation import run_experiment_with_trace_evals
    >>> from langfuse.experiment import Evaluation
    >>> def task(*, input, **kwargs):
    ...     return {"answer": input["question"]}
    >>> def exact_match(*, output, expected_output, **kwargs):
    ...     is_match = output["answer"] == expected_output["answer"]
    ...     return Evaluation(name="exact_match", value=is_match)
    >>> def trace_latency(*, trace, item_result, **kwargs):
    ...     return Evaluation(name="latency_sec", value=trace.latency or 0.0)
    >>> result = run_experiment_with_trace_evals(
    ...     "qa_dataset",
    ...     name="baseline-with-trace-evals",
    ...     task=task,
    ...     evaluators=[exact_match],
    ...     trace_evaluators=[trace_latency],
    ... )
    >>> result.trace_evaluations is not None
    True
    """
    # Pass 1 produces outputs and trace IDs; trace data itself may still be ingesting.
    experiment_result = run_experiment(
        dataset_name,
        name=name,
        task=task,
        evaluators=evaluators,
        composite_evaluator=composite_evaluator,
        run_evaluators=run_evaluators,
        description=description,
        run_name=run_name,
        max_concurrency=max_concurrency,
        metadata=metadata,
    )

    # Pass 2 waits for trace completeness before grading tool use and groundedness.
    trace_result = run_trace_evaluations(
        experiment_result,
        trace_evaluators,
        wait=trace_wait,
        max_concurrency=trace_max_concurrency,
    )

    return EvaluationResult(experiment=experiment_result, trace_evaluations=trace_result)


__all__ = ["run_experiment", "run_experiment_with_trace_evals"]

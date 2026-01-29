"""Metrics aggregation module for knowledge agent evaluation.

This module provides utilities for computing aggregate metrics across
evaluation runs, including by-dimension breakdowns and complexity analysis.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from .judges import DeepSearchQAResult, JudgeResult
from .planner import ResearchPlan, StepExecution


class EvaluationMetrics(BaseModel):
    """Aggregate metrics from an evaluation run.

    Attributes
    ----------
    total_examples : int
        Total number of examples evaluated.
    overall_f1 : float
        Average F1 score across all examples.
    avg_precision : float
        Average precision across all examples.
    avg_recall : float
        Average recall across all examples.
    outcome_distribution : dict[str, int]
        Count of each outcome category.
    by_complexity : dict[str, dict]
        Metrics broken down by question complexity.
    by_answer_type : dict[str, dict]
        Metrics broken down by answer type.
    avg_comprehensiveness : float
        Average comprehensiveness score (1-5).
    avg_causal_reasoning : float
        Average causal chain score (1-5).
    avg_exhaustiveness : float
        Average exhaustiveness score (1-5).
    avg_source_quality : float
        Average source quality score (1-5).
    avg_plan_quality : float
        Average plan quality score (1-5).
    tool_usage_distribution : dict[str, int]
        Count of how often each tool was used.
    avg_steps_per_question : float
        Average number of research steps per question.
    avg_duration_ms : float
        Average execution duration in milliseconds.
    """

    total_examples: int = 0
    overall_f1: float = 0.0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    outcome_distribution: dict[str, int] = Field(default_factory=dict)
    by_complexity: dict[str, dict] = Field(default_factory=dict)
    by_answer_type: dict[str, dict] = Field(default_factory=dict)

    # Judge dimension averages
    avg_comprehensiveness: float = 0.0
    avg_causal_reasoning: float = 0.0
    avg_exhaustiveness: float = 0.0
    avg_source_quality: float = 0.0
    avg_plan_quality: float = 0.0

    # Execution metrics
    tool_usage_distribution: dict[str, int] = Field(default_factory=dict)
    avg_steps_per_question: float = 0.0
    avg_duration_ms: float = 0.0


class EnhancedEvaluationResult(BaseModel):
    """Enhanced evaluation result with full judge outputs.

    Attributes
    ----------
    example_id : int
        The example ID that was evaluated.
    problem : str
        The original problem/question.
    ground_truth : str
        The expected answer.
    prediction : str
        The model's generated answer.
    answer_type : str
        Type of answer (Single Answer, Set Answer).
    complexity : str
        Assessed complexity (simple, moderate, complex).
    deepsearchqa_result : DeepSearchQAResult
        The DeepSearchQA evaluation result with P/R/F1.
    comprehensiveness : JudgeResult | None
        Comprehensiveness evaluation result.
    causal_reasoning : JudgeResult | None
        Causal chain evaluation result.
    exhaustiveness : JudgeResult | None
        Exhaustiveness evaluation result (for list questions).
    source_quality : JudgeResult | None
        Source quality evaluation result.
    plan_quality : JudgeResult | None
        Plan quality evaluation result.
    plan : ResearchPlan | None
        The research plan used.
    execution_trace : list[StepExecution]
        Execution trace of plan steps.
    tools_used : dict[str, int]
        Count of each tool used.
    total_duration_ms : int
        Total execution duration in milliseconds.
    human_validated : bool
        Whether this result has been human-validated.
    human_notes : str
        Notes from human validation.
    """

    example_id: int
    problem: str
    ground_truth: str
    prediction: str
    answer_type: str = "Single Answer"
    complexity: str = "moderate"

    # DeepSearchQA metrics
    deepsearchqa_result: DeepSearchQAResult | None = None

    # Judge dimension scores
    comprehensiveness: JudgeResult | None = None
    causal_reasoning: JudgeResult | None = None
    exhaustiveness: JudgeResult | None = None
    source_quality: JudgeResult | None = None
    plan_quality: JudgeResult | None = None

    # Execution details
    plan: ResearchPlan | None = None
    execution_trace: list[StepExecution] = Field(default_factory=list)
    tools_used: dict[str, int] = Field(default_factory=dict)
    total_duration_ms: int = 0

    # Human validation
    human_validated: bool = False
    human_notes: str = ""


class MetricsAggregator:
    """Computes aggregate metrics across evaluation runs.

    This class takes a list of evaluation results and computes
    various aggregate metrics for analysis and reporting.

    Examples
    --------
    >>> aggregator = MetricsAggregator()
    >>> metrics = aggregator.compute_metrics(results)
    >>> print(f"Overall F1: {metrics.overall_f1:.2f}")
    """

    def compute_metrics(
        self,
        results: list[EnhancedEvaluationResult],
    ) -> EvaluationMetrics:
        """Compute aggregate metrics from evaluation results.

        Parameters
        ----------
        results : list[EnhancedEvaluationResult]
            List of evaluation results to aggregate.

        Returns
        -------
        EvaluationMetrics
            Aggregated metrics.
        """
        if not results:
            return EvaluationMetrics()

        n = len(results)

        # DeepSearchQA metrics
        f1_scores = []
        precisions = []
        recalls = []
        outcomes = []

        for r in results:
            if r.deepsearchqa_result:
                f1_scores.append(r.deepsearchqa_result.f1_score)
                precisions.append(r.deepsearchqa_result.precision)
                recalls.append(r.deepsearchqa_result.recall)
                outcomes.append(r.deepsearchqa_result.outcome)

        # Judge dimension scores
        comprehensiveness_scores = [r.comprehensiveness.score for r in results if r.comprehensiveness]
        causal_scores = [r.causal_reasoning.score for r in results if r.causal_reasoning]
        exhaustiveness_scores = [r.exhaustiveness.score for r in results if r.exhaustiveness]
        source_scores = [r.source_quality.score for r in results if r.source_quality]
        plan_scores = [r.plan_quality.score for r in results if r.plan_quality]

        # Execution metrics
        tool_counts: Counter[str] = Counter()
        total_steps = 0
        total_duration = 0

        for r in results:
            for tool, count in r.tools_used.items():
                tool_counts[tool] += count
            total_steps += len(r.execution_trace)
            total_duration += r.total_duration_ms

        # By complexity breakdown
        by_complexity = self._compute_by_dimension(results, "complexity")

        # By answer type breakdown
        by_answer_type = self._compute_by_dimension(results, "answer_type")

        return EvaluationMetrics(
            total_examples=n,
            overall_f1=sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            avg_precision=sum(precisions) / len(precisions) if precisions else 0.0,
            avg_recall=sum(recalls) / len(recalls) if recalls else 0.0,
            outcome_distribution=dict(Counter(outcomes)),
            by_complexity=by_complexity,
            by_answer_type=by_answer_type,
            avg_comprehensiveness=sum(comprehensiveness_scores) / len(comprehensiveness_scores)
            if comprehensiveness_scores
            else 0.0,
            avg_causal_reasoning=sum(causal_scores) / len(causal_scores) if causal_scores else 0.0,
            avg_exhaustiveness=sum(exhaustiveness_scores) / len(exhaustiveness_scores)
            if exhaustiveness_scores
            else 0.0,
            avg_source_quality=sum(source_scores) / len(source_scores) if source_scores else 0.0,
            avg_plan_quality=sum(plan_scores) / len(plan_scores) if plan_scores else 0.0,
            tool_usage_distribution=dict(tool_counts),
            avg_steps_per_question=total_steps / n if n > 0 else 0.0,
            avg_duration_ms=total_duration / n if n > 0 else 0.0,
        )

    def _compute_by_dimension(
        self,
        results: list[EnhancedEvaluationResult],
        dimension: str,
    ) -> dict[str, dict]:
        """Compute metrics grouped by a dimension.

        Parameters
        ----------
        results : list[EnhancedEvaluationResult]
            Evaluation results.
        dimension : str
            Attribute to group by ('complexity' or 'answer_type').

        Returns
        -------
        dict[str, dict]
            Metrics for each group.
        """
        groups: dict[str, list[EnhancedEvaluationResult]] = {}

        for r in results:
            key = getattr(r, dimension, "unknown")
            if key not in groups:
                groups[key] = []
            groups[key].append(r)

        by_dimension = {}
        for key, group_results in groups.items():
            f1_scores = [r.deepsearchqa_result.f1_score for r in group_results if r.deepsearchqa_result]
            outcomes = [r.deepsearchqa_result.outcome for r in group_results if r.deepsearchqa_result]

            by_dimension[key] = {
                "count": len(group_results),
                "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
                "outcome_distribution": dict(Counter(outcomes)),
            }

        return by_dimension

    def compute_comparison(
        self,
        baseline_results: list[EnhancedEvaluationResult],
        experiment_results: list[EnhancedEvaluationResult],
    ) -> dict:
        """Compare metrics between two evaluation runs.

        Parameters
        ----------
        baseline_results : list[EnhancedEvaluationResult]
            Baseline evaluation results.
        experiment_results : list[EnhancedEvaluationResult]
            Experiment evaluation results.

        Returns
        -------
        dict
            Comparison metrics showing deltas.
        """
        baseline_metrics = self.compute_metrics(baseline_results)
        experiment_metrics = self.compute_metrics(experiment_results)

        return {
            "baseline": baseline_metrics.model_dump(),
            "experiment": experiment_metrics.model_dump(),
            "delta": {
                "overall_f1": experiment_metrics.overall_f1 - baseline_metrics.overall_f1,
                "avg_precision": experiment_metrics.avg_precision - baseline_metrics.avg_precision,
                "avg_recall": experiment_metrics.avg_recall - baseline_metrics.avg_recall,
                "avg_comprehensiveness": experiment_metrics.avg_comprehensiveness
                - baseline_metrics.avg_comprehensiveness,
                "avg_causal_reasoning": experiment_metrics.avg_causal_reasoning - baseline_metrics.avg_causal_reasoning,
                "avg_exhaustiveness": experiment_metrics.avg_exhaustiveness - baseline_metrics.avg_exhaustiveness,
                "avg_source_quality": experiment_metrics.avg_source_quality - baseline_metrics.avg_source_quality,
                "avg_plan_quality": experiment_metrics.avg_plan_quality - baseline_metrics.avg_plan_quality,
            },
        }

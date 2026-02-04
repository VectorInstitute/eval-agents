"""Tests for the metrics aggregation module."""

import pytest
from aieng.agent_evals.knowledge_qa.judges import DeepSearchQAResult, JudgeResult
from aieng.agent_evals.knowledge_qa.metrics import (
    EnhancedEvaluationResult,
    EvaluationMetrics,
    MetricsAggregator,
)


class TestEvaluationMetrics:
    """Tests for the EvaluationMetrics model."""

    def test_metrics_creation(self):
        """Test creating evaluation metrics."""
        metrics = EvaluationMetrics(
            total_examples=10,
            overall_f1=0.85,
            avg_precision=0.9,
            avg_recall=0.8,
            outcome_distribution={"fully_correct": 7, "partially_correct": 3},
        )
        assert metrics.total_examples == 10
        assert metrics.overall_f1 == 0.85
        assert metrics.avg_precision == 0.9

    def test_metrics_defaults(self):
        """Test default values."""
        metrics = EvaluationMetrics()
        assert metrics.total_examples == 0
        assert metrics.overall_f1 == 0.0
        assert metrics.outcome_distribution == {}


class TestEnhancedEvaluationResult:
    """Tests for the EnhancedEvaluationResult model."""

    def test_result_creation(self):
        """Test creating an enhanced evaluation result."""
        result = EnhancedEvaluationResult(
            example_id=1,
            problem="What is GDP?",
            ground_truth="Gross Domestic Product",
            prediction="GDP stands for Gross Domestic Product",
            answer_type="Single Answer",
            complexity="simple",
        )
        assert result.example_id == 1
        assert result.problem == "What is GDP?"
        assert result.answer_type == "Single Answer"

    def test_result_with_judges(self):
        """Test result with judge evaluations."""
        deepsearchqa = DeepSearchQAResult(
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            outcome="fully_correct",
        )
        comprehensiveness = JudgeResult(
            dimension="comprehensiveness",
            score=4.5,
            explanation="Good coverage",
        )

        result = EnhancedEvaluationResult(
            example_id=1,
            problem="Test",
            ground_truth="Answer",
            prediction="Answer",
            deepsearchqa_result=deepsearchqa,
            comprehensiveness=comprehensiveness,
        )
        assert result.deepsearchqa_result.f1_score == 1.0
        assert result.comprehensiveness.score == 4.5


class TestMetricsAggregator:
    """Tests for the MetricsAggregator class."""

    def test_empty_results(self):
        """Test with empty results list."""
        aggregator = MetricsAggregator()
        metrics = aggregator.compute_metrics([])
        assert metrics.total_examples == 0
        assert metrics.overall_f1 == 0.0

    def test_single_result(self):
        """Test with single result."""
        result = EnhancedEvaluationResult(
            example_id=1,
            problem="Test",
            ground_truth="Answer",
            prediction="Answer",
            deepsearchqa_result=DeepSearchQAResult(
                precision=1.0,
                recall=1.0,
                f1_score=1.0,
                outcome="fully_correct",
            ),
        )

        aggregator = MetricsAggregator()
        metrics = aggregator.compute_metrics([result])

        assert metrics.total_examples == 1
        assert metrics.overall_f1 == 1.0
        assert metrics.avg_precision == 1.0
        assert metrics.avg_recall == 1.0
        assert metrics.outcome_distribution == {"fully_correct": 1}

    def test_multiple_results(self):
        """Test with multiple results."""
        results = [
            EnhancedEvaluationResult(
                example_id=1,
                problem="Test 1",
                ground_truth="A",
                prediction="A",
                complexity="simple",
                deepsearchqa_result=DeepSearchQAResult(
                    precision=1.0, recall=1.0, f1_score=1.0, outcome="fully_correct"
                ),
            ),
            EnhancedEvaluationResult(
                example_id=2,
                problem="Test 2",
                ground_truth="B, C",
                prediction="B",
                complexity="moderate",
                deepsearchqa_result=DeepSearchQAResult(
                    precision=1.0, recall=0.5, f1_score=0.67, outcome="partially_correct"
                ),
            ),
            EnhancedEvaluationResult(
                example_id=3,
                problem="Test 3",
                ground_truth="D",
                prediction="X",
                complexity="complex",
                deepsearchqa_result=DeepSearchQAResult(
                    precision=0.0, recall=0.0, f1_score=0.0, outcome="fully_incorrect"
                ),
            ),
        ]

        aggregator = MetricsAggregator()
        metrics = aggregator.compute_metrics(results)

        assert metrics.total_examples == 3
        assert metrics.overall_f1 == pytest.approx((1.0 + 0.67 + 0.0) / 3, rel=0.01)
        assert metrics.outcome_distribution["fully_correct"] == 1
        assert metrics.outcome_distribution["partially_correct"] == 1
        assert metrics.outcome_distribution["fully_incorrect"] == 1

    def test_by_complexity_breakdown(self):
        """Test metrics breakdown by complexity."""
        results = [
            EnhancedEvaluationResult(
                example_id=1,
                problem="Simple question",
                ground_truth="A",
                prediction="A",
                complexity="simple",
                deepsearchqa_result=DeepSearchQAResult(
                    precision=1.0, recall=1.0, f1_score=1.0, outcome="fully_correct"
                ),
            ),
            EnhancedEvaluationResult(
                example_id=2,
                problem="Another simple",
                ground_truth="B",
                prediction="B",
                complexity="simple",
                deepsearchqa_result=DeepSearchQAResult(
                    precision=1.0, recall=1.0, f1_score=1.0, outcome="fully_correct"
                ),
            ),
            EnhancedEvaluationResult(
                example_id=3,
                problem="Complex question",
                ground_truth="C",
                prediction="X",
                complexity="complex",
                deepsearchqa_result=DeepSearchQAResult(
                    precision=0.0, recall=0.0, f1_score=0.0, outcome="fully_incorrect"
                ),
            ),
        ]

        aggregator = MetricsAggregator()
        metrics = aggregator.compute_metrics(results)

        assert "simple" in metrics.by_complexity
        assert "complex" in metrics.by_complexity
        assert metrics.by_complexity["simple"]["count"] == 2
        assert metrics.by_complexity["simple"]["avg_f1"] == 1.0
        assert metrics.by_complexity["complex"]["count"] == 1
        assert metrics.by_complexity["complex"]["avg_f1"] == 0.0

    def test_judge_dimension_averages(self):
        """Test averaging of judge dimension scores."""
        results = [
            EnhancedEvaluationResult(
                example_id=1,
                problem="Test",
                ground_truth="A",
                prediction="A",
                comprehensiveness=JudgeResult(dimension="comprehensiveness", score=4.0),
                causal_reasoning=JudgeResult(dimension="causal_chain", score=5.0),
            ),
            EnhancedEvaluationResult(
                example_id=2,
                problem="Test",
                ground_truth="B",
                prediction="B",
                comprehensiveness=JudgeResult(dimension="comprehensiveness", score=3.0),
                # No causal_reasoning for this one
            ),
        ]

        aggregator = MetricsAggregator()
        metrics = aggregator.compute_metrics(results)

        assert metrics.avg_comprehensiveness == 3.5  # (4.0 + 3.0) / 2
        assert metrics.avg_causal_reasoning == 5.0  # Only one result has it

    def test_tool_usage_aggregation(self):
        """Test aggregation of tool usage."""
        results = [
            EnhancedEvaluationResult(
                example_id=1,
                problem="Test",
                ground_truth="A",
                prediction="A",
                tools_used={"web_search": 3, "finance_knowledge": 1},
            ),
            EnhancedEvaluationResult(
                example_id=2,
                problem="Test",
                ground_truth="B",
                prediction="B",
                tools_used={"web_search": 2, "synthesis": 1},
            ),
        ]

        aggregator = MetricsAggregator()
        metrics = aggregator.compute_metrics(results)

        assert metrics.tool_usage_distribution["web_search"] == 5
        assert metrics.tool_usage_distribution["finance_knowledge"] == 1
        assert metrics.tool_usage_distribution["synthesis"] == 1

    def test_comparison(self):
        """Test comparing two evaluation runs."""
        baseline_results = [
            EnhancedEvaluationResult(
                example_id=1,
                problem="Test",
                ground_truth="A",
                prediction="A",
                deepsearchqa_result=DeepSearchQAResult(
                    precision=0.8, recall=0.8, f1_score=0.8, outcome="partially_correct"
                ),
            ),
        ]

        experiment_results = [
            EnhancedEvaluationResult(
                example_id=1,
                problem="Test",
                ground_truth="A",
                prediction="A",
                deepsearchqa_result=DeepSearchQAResult(
                    precision=1.0, recall=1.0, f1_score=1.0, outcome="fully_correct"
                ),
            ),
        ]

        aggregator = MetricsAggregator()
        comparison = aggregator.compute_comparison(baseline_results, experiment_results)

        assert comparison["delta"]["overall_f1"] == pytest.approx(0.2)
        assert comparison["delta"]["avg_precision"] == pytest.approx(0.2)
        assert comparison["delta"]["avg_recall"] == pytest.approx(0.2)

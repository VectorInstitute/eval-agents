"""Tests for DeepSearchQA evaluation utilities."""

from aieng.agent_evals.knowledge_qa.evaluation import EvaluationResult


class TestEvaluationResult:
    """Tests for the EvaluationResult model."""

    def test_result_creation(self):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            example_id=1,
            problem="Test question",
            ground_truth="Expected answer",
            prediction="Model answer",
            search_queries=["query1"],
            sources_used=2,
        )
        assert result.example_id == 1
        assert result.ground_truth == "Expected answer"
        assert result.prediction == "Model answer"
        assert len(result.search_queries) == 1
        assert result.sources_used == 2
        assert result.is_correct is None

    def test_result_defaults(self):
        """Test default values for evaluation result."""
        result = EvaluationResult(
            example_id=0,
            problem="Q",
            ground_truth="A",
            prediction="B",
        )
        assert result.search_queries == []
        assert result.sources_used == 0
        assert result.is_correct is None
        assert result.evaluation_notes == ""

    def test_result_with_correctness(self):
        """Test evaluation result with correctness flag."""
        result = EvaluationResult(
            example_id=2,
            problem="What is 2+2?",
            ground_truth="4",
            prediction="4",
            is_correct=True,
        )
        assert result.is_correct is True

    def test_result_with_notes(self):
        """Test evaluation result with evaluation notes."""
        result = EvaluationResult(
            example_id=3,
            problem="Complex question",
            ground_truth="Complex answer",
            prediction="Model's answer",
            evaluation_notes="Partial match detected",
        )
        assert result.evaluation_notes == "Partial match detected"

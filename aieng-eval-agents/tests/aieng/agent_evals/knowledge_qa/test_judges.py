"""Tests for the LLM-as-judge evaluators."""

from unittest.mock import MagicMock, patch

import pytest
from aieng.agent_evals.knowledge_qa.judges import (
    DeepSearchQAJudge,
    DeepSearchQAResult,
    EvaluationOutcome,
    JudgeResult,
    _calculate_metrics_from_grader,
)
from pydantic import SecretStr


class TestJudgeResult:
    """Tests for the JudgeResult model."""

    def test_judge_result_creation(self):
        """Test creating a judge result."""
        result = JudgeResult(
            dimension="comprehensiveness",
            score=4.5,
            explanation="Good coverage of all aspects",
            evidence=["Covered point A", "Covered point B"],
        )
        assert result.dimension == "comprehensiveness"
        assert result.score == 4.5
        assert result.explanation == "Good coverage of all aspects"
        assert len(result.evidence) == 2

    def test_judge_result_defaults(self):
        """Test default values for judge result."""
        result = JudgeResult(dimension="test", score=3.0)
        assert result.explanation == ""
        assert result.evidence == []


class TestDeepSearchQAResult:
    """Tests for the DeepSearchQAResult model."""

    def test_result_creation(self):
        """Test creating a DeepSearchQA result."""
        result = DeepSearchQAResult(
            precision=0.8,
            recall=0.9,
            f1_score=0.847,
            outcome=EvaluationOutcome.CORRECT_WITH_EXTRANEOUS,
            correctness_details={"item1": True, "item2": True, "item3": False},
            extraneous_items=["extra1"],
            explanation="Found 2 out of 3 items with 1 extraneous",
        )
        assert result.precision == 0.8
        assert result.recall == 0.9
        assert result.f1_score == 0.847
        assert result.outcome == EvaluationOutcome.CORRECT_WITH_EXTRANEOUS
        assert result.correctness_details["item1"] is True
        assert len(result.extraneous_items) == 1

    def test_result_defaults(self):
        """Test default values."""
        result = DeepSearchQAResult()
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1_score == 0.0
        assert result.outcome == EvaluationOutcome.FULLY_INCORRECT
        assert result.correctness_details == {}
        assert result.extraneous_items == []


class TestCalculateMetrics:
    """Tests for the _calculate_metrics_from_grader function."""

    def test_calculate_metrics_perfect_match(self):
        """Test metrics calculation with perfect match (fully_correct)."""
        # Simulate grader output for perfect match
        grader_result = {
            "Explanation": "All items found correctly",
            "Correctness Details": {"A": True, "B": True, "C": True},
            "Excessive Answers": [],
        }

        result = _calculate_metrics_from_grader(grader_result)

        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1_score == 1.0
        assert result.outcome == EvaluationOutcome.FULLY_CORRECT

    def test_calculate_metrics_with_extraneous(self):
        """Test metrics calculation with extraneous items (correct_with_extraneous)."""
        # Simulate grader output: all ground truth found + extra item
        grader_result = {
            "Explanation": "All items found but includes extra",
            "Correctness Details": {"A": True, "B": True, "C": True},
            "Excessive Answers": ["D"],
        }

        result = _calculate_metrics_from_grader(grader_result)

        assert result.precision == 0.75  # 3/(3+1)
        assert result.recall == 1.0  # 3/3
        assert result.outcome == EvaluationOutcome.CORRECT_WITH_EXTRANEOUS
        assert "D" in result.extraneous_items

    def test_calculate_metrics_with_missed(self):
        """Test metrics calculation with missed items (partially_correct)."""
        # Simulate grader output: only 2 of 3 ground truth found
        grader_result = {
            "Explanation": "Found A and B but missed C",
            "Correctness Details": {"A": True, "B": True, "C": False},
            "Excessive Answers": [],
        }

        result = _calculate_metrics_from_grader(grader_result)

        assert result.precision == 1.0  # 2/2 (no extraneous)
        assert result.recall == pytest.approx(2 / 3)  # 2/3
        assert result.outcome == EvaluationOutcome.PARTIALLY_CORRECT
        assert result.correctness_details["C"] is False

    def test_calculate_metrics_fully_incorrect(self):
        """Test metrics calculation with no matches (fully_incorrect)."""
        # Simulate grader output: no correct items
        grader_result = {
            "Explanation": "No correct items found",
            "Correctness Details": {"A": False, "B": False},
            "Excessive Answers": ["X", "Y"],
        }

        result = _calculate_metrics_from_grader(grader_result)

        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1_score == 0.0
        assert result.outcome == EvaluationOutcome.FULLY_INCORRECT

    def test_calculate_metrics_empty_ground_truth(self):
        """Test metrics calculation with empty ground truth."""
        # Edge case: no ground truth items
        grader_result = {
            "Explanation": "No ground truth to check",
            "Correctness Details": {},
            "Excessive Answers": [],
        }

        result = _calculate_metrics_from_grader(grader_result)

        assert result.recall == 1.0  # Edge case handling
        assert result.outcome == EvaluationOutcome.FULLY_CORRECT


@pytest.fixture
def mock_configs():
    """Fixture to mock the Configs class."""
    mock_config = MagicMock()
    mock_config.openai_api_key = SecretStr("test-api-key")
    mock_config.default_evaluator_model = "gemini-2.5-pro"
    mock_config.default_evaluator_temperature = 0.0
    return mock_config


@patch("aieng.agent_evals.knowledge_qa.judges.Configs")
class TestDeepSearchQAJudge:
    """Tests for the DeepSearchQAJudge."""

    @patch("aieng.agent_evals.knowledge_qa.judges.evaluate_deepsearchqa_async")
    def test_evaluate_full(self, mock_evaluate_async, mock_configs_cls, mock_configs):
        """Test full evaluation flow."""
        # Configure the Configs mock to return our mock_configs
        mock_configs_cls.return_value = mock_configs

        # Mock the async evaluator to return a result
        mock_result = DeepSearchQAResult(
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            outcome=EvaluationOutcome.FULLY_CORRECT,
            correctness_details={"USA": True, "UK": True},
            extraneous_items=[],
            explanation="Both USA and UK found correctly",
        )
        mock_evaluate_async.return_value = mock_result

        judge = DeepSearchQAJudge()
        result = judge.evaluate(
            question="Name two G7 countries",
            answer="USA and UK",
            ground_truth="USA, UK",
            answer_type="Set Answer",
        )

        assert result.dimension == "deepsearchqa"
        assert result.score == 5.0  # F1=1.0 -> score=5
        assert "Precision: 1.00" in result.evidence[0]
        assert "Recall: 1.00" in result.evidence[1]
        assert EvaluationOutcome.FULLY_CORRECT.value in result.evidence[3]

    @patch("aieng.agent_evals.knowledge_qa.judges.evaluate_deepsearchqa_async")
    def test_evaluate_partial_match(self, mock_evaluate_async, mock_configs_cls, mock_configs):
        """Test evaluation with partial match."""
        # Configure the Configs mock
        mock_configs_cls.return_value = mock_configs

        # Mock partial match result
        mock_result = DeepSearchQAResult(
            precision=1.0,
            recall=2 / 3,
            f1_score=0.8,
            outcome=EvaluationOutcome.PARTIALLY_CORRECT,
            correctness_details={"George Washington": True, "John Adams": True, "Thomas Jefferson": False},
            extraneous_items=[],
            explanation="Found Washington and Adams, missed Jefferson",
        )
        mock_evaluate_async.return_value = mock_result

        judge = DeepSearchQAJudge()
        result = judge.evaluate(
            question="Name the first three US presidents",
            answer="George Washington and John Adams",
            ground_truth="George Washington, John Adams, Thomas Jefferson",
            answer_type="Set Answer",
        )

        assert result.dimension == "deepsearchqa"
        assert EvaluationOutcome.PARTIALLY_CORRECT.value in result.evidence[3]
        assert result.score < 5.0  # Not perfect

    @patch("aieng.agent_evals.knowledge_qa.judges.evaluate_deepsearchqa_async")
    def test_evaluate_with_details(self, mock_evaluate_async, mock_configs_cls, mock_configs):
        """Test evaluation with detailed results."""
        # Configure the Configs mock
        mock_configs_cls.return_value = mock_configs

        mock_result = DeepSearchQAResult(
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            outcome=EvaluationOutcome.FULLY_CORRECT,
            correctness_details={"Paris": True},
            extraneous_items=[],
            explanation="Correct answer found",
        )
        mock_evaluate_async.return_value = mock_result

        judge = DeepSearchQAJudge()
        judge_result, detailed_result = judge.evaluate_with_details(
            question="What is the capital of France?",
            answer="Paris",
            ground_truth="Paris",
            answer_type="Single Answer",
        )

        assert isinstance(judge_result, JudgeResult)
        assert isinstance(detailed_result, DeepSearchQAResult)
        assert detailed_result.f1_score == 1.0
        assert detailed_result.outcome == EvaluationOutcome.FULLY_CORRECT

    @patch("aieng.agent_evals.knowledge_qa.judges.evaluate_deepsearchqa_async")
    def test_evaluate_single_answer_type(self, mock_evaluate_async, mock_configs_cls, mock_configs):
        """Test evaluation with Single Answer type."""
        # Configure the Configs mock
        mock_configs_cls.return_value = mock_configs

        mock_result = DeepSearchQAResult(
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            outcome=EvaluationOutcome.FULLY_CORRECT,
            correctness_details={"42": True},
            extraneous_items=[],
            explanation="Answer is semantically equivalent",
        )
        mock_evaluate_async.return_value = mock_result

        judge = DeepSearchQAJudge()
        result = judge.evaluate(
            question="What is the answer to life, the universe, and everything?",
            answer="The answer is 42.",
            ground_truth="42",
            answer_type="Single Answer",
        )

        assert result.dimension == "deepsearchqa"
        assert result.score == 5.0  # Perfect match

    @pytest.mark.asyncio
    async def test_evaluate_async(self, mock_configs_cls, mock_configs):
        """Test async evaluation."""
        # Configure the Configs mock
        mock_configs_cls.return_value = mock_configs

        with patch("aieng.agent_evals.knowledge_qa.judges.evaluate_deepsearchqa_async") as mock_evaluate:
            mock_result = DeepSearchQAResult(
                precision=1.0,
                recall=1.0,
                f1_score=1.0,
                outcome=EvaluationOutcome.FULLY_CORRECT,
                correctness_details={"test": True},
                extraneous_items=[],
                explanation="Test passed",
            )
            mock_evaluate.return_value = mock_result

            judge = DeepSearchQAJudge()
            result = await judge.evaluate_async(
                question="Test question?",
                answer="Test answer",
                ground_truth="Test answer",
                answer_type="Single Answer",
            )

            assert result.dimension == "deepsearchqa"
            assert result.score == 5.0


@pytest.mark.integration_test
class TestJudgesIntegration:
    """Integration tests for judges.

    These tests require valid API keys (OPENAI_API_KEY or GOOGLE_API_KEY).
    """

    def test_deepsearchqa_judge_real(self):
        """Test DeepSearchQA judge with real LLM."""
        judge = DeepSearchQAJudge()
        result = judge.evaluate(
            question="Name the first three US presidents",
            answer="George Washington, John Adams, Thomas Jefferson",
            ground_truth="George Washington, John Adams, Thomas Jefferson",
            answer_type="Set Answer",
        )

        assert result.dimension == "deepsearchqa"
        assert result.score >= 4.0  # Should be high for correct answer

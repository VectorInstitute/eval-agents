"""Tests for the DeepSearchQA grader."""

import pytest
from aieng.agent_evals.knowledge_qa.deepsearchqa_grader import (
    DeepSearchQAResult,
    EvaluationOutcome,
    _calculate_metrics_from_grader,
)


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

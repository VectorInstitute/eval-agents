"""Tests for the LLM-as-judge evaluators."""

import json
from unittest.mock import MagicMock, patch

import pytest
from aieng.agent_evals.knowledge_agent.judges import (
    BaseJudge,
    DeepSearchQAJudge,
    DeepSearchQAResult,
    JudgeResult,
)


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
            outcome="correct_with_extraneous",
            correctness_details={"item1": True, "item2": True, "item3": False},
            extraneous_items=["extra1"],
            explanation="Found 2 out of 3 items with 1 extraneous",
        )
        assert result.precision == 0.8
        assert result.recall == 0.9
        assert result.f1_score == 0.847
        assert result.outcome == "correct_with_extraneous"
        assert result.correctness_details["item1"] is True
        assert len(result.extraneous_items) == 1

    def test_result_defaults(self):
        """Test default values."""
        result = DeepSearchQAResult()
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1_score == 0.0
        assert result.outcome == "fully_incorrect"
        assert result.correctness_details == {}
        assert result.extraneous_items == []


def _create_mock_config():
    """Create a mock Configs for testing."""
    mock_config = MagicMock()
    mock_config.default_evaluator_model = "gemini-2.5-flash"
    return mock_config


class TestBaseJudge:
    """Tests for the BaseJudge class."""

    @patch("aieng.agent_evals.knowledge_agent.judges.Configs")
    @patch("aieng.agent_evals.knowledge_agent.judges.genai.Client")
    def test_initialization_without_config(self, mock_client_class, mock_config_class):
        """Test judge initialization without config."""
        mock_config_class.return_value = _create_mock_config()
        judge = BaseJudge()
        assert judge._model == "gemini-2.5-flash"

    @patch("aieng.agent_evals.knowledge_agent.judges.Configs")
    @patch("aieng.agent_evals.knowledge_agent.judges.genai.Client")
    def test_initialization_with_model(self, mock_client_class, mock_config_class):
        """Test judge initialization with explicit model."""
        mock_config_class.return_value = _create_mock_config()
        judge = BaseJudge(model="gemini-2.5-pro")
        assert judge._model == "gemini-2.5-pro"

    @patch("aieng.agent_evals.knowledge_agent.judges.genai.Client")
    def test_initialization_with_config(self, mock_client_class):
        """Test judge initialization with config."""
        mock_config = MagicMock()
        mock_config.default_evaluator_model = "gemini-2.5-pro"

        judge = BaseJudge(config=mock_config)
        assert judge._model == "gemini-2.5-pro"


class TestDeepSearchQAJudge:
    """Tests for the DeepSearchQAJudge."""

    def test_calculate_metrics_perfect_match(self):
        """Test metrics calculation with perfect match (fully_correct)."""
        with (
            patch("aieng.agent_evals.knowledge_agent.judges.genai.Client"),
            patch("aieng.agent_evals.knowledge_agent.judges.Configs") as mock_config,
        ):
            mock_config.return_value = _create_mock_config()
            judge = DeepSearchQAJudge()

        # Simulate grader output for perfect match
        grader_result = {
            "Explanation": "All items found correctly",
            "Correctness Details": {"A": True, "B": True, "C": True},
            "Excessive Answers": [],
        }

        result = judge._calculate_metrics_from_grader(grader_result)

        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1_score == 1.0
        assert result.outcome == "fully_correct"

    def test_calculate_metrics_with_extraneous(self):
        """Test metrics calculation with extraneous items (correct_with_extraneous)."""
        with (
            patch("aieng.agent_evals.knowledge_agent.judges.genai.Client"),
            patch("aieng.agent_evals.knowledge_agent.judges.Configs") as mock_config,
        ):
            mock_config.return_value = _create_mock_config()
            judge = DeepSearchQAJudge()

        # Simulate grader output: all ground truth found + extra item
        grader_result = {
            "Explanation": "All items found but includes extra",
            "Correctness Details": {"A": True, "B": True, "C": True},
            "Excessive Answers": ["D"],
        }

        result = judge._calculate_metrics_from_grader(grader_result)

        assert result.precision == 0.75  # 3/(3+1)
        assert result.recall == 1.0  # 3/3
        assert result.outcome == "correct_with_extraneous"
        assert "D" in result.extraneous_items

    def test_calculate_metrics_with_missed(self):
        """Test metrics calculation with missed items (partially_correct)."""
        with (
            patch("aieng.agent_evals.knowledge_agent.judges.genai.Client"),
            patch("aieng.agent_evals.knowledge_agent.judges.Configs") as mock_config,
        ):
            mock_config.return_value = _create_mock_config()
            judge = DeepSearchQAJudge()

        # Simulate grader output: only 2 of 3 ground truth found
        grader_result = {
            "Explanation": "Found A and B but missed C",
            "Correctness Details": {"A": True, "B": True, "C": False},
            "Excessive Answers": [],
        }

        result = judge._calculate_metrics_from_grader(grader_result)

        assert result.precision == 1.0  # 2/2 (no extraneous)
        assert result.recall == pytest.approx(2 / 3)  # 2/3
        assert result.outcome == "partially_correct"
        assert result.correctness_details["C"] is False

    def test_calculate_metrics_fully_incorrect(self):
        """Test metrics calculation with no matches (fully_incorrect)."""
        with (
            patch("aieng.agent_evals.knowledge_agent.judges.genai.Client"),
            patch("aieng.agent_evals.knowledge_agent.judges.Configs") as mock_config,
        ):
            mock_config.return_value = _create_mock_config()
            judge = DeepSearchQAJudge()

        # Simulate grader output: no correct items
        grader_result = {
            "Explanation": "No correct items found",
            "Correctness Details": {"A": False, "B": False},
            "Excessive Answers": ["X", "Y"],
        }

        result = judge._calculate_metrics_from_grader(grader_result)

        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1_score == 0.0
        assert result.outcome == "fully_incorrect"

    def test_calculate_metrics_empty_ground_truth(self):
        """Test metrics calculation with empty ground truth."""
        with (
            patch("aieng.agent_evals.knowledge_agent.judges.genai.Client"),
            patch("aieng.agent_evals.knowledge_agent.judges.Configs") as mock_config,
        ):
            mock_config.return_value = _create_mock_config()
            judge = DeepSearchQAJudge()

        # Edge case: no ground truth items
        grader_result = {
            "Explanation": "No ground truth to check",
            "Correctness Details": {},
            "Excessive Answers": [],
        }

        result = judge._calculate_metrics_from_grader(grader_result)

        assert result.recall == 1.0  # Edge case handling
        assert result.outcome == "fully_correct"

    @patch("aieng.agent_evals.knowledge_agent.judges.Configs")
    @patch("aieng.agent_evals.knowledge_agent.judges.genai.Client")
    def test_call_grader(self, mock_client_class, mock_config_class):
        """Test that _call_grader properly calls the LLM."""
        mock_config_class.return_value = _create_mock_config()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "Answer Correctness": {
                    "Explanation": "Belgium and France found correctly",
                    "Correctness Details": {"Belgium": True, "France": True},
                    "Excessive Answers": ["Italy"],
                }
            }
        )
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        judge = DeepSearchQAJudge()
        result = judge._call_grader(
            prompt="What countries border Luxembourg?",
            response="Belgium, France, and Italy",
            answer="Belgium, France",
            prompt_type="Set Answer",
        )

        assert result["Correctness Details"]["Belgium"] is True
        assert result["Correctness Details"]["France"] is True
        assert "Italy" in result["Excessive Answers"]
        mock_client.models.generate_content.assert_called_once()

    @patch("aieng.agent_evals.knowledge_agent.judges.Configs")
    @patch("aieng.agent_evals.knowledge_agent.judges.genai.Client")
    def test_call_grader_handles_error(self, mock_client_class, mock_config_class):
        """Test that _call_grader handles errors gracefully."""
        mock_config_class.return_value = _create_mock_config()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        judge = DeepSearchQAJudge()
        result = judge._call_grader(
            prompt="Test",
            response="Test",
            answer="Test",
            prompt_type="Single Answer",
        )

        assert "Grader error" in result["Explanation"]
        assert result["Correctness Details"] == {}

    @patch("aieng.agent_evals.knowledge_agent.judges.Configs")
    @patch("aieng.agent_evals.knowledge_agent.judges.genai.Client")
    def test_evaluate_full(self, mock_client_class, mock_config_class):
        """Test full evaluation flow with official grader prompt."""
        mock_config_class.return_value = _create_mock_config()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "Answer Correctness": {
                    "Explanation": "Both USA and UK found correctly",
                    "Correctness Details": {"USA": True, "UK": True},
                    "Excessive Answers": [],
                }
            }
        )
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

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
        assert "fully_correct" in result.evidence[3]

    @patch("aieng.agent_evals.knowledge_agent.judges.Configs")
    @patch("aieng.agent_evals.knowledge_agent.judges.genai.Client")
    def test_evaluate_partial_match(self, mock_client_class, mock_config_class):
        """Test evaluation with partial match."""
        mock_config_class.return_value = _create_mock_config()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "Answer Correctness": {
                    "Explanation": "Found Washington and Adams, missed Jefferson",
                    "Correctness Details": {
                        "George Washington": True,
                        "John Adams": True,
                        "Thomas Jefferson": False,
                    },
                    "Excessive Answers": [],
                }
            }
        )
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        judge = DeepSearchQAJudge()
        result = judge.evaluate(
            question="Name the first three US presidents",
            answer="George Washington and John Adams",
            ground_truth="George Washington, John Adams, Thomas Jefferson",
            answer_type="Set Answer",
        )

        assert result.dimension == "deepsearchqa"
        # F1 = 2 * 1.0 * (2/3) / (1.0 + 2/3) = 0.8
        assert "partially_correct" in result.evidence[3]
        assert result.score < 5.0  # Not perfect

    @patch("aieng.agent_evals.knowledge_agent.judges.Configs")
    @patch("aieng.agent_evals.knowledge_agent.judges.genai.Client")
    def test_evaluate_with_details(self, mock_client_class, mock_config_class):
        """Test evaluation with detailed results."""
        mock_config_class.return_value = _create_mock_config()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "Answer Correctness": {
                    "Explanation": "Correct answer found",
                    "Correctness Details": {"Paris": True},
                    "Excessive Answers": [],
                }
            }
        )
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

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
        assert detailed_result.outcome == "fully_correct"

    @patch("aieng.agent_evals.knowledge_agent.judges.Configs")
    @patch("aieng.agent_evals.knowledge_agent.judges.genai.Client")
    def test_evaluate_single_answer_type(self, mock_client_class, mock_config_class):
        """Test evaluation with Single Answer type."""
        mock_config_class.return_value = _create_mock_config()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "Answer Correctness": {
                    "Explanation": "Answer is semantically equivalent",
                    "Correctness Details": {"42": True},
                    "Excessive Answers": [],
                }
            }
        )
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        judge = DeepSearchQAJudge()
        result = judge.evaluate(
            question="What is the answer to life, the universe, and everything?",
            answer="The answer is 42.",
            ground_truth="42",
            answer_type="Single Answer",
        )

        assert result.dimension == "deepsearchqa"
        assert result.score == 5.0  # Perfect match


@pytest.mark.integration_test
class TestJudgesIntegration:
    """Integration tests for judges.

    These tests require a valid GOOGLE_API_KEY environment variable.
    """

    def test_deepsearchqa_judge_real(self):
        """Test DeepSearchQA judge with real LLM."""
        judge = DeepSearchQAJudge()
        result = judge.evaluate(
            question="Name the first three US presidents",
            answer="George Washington, John Adams, Thomas Jefferson",
            ground_truth="George Washington, John Adams, Thomas Jefferson",
            answer_type="List",
        )

        assert result.dimension == "deepsearchqa"
        assert result.score >= 4.0  # Should be high for correct answer

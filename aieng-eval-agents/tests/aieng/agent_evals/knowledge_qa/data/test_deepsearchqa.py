"""Tests for DeepSearchQA dataset loading and management."""

from unittest.mock import patch

import pandas as pd
import pytest
from aieng.agent_evals.knowledge_qa.data import DeepSearchQADataset, DSQAExample


class TestDSQAExample:
    """Tests for the DSQAExample model."""

    def test_example_creation(self):
        """Test creating an example."""
        example = DSQAExample(
            example_id=0,
            problem="What is the capital of France?",
            problem_category="Geography",
            answer="Paris",
            answer_type="Single Answer",
        )
        assert example.example_id == 0
        assert example.problem == "What is the capital of France?"
        assert example.problem_category == "Geography"
        assert example.answer == "Paris"
        assert example.answer_type == "Single Answer"


class TestDeepSearchQADataset:
    """Tests for the DeepSearchQADataset class."""

    @pytest.fixture
    def mock_csv_data(self):
        """Create mock CSV data."""
        return {
            "example_id": [0, 1, 2],
            "problem": ["Q1", "Q2", "Q3"],
            "problem_category": ["Cat A", "Cat B", "Cat A"],
            "answer": ["A1", "A2", "A3"],
            "answer_type": ["Single Answer", "List", "Single Answer"],
        }

    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.kagglehub.dataset_download")
    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.pd.read_csv")
    def test_dataset_loading(self, mock_read_csv, mock_download, mock_csv_data):
        """Test loading the dataset."""
        mock_download.return_value = "/fake/path"
        mock_read_csv.return_value = pd.DataFrame(mock_csv_data)

        with patch("pathlib.Path.exists", return_value=True):
            dataset = DeepSearchQADataset()
            examples = dataset.examples

        assert len(examples) == 3
        assert examples[0].problem == "Q1"

    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.kagglehub.dataset_download")
    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.pd.read_csv")
    def test_dataset_length(self, mock_read_csv, mock_download, mock_csv_data):
        """Test getting dataset length."""
        mock_download.return_value = "/fake/path"
        mock_read_csv.return_value = pd.DataFrame(mock_csv_data)

        with patch("pathlib.Path.exists", return_value=True):
            dataset = DeepSearchQADataset()
            assert len(dataset) == 3

    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.kagglehub.dataset_download")
    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.pd.read_csv")
    def test_dataset_indexing(self, mock_read_csv, mock_download, mock_csv_data):
        """Test indexing into the dataset."""
        mock_download.return_value = "/fake/path"
        mock_read_csv.return_value = pd.DataFrame(mock_csv_data)

        with patch("pathlib.Path.exists", return_value=True):
            dataset = DeepSearchQADataset()
            example = dataset[1]

        assert example.example_id == 1
        assert example.problem == "Q2"

    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.kagglehub.dataset_download")
    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.pd.read_csv")
    def test_get_by_category(self, mock_read_csv, mock_download, mock_csv_data):
        """Test filtering by category."""
        mock_download.return_value = "/fake/path"
        mock_read_csv.return_value = pd.DataFrame(mock_csv_data)

        with patch("pathlib.Path.exists", return_value=True):
            dataset = DeepSearchQADataset()
            cat_a_examples = dataset.get_by_category("Cat A")

        assert len(cat_a_examples) == 2
        assert all(ex.problem_category == "Cat A" for ex in cat_a_examples)

    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.kagglehub.dataset_download")
    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.pd.read_csv")
    def test_get_by_id(self, mock_read_csv, mock_download, mock_csv_data):
        """Test getting a single example by ID."""
        mock_download.return_value = "/fake/path"
        mock_read_csv.return_value = pd.DataFrame(mock_csv_data)

        with patch("pathlib.Path.exists", return_value=True):
            dataset = DeepSearchQADataset()
            example = dataset.get_by_id(1)

        assert example is not None
        assert example.example_id == 1
        assert example.problem == "Q2"

        # Test non-existent ID
        assert dataset.get_by_id(999) is None

    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.kagglehub.dataset_download")
    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.pd.read_csv")
    def test_get_by_ids(self, mock_read_csv, mock_download, mock_csv_data):
        """Test getting multiple examples by IDs."""
        mock_download.return_value = "/fake/path"
        mock_read_csv.return_value = pd.DataFrame(mock_csv_data)

        with patch("pathlib.Path.exists", return_value=True):
            dataset = DeepSearchQADataset()
            examples = dataset.get_by_ids([0, 2])

        assert len(examples) == 2
        assert examples[0].example_id == 0
        assert examples[1].example_id == 2

        # Test with missing IDs (should skip them)
        examples = dataset.get_by_ids([0, 999, 1])
        assert len(examples) == 2

    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.kagglehub.dataset_download")
    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.pd.read_csv")
    def test_get_categories(self, mock_read_csv, mock_download, mock_csv_data):
        """Test getting unique categories."""
        mock_download.return_value = "/fake/path"
        mock_read_csv.return_value = pd.DataFrame(mock_csv_data)

        with patch("pathlib.Path.exists", return_value=True):
            dataset = DeepSearchQADataset()
            categories = dataset.get_categories()

        assert "Cat A" in categories
        assert "Cat B" in categories

    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.kagglehub.dataset_download")
    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.pd.read_csv")
    def test_sample(self, mock_read_csv, mock_download, mock_csv_data):
        """Test random sampling."""
        mock_download.return_value = "/fake/path"
        mock_read_csv.return_value = pd.DataFrame(mock_csv_data)

        with patch("pathlib.Path.exists", return_value=True):
            dataset = DeepSearchQADataset()
            sample = dataset.sample(n=2, random_state=42)

        assert len(sample) == 2
        assert all(isinstance(ex, DSQAExample) for ex in sample)

    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.kagglehub.dataset_download")
    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.pd.read_csv")
    def test_dataframe_property(self, mock_read_csv, mock_download, mock_csv_data):
        """Test accessing the raw dataframe."""
        mock_download.return_value = "/fake/path"
        mock_read_csv.return_value = pd.DataFrame(mock_csv_data)

        with patch("pathlib.Path.exists", return_value=True):
            dataset = DeepSearchQADataset()
            df = dataset.dataframe

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "problem" in df.columns

    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.kagglehub.dataset_download")
    @patch("aieng.agent_evals.knowledge_qa.data.deepsearchqa.pd.read_csv")
    def test_filter_missing_answers(self, mock_read_csv, mock_download):
        """Test that rows with missing answers are filtered out."""
        mock_download.return_value = "/fake/path"
        data_with_na = pd.DataFrame(
            {
                "example_id": [0, 1, 2],
                "problem": ["Q1", "Q2", "Q3"],
                "problem_category": ["Cat A", "Cat B", "Cat A"],
                "answer": ["A1", None, "A3"],  # One missing answer
                "answer_type": ["Single Answer", "List", "Single Answer"],
            }
        )
        mock_read_csv.return_value = data_with_na

        with patch("pathlib.Path.exists", return_value=True):
            dataset = DeepSearchQADataset()
            examples = dataset.examples

        # Should only have 2 examples after filtering
        assert len(examples) == 2
        assert examples[0].example_id == 0
        assert examples[1].example_id == 2


@pytest.mark.integration_test
class TestDeepSearchQADatasetIntegration:
    """Integration tests for DeepSearchQADataset.

    These tests download the actual dataset from Kaggle.
    """

    def test_load_real_dataset(self):
        """Test loading the real dataset."""
        dataset = DeepSearchQADataset()

        # Dataset may have fewer than 900 examples after filtering NaN answers
        assert len(dataset) > 800  # Should have most examples
        assert dataset[0].example_id == 0
        assert len(dataset.get_categories()) > 0

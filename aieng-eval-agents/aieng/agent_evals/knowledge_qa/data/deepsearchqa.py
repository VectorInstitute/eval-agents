"""DeepSearchQA dataset loader.

This module provides classes for loading and accessing the DeepSearchQA
benchmark dataset from Kaggle.
"""

import logging
from pathlib import Path

import kagglehub
import pandas as pd
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class DSQAExample(BaseModel):
    """A single example from the DeepSearchQA dataset."""

    example_id: int = Field(description="Unique identifier for the example.")
    problem: str = Field(description="The research question/problem to solve.")
    problem_category: str = Field(description="Category of the problem (e.g., 'Politics & Government').")
    answer: str = Field(description="The ground truth answer.")
    answer_type: str = Field(description="Type of answer (e.g., 'Single Answer', 'List').")


class DeepSearchQADataset:
    """Loader and manager for the DeepSearchQA dataset.

    This class handles downloading, loading, and accessing examples from
    the DeepSearchQA benchmark dataset.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory to cache the dataset. If not provided, uses kagglehub default.

    Examples
    --------
    >>> dataset = DeepSearchQADataset()
    >>> print(f"Total examples: {len(dataset)}")
    >>> example = dataset[0]
    >>> print(example.problem)
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Initialize the dataset loader.

        Parameters
        ----------
        cache_dir : str or Path, optional
            Directory to cache the dataset.
        """
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._df: pd.DataFrame | None = None
        self._examples: list[DSQAExample] | None = None

    def _download_dataset(self) -> Path:
        """Download the dataset using kagglehub.

        Returns
        -------
        Path
            Path to the downloaded dataset directory.
        """
        logger.info("Downloading DeepSearchQA dataset...")
        path = kagglehub.dataset_download("deepmind/deepsearchqa")
        return Path(path)

    def _load_data(self) -> None:
        """Load the dataset into memory."""
        if self._df is not None:
            return

        dataset_path = self._download_dataset()
        csv_path = dataset_path / "DSQA-full.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        self._df = pd.read_csv(csv_path)

        # Filter out rows with missing answers
        original_count = len(self._df)
        self._df = self._df.dropna(subset=["answer"])
        dropped_count = original_count - len(self._df)
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count} examples with missing answers")

        logger.info(f"Loaded {len(self._df)} examples from DeepSearchQA")

        # Convert to examples
        self._examples = [
            DSQAExample(
                example_id=row["example_id"],
                problem=row["problem"],
                problem_category=row["problem_category"],
                answer=str(row["answer"]),  # Ensure string type
                answer_type=row["answer_type"],
            )
            for _, row in self._df.iterrows()
        ]

    @property
    def dataframe(self) -> pd.DataFrame:
        """Get the raw pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The full dataset as a DataFrame.
        """
        self._load_data()
        assert self._df is not None
        return self._df

    @property
    def examples(self) -> list[DSQAExample]:
        """Get all examples as DSQAExample objects.

        Returns
        -------
        list[DSQAExample]
            All examples in the dataset.
        """
        self._load_data()
        assert self._examples is not None
        return self._examples

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        self._load_data()
        assert self._examples is not None
        return len(self._examples)

    def __getitem__(self, index: int) -> DSQAExample:
        """Get an example by index.

        Parameters
        ----------
        index : int
            The index of the example to retrieve.

        Returns
        -------
        DSQAExample
            The example at the given index.
        """
        self._load_data()
        assert self._examples is not None
        return self._examples[index]

    def get_by_category(self, category: str) -> list[DSQAExample]:
        """Get all examples in a specific category.

        Parameters
        ----------
        category : str
            The problem category to filter by.

        Returns
        -------
        list[DSQAExample]
            Examples matching the category.
        """
        return [ex for ex in self.examples if ex.problem_category == category]

    def get_by_id(self, example_id: int) -> DSQAExample | None:
        """Get a single example by its ID.

        Parameters
        ----------
        example_id : int
            The unique identifier of the example.

        Returns
        -------
        DSQAExample or None
            The example with the given ID, or None if not found.
        """
        for ex in self.examples:
            if ex.example_id == example_id:
                return ex
        return None

    def get_by_ids(self, example_ids: list[int]) -> list[DSQAExample]:
        """Get multiple examples by their IDs.

        Parameters
        ----------
        example_ids : list[int]
            List of example IDs to retrieve.

        Returns
        -------
        list[DSQAExample]
            Examples matching the given IDs, in the order requested.
            Missing IDs are silently skipped.
        """
        id_to_example = {ex.example_id: ex for ex in self.examples}
        return [id_to_example[eid] for eid in example_ids if eid in id_to_example]

    def get_categories(self) -> list[str]:
        """Get all unique problem categories.

        Returns
        -------
        list[str]
            List of unique category names.
        """
        return list(self.dataframe["problem_category"].unique())

    def sample(self, n: int = 10, random_state: int | None = None) -> list[DSQAExample]:
        """Get a random sample of examples.

        Parameters
        ----------
        n : int, optional
            Number of examples to sample, by default 10.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        list[DSQAExample]
            Randomly sampled examples.
        """
        sampled_df = self.dataframe.sample(n=min(n, len(self)), random_state=random_state)
        return [
            DSQAExample(
                example_id=row["example_id"],
                problem=row["problem"],
                problem_category=row["problem_category"],
                answer=row["answer"],
                answer_type=row["answer_type"],
            )
            for _, row in sampled_df.iterrows()
        ]

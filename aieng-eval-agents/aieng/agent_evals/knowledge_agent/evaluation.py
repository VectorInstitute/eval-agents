"""Evaluation utilities for DeepSearchQA benchmark.

This module provides tools for loading, running, and evaluating the
DeepSearchQA benchmark dataset.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import kagglehub
import pandas as pd
from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from .agent import KnowledgeGroundedAgent


logger = logging.getLogger(__name__)


class DSQAExample(BaseModel):
    """A single example from the DeepSearchQA dataset."""

    example_id: int = Field(description="Unique identifier for the example.")
    problem: str = Field(description="The research question/problem to solve.")
    problem_category: str = Field(description="Category of the problem (e.g., 'Politics & Government').")
    answer: str = Field(description="The ground truth answer.")
    answer_type: str = Field(description="Type of answer (e.g., 'Single Answer', 'List').")


class EvaluationResult(BaseModel):
    """Result of evaluating a single example."""

    example_id: int = Field(description="The example ID that was evaluated.")
    problem: str = Field(description="The original problem/question.")
    ground_truth: str = Field(description="The expected answer.")
    prediction: str = Field(description="The model's generated answer.")
    search_queries: list[str] = Field(default_factory=list, description="Search queries executed by the model.")
    sources_used: int = Field(default=0, description="Number of sources cited in the response.")
    is_correct: bool | None = Field(default=None, description="Whether the answer is correct (None if not evaluated).")
    evaluation_notes: str = Field(default="", description="Additional notes about the evaluation.")


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


class DeepSearchQAEvaluator:
    """Evaluator for running and scoring DeepSearchQA benchmark.

    This class manages the evaluation pipeline for testing agents on the
    DeepSearchQA benchmark.

    Parameters
    ----------
    agent : KnowledgeGroundedAgent
        The agent to evaluate.
    dataset : DeepSearchQADataset, optional
        The dataset to use. If not provided, creates a new one.

    Examples
    --------
    >>> from aieng.agent_evals.knowledge_agent import (
    ...     KnowledgeGroundedAgent,
    ...     DeepSearchQAEvaluator,
    ... )
    >>> agent = KnowledgeGroundedAgent()
    >>> evaluator = DeepSearchQAEvaluator(agent)
    >>> results = evaluator.evaluate_sample(n=5)
    """

    def __init__(
        self,
        agent: "KnowledgeGroundedAgent",
        dataset: DeepSearchQADataset | None = None,
    ) -> None:
        """Initialize the evaluator.

        Parameters
        ----------
        agent : KnowledgeGroundedAgent
            The agent to evaluate.
        dataset : DeepSearchQADataset, optional
            The dataset to use. If not provided, a new DeepSearchQADataset
            will be created, which downloads the DSQA dataset from Kaggle.
        """
        self.agent = agent
        self.dataset = dataset or DeepSearchQADataset()

    def evaluate_example(self, example: DSQAExample) -> EvaluationResult:
        """Evaluate a single example.

        Parameters
        ----------
        example : DSQAExample
            The example to evaluate.

        Returns
        -------
        EvaluationResult
            The evaluation result.
        """
        logger.info(f"Evaluating example {example.example_id}...")

        try:
            response = self.agent.answer(example.problem)
            prediction = response.text
            search_queries = response.search_queries
            sources_used = len(response.sources)
        except Exception as e:
            logger.error(f"Error evaluating example {example.example_id}: {e}")
            return EvaluationResult(
                example_id=example.example_id,
                problem=example.problem,
                ground_truth=example.answer,
                prediction=f"ERROR: {e}",
                evaluation_notes=f"Evaluation failed: {e}",
            )

        return EvaluationResult(
            example_id=example.example_id,
            problem=example.problem,
            ground_truth=example.answer,
            prediction=prediction,
            search_queries=search_queries,
            sources_used=sources_used,
        )

    async def evaluate_example_async(self, example: DSQAExample) -> EvaluationResult:
        """Async version of evaluate_example.

        Parameters
        ----------
        example : DSQAExample
            The example to evaluate.

        Returns
        -------
        EvaluationResult
            The evaluation result.
        """
        logger.info(f"Evaluating example {example.example_id} (async)...")

        try:
            response = await self.agent.answer_async(example.problem)
            prediction = response.text
            search_queries = response.search_queries
            sources_used = len(response.sources)
        except Exception as e:
            logger.error(f"Error evaluating example {example.example_id}: {e}")
            return EvaluationResult(
                example_id=example.example_id,
                problem=example.problem,
                ground_truth=example.answer,
                prediction=f"ERROR: {e}",
                evaluation_notes=f"Evaluation failed: {e}",
            )

        return EvaluationResult(
            example_id=example.example_id,
            problem=example.problem,
            ground_truth=example.answer,
            prediction=prediction,
            search_queries=search_queries,
            sources_used=sources_used,
        )

    def evaluate_sample(self, n: int = 10, random_state: int | None = None) -> list[EvaluationResult]:
        """Evaluate a random sample of examples.

        Parameters
        ----------
        n : int, optional
            Number of examples to evaluate, by default 10.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        list[EvaluationResult]
            Results for all evaluated examples.
        """
        examples = self.dataset.sample(n=n, random_state=random_state)
        return [self.evaluate_example(ex) for ex in examples]

    async def evaluate_sample_async(
        self,
        n: int = 10,
        random_state: int | None = None,
        max_concurrency: int = 3,
    ) -> list[EvaluationResult]:
        """Async evaluation of a random sample with concurrency control.

        Parameters
        ----------
        n : int, optional
            Number of examples to evaluate, by default 10.
        random_state : int, optional
            Random seed for reproducibility.
        max_concurrency : int, optional
            Maximum concurrent evaluations, by default 3.

        Returns
        -------
        list[EvaluationResult]
            Results for all evaluated examples.
        """
        examples = self.dataset.sample(n=n, random_state=random_state)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def eval_with_semaphore(ex: DSQAExample) -> EvaluationResult:
            async with semaphore:
                return await self.evaluate_example_async(ex)

        tasks = [eval_with_semaphore(ex) for ex in examples]
        return await asyncio.gather(*tasks)

    def results_to_dataframe(self, results: list[EvaluationResult]) -> pd.DataFrame:
        """Convert evaluation results to a DataFrame.

        Parameters
        ----------
        results : list[EvaluationResult]
            The evaluation results.

        Returns
        -------
        pd.DataFrame
            Results as a DataFrame.
        """
        return pd.DataFrame([r.model_dump() for r in results])

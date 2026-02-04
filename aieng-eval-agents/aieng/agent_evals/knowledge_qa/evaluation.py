"""Evaluation utilities for DeepSearchQA benchmark.

This module provides tools for running and evaluating agents on the
DeepSearchQA benchmark dataset.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, Field

from .data import DeepSearchQADataset, DSQAExample


if TYPE_CHECKING:
    from .agent import KnowledgeGroundedAgent


logger = logging.getLogger(__name__)


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
    >>> from aieng.agent_evals.knowledge_qa import (
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

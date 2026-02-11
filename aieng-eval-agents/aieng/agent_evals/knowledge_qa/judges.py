"""LLM-as-judge evaluators for knowledge agent responses.

This module provides comprehensive evaluation using LLM judges for the
DeepSearchQA benchmark. The implementation follows the official DeepSearchQA
evaluation methodology with precision, recall, and F1 metrics.

The evaluator has been refactored to use shared evaluation infrastructure
while maintaining backward compatibility with the original API.
"""

import asyncio
import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.configs import Configs
from aieng.agent_evals.evaluation.graders._utils import run_structured_parse_call
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.evaluation.types import Evaluation
from pydantic import BaseModel, Field


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class EvaluationOutcome(str, Enum):
    """Possible outcomes for DeepSearchQA evaluation.

    The four disjoint categories represent the relationship between
    the submitted answer set (S) and ground truth set (G).
    """

    FULLY_CORRECT = "fully_correct"
    CORRECT_WITH_EXTRANEOUS = "correct_with_extraneous"
    PARTIALLY_CORRECT = "partially_correct"
    FULLY_INCORRECT = "fully_incorrect"


class JudgeResult(BaseModel):
    """Result from an LLM judge evaluation."""

    dimension: str = Field(description='The evaluation dimension (e.g., "comprehensiveness", "causal_chain")')
    score: float = Field(description="Score on a 1-5 scale")
    explanation: str = Field(default="", description="Detailed explanation of the score")
    evidence: list[str] = Field(default_factory=list, description="Specific examples supporting the score")


class DeepSearchQAResult(BaseModel):
    """Result from DeepSearchQA evaluation with IR metrics.

    This follows the official DeepSearchQA evaluation methodology from:
    https://www.kaggle.com/benchmarks/google/dsqa
    """

    precision: float = Field(
        default=0.0, description="Fraction of predicted items that are correct (0-1). P = |S ∩ G| / |S|"
    )
    recall: float = Field(
        default=0.0, description="Fraction of ground truth items that were found (0-1). R = |S ∩ G| / |G|"
    )
    f1_score: float = Field(
        default=0.0, description="Harmonic mean of precision and recall (0-1). F1 = 2 * P * R / (P + R)"
    )
    outcome: EvaluationOutcome = Field(
        default=EvaluationOutcome.FULLY_INCORRECT,
        description="Evaluation outcome indicating the relationship between submitted and ground truth answer sets",
    )
    correctness_details: dict[str, bool] = Field(
        default_factory=dict, description="For each ground truth item, whether it was found in the response"
    )
    extraneous_items: list[str] = Field(
        default_factory=list, description="Items in the response that are not in the ground truth"
    )
    explanation: str = Field(default="", description="Explanation from the judge about the evaluation")

    def to_evaluations(self) -> list[Evaluation]:
        """Convert this result to Langfuse Evaluation objects.

        Returns
        -------
        list[Evaluation]
            Four evaluations: Outcome (categorical), F1, Precision, Recall (numeric).
        """
        comment_parts = [
            f"Outcome: {self.outcome}",
            f"Precision: {self.precision:.2f}",
            f"Recall: {self.recall:.2f}",
            f"F1: {self.f1_score:.2f}",
        ]

        if self.explanation:
            comment_parts.append(f"\nExplanation: {self.explanation}")

        if self.correctness_details:
            found = sum(1 for v in self.correctness_details.values() if v)
            total = len(self.correctness_details)
            comment_parts.append(f"\nCorrectness: {found}/{total} items found")

        if self.extraneous_items:
            comment_parts.append(f"\nExtraneous: {len(self.extraneous_items)} items")

        comment = "\n".join(comment_parts)

        outcome_display = {
            EvaluationOutcome.FULLY_CORRECT: "Fully Correct",
            EvaluationOutcome.CORRECT_WITH_EXTRANEOUS: "Correct with Extraneous",
            EvaluationOutcome.PARTIALLY_CORRECT: "Partially Correct",
            EvaluationOutcome.FULLY_INCORRECT: "Fully Incorrect",
        }

        return [
            Evaluation(
                name="Outcome",
                value=outcome_display.get(self.outcome, self.outcome.value),
                comment=self.explanation,
            ),
            Evaluation(name="F1", value=self.f1_score, comment=comment),
            Evaluation(name="Precision", value=self.precision, comment=comment),
            Evaluation(name="Recall", value=self.recall, comment=comment),
        ]

    @staticmethod
    def error_evaluations(error_msg: str) -> list[Evaluation]:
        """Create error evaluations when evaluation fails.

        Parameters
        ----------
        error_msg : str
            Description of the error that occurred.

        Returns
        -------
        list[Evaluation]
            Three evaluations (F1, Precision, Recall) all set to 0.0.
        """
        comment = f"Evaluation error: {error_msg}"
        return [
            Evaluation(name="F1", value=0.0, comment=comment),
            Evaluation(name="Precision", value=0.0, comment=comment),
            Evaluation(name="Recall", value=0.0, comment=comment),
        ]


class DeepSearchQAGraderResponse(BaseModel):
    """Structured response from the DeepSearchQA grader.

    This matches the official DeepSearchQA grader output format.
    """

    answer_correctness: dict[str, Any] = Field(
        alias="Answer Correctness",
        description=(
            "Dictionary containing: "
            "Explanation (str) - Explanation of the evaluation; "
            "Correctness Details (dict[str, bool]) - Per-item correctness; "
            "Excessive Answers (list[str]) - Extra items not in ground truth"
        ),
    )


# Official DeepSearchQA grader prompt from Appendix A of the paper
DEEPSEARCHQA_GRADER_PROMPT = """\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt"
arrived at the correct answer.

**Answer Correctness Task**
* **Purpose:** Assess whether the AI response provides the correct answer(s) based on
the provided "Correct Answer" and "Prompt Type".
* **Process:**
  * Identify the "Prompt Type": "{prompt_type}".
  * Refer to the "Correct Answer": "{answer}".
  * Based on the "Prompt Type", determine if the "AI Response" contains the expected
answer(s).
    * **'Single Answer'**: Check if the response provides the answer that addresses
the user's question. It does not have to match the exact wording of the provided
answer.
    * **'Set Answer'**: Check if the response includes *each* item from the provided
ground truth answers. The order might not matter unless specified otherwise. The
response might include more answers than the list. Determine the correctness *
only* based on the list first and then check if the response includes answers not
in the list.
* **Explanation:** Provide a brief explanation justifying your assessment of answer
correctness, referencing specific parts of the AI response and the correct answer.
* **Correctness Details:** Provide a dictionary, one key for each expected answer
part, and value is a boolean indicating whether each expected answer part was found.
  * For 'Set Answer', this will be a list of attributes, one for each item/part in
the "Correct Answer". Each key will be a string indicating the expected answer
part, and the value will be a boolean indicating whether that part was found in
the response.
* **Excessive Answers:** Provide a list of strings, each indicating an excessive
answer part. If the response provides answers that are **not** in the "Correct Answer
" list, add these answers as excessive answers. Return an empty list when there's no
excessive answers in the response.

**Output Format:**
Your evaluation *must* be structured as a nested JSON dictionary with the following top-
level keys: '"Answer Correctness"'. Please return NULL if any of "Prompt", "AI Response"
or "Correct Answer" is empty.
The value for '"Answer Correctness"' should be a dictionary containing '"Explanation"' (
a string), '"Correctness Details"' (a dictionary where each key is the expected correct
answer, and the value is a boolean indicating whether the response contains the correct
answer), and '"Excessive Answers"' (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and
special characters in the JSON string. Make sure to escape all special characters and
quotes in the JSON string.

**Example (Partial):**
```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also
includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and
Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
** Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>
--------------------
Rating:
"""


def _calculate_metrics_from_grader(
    grader_result: dict[str, Any],
) -> DeepSearchQAResult:
    """Calculate precision, recall, F1 from grader output.

    This follows the exact methodology from the paper:
    - Precision = |S∩G| / |S|
    - Recall = |S∩G| / |G|
    - F1 = 2*P*R / (P+R)

    Parameters
    ----------
    grader_result : dict
        Output from the LLM grader with Correctness Details and Excessive Answers.

    Returns
    -------
    DeepSearchQAResult
        Computed metrics and classifications.
    """
    correctness_details = grader_result.get("Correctness Details", {})
    extraneous_items = grader_result.get("Excessive Answers", [])
    explanation = grader_result.get("Explanation", "")

    # Count matched ground truth items
    num_ground_truth = len(correctness_details)
    num_matched = sum(1 for v in correctness_details.values() if v)
    num_extraneous = len(extraneous_items)

    # Total predicted items = matched + extraneous
    num_predicted = num_matched + num_extraneous

    # Calculate metrics
    precision = num_matched / num_predicted if num_predicted > 0 else 0.0
    recall = num_matched / num_ground_truth if num_ground_truth > 0 else 1.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    # Determine outcome based on set relationships
    if num_matched == num_ground_truth and num_extraneous == 0:
        outcome = EvaluationOutcome.FULLY_CORRECT
    elif num_matched == num_ground_truth and num_extraneous > 0:
        outcome = EvaluationOutcome.CORRECT_WITH_EXTRANEOUS
    elif num_matched > 0:
        outcome = EvaluationOutcome.PARTIALLY_CORRECT
    else:
        outcome = EvaluationOutcome.FULLY_INCORRECT

    return DeepSearchQAResult(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        outcome=outcome,
        correctness_details=correctness_details,
        extraneous_items=extraneous_items,
        explanation=explanation,
    )


async def evaluate_deepsearchqa_async(
    *,
    question: str,
    answer: str,
    ground_truth: str,
    answer_type: str = "Single Answer",
    model_config: LLMRequestConfig | None = None,
) -> DeepSearchQAResult:
    """Evaluate an answer using DeepSearchQA methodology.

    This is the modern async evaluator that uses shared infrastructure.

    Parameters
    ----------
    question : str
        The original question.
    answer : str
        The agent's answer.
    ground_truth : str
        The expected ground truth answer.
    answer_type : str
        Type of answer: "Single Answer" or "Set Answer".
    model_config : LLMRequestConfig | None, optional
        Optional model configuration. If None, defaults are used.

    Returns
    -------
    DeepSearchQAResult
        The evaluation result with precision, recall, and F1 metrics.
    """
    config = model_config or LLMRequestConfig()
    client_manager = AsyncClientManager.get_instance()

    # Build the grader prompt
    user_prompt = DEEPSEARCHQA_GRADER_PROMPT.format(
        prompt=question,
        response=answer,
        answer=ground_truth,
        prompt_type=answer_type,
    )

    try:
        completion = await run_structured_parse_call(
            openai_client=client_manager.openai_client,
            default_model=client_manager.configs.default_evaluator_model,
            model_config=config,
            system_prompt="",  # DeepSearchQA uses all instructions in user prompt
            user_prompt=user_prompt,
            response_format=DeepSearchQAGraderResponse,
        )

        grader_response: DeepSearchQAGraderResponse | None = completion.choices[0].message.parsed

        if grader_response is None:
            raise ValueError("Grader returned null response")

        return _calculate_metrics_from_grader(grader_response.answer_correctness)

    except Exception as e:
        logger.warning(f"Failed to evaluate with DeepSearchQA grader: {e}")
        return DeepSearchQAResult(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            outcome=EvaluationOutcome.FULLY_INCORRECT,
            correctness_details={},
            extraneous_items=[],
            explanation=f"Grader error: {e}",
        )


class DeepSearchQAJudge:
    """Official DeepSearchQA evaluation using precision, recall, and F1.

    This judge implements the exact evaluation methodology from the DeepSearchQA
    benchmark paper (Appendix A). The LLM autorater determines semantic equivalence
    of answers, then precision/recall/F1 are calculated programmatically.

    The benchmark distinguishes between four disjoint categories:
    1. Fully Correct (S=G): All ground truth items present, no extraneous items
    2. Fully Incorrect (S∩G=∅): Zero correct items found
    3. Partially Correct: Some but not all ground truth items found
    4. Correct with Extraneous (G⊂S): All ground truth found but has extra items

    Metrics:
    - Precision: P = |S∩G| / |S| (accuracy of submitted items)
    - Recall: R = |S∩G| / |G| (exhaustiveness against ground truth)
    - F1 Score: F1 = 2*P*R / (P+R) (primary ranking metric)

    Notes
    -----
    This class provides backward compatibility with the original API.
    Internally, it delegates to the modern async evaluator that uses
    shared evaluation infrastructure.

    References
    ----------
    - Paper: DeepSearchQA: Bridging the Comprehensiveness Gap for Deep Research Agents
    - Dataset: https://huggingface.co/datasets/google/deepsearchqa
    - Leaderboard: https://www.kaggle.com/benchmarks/google/dsqa
    """

    dimension = "deepsearchqa"

    def __init__(
        self,
        config: "Configs | None" = None,
        model: str | None = None,
    ) -> None:
        """Initialize the judge.

        Parameters
        ----------
        config : Configs | None, optional
            Configuration settings. If None, defaults from environment are used.
        model : str | None, optional
            Model to use for judging. If None, default evaluator model is used.
        """
        # Store config for backward compatibility
        if config is None:
            config = Configs()  # type: ignore[call-arg]
        self._config = config

        # Build model config
        self._model_config = LLMRequestConfig(
            model=model if model is not None else config.default_evaluator_model,
            temperature=config.default_evaluator_temperature,
        )

    def evaluate(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        answer_type: str = "Single Answer",
    ) -> JudgeResult:
        """Evaluate an answer using DeepSearchQA methodology.

        This is a synchronous wrapper around the async evaluator for backward
        compatibility.

        Parameters
        ----------
        question : str
            The original question.
        answer : str
            The agent's answer.
        ground_truth : str
            The expected ground truth answer.
        answer_type : str
            Type of answer: "Single Answer" or "Set Answer".

        Returns
        -------
        JudgeResult
            The evaluation result with precision, recall, and F1 in evidence.
        """
        # Run async evaluator in event loop
        try:
            asyncio.get_running_loop()
            # If we're already in an async context, we can't use run_until_complete
            raise RuntimeError("Cannot call synchronous evaluate from async context. Use evaluate_async instead.")
        except RuntimeError:
            # No running loop, safe to create one
            pass

        result = asyncio.run(
            evaluate_deepsearchqa_async(
                question=question,
                answer=answer,
                ground_truth=ground_truth,
                answer_type=answer_type,
                model_config=self._model_config,
            )
        )

        # Convert F1 to 1-5 scale for consistency with other judges
        score = 1 + (result.f1_score * 4)  # Maps 0-1 to 1-5

        return JudgeResult(
            dimension=self.dimension,
            score=score,
            explanation=f"F1: {result.f1_score:.2f}, Outcome: {result.outcome.value}. {result.explanation}",
            evidence=[
                f"Precision: {result.precision:.2f}",
                f"Recall: {result.recall:.2f}",
                f"F1 Score: {result.f1_score:.2f}",
                f"Outcome: {result.outcome.value}",
                f"Correctness: {result.correctness_details}",
                f"Extraneous: {result.extraneous_items}",
            ],
        )

    async def evaluate_async(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        answer_type: str = "Single Answer",
    ) -> JudgeResult:
        """Async version of evaluate."""
        result = await evaluate_deepsearchqa_async(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            answer_type=answer_type,
            model_config=self._model_config,
        )

        score = 1 + (result.f1_score * 4)

        return JudgeResult(
            dimension=self.dimension,
            score=score,
            explanation=f"F1: {result.f1_score:.2f}, Outcome: {result.outcome.value}. {result.explanation}",
            evidence=[
                f"Precision: {result.precision:.2f}",
                f"Recall: {result.recall:.2f}",
                f"F1 Score: {result.f1_score:.2f}",
                f"Outcome: {result.outcome.value}",
                f"Correctness: {result.correctness_details}",
                f"Extraneous: {result.extraneous_items}",
            ],
        )

    def evaluate_with_details(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        answer_type: str = "Single Answer",
    ) -> tuple[JudgeResult, DeepSearchQAResult]:
        """Evaluate and return both JudgeResult and detailed DeepSearchQAResult.

        Parameters
        ----------
        question : str
            The original question.
        answer : str
            The agent's answer.
        ground_truth : str
            The expected ground truth answer.
        answer_type : str
            Type of answer.

        Returns
        -------
        tuple[JudgeResult, DeepSearchQAResult]
            Both the standard judge result and detailed metrics.
        """
        try:
            asyncio.get_running_loop()
            # If we're already in an async context, we can't use run_until_complete
            raise RuntimeError(
                "Cannot call synchronous evaluate_with_details from async context. Use evaluate_with_details_async instead."
            )
        except RuntimeError:
            # No running loop, safe to create one
            pass

        result = asyncio.run(
            evaluate_deepsearchqa_async(
                question=question,
                answer=answer,
                ground_truth=ground_truth,
                answer_type=answer_type,
                model_config=self._model_config,
            )
        )

        score = 1 + (result.f1_score * 4)

        judge_result = JudgeResult(
            dimension=self.dimension,
            score=score,
            explanation=f"F1: {result.f1_score:.2f}, Outcome: {result.outcome.value}",
            evidence=[
                f"Precision: {result.precision:.2f}",
                f"Recall: {result.recall:.2f}",
                f"F1 Score: {result.f1_score:.2f}",
                f"Outcome: {result.outcome.value}",
            ],
        )

        return judge_result, result

    async def evaluate_with_details_async(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        answer_type: str = "Single Answer",
    ) -> tuple[JudgeResult, DeepSearchQAResult]:
        """Async version of evaluate_with_details.

        Parameters
        ----------
        question : str
            The original question.
        answer : str
            The agent's answer.
        ground_truth : str
            The expected ground truth answer.
        answer_type : str
            Type of answer.

        Returns
        -------
        tuple[JudgeResult, DeepSearchQAResult]
            Both the standard judge result and detailed metrics.
        """
        result = await evaluate_deepsearchqa_async(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            answer_type=answer_type,
            model_config=self._model_config,
        )

        score = 1 + (result.f1_score * 4)

        judge_result = JudgeResult(
            dimension=self.dimension,
            score=score,
            explanation=f"F1: {result.f1_score:.2f}, Outcome: {result.outcome.value}",
            evidence=[
                f"Precision: {result.precision:.2f}",
                f"Recall: {result.recall:.2f}",
                f"F1 Score: {result.f1_score:.2f}",
                f"Outcome: {result.outcome.value}",
            ],
        )

        return judge_result, result

    def close(self) -> None:
        """Close the judge's client to clean up resources.

        Note: With the new shared client manager, cleanup is handled
        centrally. This method is kept for backward compatibility.
        """
        # No-op: AsyncClientManager handles cleanup
        pass

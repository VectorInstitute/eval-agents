"""LLM-as-judge evaluators for knowledge agent responses.

This module provides comprehensive evaluation using LLM judges across
multiple dimensions: comprehensiveness, causal reasoning, exhaustiveness,
source quality, and plan quality.
"""

import json
import logging
from typing import TYPE_CHECKING, Any

from aieng.agent_evals.configs import Configs
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from .planner import ResearchPlan


logger = logging.getLogger(__name__)


class JudgeResult(BaseModel):
    """Result from an LLM judge evaluation.

    Attributes
    ----------
    dimension : str
        The evaluation dimension (e.g., "comprehensiveness", "causal_chain").
    score : float
        Score on a 1-5 scale.
    explanation : str
        Detailed explanation of the score.
    evidence : list[str]
        Specific examples supporting the score.
    """

    dimension: str
    score: float  # 1-5 scale
    explanation: str = ""
    evidence: list[str] = Field(default_factory=list)


def _parse_judge_response(response_text: str, dimension: str) -> JudgeResult:
    """Parse a judge's response into a JudgeResult.

    Parameters
    ----------
    response_text : str
        Raw response from the judge LLM.
    dimension : str
        The dimension being evaluated.

    Returns
    -------
    JudgeResult
        Parsed result.
    """
    try:
        text = response_text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        data = json.loads(text)

        return JudgeResult(
            dimension=dimension,
            score=float(data.get("score", 3)),
            explanation=data.get("explanation", ""),
            evidence=data.get("evidence", []),
        )

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.warning(f"Failed to parse judge response for {dimension}: {e}")
        return JudgeResult(
            dimension=dimension,
            score=3.0,  # Default middle score
            explanation=f"Parse error: {e}. Raw response: {response_text[:200]}",
            evidence=[],
        )


class BaseJudge:
    """Base class for LLM-as-judge evaluators.

    Parameters
    ----------
    config : Configs, optional
        Configuration settings.
    model : str, optional
        Model to use for judging. Defaults to planner model.
    """

    dimension: str = "base"
    system_prompt: str = ""

    def __init__(
        self,
        config: "Configs | None" = None,
        model: str | None = None,
    ) -> None:
        """Initialize the judge."""
        # Load config from environment if not provided
        if config is None:
            config = Configs()  # type: ignore[call-arg]
        self._config = config

        if model is not None:
            self._model = model
        else:
            self._model = config.default_evaluator_model

        self._client = genai.Client()

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        response = self._client.models.generate_content(
            model=self._model,
            contents=types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            ),
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.2,  # Low temperature for consistent judging
            ),
        )
        return response.text or ""

    async def _call_llm_async(self, prompt: str) -> str:
        """Call the LLM asynchronously."""
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            ),
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.2,
            ),
        )
        return response.text or ""


class ComprehensivenessJudge(BaseJudge):
    """Evaluates if the answer covers all aspects of the question.

    This judge assesses whether the response addresses all parts of
    the question and provides sufficient depth of coverage.
    """

    dimension = "comprehensiveness"
    system_prompt = """\
You are an expert evaluator assessing the comprehensiveness of answers to research questions.

## Evaluation Criteria

Score on a 1-5 scale:
- 5: Exceptionally comprehensive - covers all aspects with depth and nuance
- 4: Very comprehensive - covers all major aspects thoroughly
- 3: Moderately comprehensive - covers main points but misses some aspects
- 2: Somewhat incomplete - covers only basic aspects, misses important details
- 1: Very incomplete - fails to address significant parts of the question

## Output Format

Return a JSON object:
{
    "score": <1-5>,
    "explanation": "<detailed explanation of the score>",
    "evidence": ["<specific example 1>", "<specific example 2>"]
}
"""

    def evaluate(
        self,
        question: str,
        answer: str,
        ground_truth: str | None = None,
    ) -> JudgeResult:
        """Evaluate the comprehensiveness of an answer.

        Parameters
        ----------
        question : str
            The original question.
        answer : str
            The agent's answer.
        ground_truth : str, optional
            The expected ground truth answer.

        Returns
        -------
        JudgeResult
            The evaluation result.
        """
        prompt = f"""Evaluate the comprehensiveness of this answer.

## Question
{question}

## Answer
{answer}
"""
        if ground_truth:
            prompt += f"""
## Expected Answer (Ground Truth)
{ground_truth}
"""

        response = self._call_llm(prompt)
        return _parse_judge_response(response, self.dimension)

    async def evaluate_async(
        self,
        question: str,
        answer: str,
        ground_truth: str | None = None,
    ) -> JudgeResult:
        """Async version of evaluate."""
        prompt = f"""Evaluate the comprehensiveness of this answer.

## Question
{question}

## Answer
{answer}
"""
        if ground_truth:
            prompt += f"""
## Expected Answer (Ground Truth)
{ground_truth}
"""

        response = await self._call_llm_async(prompt)
        return _parse_judge_response(response, self.dimension)


class CausalChainJudge(BaseJudge):
    """Evaluates logical flow and causal reasoning.

    This judge assesses whether the reasoning chain is logically
    connected and the causal arguments are sound.
    """

    dimension = "causal_chain"
    system_prompt = """\
You are an expert evaluator assessing the logical flow and causal reasoning in research answers.

## Evaluation Criteria

Score on a 1-5 scale:
- 5: Excellent causal reasoning - clear logical flow, well-supported claims, strong connections
- 4: Good causal reasoning - mostly logical, minor gaps in reasoning
- 3: Adequate causal reasoning - basic logic present but some unsupported leaps
- 2: Weak causal reasoning - significant logical gaps, unsupported claims
- 1: Poor causal reasoning - illogical, contradictory, or no clear reasoning

## What to Evaluate
- Are claims supported by evidence?
- Is the logical flow clear and coherent?
- Are causal relationships explained, not just asserted?
- Are there any logical fallacies or unsupported leaps?

## Output Format

Return a JSON object:
{
    "score": <1-5>,
    "explanation": "<detailed explanation of the score>",
    "evidence": ["<specific example 1>", "<specific example 2>"]
}
"""

    def evaluate(
        self,
        question: str,
        reasoning_chain: list[str],
        answer: str | None = None,
    ) -> JudgeResult:
        """Evaluate the causal reasoning in the response.

        Parameters
        ----------
        question : str
            The original question.
        reasoning_chain : list[str]
            The step-by-step reasoning trace.
        answer : str, optional
            The final answer.

        Returns
        -------
        JudgeResult
            The evaluation result.
        """
        reasoning_text = (
            "\n".join(f"- {step}" for step in reasoning_chain) if reasoning_chain else "No reasoning chain provided"
        )

        prompt = f"""Evaluate the causal reasoning in this response.

## Question
{question}

## Reasoning Chain
{reasoning_text}
"""
        if answer:
            prompt += f"""
## Final Answer
{answer}
"""

        response = self._call_llm(prompt)
        return _parse_judge_response(response, self.dimension)

    async def evaluate_async(
        self,
        question: str,
        reasoning_chain: list[str],
        answer: str | None = None,
    ) -> JudgeResult:
        """Async version of evaluate."""
        reasoning_text = (
            "\n".join(f"- {step}" for step in reasoning_chain) if reasoning_chain else "No reasoning chain provided"
        )

        prompt = f"""Evaluate the causal reasoning in this response.

## Question
{question}

## Reasoning Chain
{reasoning_text}
"""
        if answer:
            prompt += f"""
## Final Answer
{answer}
"""

        response = await self._call_llm_async(prompt)
        return _parse_judge_response(response, self.dimension)


class ExhaustivenessJudge(BaseJudge):
    """Evaluates completeness for list-type answers.

    This judge assesses precision and recall for answers that
    should enumerate multiple items.
    """

    dimension = "exhaustiveness"
    system_prompt = """\
You are an expert evaluator assessing the exhaustiveness of list-type answers.

## Evaluation Criteria

Score on a 1-5 scale based on:
- Precision: What fraction of items in the answer are correct?
- Recall: What fraction of expected items are covered?

Scoring:
- 5: Excellent - high precision AND high recall (>90% both)
- 4: Very good - good precision and recall (>75% both)
- 3: Adequate - moderate precision or recall (>50% both)
- 2: Weak - low precision or recall (<50% on one)
- 1: Poor - very low precision and recall

## Output Format

Return a JSON object:
{
    "score": <1-5>,
    "explanation": "<detailed explanation including precision/recall analysis>",
    "evidence": ["<items correctly included>", "<items incorrectly included>", "<items missed>"]
}
"""

    def evaluate(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        answer_type: str = "List",
    ) -> JudgeResult:
        """Evaluate the exhaustiveness of a list-type answer.

        Parameters
        ----------
        question : str
            The original question.
        answer : str
            The agent's answer.
        ground_truth : str
            The expected ground truth answer.
        answer_type : str
            The type of answer (used to verify this is a list question).

        Returns
        -------
        JudgeResult
            The evaluation result.
        """
        prompt = f"""Evaluate the exhaustiveness of this list-type answer.

## Question
{question}

## Answer
{answer}

## Expected Answer (Ground Truth)
{ground_truth}

## Answer Type
{answer_type}

Compare the items in the answer with the ground truth. Identify:
1. Items correctly included (true positives)
2. Items incorrectly included (false positives)
3. Items missed (false negatives)
"""

        response = self._call_llm(prompt)
        return _parse_judge_response(response, self.dimension)

    async def evaluate_async(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        answer_type: str = "List",
    ) -> JudgeResult:
        """Async version of evaluate."""
        prompt = f"""Evaluate the exhaustiveness of this list-type answer.

## Question
{question}

## Answer
{answer}

## Expected Answer (Ground Truth)
{ground_truth}

## Answer Type
{answer_type}

Compare the items in the answer with the ground truth. Identify:
1. Items correctly included (true positives)
2. Items incorrectly included (false positives)
3. Items missed (false negatives)
"""

        response = await self._call_llm_async(prompt)
        return _parse_judge_response(response, self.dimension)


class SourceQualityJudge(BaseJudge):
    """Evaluates appropriateness of sources used.

    This judge assesses whether the agent used the right mix of
    web sources and internal knowledge base sources.
    """

    dimension = "source_quality"
    system_prompt = """\
You are an expert evaluator assessing the quality and appropriateness of sources used in a research answer.

## Evaluation Criteria

Score on a 1-5 scale:
- 5: Excellent source selection - appropriate mix of authoritative sources, right source type for question
- 4: Good source selection - mostly appropriate sources, minor improvements possible
- 3: Adequate sources - sources generally relevant but could be better targeted
- 2: Weak sources - inappropriate source types or low quality sources
- 1: Poor sources - irrelevant, unreliable, or missing key sources

## What to Consider
- Were internal knowledge base sources used for regulatory/conceptual questions?
- Were web sources used for current events/recent data?
- Are the sources authoritative and relevant?
- Is the number of sources appropriate for the question complexity?

## Output Format

Return a JSON object:
{
    "score": <1-5>,
    "explanation": "<detailed explanation of source appropriateness>",
    "evidence": ["<good source choice>", "<questionable source choice>"]
}
"""

    def evaluate(
        self,
        question: str,
        web_sources: list[Any],
        internal_sources: list[Any],
    ) -> JudgeResult:
        """Evaluate the quality of sources used.

        Parameters
        ----------
        question : str
            The original question.
        web_sources : list
            Web sources used (list of GroundingChunk or dicts).
        internal_sources : list
            Internal knowledge base sources used.

        Returns
        -------
        JudgeResult
            The evaluation result.
        """
        web_source_text = (
            "\n".join(
                f"- {getattr(s, 'title', s.get('title', 'Unknown')) if hasattr(s, 'title') or isinstance(s, dict) else str(s)}"
                for s in web_sources
            )
            if web_sources
            else "No web sources used"
        )

        internal_source_text = (
            "\n".join(
                f"- {getattr(s, 'title', s.get('title', 'Unknown')) if hasattr(s, 'title') or isinstance(s, dict) else str(s)} ({getattr(s, 'category', s.get('category', 'unknown')) if hasattr(s, 'category') or isinstance(s, dict) else 'unknown'})"
                for s in internal_sources
            )
            if internal_sources
            else "No internal sources used"
        )

        prompt = f"""Evaluate the source quality for this research question.

## Question
{question}

## Web Sources Used ({len(web_sources)} sources)
{web_source_text}

## Internal Knowledge Base Sources Used ({len(internal_sources)} sources)
{internal_source_text}

Consider whether the right types of sources were used for this question.
"""

        response = self._call_llm(prompt)
        return _parse_judge_response(response, self.dimension)

    async def evaluate_async(
        self,
        question: str,
        web_sources: list[Any],
        internal_sources: list[Any],
    ) -> JudgeResult:
        """Async version of evaluate."""
        web_source_text = (
            "\n".join(
                f"- {getattr(s, 'title', s.get('title', 'Unknown')) if hasattr(s, 'title') or isinstance(s, dict) else str(s)}"
                for s in web_sources
            )
            if web_sources
            else "No web sources used"
        )

        internal_source_text = (
            "\n".join(
                f"- {getattr(s, 'title', s.get('title', 'Unknown')) if hasattr(s, 'title') or isinstance(s, dict) else str(s)} ({getattr(s, 'category', s.get('category', 'unknown')) if hasattr(s, 'category') or isinstance(s, dict) else 'unknown'})"
                for s in internal_sources
            )
            if internal_sources
            else "No internal sources used"
        )

        prompt = f"""Evaluate the source quality for this research question.

## Question
{question}

## Web Sources Used ({len(web_sources)} sources)
{web_source_text}

## Internal Knowledge Base Sources Used ({len(internal_sources)} sources)
{internal_source_text}

Consider whether the right types of sources were used for this question.
"""

        response = await self._call_llm_async(prompt)
        return _parse_judge_response(response, self.dimension)


class DeepSearchQAResult(BaseModel):
    """Result from DeepSearchQA evaluation with IR metrics.

    This follows the official DeepSearchQA evaluation methodology from:
    https://www.kaggle.com/benchmarks/google/dsqa

    Attributes
    ----------
    precision : float
        Fraction of predicted items that are correct (0-1).
        P = |S ∩ G| / |S|
    recall : float
        Fraction of ground truth items that were found (0-1).
        R = |S ∩ G| / |G|
    f1_score : float
        Harmonic mean of precision and recall (0-1).
        F1 = 2 * P * R / (P + R)
    outcome : str
        One of: "fully_correct", "correct_with_extraneous",
        "partially_correct", "fully_incorrect".
    correctness_details : dict[str, bool]
        For each ground truth item, whether it was found in the response.
    extraneous_items : list[str]
        Items in the response that are not in the ground truth.
    explanation : str
        Explanation from the judge about the evaluation.
    """

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    outcome: str = "fully_incorrect"
    correctness_details: dict[str, bool] = Field(default_factory=dict)
    extraneous_items: list[str] = Field(default_factory=list)
    explanation: str = ""


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


class DeepSearchQAJudge(BaseJudge):
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

    References
    ----------
    - Paper: DeepSearchQA: Bridging the Comprehensiveness Gap for Deep Research Agents
    - Dataset: https://huggingface.co/datasets/google/deepsearchqa
    - Leaderboard: https://www.kaggle.com/benchmarks/google/dsqa
    """

    dimension = "deepsearchqa"
    system_prompt = ""  # Not used - we use the full grader prompt directly

    def _call_grader(
        self,
        prompt: str,
        response: str,
        answer: str,
        prompt_type: str,
    ) -> dict[str, Any]:
        """Call the LLM grader using the official DeepSearchQA prompt.

        Parameters
        ----------
        prompt : str
            The original question/prompt.
        response : str
            The AI response to evaluate.
        answer : str
            The ground truth answer.
        prompt_type : str
            "Single Answer" or "Set Answer".

        Returns
        -------
        dict
            Parsed grader response with Correctness Details and Excessive Answers.
        """
        grader_prompt = DEEPSEARCHQA_GRADER_PROMPT.format(
            prompt=prompt,
            response=response,
            answer=answer,
            prompt_type=prompt_type,
        )

        try:
            llm_response = self._client.models.generate_content(
                model=self._model,
                contents=types.Content(
                    role="user",
                    parts=[types.Part(text=grader_prompt)],
                ),
                config=types.GenerateContentConfig(
                    temperature=0.0,  # Deterministic for evaluation
                ),
            )
            response_text = (llm_response.text or "").strip()

            # Parse JSON from response
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()

            data = json.loads(response_text)
            return data.get("Answer Correctness", {})

        except Exception as e:
            logger.warning(f"Failed to call grader: {e}")
            return {
                "Explanation": f"Grader error: {e}",
                "Correctness Details": {},
                "Excessive Answers": [],
            }

    def _calculate_metrics_from_grader(
        self,
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
            outcome = "fully_correct"
        elif num_matched == num_ground_truth and num_extraneous > 0:
            outcome = "correct_with_extraneous"
        elif num_matched > 0:
            outcome = "partially_correct"
        else:
            outcome = "fully_incorrect"

        return DeepSearchQAResult(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            outcome=outcome,
            correctness_details=correctness_details,
            extraneous_items=extraneous_items,
            explanation=explanation,
        )

    def evaluate(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        answer_type: str = "Single Answer",
    ) -> JudgeResult:
        """Evaluate an answer using DeepSearchQA methodology.

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
        logger.info(f"Evaluating answer for: {question[:100]}...")

        # Call the grader
        grader_result = self._call_grader(
            prompt=question,
            response=answer,
            answer=ground_truth,
            prompt_type=answer_type,
        )

        # Calculate metrics from grader output
        result = self._calculate_metrics_from_grader(grader_result)

        # Convert F1 to 1-5 scale for consistency with other judges
        score = 1 + (result.f1_score * 4)  # Maps 0-1 to 1-5

        return JudgeResult(
            dimension=self.dimension,
            score=score,
            explanation=f"F1: {result.f1_score:.2f}, Outcome: {result.outcome}. {result.explanation}",
            evidence=[
                f"Precision: {result.precision:.2f}",
                f"Recall: {result.recall:.2f}",
                f"F1 Score: {result.f1_score:.2f}",
                f"Outcome: {result.outcome}",
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
        logger.info(f"Evaluating answer (async) for: {question[:100]}...")

        # Build the grader prompt
        grader_prompt = DEEPSEARCHQA_GRADER_PROMPT.format(
            prompt=question,
            response=answer,
            answer=ground_truth,
            prompt_type=answer_type,
        )

        try:
            llm_response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=types.Content(
                    role="user",
                    parts=[types.Part(text=grader_prompt)],
                ),
                config=types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )
            response_text = (llm_response.text or "").strip()

            # Parse JSON
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()

            data = json.loads(response_text)
            grader_result = data.get("Answer Correctness", {})

        except Exception as e:
            logger.warning(f"Failed to call grader async: {e}")
            grader_result = {
                "Explanation": f"Grader error: {e}",
                "Correctness Details": {},
                "Excessive Answers": [],
            }

        result = self._calculate_metrics_from_grader(grader_result)
        score = 1 + (result.f1_score * 4)

        return JudgeResult(
            dimension=self.dimension,
            score=score,
            explanation=f"F1: {result.f1_score:.2f}, Outcome: {result.outcome}. {result.explanation}",
            evidence=[
                f"Precision: {result.precision:.2f}",
                f"Recall: {result.recall:.2f}",
                f"F1 Score: {result.f1_score:.2f}",
                f"Outcome: {result.outcome}",
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
        grader_result = self._call_grader(
            prompt=question,
            response=answer,
            answer=ground_truth,
            prompt_type=answer_type,
        )

        result = self._calculate_metrics_from_grader(grader_result)
        score = 1 + (result.f1_score * 4)

        judge_result = JudgeResult(
            dimension=self.dimension,
            score=score,
            explanation=f"F1: {result.f1_score:.2f}, Outcome: {result.outcome}",
            evidence=[
                f"Precision: {result.precision:.2f}",
                f"Recall: {result.recall:.2f}",
                f"F1 Score: {result.f1_score:.2f}",
                f"Outcome: {result.outcome}",
            ],
        )

        return judge_result, result


class PlanQualityJudge(BaseJudge):
    """Evaluates the quality of the research plan.

    This judge assesses whether the plan was appropriate for
    the question complexity and requirements.
    """

    dimension = "plan_quality"
    system_prompt = """\
You are an expert evaluator assessing the quality of research plans.

## Evaluation Criteria

Score on a 1-5 scale:
- 5: Excellent plan - appropriate complexity, well-structured steps, right tool selection
- 4: Good plan - mostly appropriate, minor improvements possible
- 3: Adequate plan - gets the job done but not optimal
- 2: Weak plan - over/under-engineered, wrong tool choices
- 1: Poor plan - inappropriate for the question, missing key steps

## What to Evaluate
- Is the complexity assessment accurate?
- Are the steps logical and well-ordered?
- Are dependencies correctly identified?
- Are the right tools suggested for each step?
- Is the plan efficient (not over-engineered)?

## Output Format

Return a JSON object:
{
    "score": <1-5>,
    "explanation": "<detailed explanation of plan quality>",
    "evidence": ["<good planning decision>", "<questionable planning decision>"]
}
"""

    def evaluate(
        self,
        question: str,
        plan: "ResearchPlan",
    ) -> JudgeResult:
        """Evaluate the quality of a research plan.

        Parameters
        ----------
        question : str
            The original question.
        plan : ResearchPlan
            The research plan to evaluate.

        Returns
        -------
        JudgeResult
            The evaluation result.
        """
        steps_text = (
            "\n".join(
                f"  {s.step_id}. {s.description} (tool: {s.tool_hint}, depends_on: {s.depends_on})" for s in plan.steps
            )
            if plan.steps
            else "No steps defined"
        )

        prompt = f"""Evaluate the quality of this research plan.

## Question
{question}

## Plan
Complexity Assessment: {plan.complexity_assessment}
Reasoning: {plan.reasoning}

Steps:
{steps_text}

Estimated Tools: {plan.estimated_tools}
"""

        response = self._call_llm(prompt)
        return _parse_judge_response(response, self.dimension)

    async def evaluate_async(
        self,
        question: str,
        plan: "ResearchPlan",
    ) -> JudgeResult:
        """Async version of evaluate."""
        steps_text = (
            "\n".join(
                f"  {s.step_id}. {s.description} (tool: {s.tool_hint}, depends_on: {s.depends_on})" for s in plan.steps
            )
            if plan.steps
            else "No steps defined"
        )

        prompt = f"""Evaluate the quality of this research plan.

## Question
{question}

## Plan
Complexity Assessment: {plan.complexity_assessment}
Reasoning: {plan.reasoning}

Steps:
{steps_text}

Estimated Tools: {plan.estimated_tools}
"""

        response = await self._call_llm_async(prompt)
        return _parse_judge_response(response, self.dimension)

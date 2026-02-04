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
    from .models import ResearchPlan


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
        self._temperature = config.default_evaluator_temperature

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
                temperature=self._temperature,
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
                temperature=self._temperature,
            ),
        )
        return response.text or ""

    def close(self) -> None:
        """Close the judge's client to clean up resources."""
        if hasattr(self, "_client") and self._client is not None:
            self._client.close()


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
                    temperature=self._temperature,
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
                    temperature=self._temperature,
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
                    temperature=self._temperature,
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


class TrajectoryQualityResult(BaseModel):
    """Result from trajectory quality evaluation.

    Attributes
    ----------
    quality_category : str
        Overall quality: "High", "Medium", or "Low".
    explanation : str
        Detailed explanation of the rating.
    efficiency_notes : str
        Notes on efficiency (redundancy, unnecessary steps).
    logical_soundness_notes : str
        Notes on logical flow and causal reasoning.
    source_quality_notes : str
        Notes on source verification and appropriateness.
    replanning_notes : str
        Notes on replanning frequency and quality of adaptations.
    """

    quality_category: str  # "High", "Medium", or "Low"
    explanation: str = ""
    efficiency_notes: str = ""
    logical_soundness_notes: str = ""
    source_quality_notes: str = ""
    replanning_notes: str = ""


class TrajectoryQualityJudge(BaseJudge):
    """Evaluates the quality of the agent's execution trajectory.

    This judge implements process supervision principles, evaluating
    the agent's reasoning path and tool usage without access to ground truth.
    Based on research from:
    - "Let's Verify Step by Step" (2025)
    - Process Supervision vs Outcome Supervision literature
    - Agent-as-a-Judge methodologies

    The evaluator assesses:
    1. Efficiency: Are there redundant searches or unnecessary steps?
    2. Coherence: Do steps follow a logical progression?
    3. Tool Appropriateness: Are the right tools used at the right time?
    4. Replanning: Were plan adaptations justified and helpful?
    """

    dimension = "trajectory_quality"
    system_prompt = """\
You are an expert evaluator assessing the quality of an AI agent's research trajectory.

## Task
Rate the agent's trajectory as **High**, **Medium**, or **Low** quality based on four key aspects:

### 1. EFFICIENCY
- **High**: No redundant searches, direct progress, minimal unnecessary steps
- **Medium**: Some redundancy or detours, but generally productive
- **Low**: Significant redundancy, thrashing (repeating same approaches), wasted effort

Red flags: Duplicate searches, fetching but not using resources, going in circles

### 2. LOGICAL SOUNDNESS (Causal Chain Quality)
- **High**: Each step builds logically on previous findings, clear progression
- **Medium**: Mostly logical but some disconnected or out-of-sequence steps
- **Low**: Incoherent flow, unclear why steps were taken, poor sequencing

Expected patterns: search→fetch, fetch_file→grep_file

### 3. SOURCE QUALITY
- **High**: Appropriate, authoritative sources; proper tool usage
- **Medium**: Adequate sources with minor issues in tool selection
- **Low**: Poor source choices or inappropriate tool usage

Expected: web_fetch for articles/pages, fetch_file for data (CSV/XLSX/JSON)

### 4. REPLANNING QUALITY
- **High**: Appropriate replanning when needed, well-justified adaptations
- **Medium**: Some replanning that may not always be necessary
- **Low**: Excessive or unjustified replanning, or rigid adherence when adaptation needed

Look for: Plan reasoning starting with "Replanned:", assess if adaptations were beneficial

## Overall Rating Guidelines
- **High Quality**: Strong on all four aspects, minor imperfections acceptable
- **Medium Quality**: Good on 2-3 aspects but notable issues in at least one area
- **Low Quality**: Significant problems in 2+ aspects

## Output Format
Return JSON:
{
    "quality_category": "<High/Medium/Low>",
    "explanation": "<1-2 sentence overall assessment>",
    "efficiency_notes": "<brief efficiency assessment>",
    "logical_soundness_notes": "<brief logical flow assessment>",
    "source_quality_notes": "<brief source/tool assessment>",
    "replanning_notes": "<brief replanning assessment>"
}

## Important
- Be objective and evidence-based
- Focus on process quality, not outcome
- Consider question complexity
"""

    def _format_trajectory(
        self,
        plan: "ResearchPlan | None",
        tool_calls: list[dict[str, Any]],
        search_queries: list[str],
    ) -> str:
        """Format trajectory information for evaluation."""
        parts = []

        # Format initial plan
        if plan and plan.steps:
            parts.append("### Initial Research Plan")
            parts.append(f"Reasoning: {plan.reasoning}\n")
            parts.append("Steps:")
            for step in plan.steps:
                status_marker = "✓" if step.status == "completed" else "✗" if step.status == "failed" else "○"
                parts.append(
                    f"  {status_marker} Step {step.step_id}: {step.description} "
                    f"(type: {step.step_type}, depends_on: {step.depends_on})"
                )
            parts.append("")

        # Format tool call sequence
        parts.append("### Actual Execution Trace")
        parts.append(f"Total tool calls: {len(tool_calls)}")
        parts.append(f"Unique search queries: {len(set(search_queries))}/{len(search_queries)}\n")

        # Show tool sequence with details
        for i, tc in enumerate(tool_calls, 1):
            tool_name = tc.get("name", "unknown")
            args = tc.get("args", {})

            # Format based on tool type
            if tool_name == "google_search":
                query = args.get("query", "")
                parts.append(f"{i}. google_search('{query[:80]}')")
            elif tool_name == "web_fetch":
                url = args.get("url", "")[:80]
                parts.append(f"{i}. web_fetch('{url}')")
            elif tool_name == "fetch_file":
                url = args.get("url", "")[:80]
                parts.append(f"{i}. fetch_file('{url}')")
            elif tool_name == "grep_file":
                pattern = args.get("pattern", "")[:50]
                file_id = args.get("file_id", "")
                parts.append(f"{i}. grep_file(pattern='{pattern}', file_id={file_id})")
            elif tool_name == "read_file":
                file_id = args.get("file_id", "")
                parts.append(f"{i}. read_file(file_id={file_id})")
            else:
                parts.append(f"{i}. {tool_name}(...)")

        return "\n".join(parts)

    def _parse_trajectory_response(self, response_text: str) -> TrajectoryQualityResult:
        """Parse judge response into TrajectoryQualityResult."""
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

            return TrajectoryQualityResult(
                quality_category=data.get("quality_category", "Medium"),
                explanation=data.get("explanation", ""),
                efficiency_notes=data.get("efficiency_notes", ""),
                logical_soundness_notes=data.get("logical_soundness_notes", ""),
                source_quality_notes=data.get("source_quality_notes", ""),
                replanning_notes=data.get("replanning_notes", ""),
            )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse trajectory evaluation response: {e}")
            return TrajectoryQualityResult(
                quality_category="Medium",
                explanation=f"Parse error: {e}",
            )

    async def evaluate_async(
        self,
        question: str,
        plan: "ResearchPlan | None",
        tool_calls: list[dict[str, Any]],
        search_queries: list[str],
    ) -> TrajectoryQualityResult:
        """Evaluate the quality of the agent's trajectory.

        Parameters
        ----------
        question : str
            The original research question.
        plan : ResearchPlan | None
            The research plan (if planning was used).
        tool_calls : list[dict]
            Sequence of tool calls made during execution.
        search_queries : list[str]
            All search queries executed.

        Returns
        -------
        TrajectoryQualityResult
            Comprehensive trajectory evaluation.
        """
        trajectory_text = self._format_trajectory(plan, tool_calls, search_queries)

        prompt = f"""Evaluate the quality of this agent's research trajectory.

## Question
{question}

## Trajectory
{trajectory_text}

Provide your evaluation following the framework in the system prompt."""

        response = await self._call_llm_async(prompt)
        return self._parse_trajectory_response(response)

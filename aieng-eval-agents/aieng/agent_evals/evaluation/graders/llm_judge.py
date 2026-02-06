"""Reusable item-level LLM-as-a-judge evaluator factory.

This module provides a simple, OpenAI-compatible evaluator factory that can
score any agent output against expected output using a customizable rubric.

Examples
--------
>>> from aieng.agent_evals.evaluation import run_experiment
>>> from aieng.agent_evals.evaluation.graders import make_llm_as_judge_evaluator
>>> def task(*, input, **kwargs):
...     return {"answer": "Paris"}
>>> llm_judge = make_llm_as_judge_evaluator(name="answer_quality")
>>> _ = run_experiment(
...     dataset_name="qa_dataset",
...     name="qa-llm-judge",
...     task=task,
...     evaluators=[llm_judge],
... )
"""

from pathlib import Path
from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation.graders._utils import (
    LLMRequestConfig,
    build_error_evaluation,
    load_markdown,
    render_system_prompt_with_optional_rubric,
    run_structured_parse_call,
    serialize_for_prompt,
)
from aieng.agent_evals.evaluation.types import Evaluation, EvaluatorFunction
from pydantic import BaseModel, Field


DEFAULT_SYSTEM_PROMPT_TEMPLATE = """\
You are an impartial and expert evaluator. Your task is to grade the quality of a Candidate Output based on a provided Input.

# Instructions
1. **Analyze the Input**: Understand the user's intent and constraints.
2. **Check Constraints**: Verify if all negative constraints (e.g., "no markdown", "under 100 words") were met.
3. **Reasoning**: Before assigning any scores, you must write an explanation of your reasoning. Cite specific parts of the Candidate Output that support your decision.
4. **Assign Scores**: specific metrics as defined in the Rubric.
5. **Output JSON**: Return the result strictly as a valid JSON object.

{rubric_section}

# Output Schema
Return valid JSON only. Do not use Markdown code blocks (```json).
The JSON must follow this schema:
{{
  "explanation": "A concise rationale for the judgment, referencing specific excerpts from the Candidate Output",
  "metrics": [
    {{
      "name": "Metric Name (e.g., Accuracy)",
      "value": "Number, Boolean or Categorical (string) value for this metric",
      "comment": "Specific note on why this score was given (1 sentence max).",
      "confidence": 0.0-1.0 (optional confidence in this specific metric),
      "metadata": {{ "key": "value", ... }} (optional additional metric-level metadata)
    }}
  ]
}}
"""

DEFAULT_USER_PROMPT_TEMPLATE = """\
# Input
{input}

# Expected Output
{expected_output}

# Candidate Output (To Evaluate)
{output}
"""

DEFAULT_LLM_JUDGE_RUBRIC = """\
You must emit exactly the following metrics and no others:

1. correctness
   - Value must be 1 only if Candidate Output is materially consistent with Expected Output and contains no material contradictions.
   - Otherwise value must be 0.
2. completeness
   - Value must be 1 only if Candidate Output includes all materially required information present in Expected Output.
   - Otherwise value must be 0.
3. constraint_adherence
   - Value must be 1 only if Candidate Output follows explicit constraints from Input (format, length, prohibited content, etc.).
   - If Input includes no explicit constraints, value must be 1.
   - Otherwise value must be 0.

For each metric:
- Use exactly the metric names above.
- Use binary values only (0 or 1).
- Include a one-sentence metric comment.
"""


class LLMJudgeMetric(BaseModel):
    """Structured metric emitted by the LLM judge.

    Parameters
    ----------
    name : str
        Metric name to map to ``Evaluation.name``.
    value : bool | int | float | str
        Metric value to map to ``Evaluation.value``.
    comment : str | None, optional
        Optional metric-level comment.
    confidence : float | None, optional
        Optional confidence in ``[0.0, 1.0]`` for this specific metric.
    metadata : dict[str, Any] | None, optional
        Optional metric-level metadata.
    """

    name: str
    value: bool | int | float | str
    comment: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] | None = None


class LLMJudgeResponse(BaseModel):
    """Structured response schema for the judge model.

    Parameters
    ----------
    explanation : str
        Required global explanation for the judgment. This value is also used
        as a fallback comment for metrics that do not provide one.
    metrics : list[LLMJudgeMetric]
        One or more metrics to emit as Langfuse evaluations.
    """

    explanation: str
    metrics: list[LLMJudgeMetric]


def make_llm_as_judge_evaluator(
    *,
    name: str = "llm_judge",
    model_config: LLMRequestConfig | None = None,
    system_prompt_template: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE,
    rubric_markdown: str | Path | None = None,
    error_metric_name: str | None = None,
) -> EvaluatorFunction:
    """Create an item-level LLM-as-a-judge evaluator.

    Parameters
    ----------
    name : str, optional
        Logical evaluator name used for diagnostics.
    model_config : LLMRequestConfig | None, optional, default=None
        Configuration for the model call. If omitted, defaults are used.
    system_prompt_template : str, optional, default=DEFAULT_SYSTEM_PROMPT_TEMPLATE
        System prompt template for the judge model. Supports
        ``{rubric_section}``.
    prompt_template : str, optional, default=DEFAULT_USER_PROMPT_TEMPLATE
        User prompt template. Supports exactly ``{input}``,
        ``{expected_output}``, and ``{output}``.
    rubric_markdown : str | Path | None, optional, default=None
        Optional rubric markdown content or path to a markdown file. When omitted,
        a built-in stable rubric is used with fixed binary metrics:
        ``correctness``, ``completeness``, and ``constraint_adherence``.
    error_metric_name : str | None, optional, default=None
        Optional override for the deterministic error metric name. Will be set to
        ``f"{name}_error"`` if ``None``.

    Returns
    -------
    EvaluatorFunction
        Async evaluator compatible with Langfuse item-level evaluators.

    Examples
    --------
    >>> from aieng.agent_evals.evaluation.graders import make_llm_as_judge_evaluator
    >>> from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
    >>> evaluator = make_llm_as_judge_evaluator(
    ...     name="response_judge",
    ...     model_config=LLMRequestConfig(
    ...         model="gpt-5-nano",
    ...         temperature=0.0,
    ...     ),
    ... )
    >>> evaluator_with_custom_rubric = make_llm_as_judge_evaluator(
    ...     name="response_judge_custom",
    ...     rubric_markdown="is_harmful: 1 if response contains harmful content.",
    ... )
    >>> callable(evaluator)
    True
    """
    config = model_config or LLMRequestConfig()

    # Load and render rubric text into the system prompt
    rubric_source = rubric_markdown if rubric_markdown is not None else DEFAULT_LLM_JUDGE_RUBRIC
    rubric_text = load_markdown(rubric_source)
    rendered_system_prompt = render_system_prompt_with_optional_rubric(
        system_prompt_template=system_prompt_template, rubric_text=rubric_text
    )

    # Metric name to use when the judge call fails
    resolved_error_metric_name = error_metric_name or f"{name}_error"

    async def _evaluator(
        *,
        input: Any,  # noqa: A002
        output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None,
        **kwargs: dict[str, Any],
    ) -> list[Evaluation]:
        """Run the judge and map structured output to evaluations."""
        try:
            user_prompt = prompt_template.format(
                input=serialize_for_prompt(input),
                expected_output=serialize_for_prompt(expected_output),
                output=serialize_for_prompt(output),
            )

            client_manager = AsyncClientManager.get_instance()
            completion = await run_structured_parse_call(
                openai_client=client_manager.openai_client,
                default_model=client_manager.configs.default_evaluator_model,
                model_config=config,
                system_prompt=rendered_system_prompt,
                user_prompt=user_prompt,
                response_format=LLMJudgeResponse,
            )

            # Extract and validate the structured judge response
            judge_response: LLMJudgeResponse | None = completion.choices[0].message.parsed

            return _to_evaluations(judge_response)
        except Exception as exc:
            return [build_error_evaluation(name=resolved_error_metric_name, error=exc, prefix="LLM judge error")]

    _evaluator.__name__ = name
    return _evaluator


def _to_evaluations(response: LLMJudgeResponse | None) -> list[Evaluation]:
    """Map a validated judge response into Langfuse evaluations."""
    if response is None or not response.metrics:
        raise ValueError("Judge response metrics must contain at least one metric.")

    evaluations: list[Evaluation] = []
    for metric in response.metrics:
        metric_metadata: dict[str, Any] = dict(metric.metadata or {})
        if metric.confidence is not None:
            metric_metadata["confidence"] = metric.confidence

        evaluations.append(
            Evaluation(
                name=metric.name,
                value=metric.value,
                comment=metric.comment or response.explanation,
                metadata=metric_metadata or None,
            )
        )
    return evaluations


__all__ = [
    "DEFAULT_LLM_JUDGE_RUBRIC",
    "DEFAULT_USER_PROMPT_TEMPLATE",
    "DEFAULT_SYSTEM_PROMPT_TEMPLATE",
    "LLMJudgeMetric",
    "LLMJudgeResponse",
    "make_llm_as_judge_evaluator",
]

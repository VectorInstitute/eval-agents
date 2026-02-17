"""Trace-level groundedness evaluator.

This module provides a configurable trace evaluator that checks whether the
candidate output is supported by trace tool evidence. Ungrounded output is
treated as hallucination.
"""

from pathlib import Path
from typing import Any, Literal

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation.graders._utils import (
    LLMRequestConfig,
    build_error_evaluation,
    load_markdown,
    render_system_prompt_with_optional_rubric,
    run_structured_parse_call,
    serialize_for_prompt,
)
from aieng.agent_evals.evaluation.trace import _default_tool_call_predicate
from aieng.agent_evals.evaluation.types import Evaluation, TraceEvaluatorFunction, TraceObservationPredicate
from langfuse.api import ScoreDataType
from langfuse.api.resources import ObservationsView
from langfuse.api.resources.commons.types.trace_with_full_details import TraceWithFullDetails
from langfuse.experiment import ExperimentItemResult
from pydantic import BaseModel, Field


DEFAULT_GROUNDEDNESS_SYSTEM_PROMPT = """\
You are a Fact-Checking Judge. Your ONLY function is to verify if the Candidate Output is factually supported by the provided Context.

# Ground Rules
1. **Context is King**: You must ignore your own external knowledge. If a claim is true in the real world but not mentioned in the Context, it is "Unsupported".
2. **Atomic Claims**: Break the Candidate Output into separate, short facts (claims).
3. **Verdict definitions**:
   - **Supported**: The claim is explicitly stated or directly implied by the Context.
   - **Unsupported**: The claim contradicts the Context OR is simply missing from the Context.

{rubric_section}

# Output Schema
Return valid JSON only (no markdown).
{{
  "explanation": "Brief summary of the analysis...",
  "claims": [
    {{
      "text": "The exact claim statement from the candidate.",
      "verdict": "Supported" | "Unsupported",
      "reason": "Quote from Context proving/disproving this."
    }}
  ],
  "score": float (0.0 to 1.0)
}}
"""

DEFAULT_GROUNDEDNESS_USER_PROMPT = """\
# Context (The Source of Truth)
{context}

# Candidate Output (To Verify)
{output}

# Task
1. Extract all verifiable claims from the Candidate Output.
2. Verify each against the Context.
3. Calculate the score as: (Number of Supported Claims) / (Total Claims).
"""

DEFAULT_GROUNDEDNESS_EXCLUDED_TOOL_NAMES: frozenset[str] = frozenset({"set_model_response"})


class TraceGroundednessClaim(BaseModel):
    """Single claim verdict returned by the groundedness judge.

    Parameters
    ----------
    text : str
        Claim text extracted from candidate output.
    verdict : Literal["Supported", "Unsupported"]
        Verdict for the claim against trace evidence context.
    reason : str
        Short rationale citing support or lack of support in context.
    """

    text: str
    verdict: Literal["Supported", "Unsupported"]
    reason: str


class TraceGroundednessResponse(BaseModel):
    """Structured response for trace groundedness judgment.

    Parameters
    ----------
    explanation : str
        Brief reasoning summary for the overall judgment.
    claims : list[TraceGroundednessClaim]
        Claim-level verdicts used for deterministic score computation.
    score : float
        Raw score produced by the judge model in the range ``[0.0, 1.0]``.
    """

    explanation: str
    claims: list[TraceGroundednessClaim]
    score: float = Field(ge=0.0, le=1.0)


def create_trace_groundedness_evaluator(
    *,
    name: str = "trace_groundedness",
    model_config: LLMRequestConfig | None = None,
    system_prompt_template: str = DEFAULT_GROUNDEDNESS_SYSTEM_PROMPT,
    prompt_template: str = DEFAULT_GROUNDEDNESS_USER_PROMPT,
    rubric_markdown: str | Path | None = None,
    error_metric_name: str | None = None,
    max_tool_observations: int = 100,
    max_field_chars: int | None = None,
    max_unsupported_claims_in_metadata: int = 25,
    tool_observation_predicate: TraceObservationPredicate | None = None,
) -> TraceEvaluatorFunction:
    """Create a trace evaluator for output groundedness against tool evidence.

    Parameters
    ----------
    name : str, optional, default="trace_groundedness"
        Logical evaluator name used for diagnostics.
    model_config : LLMRequestConfig | None, optional, default=None
        Model request and retry configuration reused from ``llm_judge``.
    system_prompt_template : str, optional, default=DEFAULT_GROUNDEDNESS_SYSTEM_PROMPT
        System prompt template for the groundedness judge. If it contains
        ``{rubric_section}``, rubric text is inserted at that location;
        otherwise the rubric section is appended to the end.
    prompt_template : str, optional, default=DEFAULT_GROUNDEDNESS_USER_PROMPT
        User prompt template supporting ``{context}`` and ``{output}``.
    rubric_markdown : str | Path | None, optional, default=None
        Optional rubric markdown text or path. This is rendered and injected into
        the system prompt to provide additional guidance to the judge without
        requiring users to fully rewrite the system prompt when customizing
        evaluation guidance.
    error_metric_name : str | None, optional, default=None
        Optional override for deterministic error metric name.
    max_tool_observations : int, optional, default=100
        Maximum number of tool observations to include in prompt context.
        When more are present, the most recent observations are kept.
    max_field_chars : int | None, optional, default=None
        Maximum character length for each serialized tool input/output field.
        Use ``None`` for no truncation.
    max_unsupported_claims_in_metadata : int, optional, default=25
        Maximum number of unsupported claims to include in metric metadata.
    tool_observation_predicate : TraceObservationPredicate | None, optional,
        default=None
        Optional predicate for selecting tool observations. When omitted, a
        groundedness-specific default is used: it keeps tool-like
        observations while excluding framework output-normalization helpers
        such as ``set_model_response`` to avoid target leakage.

    Returns
    -------
    TraceEvaluatorFunction
        Async trace evaluator that emits one groundedness metric or one error metric.

    Raises
    ------
    ValueError
        If the judge returns no claims or if ``max_unsupported_claims_in_metadata`` is
        negative.
    """
    if max_unsupported_claims_in_metadata < 0:
        raise ValueError("``max_unsupported_claims_in_metadata`` must be non-negative.")

    resolved_model_config = model_config or LLMRequestConfig()

    # Load and render rubric text into the system prompt
    rubric = load_markdown(rubric_markdown)
    rendered_system_prompt = render_system_prompt_with_optional_rubric(
        system_prompt_template=system_prompt_template, rubric=rubric
    )

    # Error metric name is deterministic to keep failed evaluations analyzable
    # without dropping traces.
    resolved_error_metric_name = error_metric_name or f"{name}_error"

    async def _evaluator(
        *, trace: TraceWithFullDetails, item_result: ExperimentItemResult, **kwargs: Any
    ) -> Evaluation:
        """Evaluate groundedness for a single trace result."""
        try:
            context_text, tool_observation_count = _build_tool_context(
                trace=trace,
                max_tool_observations=max_tool_observations,
                max_field_chars=max_field_chars,
                tool_observation_predicate=tool_observation_predicate,
            )
            user_prompt = prompt_template.format(context=context_text, output=serialize_for_prompt(item_result.output))

            client_manager = AsyncClientManager.get_instance()
            completion = await run_structured_parse_call(
                openai_client=client_manager.openai_client,
                default_model=client_manager.configs.default_evaluator_model,
                model_config=resolved_model_config,
                system_prompt=rendered_system_prompt,
                user_prompt=user_prompt,
                response_format=TraceGroundednessResponse,
            )

            judge_response: TraceGroundednessResponse | None = completion.choices[0].message.parsed

            return _to_groundedness_evaluation(
                response=judge_response,
                tool_observation_count=tool_observation_count,
                max_unsupported_claims_in_metadata=max_unsupported_claims_in_metadata,
            )
        except Exception as exc:
            # Deterministic error scores keep rows analyzable without dropping traces.
            return build_error_evaluation(name=resolved_error_metric_name, error=exc, prefix="Trace groundedness error")

    _evaluator.__name__ = name
    return _evaluator


def _to_groundedness_evaluation(
    *, response: TraceGroundednessResponse | None, tool_observation_count: int, max_unsupported_claims_in_metadata: int
) -> Evaluation:
    """Convert groundedness judge response to Langfuse evaluation."""
    if response is None:
        raise ValueError("Groundedness judge returned no parsed response.")

    claims = response.claims
    if not claims:
        raise ValueError("Groundedness judge returned no verifiable claims.")

    supported_claims = [claim for claim in claims if claim.verdict == "Supported"]
    unsupported_claims = [claim for claim in claims if claim.verdict == "Unsupported"]
    groundedness_score = len(supported_claims) / len(claims)

    unsupported_claim_metadata = [
        {"text": claim.text, "reason": claim.reason}
        for claim in unsupported_claims[:max_unsupported_claims_in_metadata]
    ]

    metadata: dict[str, Any] = {
        "claim_count": len(claims),
        "supported_claim_count": len(supported_claims),
        "unsupported_claim_count": len(unsupported_claims),
        "tool_observation_count": tool_observation_count,
        "model_score_raw": response.score,
    }
    if unsupported_claim_metadata:
        # Keep unsupported claim detail in metadata only to avoid extra score rows.
        metadata["unsupported_claims"] = unsupported_claim_metadata

    return Evaluation(
        name="groundedness_score",
        value=groundedness_score,
        comment=response.explanation,
        data_type=ScoreDataType.NUMERIC,
        metadata=metadata,
    )


def _build_tool_context(
    *,
    trace: TraceWithFullDetails,
    max_tool_observations: int,
    max_field_chars: int | None,
    tool_observation_predicate: TraceObservationPredicate | None,
) -> tuple[str, int]:
    """Build serialized tool-evidence context from a trace."""
    observations = trace.observations or []
    predicate = tool_observation_predicate or _default_groundedness_tool_observation_predicate
    tool_observations = [observation for observation in observations if predicate(observation)]

    if not tool_observations:
        raise ValueError("No tool observations available for groundedness evaluation.")

    tool_observations.sort(key=_observation_sort_key)
    if len(tool_observations) > max_tool_observations:
        tool_observations = tool_observations[-max_tool_observations:]

    evidence_rows: list[dict[str, Any]] = []
    for observation in tool_observations:
        evidence_rows.append(
            {
                "id": observation.id,
                "type": observation.type,
                "name": observation.name,
                "input": _truncate_text(serialize_for_prompt(observation.input), max_chars=max_field_chars),
                "output": _truncate_text(serialize_for_prompt(observation.output), max_chars=max_field_chars),
            }
        )

    # Only tool evidence is included to avoid contaminating fact checks with
    # model thought text or speculative intermediate generations.
    context_text = serialize_for_prompt({"tool_observations": evidence_rows})
    return context_text, len(tool_observations)


def _default_groundedness_tool_observation_predicate(observation: ObservationsView) -> bool:
    """Default groundedness predicate for selecting evidence-bearing tools.

    This wraps the generic tool-call heuristic and excludes framework
    output-normalization tools that can leak the final model answer into the
    evidence context.
    """
    if not _default_tool_call_predicate(observation):
        return False

    return not _observation_is_excluded_for_groundedness(observation)


def _observation_is_excluded_for_groundedness(observation: ObservationsView) -> bool:
    """Return True when observation should be excluded from groundedness."""
    observation_name = (observation.name or "").strip().lower()
    if observation_name in DEFAULT_GROUNDEDNESS_EXCLUDED_TOOL_NAMES:
        return True

    metadata = observation.metadata
    if not isinstance(metadata, dict):
        return False

    metadata_candidates: list[Any] = [
        metadata.get("tool_name"),
        metadata.get("tool"),
        metadata.get("function_name"),
        metadata.get("function"),
    ]
    for candidate in metadata_candidates:
        if isinstance(candidate, str):
            if candidate.strip().lower() in DEFAULT_GROUNDEDNESS_EXCLUDED_TOOL_NAMES:
                return True
            continue

        if isinstance(candidate, dict):
            nested_name = candidate.get("name")
            if isinstance(nested_name, str) and nested_name.strip().lower() in DEFAULT_GROUNDEDNESS_EXCLUDED_TOOL_NAMES:
                return True

    return False


def _truncate_text(text: str, *, max_chars: int | None) -> str:
    """Truncate text to ``max_chars`` with explicit marker."""
    if max_chars is None:
        return text
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...[truncated]"


def _observation_sort_key(observation: Any) -> str:
    """Return a stable key for chronological observation sorting."""
    start_time = getattr(observation, "start_time", None)
    if start_time is None:
        return ""
    if hasattr(start_time, "isoformat"):
        return start_time.isoformat()
    return str(start_time)


__all__ = [
    "DEFAULT_GROUNDEDNESS_EXCLUDED_TOOL_NAMES",
    "DEFAULT_GROUNDEDNESS_SYSTEM_PROMPT",
    "DEFAULT_GROUNDEDNESS_USER_PROMPT",
    "TraceGroundednessClaim",
    "TraceGroundednessResponse",
    "create_trace_groundedness_evaluator",
]

"""Offline evaluators for the legislative content extraction agent."""

import logging
from typing import Any

from langfuse.experiment import Evaluation

logger = logging.getLogger(__name__)

# The task output is the parsed JSON dict extracted from AgentResponse.text,
# or None if parsing failed.
# {
#   "jurisdiction_code": str,
#   "session_code": str,
#   "chamber_code": str,
#   "measure_type_code": str,
#   "measure_number": str,
#   "title": str,
#   "summary": str,
#   "sponsors": list[str],
#   "sections_affected": list[dict],
# }
EvaluationOutput = dict[str, Any] | None

EXACT_MATCH_FIELDS = [
    "jurisdiction_code",
    "session_code",
    "chamber_code",
    "measure_type_code",
    "measure_number",
    "title"
]


def _exact_match_evaluation(
    field: str,
    output: EvaluationOutput,
    expected_output: EvaluationOutput,
) -> Evaluation:
    """Return an Evaluation for a single field using exact string comparison.

    Parameters
    ----------
    field : str
        The field name to compare.
    output : EvaluationOutput
        The parsed JSON dict produced by the agent, or None.
    expected_output : EvaluationOutput
        The ground-truth dict from the dataset item.

    Returns
    -------
    Evaluation
        value = 1 if exact match, 0 otherwise.
    """
    # --- guard: missing expected value ---
    if expected_output is None or field not in expected_output:
        return Evaluation(
            name=field,
            value=0,
            comment=f"Expected output is missing or does not contain '{field}'.",
        )

    expected_value: str = expected_output[field]

    # --- guard: agent returned nothing parseable ---
    if output is None:
        return Evaluation(
            name=field,
            value=0,
            comment=(
                f"Agent output could not be parsed as JSON. "
                f"Expected '{expected_value}'."
            ),
        )

    # --- guard: field missing from agent output ---
    if field not in output:
        return Evaluation(
            name=field,
            value=0,
            comment=(
                f"Field '{field}' not found in agent output. "
                f"Expected '{expected_value}'."
            ),
        )

    actual_value: str = output[field]
    is_match = actual_value == expected_value

    if is_match:
        comment = f"Exact match: '{actual_value}'."
    else:
        comment = f"Mismatch — expected '{expected_value}', got '{actual_value}'."

    return Evaluation(
        name=field,
        value=1 if is_match else 0,
        comment=comment,
    )


async def exact_match_evaluator(
    *,
    input: str,
    output: EvaluationOutput,
    expected_output: EvaluationOutput,
    **kwargs: Any,
) -> list[Evaluation]:
    """Evaluate key fields using exact string comparison.

    No LLM calls are made. Each field in EXACT_MATCH_FIELDS is compared
    directly (case-sensitive) against the expected value from the dataset.

    Fields evaluated:
        - measure_type_code
        - chamber_code
        - session_code
        - measure_number

    Parameters
    ----------
    input : str
        The original prompt sent to the agent.
    output : EvaluationOutput
        The parsed JSON dict produced by the agent, or None if the agent
        returned unparseable output.
    expected_output : EvaluationOutput
        The ground-truth dict from the Langfuse dataset item.
    **kwargs
        Ignored additional keyword arguments passed by the experiment runner.

    Returns
    -------
    list[Evaluation]
        One Evaluation per field.
        value = 1 if exact match, 0 otherwise.
    """
    return [
        _exact_match_evaluation(field, output, expected_output)
        for field in EXACT_MATCH_FIELDS
    ]

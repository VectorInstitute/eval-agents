"""Tests for the trace groundedness evaluator factory."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.evaluation.graders.trace_groundedness import (
    TraceGroundednessClaim,
    TraceGroundednessResponse,
    create_trace_groundedness_evaluator,
)
from langfuse.api import ScoreDataType
from pydantic import ValidationError


def _completion(parsed_response: TraceGroundednessResponse | None) -> SimpleNamespace:
    """Build a minimal parse-completion object."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed_response))])


@pytest.fixture
def fake_manager(monkeypatch) -> SimpleNamespace:
    """Patch AsyncClientManager singleton for deterministic tests."""
    manager = SimpleNamespace(
        openai_client=object(), configs=SimpleNamespace(default_evaluator_model="gpt-default-evaluator")
    )
    monkeypatch.setattr(
        "aieng.agent_evals.evaluation.graders.trace_groundedness.AsyncClientManager.get_instance", lambda: manager
    )
    return manager


def _make_observation(
    *,
    obs_id: str,
    obs_type: str,
    name: str,
    input_payload: object,
    output_payload: object,
    start_time: datetime,
    metadata: dict[str, object] | None = None,
) -> SimpleNamespace:
    """Build a minimal observation-like object for trace context generation."""
    return SimpleNamespace(
        id=obs_id,
        type=obs_type,
        name=name,
        input=input_payload,
        output=output_payload,
        start_time=start_time,
        metadata=metadata,
    )


def _make_trace(observations: list[SimpleNamespace]) -> SimpleNamespace:
    """Build a minimal trace-like object."""
    return SimpleNamespace(observations=observations)


def _make_item_result(output_payload: object) -> SimpleNamespace:
    """Build a minimal item-result-like object."""
    return SimpleNamespace(output=output_payload)


@pytest.mark.asyncio
async def test_create_trace_groundedness_evaluator_success_wires_parse_call_and_computes_score(
    fake_manager, monkeypatch
) -> None:
    """Compute groundedness score and wire parse call arguments correctly."""
    captured_kwargs: dict[str, object] = {}

    async def fake_parse_call(**kwargs) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        return _completion(
            TraceGroundednessResponse(
                explanation="Most claims are grounded.",
                claims=[
                    TraceGroundednessClaim(text="Claim 1", verdict="Supported", reason="Tool output confirms."),
                    TraceGroundednessClaim(text="Claim 2", verdict="Supported", reason="Search result confirms."),
                    TraceGroundednessClaim(text="Claim 3", verdict="Unsupported", reason="Not in tool evidence."),
                ],
                score=0.42,
            )
        )

    monkeypatch.setattr(
        "aieng.agent_evals.evaluation.graders.trace_groundedness.run_structured_parse_call", fake_parse_call
    )

    evaluator_config = LLMRequestConfig(model="gpt-test-groundedness", temperature=0.0)
    evaluator = create_trace_groundedness_evaluator(
        name="trace_groundedness_custom",
        model_config=evaluator_config,
        rubric_markdown="- Use only tool evidence.",
        max_tool_observations=2,
        max_field_chars=20,
        max_unsupported_claims_in_metadata=1,
    )

    trace = _make_trace(
        observations=[
            _make_observation(
                obs_id="obs-old",
                obs_type="tool_call",
                name="payments_tool",
                input_payload={"query": "old"},
                output_payload={"result": "A" * 120},
                start_time=datetime(2024, 1, 1, 10, 0, 0),
            ),
            _make_observation(
                obs_id="obs-mid",
                obs_type="tool_call",
                name="accounts_tool",
                input_payload={"query": "mid"},
                output_payload={"result": "B" * 120},
                start_time=datetime(2024, 1, 1, 11, 0, 0),
            ),
            _make_observation(
                obs_id="obs-new",
                obs_type="tool_call",
                name="web_tool",
                input_payload={"query": "new"},
                output_payload={"result": "C" * 120},
                start_time=datetime(2024, 1, 1, 12, 0, 0),
            ),
            _make_observation(
                obs_id="obs-excluded",
                obs_type="tool_call",
                name="set_model_response",
                input_payload={"query": "ignore"},
                output_payload={"result": "ignore"},
                start_time=datetime(2024, 1, 1, 13, 0, 0),
            ),
        ]
    )
    item_result = _make_item_result({"answer": "Final answer from agent."})

    evaluation = await evaluator(trace=trace, item_result=item_result)

    assert evaluator.__name__ == "trace_groundedness_custom"
    assert evaluation.name == "groundedness_score"
    assert evaluation.value == pytest.approx(2 / 3)
    assert evaluation.comment == "Most claims are grounded."
    assert evaluation.data_type == ScoreDataType.NUMERIC

    assert evaluation.metadata == {
        "claim_count": 3,
        "supported_claim_count": 2,
        "unsupported_claim_count": 1,
        "tool_observation_count": 2,
        "model_score_raw": 0.42,
        "unsupported_claims": [{"text": "Claim 3", "reason": "Not in tool evidence."}],
    }

    assert captured_kwargs["openai_client"] is fake_manager.openai_client
    assert captured_kwargs["default_model"] == "gpt-default-evaluator"
    assert captured_kwargs["model_config"] is evaluator_config
    assert captured_kwargs["response_format"] is TraceGroundednessResponse
    assert "- Use only tool evidence." in str(captured_kwargs["system_prompt"])

    user_prompt = str(captured_kwargs["user_prompt"])
    assert "...[truncated]" in user_prompt
    assert "set_model_response" not in user_prompt
    assert "payments_tool" not in user_prompt
    assert "accounts_tool" in user_prompt
    assert "web_tool" in user_prompt


@pytest.mark.asyncio
async def test_create_trace_groundedness_evaluator_default_has_no_tool_field_truncation(
    fake_manager, monkeypatch
) -> None:
    """Do not truncate tool fields when ``max_field_chars`` is left as default."""
    captured_kwargs: dict[str, object] = {}
    long_tool_output = "LONG-EVIDENCE-" + ("X" * 200)

    async def fake_parse_call(**kwargs) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        return _completion(
            TraceGroundednessResponse(
                explanation="All claims grounded.",
                claims=[TraceGroundednessClaim(text="Claim 1", verdict="Supported", reason="Evidence present.")],
                score=1.0,
            )
        )

    monkeypatch.setattr(
        "aieng.agent_evals.evaluation.graders.trace_groundedness.run_structured_parse_call", fake_parse_call
    )

    evaluator = create_trace_groundedness_evaluator()
    trace = _make_trace(
        observations=[
            _make_observation(
                obs_id="obs-1",
                obs_type="tool_call",
                name="search_tool",
                input_payload={"query": "evidence"},
                output_payload={"result": long_tool_output},
                start_time=datetime(2024, 1, 1, 12, 0, 0),
            )
        ]
    )

    await evaluator(trace=trace, item_result=_make_item_result({"answer": "candidate"}))

    user_prompt = str(captured_kwargs["user_prompt"])
    assert "...[truncated]" not in user_prompt
    assert long_tool_output in user_prompt
    assert captured_kwargs["openai_client"] is fake_manager.openai_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "error_metric_name", "expected_error_type", "expected_metric_name", "expect_parse_called"),
    [
        (
            "no_tool_observations",
            None,
            "ValueError",
            "trace_groundedness_test_error",
            False,
        ),
        (
            "parse_runtime_error",
            "custom_trace_groundedness_error",
            "RuntimeError",
            "custom_trace_groundedness_error",
            True,
        ),
        (
            "empty_claims_response",
            None,
            "ValueError",
            "trace_groundedness_test_error",
            True,
        ),
    ],
)
async def test_create_trace_groundedness_evaluator_error_paths_return_deterministic_error_metric(
    fake_manager,
    monkeypatch,
    scenario: str,
    error_metric_name: str | None,
    expected_error_type: str,
    expected_metric_name: str,
    expect_parse_called: bool,
) -> None:
    """Return deterministic error metrics for context, parse, and response failures."""
    del fake_manager

    if scenario == "parse_runtime_error":
        parse_mock = AsyncMock(side_effect=RuntimeError("judge service unavailable"))
    elif scenario == "empty_claims_response":
        parse_mock = AsyncMock(
            return_value=_completion(TraceGroundednessResponse(explanation="No claims", claims=[], score=0.0))
        )
    else:
        parse_mock = AsyncMock(return_value=_completion(None))

    monkeypatch.setattr("aieng.agent_evals.evaluation.graders.trace_groundedness.run_structured_parse_call", parse_mock)

    evaluator = create_trace_groundedness_evaluator(name="trace_groundedness_test", error_metric_name=error_metric_name)

    if scenario == "no_tool_observations":
        trace = _make_trace(observations=[])
    else:
        trace = _make_trace(
            observations=[
                _make_observation(
                    obs_id="obs-tool",
                    obs_type="tool_call",
                    name="search_tool",
                    input_payload={"query": "evidence"},
                    output_payload={"result": "found"},
                    start_time=datetime(2024, 1, 1, 12, 0, 0),
                )
            ]
        )

    evaluation = await evaluator(trace=trace, item_result=_make_item_result({"answer": "candidate"}))

    assert evaluation.name == expected_metric_name
    assert evaluation.value is True
    assert str(evaluation.comment).startswith("Trace groundedness error: ")
    assert evaluation.metadata["error_type"] == expected_error_type

    if expect_parse_called:
        parse_mock.assert_awaited_once()
    else:
        parse_mock.assert_not_awaited()


def test_trace_groundedness_models_validate_bounds_and_literals() -> None:
    """Validate literal verdicts and score bounds for public models."""
    claim = TraceGroundednessClaim(text="A supported claim", verdict="Supported", reason="Present in tool context")
    response_low = TraceGroundednessResponse(explanation="ok", claims=[claim], score=0.0)
    response_high = TraceGroundednessResponse(explanation="ok", claims=[claim], score=1.0)

    assert claim.verdict == "Supported"
    assert response_low.score == 0.0
    assert response_high.score == 1.0

    with pytest.raises(ValidationError):
        TraceGroundednessClaim(text="bad", verdict="Unknown", reason="invalid literal")

    with pytest.raises(ValidationError):
        TraceGroundednessResponse(explanation="bad", claims=[claim], score=1.1)

    with pytest.raises(ValueError, match="must be non-negative"):
        create_trace_groundedness_evaluator(max_unsupported_claims_in_metadata=-1)

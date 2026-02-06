"""Tests for the LLM-as-a-judge evaluator factory."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aieng.agent_evals.evaluation import graders as graders_package
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.evaluation.graders.llm_judge import (
    DEFAULT_LLM_JUDGE_RUBRIC,
    LLMJudgeMetric,
    LLMJudgeResponse,
    _to_evaluations,
    make_llm_as_judge_evaluator,
)
from pydantic import ValidationError


def _completion(parsed_response: LLMJudgeResponse | None) -> SimpleNamespace:
    """Build a minimal parse-completion object."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed_response))])


@pytest.fixture
def fake_manager(monkeypatch) -> SimpleNamespace:
    """Patch AsyncClientManager singleton for deterministic tests."""
    manager = SimpleNamespace(
        openai_client=object(), configs=SimpleNamespace(default_evaluator_model="gpt-default-evaluator")
    )
    monkeypatch.setattr(
        "aieng.agent_evals.evaluation.graders.llm_judge.AsyncClientManager.get_instance", lambda: manager
    )
    return manager


@pytest.mark.asyncio
async def test_make_evaluator_success_with_custom_rubric_maps_and_wires_calls(fake_manager, monkeypatch) -> None:
    """Map metrics correctly and pass expected parse call arguments."""
    captured_kwargs: dict[str, object] = {}

    async def fake_parse_call(**kwargs) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        return _completion(
            LLMJudgeResponse(
                explanation="Global explanation",
                metrics=[
                    LLMJudgeMetric(
                        name="accuracy", value=1, comment=None, confidence=0.9, metadata={"source": "judge"}
                    ),
                    LLMJudgeMetric(
                        name="style_ok", value=True, comment="Clear and concise.", confidence=None, metadata=None
                    ),
                ],
            )
        )

    monkeypatch.setattr("aieng.agent_evals.evaluation.graders.llm_judge.run_structured_parse_call", fake_parse_call)

    config = LLMRequestConfig(model="gpt-test-judge", temperature=0.0)
    evaluator = make_llm_as_judge_evaluator(
        name="quality_judge", model_config=config, rubric_markdown="- Reward factual correctness."
    )

    evaluations = await evaluator(
        input={"question": "What is the capital of France?"},
        output={"answer": "Paris"},
        expected_output={"answer": "Paris"},
        metadata={"dataset": "qa"},
    )

    assert evaluator.__name__ == "quality_judge"
    assert len(evaluations) == 2

    first_eval = evaluations[0]
    assert first_eval.name == "accuracy"
    assert first_eval.value == 1
    assert first_eval.comment == "Global explanation"
    assert first_eval.metadata == {"source": "judge", "confidence": 0.9}

    second_eval = evaluations[1]
    assert second_eval.name == "style_ok"
    assert second_eval.value is True
    assert second_eval.comment == "Clear and concise."
    assert second_eval.metadata is None

    assert captured_kwargs["openai_client"] is fake_manager.openai_client
    assert captured_kwargs["default_model"] == "gpt-default-evaluator"
    assert captured_kwargs["model_config"] is config
    assert captured_kwargs["response_format"] is LLMJudgeResponse
    assert "- Reward factual correctness." in str(captured_kwargs["system_prompt"])

    user_prompt = str(captured_kwargs["user_prompt"])
    assert "# Input" in user_prompt
    assert "# Expected Output" in user_prompt
    assert "# Candidate Output (To Evaluate)" in user_prompt
    assert '"question": "What is the capital of France?"' in user_prompt


@pytest.mark.asyncio
async def test_make_evaluator_uses_default_rubric_when_none(fake_manager, monkeypatch) -> None:
    """Inject DEFAULT_LLM_JUDGE_RUBRIC when rubric_markdown is omitted."""
    captured_kwargs: dict[str, object] = {}

    async def fake_parse_call(**kwargs) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        return _completion(
            LLMJudgeResponse(
                explanation="Free-form metric names are still passed through.",
                metrics=[LLMJudgeMetric(name="custom_metric", value=1, comment="ok")],
            )
        )

    monkeypatch.setattr("aieng.agent_evals.evaluation.graders.llm_judge.run_structured_parse_call", fake_parse_call)

    evaluator = make_llm_as_judge_evaluator(name="default_rubric")
    evaluations = await evaluator(
        input={"prompt": "hello"},
        output={"answer": "world"},
        expected_output={"answer": "world"},
        metadata=None,
    )

    assert evaluations[0].name == "custom_metric"
    assert DEFAULT_LLM_JUDGE_RUBRIC.strip() in str(captured_kwargs["system_prompt"])
    assert graders_package.DEFAULT_LLM_JUDGE_RUBRIC == DEFAULT_LLM_JUDGE_RUBRIC


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "error_metric_name", "expected_error_type", "expected_metric_name", "expect_parse_called"),
    [
        ("parse_error", None, "RuntimeError", "quality_judge_error", True),
        ("prompt_template_key_error", "custom_error_metric", "KeyError", "custom_error_metric", False),
    ],
)
async def test_make_evaluator_error_paths_return_deterministic_error_metric(
    fake_manager,
    monkeypatch,
    scenario: str,
    error_metric_name: str | None,
    expected_error_type: str,
    expected_metric_name: str,
    expect_parse_called: bool,
) -> None:
    """Return deterministic error metrics for parser and prompt formatting failures."""
    del fake_manager
    parse_mock = AsyncMock(side_effect=RuntimeError("judge service unavailable"))
    monkeypatch.setattr("aieng.agent_evals.evaluation.graders.llm_judge.run_structured_parse_call", parse_mock)

    if scenario == "parse_error":
        evaluator = make_llm_as_judge_evaluator(
            name="quality_judge", model_config=None, error_metric_name=error_metric_name
        )
    else:
        evaluator = make_llm_as_judge_evaluator(
            name="quality_judge",
            model_config=LLMRequestConfig(),
            prompt_template="Broken template: {missing_required_key}",
            error_metric_name=error_metric_name,
        )

    evaluations = await evaluator(
        input={"prompt": "hello"},
        output={"answer": "world"},
        expected_output={"answer": "world"},
        metadata=None,
    )

    assert len(evaluations) == 1
    error_eval = evaluations[0]
    assert error_eval.name == expected_metric_name
    assert error_eval.value is True
    assert error_eval.comment.startswith("LLM judge error: ")
    assert error_eval.metadata["error_type"] == expected_error_type

    if scenario == "parse_error":
        assert isinstance(parse_mock.await_args.kwargs["model_config"], LLMRequestConfig)

    if expect_parse_called:
        parse_mock.assert_awaited_once()
    else:
        parse_mock.assert_not_awaited()


@pytest.mark.parametrize("response", [None, LLMJudgeResponse(explanation="No metrics", metrics=[])])
def test_to_evaluations_rejects_missing_metrics(response: LLMJudgeResponse | None) -> None:
    """Reject parsed responses with no metrics."""
    with pytest.raises(ValueError, match="must contain at least one metric"):
        _to_evaluations(response)


def test_llm_judge_metric_confidence_validation_bounds() -> None:
    """Accept confidence at boundaries and reject out-of-range values."""
    low = LLMJudgeMetric(name="score", value=1, confidence=0.0)
    high = LLMJudgeMetric(name="score", value=1, confidence=1.0)

    assert low.confidence == 0.0
    assert high.confidence == 1.0

    with pytest.raises(ValidationError):
        LLMJudgeMetric(name="score", value=1, confidence=1.1)

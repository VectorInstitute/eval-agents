"""Tests for token usage tracking."""

from unittest.mock import MagicMock, patch

import pytest
from aieng.agent_evals.knowledge_qa.agent import KnowledgeGroundedAgent
from aieng.agent_evals.knowledge_qa.token_tracker import (
    DEFAULT_MODEL,
    KNOWN_MODEL_LIMITS,
    TokenTracker,
    TokenUsage,
)
from google.adk.events import Event
from google.genai import types


def _make_event(
    prompt: int = 0,
    completion: int = 0,
    total: int = 0,
    cached: int = 0,
) -> Event:
    """Build an ADK Event carrying usage_metadata."""
    return Event(
        author="model",
        usageMetadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt,
            candidates_token_count=completion,
            total_token_count=total,
            cached_content_token_count=cached,
        ),
    )


def _make_tracker(model: str = "gemini-2.5-flash", context_limit: int = 1_000_000) -> TokenTracker:
    """Build a TokenTracker with a mocked API call so no network I/O occurs."""
    mock_model_info = MagicMock()
    mock_model_info.input_token_limit = context_limit
    mock_client = MagicMock()
    mock_client.models.get.return_value = mock_model_info

    with patch("aieng.agent_evals.knowledge_qa.token_tracker.Client", return_value=mock_client):
        return TokenTracker(model=model)


# =============================================================================
# TokenUsage model
# =============================================================================


class TestTokenUsage:
    """Tests for the TokenUsage Pydantic model."""

    def test_defaults(self):
        """Test all fields start at zero with a sensible context_limit default."""
        usage = TokenUsage()
        assert usage.latest_prompt_tokens == 0
        assert usage.latest_cached_tokens == 0
        assert usage.total_prompt_tokens == 0
        assert usage.total_completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.context_limit == 1_000_000

    def test_context_used_percent_proportional(self):
        """Test context_used_percent equals prompt / limit * 100."""
        usage = TokenUsage(latest_prompt_tokens=250_000, context_limit=1_000_000)
        assert usage.context_used_percent == pytest.approx(25.0)

    def test_context_used_percent_full(self):
        """Test context_used_percent is 100 when prompt equals limit."""
        usage = TokenUsage(latest_prompt_tokens=500_000, context_limit=500_000)
        assert usage.context_used_percent == pytest.approx(100.0)

    def test_context_used_percent_zero_limit(self):
        """Test context_used_percent is 0.0 when context_limit is zero."""
        usage = TokenUsage(latest_prompt_tokens=1000, context_limit=0)
        assert usage.context_used_percent == 0.0

    def test_context_remaining_percent_complements_used(self):
        """Test context_remaining_percent sums to 100 with context_used_percent."""
        usage = TokenUsage(latest_prompt_tokens=300_000, context_limit=1_000_000)
        assert usage.context_used_percent + usage.context_remaining_percent == pytest.approx(100.0)

    def test_context_remaining_percent_clamps_at_zero(self):
        """Test context_remaining_percent never goes negative when over limit."""
        usage = TokenUsage(latest_prompt_tokens=2_000_000, context_limit=1_000_000)
        assert usage.context_remaining_percent == 0.0

    def test_context_remaining_percent_zero_limit(self):
        """Test context_remaining_percent is 100 when limit is zero (used = 0%)."""
        usage = TokenUsage(context_limit=0)
        assert usage.context_remaining_percent == pytest.approx(100.0)


# =============================================================================
# TokenTracker initialisation & model limit fetching
# =============================================================================


class TestTokenTrackerInit:
    """Tests for TokenTracker initialisation and model limit resolution."""

    def test_uses_api_limit_when_available(self):
        """Test the context limit is taken from the API when it succeeds."""
        tracker = _make_tracker(model="gemini-2.5-pro", context_limit=2_000_000)
        assert tracker.usage.context_limit == 2_000_000

    def test_falls_back_to_known_limit_on_api_error(self):
        """Test falls back to KNOWN_MODEL_LIMITS when the API call raises."""
        with patch("aieng.agent_evals.knowledge_qa.token_tracker.Client", side_effect=Exception("network error")):
            tracker = TokenTracker(model="gemini-2.5-flash")
        assert tracker.usage.context_limit == KNOWN_MODEL_LIMITS["gemini-2.5-flash"]

    def test_falls_back_to_default_for_unknown_model(self):
        """Test uses TokenUsage default limit for a model not in KNOWN_MODEL_LIMITS."""
        with patch("aieng.agent_evals.knowledge_qa.token_tracker.Client", side_effect=Exception("network error")):
            tracker = TokenTracker(model="gemini-unknown-model")
        assert tracker.usage.context_limit == 1_000_000  # TokenUsage default

    def test_api_client_is_closed_after_successful_fetch(self):
        """Test the Google API client is always closed after a successful fetch."""
        mock_model_info = MagicMock()
        mock_model_info.input_token_limit = 500_000
        mock_client = MagicMock()
        mock_client.models.get.return_value = mock_model_info

        with patch("aieng.agent_evals.knowledge_qa.token_tracker.Client", return_value=mock_client):
            TokenTracker(model="gemini-2.5-flash")

        mock_client.close.assert_called_once()

    def test_api_client_is_closed_after_failed_fetch(self):
        """Test the Google API client is closed even when models.get raises."""
        mock_client = MagicMock()
        mock_client.models.get.side_effect = Exception("timeout")

        with patch("aieng.agent_evals.knowledge_qa.token_tracker.Client", return_value=mock_client):
            TokenTracker(model="gemini-2.5-flash")

        mock_client.close.assert_called_once()

    def test_uses_default_model_when_none_given(self):
        """Test the model defaults to DEFAULT_MODEL when none is provided."""
        mock_model_info = MagicMock()
        mock_model_info.input_token_limit = 1_000_000
        mock_client = MagicMock()
        mock_client.models.get.return_value = mock_model_info

        with patch("aieng.agent_evals.knowledge_qa.token_tracker.Client", return_value=mock_client):
            tracker = TokenTracker()

        assert tracker._model == DEFAULT_MODEL

    def test_api_none_input_token_limit_falls_back_to_known(self):
        """Test falls back to KNOWN_MODEL_LIMITS if API returns None for token limit."""
        mock_model_info = MagicMock()
        mock_model_info.input_token_limit = None
        mock_client = MagicMock()
        mock_client.models.get.return_value = mock_model_info

        with patch("aieng.agent_evals.knowledge_qa.token_tracker.Client", return_value=mock_client):
            tracker = TokenTracker(model="gemini-2.5-flash")

        assert tracker.usage.context_limit == KNOWN_MODEL_LIMITS["gemini-2.5-flash"]

    def test_initial_usage_all_zero(self):
        """Test all token counts start at zero after initialisation."""
        tracker = _make_tracker()
        u = tracker.usage
        assert u.latest_prompt_tokens == 0
        assert u.latest_cached_tokens == 0
        assert u.total_prompt_tokens == 0
        assert u.total_completion_tokens == 0
        assert u.total_tokens == 0


# =============================================================================
# add_from_event
# =============================================================================


class TestAddFromEvent:
    """Tests for TokenTracker.add_from_event."""

    def test_ignores_event_without_usage_metadata(self):
        """Test that events without usage_metadata leave all counts at zero."""
        tracker = _make_tracker()
        event = Event(author="model")
        tracker.add_from_event(event)
        assert tracker.usage.total_tokens == 0

    def test_ignores_event_with_none_usage_metadata(self):
        """Test that an event with usage_metadata=None leaves all counts at zero."""
        tracker = _make_tracker()
        event = MagicMock(spec=[])  # no usage_metadata attribute at all
        tracker.add_from_event(event)
        assert tracker.usage.total_tokens == 0

    def test_single_event_sets_latest_and_totals(self):
        """Test a single event correctly populates both latest and cumulative fields."""
        tracker = _make_tracker()
        tracker.add_from_event(_make_event(prompt=100, completion=50, total=150, cached=10))
        u = tracker.usage
        assert u.latest_prompt_tokens == 100
        assert u.latest_cached_tokens == 10
        assert u.total_prompt_tokens == 100
        assert u.total_completion_tokens == 50
        assert u.total_tokens == 150

    def test_latest_tokens_reflect_most_recent_event(self):
        """Test latest_* fields are overwritten (not accumulated) on each event."""
        tracker = _make_tracker()
        tracker.add_from_event(_make_event(prompt=100, completion=40, total=140, cached=5))
        tracker.add_from_event(_make_event(prompt=200, completion=60, total=260, cached=20))
        u = tracker.usage
        # latest should reflect only the second event
        assert u.latest_prompt_tokens == 200
        assert u.latest_cached_tokens == 20

    def test_totals_accumulate_across_events(self):
        """Test total_* fields accumulate across multiple events."""
        tracker = _make_tracker()
        tracker.add_from_event(_make_event(prompt=100, completion=40, total=140))
        tracker.add_from_event(_make_event(prompt=200, completion=60, total=260))
        u = tracker.usage
        assert u.total_prompt_tokens == 300
        assert u.total_completion_tokens == 100
        assert u.total_tokens == 400

    def test_none_field_values_treated_as_zero(self):
        """Test that None values in usage_metadata fields are treated as zero."""
        tracker = _make_tracker()
        # Build an event whose metadata returns None for every attribute
        mock_meta = MagicMock()
        mock_meta.prompt_token_count = None
        mock_meta.cached_content_token_count = None
        mock_meta.candidates_token_count = None
        mock_meta.total_token_count = None

        event = MagicMock()
        event.usage_metadata = mock_meta
        tracker.add_from_event(event)

        u = tracker.usage
        assert u.latest_prompt_tokens == 0
        assert u.latest_cached_tokens == 0
        assert u.total_completion_tokens == 0
        assert u.total_tokens == 0

    def test_context_used_percent_updates_after_event(self):
        """Test context_used_percent reflects latest prompt tokens after an event."""
        tracker = _make_tracker(context_limit=1_000_000)
        tracker.add_from_event(_make_event(prompt=500_000, total=500_000))
        assert tracker.usage.context_used_percent == pytest.approx(50.0)


# =============================================================================
# reset
# =============================================================================


class TestTokenTrackerReset:
    """Tests for TokenTracker.reset."""

    def test_reset_clears_all_counts(self):
        """Test reset brings all token counts back to zero."""
        tracker = _make_tracker(context_limit=1_048_576)
        tracker.add_from_event(_make_event(prompt=100, completion=50, total=150, cached=5))
        tracker.reset()
        u = tracker.usage
        assert u.latest_prompt_tokens == 0
        assert u.latest_cached_tokens == 0
        assert u.total_prompt_tokens == 0
        assert u.total_completion_tokens == 0
        assert u.total_tokens == 0

    def test_reset_preserves_context_limit(self):
        """Test reset keeps the context_limit that was fetched at initialisation."""
        tracker = _make_tracker(context_limit=1_048_576)
        tracker.add_from_event(_make_event(prompt=100, total=100))
        tracker.reset()
        assert tracker.usage.context_limit == 1_048_576

    def test_accumulation_continues_correctly_after_reset(self):
        """Test token counts accumulate normally after a reset."""
        tracker = _make_tracker()
        tracker.add_from_event(_make_event(prompt=100, completion=40, total=140))
        tracker.reset()
        tracker.add_from_event(_make_event(prompt=200, completion=60, total=260))
        u = tracker.usage
        assert u.total_prompt_tokens == 200
        assert u.total_completion_tokens == 60
        assert u.total_tokens == 260


# =============================================================================
# Integration test — real Gemini model
# =============================================================================


@pytest.mark.integration_test
class TestTokenTrackerIntegration:
    """Integration tests that validate token tracking against the live Gemini API.

    Requires GOOGLE_API_KEY to be set in the environment / .env file.
    Run with: cd aieng-eval-agents &&
    uv run --env-file ../.env pytest -m integration_test tests -v
    """

    def test_fetch_model_limits_from_real_api(self):
        """Test that _fetch_model_limits contacts the real API and returns a limit."""
        tracker = TokenTracker(model="gemini-2.5-flash")
        assert tracker.usage.context_limit > 0

    @pytest.mark.asyncio
    async def test_agent_populates_token_tracker_after_answer(self):
        """Test that running a real agent query results in non-zero token counts.

        This end-to-end test exercises the full path:
          Agent.answer_async()
            -> Runner emits Event(usageMetadata=...)
              -> _process_event()
                -> TokenTracker.add_from_event()
                  -> usage fields updated
        """
        agent = KnowledgeGroundedAgent(enable_planning=False, enable_caching=False, enable_compaction=False)
        await agent.answer_async("What is the capital of France?")

        usage = agent.token_tracker.usage
        # The API must have returned prompt tokens for at least one call
        assert usage.total_prompt_tokens > 0, "expected prompt tokens to be tracked"
        assert usage.total_completion_tokens > 0, "expected completion tokens to be tracked"
        assert usage.total_tokens > 0, "expected total tokens to be tracked"

        # latest_prompt_tokens should equal the last event's prompt count, which is
        # non-zero for any model call; it should also be <= total
        assert usage.latest_prompt_tokens > 0
        assert usage.latest_prompt_tokens <= usage.total_prompt_tokens

    @pytest.mark.asyncio
    async def test_context_used_percent_is_sensible_after_answer(self):
        """Test that context_used_percent is a small positive fraction after a query."""
        agent = KnowledgeGroundedAgent(enable_planning=False, enable_caching=False, enable_compaction=False)
        await agent.answer_async("What is 2 + 2?")

        usage = agent.token_tracker.usage
        # A short query must use some context but nowhere near the full window
        assert 0.0 < usage.context_used_percent < 50.0
        assert usage.context_remaining_percent == pytest.approx(100.0 - usage.context_used_percent)

    @pytest.mark.asyncio
    async def test_reset_clears_tracking_between_agent_calls(self):
        """Test that agent.reset() zeroes the token tracker before a conversation."""
        agent = KnowledgeGroundedAgent(enable_planning=False, enable_caching=False, enable_compaction=False)
        await agent.answer_async("What is the capital of France?")

        assert agent.token_tracker.usage.total_tokens > 0

        agent.reset()

        assert agent.token_tracker.usage.total_prompt_tokens == 0
        assert agent.token_tracker.usage.total_completion_tokens == 0
        assert agent.token_tracker.usage.total_tokens == 0
        # Context limit must survive the reset
        assert agent.token_tracker.usage.context_limit > 0

    @pytest.mark.asyncio
    async def test_second_call_accumulates_on_top_of_first(self):
        """Test that totals accumulate across two successive answer_async calls."""
        agent = KnowledgeGroundedAgent(enable_planning=False, enable_caching=False, enable_compaction=False)

        await agent.answer_async("What is the capital of France?")
        tokens_after_first = agent.token_tracker.usage.total_tokens

        await agent.answer_async("What is the capital of Germany?")
        tokens_after_second = agent.token_tracker.usage.total_tokens

        assert tokens_after_second > tokens_after_first, "total_tokens should grow after a second answer_async call"

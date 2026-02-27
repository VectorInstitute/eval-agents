"""Tests for retry configuration and error handling."""

from unittest.mock import patch

from aieng.agent_evals.knowledge_qa.retry import (
    API_RETRY_INITIAL_WAIT,
    API_RETRY_JITTER,
    API_RETRY_MAX_ATTEMPTS,
    API_RETRY_MAX_WAIT,
    MAX_EMPTY_RESPONSE_RETRIES,
    is_context_overflow_error,
    is_retryable_api_error,
)


class FakeClientError(Exception):
    """Fake ClientError for testing isinstance checks without API credentials."""


class TestRetryConstants:
    """Tests for retry configuration constants."""

    def test_max_empty_response_retries(self):
        """Test MAX_EMPTY_RESPONSE_RETRIES constant value."""
        assert MAX_EMPTY_RESPONSE_RETRIES == 2

    def test_api_retry_max_attempts(self):
        """Test API_RETRY_MAX_ATTEMPTS constant value."""
        assert API_RETRY_MAX_ATTEMPTS == 5

    def test_api_retry_initial_wait(self):
        """Test API_RETRY_INITIAL_WAIT constant value in seconds."""
        assert API_RETRY_INITIAL_WAIT == 1

    def test_api_retry_max_wait(self):
        """Test API_RETRY_MAX_WAIT constant value in seconds."""
        assert API_RETRY_MAX_WAIT == 60

    def test_api_retry_jitter(self):
        """Test API_RETRY_JITTER constant value in seconds."""
        assert API_RETRY_JITTER == 5


class TestIsRetryableApiError:
    """Tests for the is_retryable_api_error function."""

    def test_returns_true_for_429_error(self):
        """Test returns True when error message contains '429'."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_retryable_api_error(FakeClientError("Error 429: Too Many Requests")) is True

    def test_returns_true_for_resource_exhausted(self):
        """Test returns True when error message contains 'resource_exhausted'."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_retryable_api_error(FakeClientError("RESOURCE_EXHAUSTED: API limit hit")) is True

    def test_returns_true_for_quota_error(self):
        """Test returns True when error message contains 'quota'."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_retryable_api_error(FakeClientError("quota limit reached for this project")) is True

    def test_returns_true_for_mixed_case_429(self):
        """Test case-insensitive match for rate limit errors containing '429'."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_retryable_api_error(FakeClientError("Rate limit error: status=429")) is True

    def test_returns_true_for_mixed_case_resource_exhausted(self):
        """Test case-insensitive match for RESOURCE_EXHAUSTED errors."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_retryable_api_error(FakeClientError("resource_exhausted quota for gemini")) is True

    def test_returns_true_for_mixed_case_quota(self):
        """Test case-insensitive match for QUOTA errors."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_retryable_api_error(FakeClientError("QUOTA_EXCEEDED for this project")) is True

    def test_returns_false_for_token_count_exceeds(self):
        """Test returns False for context overflow 'token count exceeds'."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_retryable_api_error(FakeClientError("Token count exceeds the context window")) is False

    def test_returns_false_for_invalid_argument_with_token(self):
        """Test returns False for invalid_argument errors involving tokens."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_retryable_api_error(FakeClientError("INVALID_ARGUMENT: token limit exceeded")) is False

    def test_returns_false_for_cache_expired(self):
        """Test returns False for cache expiration errors."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_retryable_api_error(FakeClientError("Cache has expired for this request")) is False

    def test_returns_false_for_non_client_error_with_rate_limit_text(self):
        """Test returns False for non-ClientError even with rate limit keywords."""
        assert is_retryable_api_error(ValueError("rate limit 429")) is False

    def test_returns_false_for_base_exception(self):
        """Test returns False for plain BaseException."""
        assert is_retryable_api_error(Exception("quota resource_exhausted")) is False

    def test_returns_false_for_runtime_error(self):
        """Test returns False for RuntimeError with rate limit text."""
        assert is_retryable_api_error(RuntimeError("resource_exhausted quota exceeded")) is False

    def test_returns_false_for_other_client_error(self):
        """Test returns False for ClientError without any retryable keywords."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_retryable_api_error(FakeClientError("Bad request: unknown field")) is False

    def test_returns_false_for_client_error_with_token_no_rate_limit(self):
        """Test returns False for ClientError with 'token' but no rate limit."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_retryable_api_error(FakeClientError("token refresh failed")) is False

    def test_context_overflow_takes_precedence_over_rate_limit(self):
        """Test context overflow early-exit occurs before rate limit check."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            # Message matches both context overflow and rate limit patterns
            error = FakeClientError("token count exceeds limit, status 429 quota")
            assert is_retryable_api_error(error) is False

    def test_cache_expired_takes_precedence_over_rate_limit(self):
        """Test cache expiration early-exit occurs before rate limit check."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            # Message matches both cache expiration and rate limit patterns
            error = FakeClientError("cache expired and quota resource_exhausted")
            assert is_retryable_api_error(error) is False

    def test_invalid_argument_without_token_does_not_block_rate_limit(self):
        """Test invalid_argument without 'token' does not suppress rate limit retry."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            # no "token" → not context overflow → falls through to rate limit
            error = FakeClientError("INVALID_ARGUMENT: bad request quota 429")
            assert is_retryable_api_error(error) is True


class TestIsContextOverflowError:
    """Tests for the is_context_overflow_error function."""

    def test_returns_true_for_token_count_exceeds(self):
        """Test returns True when error message contains 'token count exceeds'."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_context_overflow_error(FakeClientError("Token count exceeds the context window")) is True

    def test_returns_true_for_invalid_argument_with_token(self):
        """Test returns True for invalid_argument errors with 'token' in message."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_context_overflow_error(FakeClientError("INVALID_ARGUMENT: token limit exceeded")) is True

    def test_returns_true_for_mixed_case_token_count_exceeds(self):
        """Test case-insensitive match for 'TOKEN COUNT EXCEEDS'."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_context_overflow_error(FakeClientError("TOKEN COUNT EXCEEDS maximum limit")) is True

    def test_returns_true_for_mixed_case_invalid_argument_token(self):
        """Test case-insensitive match for INVALID_ARGUMENT + TOKEN."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_context_overflow_error(FakeClientError("INVALID_ARGUMENT: TOKEN limit exceeded")) is True

    def test_returns_false_for_rate_limit_429(self):
        """Test returns False for 429 rate limit errors."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_context_overflow_error(FakeClientError("429 Too Many Requests")) is False

    def test_returns_false_for_resource_exhausted(self):
        """Test returns False for resource exhausted errors."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_context_overflow_error(FakeClientError("RESOURCE_EXHAUSTED: quota exceeded")) is False

    def test_returns_false_for_cache_expired(self):
        """Test returns False for cache expiration errors."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_context_overflow_error(FakeClientError("cache has expired")) is False

    def test_returns_false_for_invalid_argument_without_token(self):
        """Test returns False when 'invalid_argument' present but 'token' is absent."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_context_overflow_error(FakeClientError("INVALID_ARGUMENT: bad field value")) is False

    def test_returns_false_for_token_without_invalid_argument_or_token_count_exceeds(self):
        """Test returns False when 'token' appears alone without matching patterns."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_context_overflow_error(FakeClientError("token refresh failed")) is False

    def test_returns_false_for_non_client_error_with_overflow_text(self):
        """Test returns False for non-ClientError even with overflow keywords."""
        assert is_context_overflow_error(ValueError("token count exceeds limit")) is False

    def test_returns_false_for_base_exception(self):
        """Test returns False for plain Exception with context overflow text."""
        assert is_context_overflow_error(Exception("token count exceeds")) is False

    def test_returns_false_for_other_client_error(self):
        """Test returns False for ClientError without context overflow indicators."""
        with patch("aieng.agent_evals.knowledge_qa.retry.ClientError", FakeClientError):
            assert is_context_overflow_error(FakeClientError("Internal server error occurred")) is False

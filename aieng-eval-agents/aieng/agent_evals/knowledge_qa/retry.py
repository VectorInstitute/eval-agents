"""Retry configuration and error handling for API calls.

This module provides retry logic for handling rate limits, quota exhaustion,
and context overflow errors when interacting with the Gemini API.
"""

from google.genai.errors import ClientError


# Max retries for empty model responses
MAX_EMPTY_RESPONSE_RETRIES = 2

# API retry configuration for rate limit and quota exhaustion
API_RETRY_MAX_ATTEMPTS = 5
API_RETRY_INITIAL_WAIT = 1  # seconds
API_RETRY_MAX_WAIT = 60  # seconds
API_RETRY_JITTER = 5  # seconds


def is_retryable_api_error(exception: BaseException) -> bool:
    """Check if an exception is a retryable API error (rate limit/quota exhaustion).

    Does NOT retry context overflow or cache expiration - those need session
    reset instead.

    Parameters
    ----------
    exception : BaseException
        The exception to check.

    Returns
    -------
    bool
        True if the exception should trigger a retry (429/RESOURCE_EXHAUSTED errors).
    """
    if isinstance(exception, ClientError):
        error_str = str(exception).lower()

        # Don't retry context overflow - needs session reset, not retry
        if "token count exceeds" in error_str or ("invalid_argument" in error_str and "token" in error_str):
            return False

        # Don't retry cache expiration - needs session reset, not retry
        if "cache" in error_str and "expired" in error_str:
            return False

        # Check for rate limit indicators
        if "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str:
            return True
    return False


def is_context_overflow_error(exception: BaseException) -> bool:
    """Check if an exception is a context overflow error.

    Parameters
    ----------
    exception : BaseException
        The exception to check.

    Returns
    -------
    bool
        True if the exception is due to context window overflow.
    """
    if isinstance(exception, ClientError):
        error_str = str(exception).lower()
        return "token count exceeds" in error_str or ("invalid_argument" in error_str and "token" in error_str)
    return False


def is_cache_expiration_error(exception: BaseException) -> bool:
    """Check if an exception is a cache expiration error.

    Parameters
    ----------
    exception : BaseException
        The exception to check.

    Returns
    -------
    bool
        True if the exception is due to expired cache content.
    """
    if isinstance(exception, ClientError):
        error_str = str(exception).lower()
        return "cache" in error_str and "expired" in error_str
    return False

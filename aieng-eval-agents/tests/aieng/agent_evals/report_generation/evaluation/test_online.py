"""Tests for the online evaluation of the report generation agent."""

from unittest.mock import Mock, patch

import pytest
from aieng.agent_evals.report_generation.evaluation.online import report_final_response_score


@patch("aieng.agent_evals.report_generation.evaluation.online.AsyncClientManager.get_instance")
def test_report_final_response_score_positive_score(mock_async_client_manager_instance):
    """Test the report_final_response_score function with a positive score."""
    test_string_match = "string-to-match"
    test_trace_id = "test_trace_id"

    mock_langfuse_client = Mock()
    mock_langfuse_client.get_current_trace_id.return_value = test_trace_id

    mock_async_client_manager_instance.return_value = Mock()
    mock_async_client_manager_instance.return_value.langfuse_client = mock_langfuse_client

    mock_event = Mock()
    mock_event.is_final_response.return_value = True
    mock_event.content = Mock()
    mock_event.content.parts = [
        Mock(text=f"test_final_response_text {test_string_match} test_final_response_text"),
    ]

    report_final_response_score(mock_event, string_match=test_string_match)

    mock_langfuse_client.create_score.assert_called_once_with(
        name="Valid Final Response",
        value=1,
        trace_id=test_trace_id,
        comment="Final response contains the string match.",
        metadata={
            "final_response": mock_event.content.parts[0].text,
            "string_match": test_string_match,
        },
    )
    mock_langfuse_client.flush.assert_called_once()


@patch("aieng.agent_evals.report_generation.evaluation.online.AsyncClientManager.get_instance")
def test_report_final_response_score_negative_score(mock_async_client_manager_instance):
    """Test the report_final_response_score function with a negative score."""
    test_string_match = "string-to-match"
    test_trace_id = "test_trace_id"

    mock_langfuse_client = Mock()
    mock_langfuse_client.get_current_trace_id.return_value = test_trace_id

    mock_async_client_manager_instance.return_value = Mock()
    mock_async_client_manager_instance.return_value.langfuse_client = mock_langfuse_client

    mock_event = Mock()
    mock_event.is_final_response.return_value = True
    mock_event.content = Mock()
    mock_event.content.parts = [
        Mock(text="test_final_response_text test_final_response_text"),
    ]

    report_final_response_score(mock_event, string_match=test_string_match)

    mock_langfuse_client.create_score.assert_called_once_with(
        name="Valid Final Response",
        value=0,
        trace_id=test_trace_id,
        comment="Final response does not contains the string match.",
        metadata={
            "final_response": mock_event.content.parts[0].text,
            "string_match": test_string_match,
        },
    )
    mock_langfuse_client.flush.assert_called_once()


@patch("aieng.agent_evals.report_generation.evaluation.online.AsyncClientManager.get_instance")
def test_report_final_response_invalid(mock_async_client_manager_instance):
    """Test the report_final_response_score function with a negative score."""
    test_string_match = "string-to-match"
    test_trace_id = "test_trace_id"

    mock_langfuse_client = Mock()
    mock_langfuse_client.get_current_trace_id.return_value = test_trace_id

    mock_async_client_manager_instance.return_value = Mock()
    mock_async_client_manager_instance.return_value.langfuse_client = mock_langfuse_client

    mock_event = Mock()
    mock_event.is_final_response.return_value = True
    mock_event.content = Mock()
    mock_event.content.parts = [Mock(text=None)]

    report_final_response_score(mock_event, string_match=test_string_match)

    mock_langfuse_client.create_score.assert_called_once_with(
        name="Valid Final Response",
        value=0,
        trace_id=test_trace_id,
        comment="Final response not found in the event",
        metadata={
            "string_match": test_string_match,
        },
    )
    mock_langfuse_client.flush.assert_called_once()


def test_report_final_response_not_final_response():
    """Test raising an error when the event is not a final response."""
    mock_event = Mock()
    mock_event.is_final_response.return_value = False

    with pytest.raises(ValueError, match="Event is not a final response"):
        report_final_response_score(mock_event)


@patch("aieng.agent_evals.report_generation.evaluation.online.AsyncClientManager.get_instance")
def test_report_final_response_langfuse_trace_id_none(mock_async_client_manager_instance):
    """Test raising an error when the Langfuse trace ID is None."""
    mock_langfuse_client = Mock()
    mock_langfuse_client.get_current_trace_id.return_value = None

    mock_async_client_manager_instance.return_value = Mock()
    mock_async_client_manager_instance.return_value.langfuse_client = mock_langfuse_client

    with pytest.raises(ValueError, match="Langfuse trace ID is None."):
        report_final_response_score(Mock())

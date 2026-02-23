"""Tests for the report generation agent."""

import logging
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from aieng.agent_evals.configs import Configs
from aieng.agent_evals.report_generation.agent import EventParser, EventType, get_report_generation_agent


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@pytest.fixture
def setup_dotenv():
    """Copy .env.example to .env for the test run, then remove it in teardown."""
    root_dir = Path.cwd()
    env_dir = root_dir
    while env_dir.name != "eval-agents":
        env_dir = env_dir.parent

    env_exists = Path(".env").exists()
    if env_exists:
        # Moving existing .env to .env.bkp
        shutil.move(".env", ".env.bkp")

    shutil.copy(env_dir / ".env.example", ".env")

    yield

    Path(".env").unlink()
    if env_exists:
        # Moving the existing .env back
        shutil.move(".env.bkp", ".env")


@patch("aieng.agent_evals.report_generation.agent.init_tracing")
def test_get_report_generation_agent_with_langfuse(mock_init_tracing, setup_dotenv):
    """Test the get_report_generation_agent function."""
    test_instructions = "You are a report generation agent."
    test_reports_output_path = Path("reports/")
    test_after_agent_callback = Mock()

    agent = get_report_generation_agent(
        instructions=test_instructions,
        reports_output_path=test_reports_output_path,
        after_agent_callback=test_after_agent_callback,
    )

    assert agent.name == "ReportGenerationAgent"
    assert agent.model == Configs().default_worker_model
    assert agent.instruction == test_instructions
    assert [tool.__name__ for tool in agent.tools] == ["get_schema_info", "execute", "write_xlsx"]
    assert agent.tools[2].__self__.reports_output_path == test_reports_output_path
    assert agent.after_agent_callback == test_after_agent_callback

    mock_init_tracing.assert_called_once_with(service_name="ReportGenerationAgent")


@patch("aieng.agent_evals.report_generation.agent.init_tracing")
def test_get_report_generation_agent_without_langfuse(mock_init_tracing, setup_dotenv):
    """Test the get_report_generation_agent function."""
    test_instructions = "You are a report generation agent."
    test_reports_output_path = Path("reports/")

    agent = get_report_generation_agent(
        instructions=test_instructions,
        reports_output_path=test_reports_output_path,
        langfuse_tracing=False,
    )

    assert agent is not None
    mock_init_tracing.assert_not_called()


def test_parse_event_final_response():
    """Test the event parser when the event is a final response."""
    test_final_response_text = "Hello, world!"

    mock_event = Mock()
    mock_event.is_final_response.return_value = True
    mock_event.content = Mock()
    mock_event.content.parts = [Mock(text=test_final_response_text)]

    parsed_events = EventParser.parse(mock_event)

    assert len(parsed_events) == 1
    assert parsed_events[0].type == EventType.FINAL_RESPONSE
    assert parsed_events[0].text == test_final_response_text


def test_parse_event_invalid_final_response():
    """Test the event parser when the event is a final response but is invalid."""
    mock_event = Mock()
    mock_event.is_final_response.return_value = True
    mock_event.content = None

    parsed_events = EventParser.parse(mock_event)
    assert len(parsed_events) == 0

    mock_event.content = Mock()
    mock_event.content.parts = None

    parsed_events = EventParser.parse(mock_event)
    assert len(parsed_events) == 0

    mock_event.content.parts = []

    parsed_events = EventParser.parse(mock_event)
    assert len(parsed_events) == 0

    mock_event.content.parts = [Mock(text=None)]

    parsed_events = EventParser.parse(mock_event)
    assert len(parsed_events) == 0


def test_parse_event_model_response():
    """Test the event parser when the event is a model response."""
    mock_event = Mock()
    mock_event.is_final_response.return_value = False
    mock_event.content = Mock()
    mock_event.content.role = "model"
    function_call_mock = Mock()
    function_call_mock.name = "test_function_call_name"
    function_call_mock.args = "test_args"
    mock_event.content.parts = [
        Mock(function_call=function_call_mock),
        Mock(function_call=None, thought_signature="test_thought_signature", text="test thought text"),
    ]

    parsed_events = EventParser.parse(mock_event)

    assert parsed_events[0].type == EventType.TOOL_CALL
    assert parsed_events[0].text == mock_event.content.parts[0].function_call.name
    assert parsed_events[0].arguments == mock_event.content.parts[0].function_call.args
    assert parsed_events[1].type == EventType.THOUGHT
    assert parsed_events[1].text == mock_event.content.parts[1].text


def test_parse_event_user_response():
    """Test the event parser when the event is a user response."""
    mock_event = Mock()
    mock_event.is_final_response.return_value = False
    mock_event.content = Mock()
    mock_event.content.role = "user"
    function_response_mock = Mock()
    function_response_mock.name = "test_function_response_name"
    function_response_mock.response = "test_response"
    mock_event.content.parts = [Mock(function_response=function_response_mock)]

    parsed_events = EventParser.parse(mock_event)

    assert len(parsed_events) == 1
    assert parsed_events[0].type == EventType.TOOL_RESPONSE
    assert parsed_events[0].text == function_response_mock.name
    assert parsed_events[0].arguments == function_response_mock.response

"""Tests for the offline evaluation of the report generation agent."""

from pathlib import Path
from unittest.mock import ANY, Mock, patch

import pytest
from aieng.agent_evals.report_generation.evaluation.offline import (
    evaluate,
    final_result_evaluator,
    trajectory_evaluator,
)


@patch("aieng.agent_evals.report_generation.evaluation.offline.AsyncClientManager.get_instance")
@patch("aieng.agent_evals.report_generation.evaluation.offline.DbManager.get_instance")
@pytest.mark.asyncio
async def test_evaluate(mock_db_manager_instance, mock_async_client_manager_instance):
    """Test the evaluate function."""
    test_dataset_name = "test_dataset"
    test_reports_output_path = Path("reports/")
    test_max_concurrency = 5

    mock_result = Mock()
    mock_dataset = Mock()
    mock_dataset.run_experiment.return_value = mock_result
    mock_langfuse_client = Mock()
    mock_langfuse_client.get_dataset.return_value = mock_dataset
    mock_async_client_manager_instance.return_value = Mock()
    mock_async_client_manager_instance.return_value.langfuse_client = mock_langfuse_client

    mock_db_manager_instance.return_value = Mock()

    await evaluate(
        dataset_name=test_dataset_name,
        reports_output_path=test_reports_output_path,
        max_concurrency=test_max_concurrency,
    )

    mock_dataset.run_experiment.assert_called_once_with(
        name="Evaluate Report Generation Agent",
        description="Evaluate the Report Generation Agent with data from Langfuse",
        task=ANY,
        evaluators=[final_result_evaluator, trajectory_evaluator],
        max_concurrency=test_max_concurrency,
    )

    task = mock_dataset.run_experiment.call_args_list[0][1]["task"]
    assert task.__name__ == "run"
    assert task.__self__.__class__.__name__ == "ReportGenerationTask"
    assert task.__self__.reports_output_path == test_reports_output_path

    mock_db_manager_instance.return_value.close.assert_called_once()
    mock_async_client_manager_instance.return_value.close.assert_called_once()

"""Environment variables and their defaults for the report generation agent."""

from pathlib import Path

from aieng.agent_evals.async_client_manager import AsyncClientManager


DEFAULT_REPORTS_OUTPUT_PATH = "implementations/report_generation/reports/"


def get_reports_output_path() -> Path:
    """Get the reports output path.

    If no path is provided in the REPORTS_OUTPUT_PATH env var, will use the
    default path in DEFAULT_REPORTS_OUTPUT_PATH.

    Returns
    -------
    Path
        The reports output path.
    """
    output_path = AsyncClientManager.get_instance().configs.report_generation_output_path
    if output_path:
        return Path(output_path)

    return Path(DEFAULT_REPORTS_OUTPUT_PATH)

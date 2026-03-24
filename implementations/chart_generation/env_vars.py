"""Environment variable defaults for the chart generation agent."""

from pathlib import Path

from aieng.agent_evals.async_client_manager import AsyncClientManager


DEFAULT_CHARTS_OUTPUT_PATH = "implementations/chart_generation/charts/"


def get_charts_output_path() -> Path:
    """Get the charts output path.

    If no path is provided in the CHARTS_OUTPUT_PATH env var, will use the
    default path in DEFAULT_CHARTS_OUTPUT_PATH.

    Returns
    -------
    Path
        The charts output path.
    """
    output_path = AsyncClientManager.get_instance().configs.chart_generation_output_path
    if output_path:
        return Path(output_path)
    return Path(DEFAULT_CHARTS_OUTPUT_PATH)

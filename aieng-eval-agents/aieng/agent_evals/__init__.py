"""Agent evaluation utilities and implementations.

This package provides tools for building and evaluating AI agents,
including display utilities for Jupyter notebooks.
"""

from .display import (
    create_console,
    display_comparison,
    display_evaluation_result,
    display_example,
    display_info,
    display_metrics_table,
    display_response,
    display_source_table,
    display_success,
    display_warning,
)


__all__ = [
    # Display utilities
    "create_console",
    "display_response",
    "display_source_table",
    "display_comparison",
    "display_example",
    "display_evaluation_result",
    "display_metrics_table",
    "display_success",
    "display_info",
    "display_warning",
]

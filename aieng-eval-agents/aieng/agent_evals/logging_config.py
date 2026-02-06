"""Logging configuration with colors and clean output.

This module provides a clean, colored logging setup for agent evaluations
using the rich library. It reuses the console infrastructure from display.py
for consistent styling across the codebase.
"""

import logging

from rich.logging import RichHandler

from .display import create_console


def setup_logging(
    level: int = logging.INFO,
    show_time: bool = True,
    show_path: bool = False,
) -> None:
    """Configure colored logging with rich.

    Uses the same console theme as display.py for consistent styling.

    Parameters
    ----------
    level : int, optional
        Logging level, by default logging.INFO.
    show_time : bool, optional
        Whether to show timestamps, by default True.
    show_path : bool, optional
        Whether to show file path in logs, by default False.
    """
    # Reuse display console with force_jupyter=False for CLI
    console = create_console(force_jupyter=False)

    # Configure rich handler with clean formatting
    rich_handler = RichHandler(
        console=console,
        show_time=show_time,
        show_path=show_path,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        omit_repeated_times=False,
    )

    # Simple format - rich handles styling
    rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_handler],
        force=True,
    )

    # Silence noisy third-party libraries
    _silence_third_party_loggers()


def _silence_third_party_loggers() -> None:
    """Reduce noise from third-party libraries.

    Sets logging levels for common noisy libraries to WARNING or ERROR
    to keep evaluation output clean and focused on agent behavior.
    """
    # Google SDK libraries - only warnings and above
    for logger_name in [
        "google_adk",
        "google_genai",
        "google.adk",
        "google.genai",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Tracing/observability - only warnings
    logging.getLogger("langfuse").setLevel(logging.WARNING)

    # HTTP/network libraries - errors only
    for logger_name in ["httpx", "httpcore", "urllib3"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # System libraries
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("py.warnings").setLevel(logging.ERROR)

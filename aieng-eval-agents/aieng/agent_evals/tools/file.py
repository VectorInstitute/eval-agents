"""File tools for searching and reading local files.

Provides grep-style search and section reading for text files.
Designed to work with files fetched by the HTTP tools.
"""

import logging
import os
from typing import Any

from google.adk.tools.function_tool import FunctionTool


logger = logging.getLogger(__name__)


def grep_file(
    file_path: str,
    pattern: str,
    context_lines: int = 5,
    max_results: int = 10,
) -> dict[str, Any]:
    """Search a file for lines matching a pattern.

    A grep-style tool that searches text files for matching lines
    and returns matches with surrounding context.

    Parameters
    ----------
    file_path : str
        Path to the file to search.
    pattern : str
        Search pattern. Can be comma-separated for OR matching.
        Example: "operating expenses, total costs" matches either term.
    context_lines : int, optional
        Lines of context around each match (default 5).
    max_results : int, optional
        Maximum matching sections to return (default 10).

    Returns
    -------
    dict
        Contains 'matches' (list with line numbers and context),
        'total_matches', and 'patterns'. On error, contains 'error'.
    """
    logger.info(f"Grep {file_path} for: {pattern}")

    try:
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"File not found: {file_path}",
            }

        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        patterns = [p.strip().lower() for p in pattern.split(",") if p.strip()]
        if not patterns:
            return {
                "status": "error",
                "error": "No valid pattern provided.",
            }

        matches: list[dict[str, Any]] = []
        used_ranges: set[int] = set()

        for line_num, line in enumerate(lines):
            line_lower = line.lower()

            matched_patterns = [p for p in patterns if p in line_lower]
            if not matched_patterns:
                continue

            if line_num in used_ranges:
                continue

            start = max(0, line_num - context_lines)
            end = min(len(lines), line_num + context_lines + 1)

            used_ranges.update(range(start, end))

            context_text = "".join(lines[start:end]).strip()
            matches.append(
                {
                    "line_number": line_num + 1,
                    "matched_patterns": matched_patterns,
                    "context": context_text,
                }
            )

            if len(matches) >= max_results:
                break

        if not matches:
            return {
                "status": "success",
                "matches": [],
                "total_matches": 0,
                "patterns": patterns,
                "message": f"No matches found for: {', '.join(patterns)}",
            }

        return {
            "status": "success",
            "matches": matches,
            "total_matches": len(matches),
            "patterns": patterns,
        }

    except Exception as e:
        logger.exception(f"Error in grep_file {file_path}")
        return {
            "status": "error",
            "error": f"Grep failed: {str(e)}",
        }


def read_file(
    file_path: str,
    start_line: int = 1,
    num_lines: int = 100,
) -> dict[str, Any]:
    """Read a specific section of a file.

    Use this to read portions of large documents, especially after
    using grep_file to identify relevant sections.

    Parameters
    ----------
    file_path : str
        Path to the file to read.
    start_line : int, optional
        Line number to start from (1-indexed, default 1).
    num_lines : int, optional
        Number of lines to read (default 100).

    Returns
    -------
    dict
        Contains 'content', 'start_line', 'end_line', 'total_lines'.
        On error, contains 'error'.
    """
    logger.info(f"Reading {file_path} from line {start_line}")

    try:
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"File not found: {file_path}. Use fetch_url first.",
            }

        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)

        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, start_idx + num_lines)

        content = "".join(lines[start_idx:end_idx])

        return {
            "status": "success",
            "content": content,
            "start_line": start_idx + 1,
            "end_line": end_idx,
            "total_lines": total_lines,
        }

    except Exception as e:
        logger.exception(f"Error reading {file_path}")
        return {
            "status": "error",
            "error": f"Read failed: {str(e)}",
        }


def create_grep_file_tool() -> FunctionTool:
    """Create an ADK FunctionTool for grep-style file searching."""
    return FunctionTool(func=grep_file)


def create_read_file_tool() -> FunctionTool:
    """Create an ADK FunctionTool for reading file sections."""
    return FunctionTool(func=read_file)

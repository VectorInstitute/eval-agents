"""JSON validation tool for checking whether a string is valid JSON.

Provides a tool for the legislative content extraction agent to validate
that a given string is properly formatted JSON.
"""

import json
import logging
from typing import Any

from google.adk.tools.function_tool import FunctionTool


logger = logging.getLogger(__name__)


def validate_json(json_string: str) -> dict[str, Any]:
    """Validate whether a string is valid JSON.

    Attempts to parse the provided string as JSON and returns whether
    it is valid.

    Parameters
    ----------
    json_string : str
        The string to validate as JSON.

    Returns
    -------
    dict
        On success: 'status', 'valid' (True).
        On invalid JSON: 'status', 'valid' (False), 'error'.

    Examples
    --------
    >>> result = validate_json('{"key": "value"}')
    >>> print(result["valid"])
    True
    >>> result = validate_json('not valid json')
    >>> print(result["valid"])
    False
    """
    if not json_string:
        return {"status": "error", "error": "json_string is required."}

    try:
        json.loads(json_string)
        return {"status": "success", "valid": True}
    except json.JSONDecodeError as e:
        logger.debug(f"Invalid JSON: {e}")
        return {"status": "success", "valid": False, "error": str(e)}


def create_validate_json_tool() -> FunctionTool:
    """Create an ADK FunctionTool for validating JSON strings."""
    return FunctionTool(func=validate_json)

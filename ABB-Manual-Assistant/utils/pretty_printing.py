"""Pretty-Print Utils."""

import json
from typing import Any

import pydantic


def _serializer(item: Any) -> dict[str, Any] | str:
    """Serialize using heuristics."""
    if isinstance(item, pydantic.BaseModel):
        return item.model_dump()
    return str(item)


def pretty_print(data: Any) -> str:
    """Extract and JSON-dump only the 'properties' field from result objects."""
    try:
        if isinstance(data, list):
            properties_list = []
            for obj in data:
                if hasattr(obj, "properties"):
                    properties_list.append(obj.properties)
                else:
                    properties_list.append(_serializer(obj))
        else:
            properties_list = [getattr(data, "properties", _serializer(data))]

        output = json.dumps(properties_list, indent=2)
        print(output)
        return output

    except Exception as e:
        return f"Error during pretty print: {e}"

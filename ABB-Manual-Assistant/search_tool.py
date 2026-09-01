import json
import os

from dotenv import load_dotenv
from utils.tools.vertex_search import vertex_search


load_dotenv()


class VertexSearchTool:
    """Compatibility wrapper for Vertex AI Search.

    This implementation delegates to the Vertex AI Search helper and returns
    results in a JSON format compatible with the previous consumer code.
    """

    def __init__(self, data_name: str | None = None):
        self.data_name = data_name or os.getenv("COLLECTION_NAME")

    async def create_client(self):
        return None

    async def ensure_connected(self):
        return None

    async def get_knowledge(self, query: str) -> str:
        try:
            result = await vertex_search(query)
            if result.get("status") != "success":
                return f"Search error: {result.get('error', 'unknown')}"

            summary = result.get("summary", "")
            sources = result.get("sources", [])

            if not sources:
                return "No results found."

            formatted_results = []
            for src in sources:
                formatted_results.append(
                    {
                        "Document Name": src.get("title", ""),
                        "URL": src.get("uri", ""),
                        "Page Number": "",
                        "Full Text": summary,
                    }
                )

            return json.dumps(formatted_results, indent=2, sort_keys=True)

        except Exception as e:
            return f"Search error: {e}"

    async def close(self):
        return None

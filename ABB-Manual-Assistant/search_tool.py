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
                return json.dumps(
                    [
                        {
                            "source": "",
                            "url": "",
                            "page_number": "",
                            "excerpt": f"Search error: {result.get('error', 'Unknown')}",
                            "confidence": 0.0,
                        }
                    ],
                    indent=2,
                    sort_keys=True,
                )

            summary = result.get("summary", "")
            sources = result.get("sources", [])

            if not sources:
                return json.dumps([], indent=2, sort_keys=True)

            formatted_results = []
            for src in sources:
                formatted_results.append(
                    {
                        "source": src.get("title", ""),
                        "url": src.get("uri", ""),
                        "page_number": "",
                        "excerpt": summary,
                        "confidence": 1.0,
                    }
                )

            return json.dumps(formatted_results, indent=2, sort_keys=True)

        except Exception as e:
            return json.dumps(
                [
                    {
                        "source": "",
                        "url": "",
                        "page_number": "",
                        "excerpt": f"Search error: {str(e)}",
                        "confidence": 0.0,
                    }
                ],
                indent=2,
                sort_keys=True,
            )

    async def close(self):
        return None

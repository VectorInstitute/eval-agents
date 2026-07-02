# generated test file - only search tool

import os
import sys


# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio

from search_tool import VertexSearchTool


async def test():
    try:
        vertex_search_tool = VertexSearchTool()
        test_query = "Spot application weld error reported"
        result = await vertex_search_tool.get_knowledge(test_query)
        print("[TEST RESULT]")
        print(result if result else "No result returned.")
    except Exception as e:
        print(f"[TEST ERROR] {e}")


if __name__ == "__main__":
    asyncio.run(test())

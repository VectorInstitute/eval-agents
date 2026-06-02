# generated test file - only search tool

import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from search_tool import Weaviate
import asyncio

async def test():
    try:
        weaviate_search = Weaviate()
        test_query = "Spot application weld error reported"
        result = await weaviate_search.get_knowledge(test_query)
        print("[TEST RESULT]")
        print(result if result else "No result returned.")
    except Exception as e:
        print(f"[TEST ERROR] {e}")

if __name__ == "__main__":
    asyncio.run(test())

import asyncio
from search_agent import SearchAgent  # Replace with actual import path

async def test_error_code_query():
    agent = SearchAgent()
    query = "What does error code 10039 mean in ABB robot manuals?"
    
    print("Running test query...")
    result = await agent.run(query)
    
    print("\n=== Test Result ===")
    print(result)

if __name__ == "__main__":
    asyncio.run(test_error_code_query())

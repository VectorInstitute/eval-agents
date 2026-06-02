import asyncio
from search_tool import Weaviate  # Ensure this file exists in the same directory

async def main():
    search_tool = Weaviate()
    result = await search_tool.get_knowledge("error code 10039")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

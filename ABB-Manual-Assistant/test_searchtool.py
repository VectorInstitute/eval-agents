import asyncio

from search_tool import VertexSearchTool


async def main():
    search_tool = VertexSearchTool()
    result = await search_tool.get_knowledge("error code 10039")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())

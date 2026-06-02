# generated test file - agent using search tool

import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import os
from dotenv import load_dotenv
import agents
from openai import AsyncOpenAI
from search_tool import Weaviate  # your tool

load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# Initialize Weaviate search class
weaviate_search = Weaviate()

async def search_knowledgebase(query: str) -> str:
    print(f"[TOOL] Called with query: {query}")
    try:
        result = await weaviate_search.get_knowledge(query)
        print(f"[TOOL] Result length: {len(result) if result else 'None'}")
        print(f"[TOOL] Full text: {result}")
        return result
    except Exception as e:
        print(f"[TOOL] Exception: {e}")
        return f"Error during search: {e}"

knowledge_tool = agents.function_tool(search_knowledgebase)

agent = agents.Agent(
    name="Debug Knowledge Agent",
    instructions="You are a helpful assistant. Use the tool to get knowledge.",
    tools=[knowledge_tool],
    model=agents.OpenAIChatCompletionsModel(
        model="gemini-2.5-pro",
        openai_client=client,
    ),
    model_settings=agents.ModelSettings(tool_choice="required"),
)

async def main():
    test_queries = [
        # "10077, FTP server down",
        # "What is FTP?",
        # "Explain HTTP protocol",
        # "What is the largest ABB robot",
        # "What are the specs for the IRB 140?",
        # "What is \"EN ISO 12100 -1\"",
        # "What colour is the sky",
        "Spot application weld error reported"
    ]

    for q in test_queries:
        print("\n===============================")
        print(f"Running agent with prompt: {q}")
        try:
            response = await agents.Runner.run(agent, input=q)
            print("[AGENT FINAL OUTPUT]:\n", response.final_output)
        except Exception as e:
            print("[AGENT] Exception during run:", e)

    await weaviate_search.close()

if __name__ == "__main__":
    asyncio.run(main())

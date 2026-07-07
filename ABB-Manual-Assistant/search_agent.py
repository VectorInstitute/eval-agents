import inspect
import logging
import os
import traceback
from typing import Any

import agents
from dotenv import load_dotenv
from openai import AsyncOpenAI

from search_tool import VertexSearchTool


load_dotenv()

logger = logging.getLogger(__name__)


class SearchAgent:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

        self.knowledge_tool = agents.function_tool(
            self.search_knowledgebase,
            name_override="knowledge_search",
            description_override=(
                "Searches the ABB robot manual vector database for the most "
                "relevant technical sections."
            ),
        )

        self.search_agent = agents.Agent(
            name="Search Agent",
            instructions="""
You are a Search Agent specialized in retrieving exact, relevant information
from ABB robot manuals stored in a vector database.

Your ONLY purpose is to:
1. Use the provided `knowledge_search` tool to query the database.
2. Return the most relevant technical excerpts that directly answer the query.
3. Preserve the original wording. Do not paraphrase, summarize, or add advice.

Rules:
- Always use `knowledge_search`.
- Do not answer from your own knowledge.
- Do not guess or fill in missing technical details.
- Do not include unrelated or generic text.
- If no relevant information is found, return an empty list: [].
- If the search tool fails, clearly state that retrieval failed. Do not invent
  an ABB-specific answer.

Output Format:
[
  {
    "source": "<document title or section name>",
    "url": "<clickable source URL including the page number>",
    "excerpt": "<exact paragraph or sentence from the document>",
    "confidence": <numeric relevance score or rank>
  }
]

IMPORTANT: Every single object in the array must include all four fields,
including "confidence". Never omit "confidence" from any item, even when
returning many results. Double-check every item before responding.
""".strip(),
            model=agents.OpenAIChatCompletionsModel(
                model="gemini-2.5-flash",
                openai_client=self.client,
            ),
            model_settings=agents.ModelSettings(
                tool_choice="required",
                temperature=0,
            ),
            tools=[self.knowledge_tool],
        )

    @staticmethod
    def _compact_repr(value: Any, max_length: int = 4000) -> str:
        value_repr = repr(value)

        if len(value_repr) <= max_length:
            return value_repr

        return f"{value_repr[:max_length]}... <truncated>"

    @staticmethod
    async def _resolve_knowledge_result(value: Any) -> Any:
        """
        Normalize whatever VertexSearchTool.get_knowledge() returns.

        It may return:
        - a regular result value,
        - an awaitable/coroutine,
        - or an async generator.
        """
        if inspect.isasyncgen(value) or hasattr(value, "__aiter__"):
            return [item async for item in value]

        if inspect.isawaitable(value):
            return await value

        return value

    @staticmethod
    async def search_knowledgebase(query: str) -> Any:
        """
        Tool function registered with the Search Agent.

        This method must remain inside SearchAgent. The prior AttributeError
        means it was not present on the class at runtime.
        """
        query = query.strip()

        if not query:
            print("[SEARCH] Empty query received; returning no results.", flush=True)
            return []

        try:
            print(f"\n[SEARCH] Query: {query!r}", flush=True)

            vertex_tool = VertexSearchTool()

            raw_result = vertex_tool.get_knowledge(query)

            print(
                f"[SEARCH] Backend return type: {type(raw_result).__name__}",
                flush=True,
            )

            resolved_result = await SearchAgent._resolve_knowledge_result(
                raw_result
            )

            print(
                f"[SEARCH] Resolved result type: "
                f"{type(resolved_result).__name__}",
                flush=True,
            )
            print(
                f"[SEARCH] Resolved result: "
                f"{SearchAgent._compact_repr(resolved_result)}",
                flush=True,
            )

            if resolved_result is None:
                print("[SEARCH] Backend returned None; returning no results.", flush=True)
                return []

            return resolved_result

        except Exception as exc:
            print("\n[SEARCH] VERTEX RETRIEVAL FAILED", flush=True)
            print(f"[SEARCH] Query: {query!r}", flush=True)
            print(f"[SEARCH] Error: {type(exc).__name__}: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)

            logger.exception(
                "ABB manual retrieval failed. query=%r",
                query,
            )

            # Re-raise so the agent/tool layer receives a real failure rather
            # than silently converting it into an empty search result.
            raise

    async def run(self, prompt: str) -> Any:
        """
        Preserve the original return type for compatibility with Orchestrator.

        Do not return response.final_output here until we inspect how
        orchestrator_agent.py consumes SearchAgent.run().
        """
        response = await agents.Runner.run(
            self.search_agent,
            input=prompt,
        )

        return response
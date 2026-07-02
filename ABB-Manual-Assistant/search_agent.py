import os

import agents
from dotenv import load_dotenv
from openai import AsyncOpenAI
from search_tool import VertexSearchTool


load_dotenv()


class SearchAgent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

        self.knowledge_tool = agents.function_tool(
            self.search_knowledgebase,
            name_override="knowledge_search",
            description_override="Searches the ABB robot manual vector database for the most relevant technical sections.",
        )

        self.search_agent = agents.Agent(
            name="Search Agent",
            instructions="""
                            You are a Search Agent specialized in retrieving exact, relevant information from ABB robot manuals stored in a vector database.

                            Your ONLY purpose is to:
                            1. Use the provided "knowledge_search" tool to query the database.
                            2. Return the most relevant technical excerpts that directly answer the query.
                            3. Preserve the original wording — do not paraphrase, summarize, or add explanations.

                            Rules:
                            - Do NOT generate answers from your own knowledge.
                            - Do NOT guess or fill in missing details.
                            - Do NOT include unrelated or generic text.

                            Output Format:
                            [
                            {
                                "source": "<document title or section name>",
                                "url": "< a clickable url to go to the source of the information, include the page number >"
                                "excerpt": "<exact paragraph or sentence from the document>",
                                "confidence": <numeric relevance score or rank>
                            },
                            ...
                            ]

                            Guidelines:
                            - Always select the top results that are most technically accurate and useful for a technician repairing ABB robots.
                            - Avoid redundancy — do not return overlapping excerpts.
                            - If no relevant information is found, return an empty list [].
                            """,
            model=agents.OpenAIChatCompletionsModel(
                model="gemini-2.5-flash", openai_client=self.client
            ),
            model_settings=agents.ModelSettings(tool_choice="required", temperature=0),
            tools=[self.knowledge_tool],
        )

    @staticmethod
    async def search_knowledgebase(query: str):
        vertex_tool = VertexSearchTool()

        return await vertex_tool.get_knowledge(query)

    async def run(self, prompt: str) -> str:
        response = await agents.Runner.run(self.search_agent, input=prompt)
        return response

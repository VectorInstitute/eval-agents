import os

import agents
from dotenv import load_dotenv
from openai import AsyncOpenAI


load_dotenv()


class WorkorderAgent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

        self.workorder_agent = agents.Agent(
            name="Workorder Agent",
            instructions="""
                            Given a conversation with an ABB robot manual assistant agent you should create a workorder that states:

                            1) Workorder Title: A descriptive title for the workorder.
                            2) Error/Issue: the error or issue that occured.
                            3) Work completed: the work/action that the user has taken or will take to resolve the error.

                            Only use information from the user's conversation with the ABB robot assistant agent to complete the workorder.
                            Your response should include only the created workorder and nothing else. Provide as much detail as possible.
                            """,
            model=agents.OpenAIChatCompletionsModel(
                model="gemini-2.5-flash-lite-preview-06-17", openai_client=self.client
            ),
            model_settings=agents.ModelSettings(temperature=0.5),
        )

    async def run(self, prompt: str) -> str:
        response = await agents.Runner.run(self.workorder_agent, input=prompt)
        return response

import os

import agents
from dotenv import load_dotenv
from openai import AsyncOpenAI


load_dotenv()


WORKORDER_TEMPLATE = """
Workorder Title: <short technician-ready title>
Equipment or System: <robot, controller, cell, or Unknown>
Reported Issue: <core problem being reported>
Symptoms:
- <symptom or observation>
Relevant Context:
- <model, error code, recent changes, conditions, or Unknown>
Work Completed or Planned:
- <work already completed or intended next action>
Recommended Next Steps:
- <clear next step>
Safety Notes:
- <safety consideration, or None identified>
Missing Information:
- <detail needed before diagnosis, or None identified>
""".strip()


class WorkorderAgent:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

        self.workorder_agent = agents.Agent(
            name="Workorder Agent",
            instructions="""
You convert an ABB troubleshooting conversation into a concise technician-ready
work order.

Use only details explicitly present in the conversation.

Rules:
- Do not invent troubleshooting facts, completed work, equipment names, or error codes.
- If a detail is missing, write "Unknown" or "None identified" instead of guessing.
- Capture the actual issue, important symptoms, and useful context for a technician.
- Keep the work order concise and operationally useful.
- Return only the completed work order using the exact section headings below.

Required format:
"""
            + WORKORDER_TEMPLATE,
            model=agents.OpenAIChatCompletionsModel(
                model="gemini-2.5-flash",
                openai_client=self.client,
            ),
            model_settings=agents.ModelSettings(temperature=0),
        )

    @staticmethod
    def build_input(conversation: str) -> str:
        return (
            "Create a work order from the conversation below.\n\n"
            f"Conversation:\n{conversation.strip()}"
        )

    async def run(self, prompt: str) -> str:
        response = await agents.Runner.run(
            self.workorder_agent,
            input=self.build_input(prompt),
        )
        final_output = getattr(response, "final_output", None)

        if isinstance(final_output, str) and final_output.strip():
            return final_output.strip()

        return str(response).strip()

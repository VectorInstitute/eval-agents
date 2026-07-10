import os

import agents
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from search_agent import SearchAgent
from workorder_agent import WorkorderAgent


load_dotenv()


class Orchestrator:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

        search_agent_instance = SearchAgent()
        self.search_agent_tool = search_agent_instance.search_agent.as_tool(
            tool_name="search_knowledge_base",
            tool_description="Agent searches ABB robot manuals for repair, troubleshooting, maintenance instructions, etc.",
        )

        workorder_agent_instance = WorkorderAgent()
        self.workorder_agent_tool = workorder_agent_instance.workorder_agent.as_tool(
            tool_name="workorder_agent",
            tool_description="Given a conversation between the user and the orchestrator agent, the workorder agent will create a workorder.",
        )

        self.main_agent = agents.Agent(
            name="Orchestrator Agent",
            instructions="""
                            You are a helpful assistant and organizer.
                            Answer only the question asked, as briefly as possible. Prefer the manual's exact wording over rephrasing. Do not restate background or add sections.
                            Each claim must be traceable to a retrieved excerpt. If no excerpt supports a statement, omit it.
                            If nothing relevant is found, say so — do not use outside knowledge
                            Always present the search agent's findings at the bottom of your output inside a collapseable section.
                            Do not add Consequences, Prevention, Safety, or best-practice sections, and do not offer to create a work order, unless the user asked or the search results contain it.
                            Do not add causes, steps, or facts that aren't in the search results, and don't generalize beyond them.

                            If the user asks you to create a workorder, then call the workorder_agent.
                            """,
            model=agents.OpenAIChatCompletionsModel(model="gemini-2.5-pro", openai_client=self.client),
            model_settings=agents.ModelSettings(tool_choice="required", temperature=0.2),
            tools=[self.search_agent_tool, self.workorder_agent_tool],
        )

    async def run(self, prompt: str, history) -> str:
        context = ""

        # reconstruct conversation history into a string
        if history:
            for i in range(0, len(history), 2):
                if i + 1 < len(history):
                    user_msg = history[i]["content"]
                    bot_msg = history[i + 1]["content"]
                    context += f"User: {user_msg}\nAssistant: {bot_msg}\n"

        # combine history with the new prompt
        full_prompt = f"{context} User: {prompt}"

        result_stream = agents.Runner.run_streamed(self.main_agent, input=full_prompt)
        async for event in result_stream.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
                yield event.data.delta

"""Research planner for decomposing complex questions into executable plans.

This module provides planning capabilities for the knowledge agent,
enabling it to break down complex research questions into structured,
multi-step research plans that are observable and evaluable.
"""

import json
import logging
from typing import TYPE_CHECKING

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from .config import KnowledgeAgentConfig


logger = logging.getLogger(__name__)


class StepStatus:
    """Status constants for research steps."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ResearchStep(BaseModel):
    """A single step in a research plan.

    Attributes
    ----------
    step_id : int
        Unique identifier for the step within the plan.
    description : str
        Clear description of what this step accomplishes.
    tool_hint : str
        Suggested tool to use: "web_search", "fetch_url", "read_pdf", or "synthesis".
    depends_on : list[int]
        IDs of steps that must complete before this one.
    expected_output : str
        Description of what this step is expected to produce.
    status : str
        Current execution status: "pending", "in_progress", "completed", "failed",
        or "skipped".
    actual_output : str
        What was actually found/produced by this step.
    attempts : int
        Number of times this step has been attempted.
    failure_reason : str
        Reason for failure if the step failed.
    """

    step_id: int
    description: str
    tool_hint: str  # "web_search", "fetch_url", "read_pdf", "synthesis"
    depends_on: list[int] = Field(default_factory=list)
    expected_output: str = ""
    # Dynamic tracking fields
    status: str = Field(default=StepStatus.PENDING)
    actual_output: str = ""
    attempts: int = 0
    failure_reason: str = ""


class ResearchPlan(BaseModel):
    """A complete research plan for answering a complex question.

    This model represents an observable, evaluable research plan that
    decomposes a question into executable steps with clear dependencies.

    Attributes
    ----------
    original_question : str
        The original question being answered.
    complexity_assessment : str
        Assessment of question complexity: "simple", "moderate", or "complex".
    steps : list[ResearchStep]
        Ordered list of research steps to execute.
    estimated_tools : list[str]
        Tools expected to be used during execution.
    reasoning : str
        Explanation of why this plan was chosen.
    """

    original_question: str
    complexity_assessment: str  # "simple", "moderate", "complex"
    steps: list[ResearchStep] = Field(default_factory=list)
    estimated_tools: list[str] = Field(default_factory=list)
    reasoning: str = ""

    def get_step(self, step_id: int) -> ResearchStep | None:
        """Get a step by its ID.

        Parameters
        ----------
        step_id : int
            The step ID to find.

        Returns
        -------
        ResearchStep | None
            The step if found, None otherwise.
        """
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def update_step(
        self,
        step_id: int,
        status: str | None = None,
        actual_output: str | None = None,
        failure_reason: str | None = None,
        increment_attempts: bool = False,
    ) -> bool:
        """Update a step's tracking fields.

        Parameters
        ----------
        step_id : int
            The step ID to update.
        status : str, optional
            New status for the step.
        actual_output : str, optional
            What was actually found/produced.
        failure_reason : str, optional
            Reason for failure if applicable.
        increment_attempts : bool
            Whether to increment the attempts counter.

        Returns
        -------
        bool
            True if the step was found and updated, False otherwise.
        """
        step = self.get_step(step_id)
        if step is None:
            return False

        if status is not None:
            step.status = status
        if actual_output is not None:
            step.actual_output = actual_output
        if failure_reason is not None:
            step.failure_reason = failure_reason
        if increment_attempts:
            step.attempts += 1

        return True

    def get_pending_steps(self) -> list[ResearchStep]:
        """Get steps that are ready to execute (pending with no unmet dependencies).

        Returns
        -------
        list[ResearchStep]
            Steps that can be executed now.
        """
        completed_ids = {s.step_id for s in self.steps if s.status == StepStatus.COMPLETED}
        pending = []

        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            # Check if all dependencies are completed
            if all(dep_id in completed_ids for dep_id in step.depends_on):
                pending.append(step)

        return pending

    def get_steps_by_status(self, status: str) -> list[ResearchStep]:
        """Get all steps with a specific status.

        Parameters
        ----------
        status : str
            The status to filter by.

        Returns
        -------
        list[ResearchStep]
            Steps matching the status.
        """
        return [s for s in self.steps if s.status == status]

    def add_step(self, step: ResearchStep) -> None:
        """Add a new step to the plan.

        Parameters
        ----------
        step : ResearchStep
            The step to add.
        """
        self.steps.append(step)

    def is_complete(self) -> bool:
        """Check if all steps are either completed, failed, or skipped.

        Returns
        -------
        bool
            True if no steps are pending or in progress.
        """
        terminal_statuses = {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED}
        return all(s.status in terminal_statuses for s in self.steps)

    def get_next_step_id(self) -> int:
        """Get the next available step ID.

        Returns
        -------
        int
            The next step ID (max existing ID + 1).
        """
        if not self.steps:
            return 1
        return max(s.step_id for s in self.steps) + 1

    def get_progress_summary(self) -> dict[str, int]:
        """Get a summary of step statuses.

        Returns
        -------
        dict[str, int]
            Count of steps in each status.
        """
        summary: dict[str, int] = {
            StepStatus.PENDING: 0,
            StepStatus.IN_PROGRESS: 0,
            StepStatus.COMPLETED: 0,
            StepStatus.FAILED: 0,
            StepStatus.SKIPPED: 0,
        }
        for step in self.steps:
            if step.status in summary:
                summary[step.status] += 1
        return summary


class StepExecution(BaseModel):
    """Record of executing a single research step.

    This model captures the execution trace for evaluation purposes.

    Attributes
    ----------
    step_id : int
        The step ID that was executed.
    tool_used : str
        The actual tool that was used.
    input_query : str
        The query or input provided to the tool.
    output_summary : str
        Summary of what the step produced.
    sources_found : int
        Number of sources discovered in this step.
    duration_ms : int
        Execution time in milliseconds.
    raw_output : str
        Raw output from the tool for debugging.
    """

    step_id: int
    tool_used: str
    input_query: str
    output_summary: str = ""
    sources_found: int = 0
    duration_ms: int = 0
    raw_output: str = ""


PLANNER_SYSTEM_PROMPT = """\
You are a research planning expert. Your task is to analyze questions and create
structured research plans that can be executed by a knowledge-grounded QA agent.

## Available Tools

The agent has access to these tools:
1. **web_search**: Google Search for finding relevant sources, current information, news, events
2. **fetch_url**: Fetch and read the full content of a specific webpage URL
3. **read_pdf**: Read and extract text from PDF documents (SEC filings, research papers, reports)
4. **synthesis**: Combining information from multiple sources into a coherent answer

## Planning Guidelines

1. **Assess Complexity**:
   - "simple": Single fact lookup, straightforward question → 1-2 steps
   - "moderate": Multiple related facts or comparison → 2-4 steps
   - "complex": Causal reasoning, multi-part analysis → 4+ steps

2. **Choose Tools Wisely**:
   - Use web_search first to find relevant sources
   - Use fetch_url when you need to read the full content of a webpage
   - Use read_pdf for PDF documents (SEC filings, annual reports, research papers)
   - Use synthesis as the final step when combining multiple sources

3. **Create Dependencies**:
   - Steps should have clear dependencies (depends_on lists)
   - Later steps can build on earlier findings
   - Synthesis typically depends on all information-gathering steps

4. **Be Efficient**:
   - Don't create unnecessary steps
   - Combine related searches when possible
   - Simple questions don't need complex plans

## Output Format

Return a JSON object with this structure:
{
    "original_question": "the question",
    "complexity_assessment": "simple|moderate|complex",
    "steps": [
        {
            "step_id": 1,
            "description": "What this step does",
            "tool_hint": "web_search|fetch_url|read_pdf|synthesis",
            "depends_on": [],
            "expected_output": "What we expect to learn"
        }
    ],
    "estimated_tools": ["web_search", "fetch_url", "read_pdf"],
    "reasoning": "Why this plan makes sense for the question"
}
"""


def _parse_plan_response(response_text: str, question: str) -> ResearchPlan:
    """Parse the LLM response into a ResearchPlan.

    Parameters
    ----------
    response_text : str
        Raw response from the planning LLM.
    question : str
        The original question (for fallback).

    Returns
    -------
    ResearchPlan
        Parsed research plan.
    """
    # Try to extract JSON from response
    try:
        # Look for JSON block in response
        text = response_text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        plan_data = json.loads(text)

        steps = [
            ResearchStep(
                step_id=s.get("step_id", i + 1),
                description=s.get("description", ""),
                tool_hint=s.get("tool_hint", "web_search"),
                depends_on=s.get("depends_on", []),
                expected_output=s.get("expected_output", ""),
            )
            for i, s in enumerate(plan_data.get("steps", []))
        ]

        return ResearchPlan(
            original_question=plan_data.get("original_question", question),
            complexity_assessment=plan_data.get("complexity_assessment", "moderate"),
            steps=steps,
            estimated_tools=plan_data.get("estimated_tools", ["web_search"]),
            reasoning=plan_data.get("reasoning", ""),
        )

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse plan response: {e}")
        # Return a simple default plan
        return ResearchPlan(
            original_question=question,
            complexity_assessment="simple",
            steps=[
                ResearchStep(
                    step_id=1,
                    description="Search for relevant information",
                    tool_hint="web_search",
                    depends_on=[],
                    expected_output="Relevant facts and sources",
                ),
                ResearchStep(
                    step_id=2,
                    description="Synthesize findings into answer",
                    tool_hint="synthesis",
                    depends_on=[1],
                    expected_output="Complete answer with citations",
                ),
            ],
            estimated_tools=["web_search"],
            reasoning="Default plan due to parsing error",
        )


class ResearchPlanner:
    """Creates multi-step research plans for complex questions.

    This planner analyzes questions and produces structured plans that
    can be executed and evaluated. Plans include tool recommendations,
    step dependencies, and expected outputs.

    Parameters
    ----------
    config : KnowledgeAgentConfig, optional
        Configuration settings. If not provided, uses defaults.
    model : str, optional
        Model to use for planning. Defaults to planner model from config.

    Examples
    --------
    >>> from aieng.agent_evals.knowledge_agent.planner import ResearchPlanner
    >>> planner = ResearchPlanner()
    >>> plan = planner.create_plan("What caused the 2008 financial crisis?")
    >>> print(f"Complexity: {plan.complexity_assessment}")
    >>> for step in plan.steps:
    ...     print(f"  {step.step_id}. {step.description}")
    """

    def __init__(
        self,
        config: "KnowledgeAgentConfig | None" = None,
        model: str | None = None,
    ) -> None:
        """Initialize the research planner.

        Parameters
        ----------
        config : KnowledgeAgentConfig, optional
            Configuration settings.
        model : str, optional
            Model to use for planning.
        """
        self._config = config

        if model is not None:
            self._model = model
        elif config is not None:
            self._model = config.default_planner_model
        else:
            self._model = "gemini-2.5-flash"

        # Initialize Genai client
        self._client = genai.Client()

    def create_plan(self, question: str) -> ResearchPlan:
        """Create a research plan for the given question.

        Parameters
        ----------
        question : str
            The question to create a plan for.

        Returns
        -------
        ResearchPlan
            A structured research plan.
        """
        logger.info(f"Creating research plan for: {question[:100]}...")

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=types.Content(
                    role="user",
                    parts=[types.Part(text=f"Create a research plan for this question:\n\n{question}")],
                ),
                config=types.GenerateContentConfig(
                    system_instruction=PLANNER_SYSTEM_PROMPT,
                    temperature=0.3,  # Lower temperature for more consistent planning
                ),
            )

            response_text = response.text or ""
            plan = _parse_plan_response(response_text, question)

            logger.info(f"Created plan with {len(plan.steps)} steps (complexity: {plan.complexity_assessment})")
            return plan

        except Exception as e:
            logger.error(f"Error creating research plan: {e}")
            # Return a simple fallback plan
            return ResearchPlan(
                original_question=question,
                complexity_assessment="simple",
                steps=[
                    ResearchStep(
                        step_id=1,
                        description="Search for relevant information",
                        tool_hint="web_search",
                        depends_on=[],
                        expected_output="Relevant facts and sources",
                    ),
                ],
                estimated_tools=["web_search"],
                reasoning=f"Fallback plan due to error: {e}",
            )

    async def create_plan_async(self, question: str) -> ResearchPlan:
        """Async version of create_plan.

        Parameters
        ----------
        question : str
            The question to create a plan for.

        Returns
        -------
        ResearchPlan
            A structured research plan.
        """
        logger.info(f"Creating research plan (async) for: {question[:100]}...")

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=types.Content(
                    role="user",
                    parts=[types.Part(text=f"Create a research plan for this question:\n\n{question}")],
                ),
                config=types.GenerateContentConfig(
                    system_instruction=PLANNER_SYSTEM_PROMPT,
                    temperature=0.3,
                ),
            )

            response_text = response.text or ""
            plan = _parse_plan_response(response_text, question)

            logger.info(f"Created plan with {len(plan.steps)} steps (complexity: {plan.complexity_assessment})")
            return plan

        except Exception as e:
            logger.error(f"Error creating research plan: {e}")
            return ResearchPlan(
                original_question=question,
                complexity_assessment="simple",
                steps=[
                    ResearchStep(
                        step_id=1,
                        description="Search for relevant information",
                        tool_hint="web_search",
                        depends_on=[],
                        expected_output="Relevant facts and sources",
                    ),
                ],
                estimated_tools=["web_search"],
                reasoning=f"Fallback plan due to error: {e}",
            )

    def suggest_new_steps(
        self,
        plan: ResearchPlan,
        step_result: str,
        failed_step: ResearchStep | None = None,
    ) -> list[ResearchStep]:
        """Suggest new steps based on execution results.

        This method analyzes the current plan state and suggests new steps
        when a step fails or when new information suggests additional research.

        Parameters
        ----------
        plan : ResearchPlan
            The current research plan.
        step_result : str
            Description of what happened in the last step.
        failed_step : ResearchStep, optional
            The step that failed, if any.

        Returns
        -------
        list[ResearchStep]
            New steps to add to the plan.
        """
        prompt = self._build_replanning_prompt(plan, step_result, failed_step)

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)],
                ),
                config=types.GenerateContentConfig(
                    system_instruction=REPLANNING_SYSTEM_PROMPT,
                    temperature=0.3,
                ),
            )

            response_text = response.text or ""
            new_steps = _parse_new_steps_response(response_text, plan.get_next_step_id())

            logger.info(f"Suggested {len(new_steps)} new steps")
            return new_steps

        except Exception as e:
            logger.error(f"Error suggesting new steps: {e}")
            return []

    async def suggest_new_steps_async(
        self,
        plan: ResearchPlan,
        step_result: str,
        failed_step: ResearchStep | None = None,
    ) -> list[ResearchStep]:
        """Async version of suggest_new_steps.

        Parameters
        ----------
        plan : ResearchPlan
            The current research plan.
        step_result : str
            Description of what happened in the last step.
        failed_step : ResearchStep, optional
            The step that failed, if any.

        Returns
        -------
        list[ResearchStep]
            New steps to add to the plan.
        """
        prompt = self._build_replanning_prompt(plan, step_result, failed_step)

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)],
                ),
                config=types.GenerateContentConfig(
                    system_instruction=REPLANNING_SYSTEM_PROMPT,
                    temperature=0.3,
                ),
            )

            response_text = response.text or ""
            new_steps = _parse_new_steps_response(response_text, plan.get_next_step_id())

            logger.info(f"Suggested {len(new_steps)} new steps")
            return new_steps

        except Exception as e:
            logger.error(f"Error suggesting new steps: {e}")
            return []

    def _build_replanning_prompt(
        self,
        plan: ResearchPlan,
        step_result: str,
        failed_step: ResearchStep | None = None,
    ) -> str:
        """Build the prompt for replanning.

        Parameters
        ----------
        plan : ResearchPlan
            The current research plan.
        step_result : str
            Description of what happened in the last step.
        failed_step : ResearchStep, optional
            The step that failed, if any.

        Returns
        -------
        str
            The replanning prompt.
        """
        # Build step status summary
        status_lines = []
        for step in plan.steps:
            status_icon = {
                StepStatus.COMPLETED: "✓",
                StepStatus.FAILED: "✗",
                StepStatus.SKIPPED: "⊘",
                StepStatus.IN_PROGRESS: "→",
                StepStatus.PENDING: "○",
            }.get(step.status, "?")

            line = f"  {status_icon} Step {step.step_id}: {step.description} [{step.status}]"
            if step.actual_output:
                line += f"\n      Output: {step.actual_output[:200]}..."
            if step.failure_reason:
                line += f"\n      Failure: {step.failure_reason}"
            status_lines.append(line)

        status_summary = "\n".join(status_lines)

        prompt = f"""Question: {plan.original_question}

Current Plan Status:
{status_summary}

Latest Result: {step_result}
"""

        if failed_step:
            prompt += f"""
Failed Step Details:
- Step {failed_step.step_id}: {failed_step.description}
- Tool hint: {failed_step.tool_hint}
- Attempts: {failed_step.attempts}
- Failure reason: {failed_step.failure_reason}
"""

        prompt += "\nBased on this, suggest any new steps needed to complete the research."

        return prompt


REPLANNING_SYSTEM_PROMPT = """\
You are a research planning expert helping to dynamically adjust research plans.

Given the current state of a research plan and recent execution results, suggest
new steps if needed. Consider:

1. **Failed Steps**: If a step failed, suggest an alternative approach
   - Try different search terms
   - Try different sources (web vs PDF)
   - Try a different angle to find the same information

2. **Incomplete Information**: If results are partial, add steps to fill gaps

3. **New Leads**: If new information suggests additional research, add steps

4. **When NOT to add steps**:
   - If the question can be answered with current findings
   - If we've already tried multiple alternatives for the same information
   - If the failed step was optional

## Output Format

Return a JSON array of new steps (empty array if no new steps needed):
[
    {
        "description": "What this step does",
        "tool_hint": "web_search|fetch_url|read_pdf|synthesis",
        "depends_on": [],
        "expected_output": "What we expect to learn"
    }
]

Keep suggestions focused and avoid redundant steps.
"""


def _parse_new_steps_response(response_text: str, start_id: int) -> list[ResearchStep]:
    """Parse the replanning response into new steps.

    Parameters
    ----------
    response_text : str
        Raw response from the replanning LLM.
    start_id : int
        Starting step ID for new steps.

    Returns
    -------
    list[ResearchStep]
        Parsed new steps.
    """
    try:
        text = response_text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        steps_data = json.loads(text)

        if not isinstance(steps_data, list):
            return []

        new_steps = []
        for i, s in enumerate(steps_data):
            step = ResearchStep(
                step_id=start_id + i,
                description=s.get("description", ""),
                tool_hint=s.get("tool_hint", "web_search"),
                depends_on=s.get("depends_on", []),
                expected_output=s.get("expected_output", ""),
            )
            new_steps.append(step)

        return new_steps

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse new steps response: {e}")
        return []

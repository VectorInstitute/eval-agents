"""Research planner for decomposing complex questions into executable plans.

This module provides planning capabilities for the knowledge agent,
enabling it to break down complex research questions into structured,
multi-step research plans that are observable and evaluable.
"""

import logging
from typing import TYPE_CHECKING

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from aieng.agent_evals.configs import Configs


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
    step_type : str
        Type of step: "research" (uses tools to gather info) or "synthesis"
        (combines findings without tools).
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
    step_type: str = "research"  # "research" or "synthesis"
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
    reasoning : str
        Explanation of why this plan was chosen.
    """

    original_question: str
    complexity_assessment: str  # "simple", "moderate", "complex"
    steps: list[ResearchStep] = Field(default_factory=list)
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
        description: str | None = None,
        expected_output: str | None = None,
    ) -> bool:
        """Update a step's fields.

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
        description : str, optional
            New description for the step (for plan refinement).
        expected_output : str, optional
            New expected output for the step (for plan refinement).

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
        if description is not None:
            step.description = description
        if expected_output is not None:
            step.expected_output = expected_output

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


class StepUpdate(BaseModel):
    """An update to an existing step.

    Attributes
    ----------
    step_id : int
        The ID of the step to update.
    new_description : str
        The updated description for the step.
    new_expected_output : str
        The updated expected output for the step.
    """

    step_id: int
    new_description: str
    new_expected_output: str = ""


class PlanReflection(BaseModel):
    """Result of reflecting on a completed step.

    Attributes
    ----------
    can_answer_now : bool
        Whether we have enough information to synthesize the answer.
    key_findings : str
        Summary of what was learned in this step.
    reasoning : str
        Why this decision was made.
    steps_to_update : list[StepUpdate]
        Updates to pending steps based on what was learned.
        Prefer this over remove+add when refining a step's approach.
    steps_to_remove : list[int]
        IDs of pending steps that are no longer needed.
    steps_to_add : list[ResearchStep]
        New steps to add. Only use if work doesn't fit existing steps.
    """

    can_answer_now: bool = False
    key_findings: str = ""
    reasoning: str = ""
    steps_to_update: list[StepUpdate] = Field(default_factory=list)
    steps_to_remove: list[int] = Field(default_factory=list)
    steps_to_add: list[ResearchStep] = Field(default_factory=list)


class NewStepsResponse(BaseModel):
    """Response model for replanning - wrapper for list of new steps.

    Attributes
    ----------
    steps : list[ResearchStep]
        New steps to add to the plan. Empty list if no new steps needed.
    """

    steps: list[ResearchStep] = Field(default_factory=list)


PLANNER_SYSTEM_PROMPT = """\
You are a research planning expert. Your task is to analyze questions and create
structured research plans that can be executed by a knowledge-grounded QA agent.

## Available Tools

The agent has access to these tools:
1. **web_search**: Google Search for finding relevant sources, current information, news, events
2. **web_fetch**: Fetch and read the full content of a URL (HTML pages AND PDFs). Use this for
   web pages, SEC filings, research papers, and any PDF documents.
3. **fetch_file**: Download data files (CSV, XLSX, JSON) for local searching
4. **grep_file**: Search within downloaded files for specific patterns
5. **read_file**: Read specific sections of downloaded files by line numbers
6. **synthesis**: Combining information from multiple sources into a coherent answer

## CRITICAL Planning Rules

### Rule 1: Find ONE Comprehensive Source First
For questions asking about multiple related items (e.g., "which of these 3 cities...",
"compare X, Y, and Z..."), ALWAYS:
- First search for a SINGLE authoritative source that contains ALL the data
- Then use web_fetch to get that source
- Then extract ALL data points from that ONE source
- DO NOT search separately for each item!

BAD plan (wasteful):
1. Search for "data about city A"
2. Search for "data about city B"
3. Search for "data about city C"

GOOD plan (efficient):
1. Search for comprehensive dataset/source that covers all cities
2. Fetch that authoritative source
3. Extract all relevant data from that source
4. Synthesize the answer

### Rule 2: Use Specific Extraction Prompts
When using web_fetch, provide specific prompts to extract exactly the data you need.
If you need multiple data points from the same page, use one web_fetch call with a
comprehensive extraction prompt rather than multiple calls.

### Rule 3: Minimize Searches
- Aim for 1-2 well-crafted searches, not many narrow searches
- Search for authoritative/official sources (government data, official statistics)
- One good source beats many partial sources

## Complexity Assessment

- "simple": Single fact lookup (e.g., "What is X?") → 2 steps (search + synthesis)
- "moderate": Multiple related facts or comparisons → 3 steps (search + fetch/verify + synthesis)
- "complex": Multi-part analysis, historical data, or multi-source verification → 3-5 steps

IMPORTANT: Every plan MUST include a final "synthesis" step to combine findings into a complete answer.
Even simple questions need at least 2 steps: research + synthesis.

## Output Format

Return a JSON object with this structure:
{
    "original_question": "the question",
    "complexity_assessment": "simple|moderate|complex",
    "steps": [
        {
            "step_id": 1,
            "description": "What this step does",
            "step_type": "research|synthesis",
            "depends_on": [],
            "expected_output": "What we expect to learn"
        }
    ],
    "reasoning": "Why this plan makes sense for the question"
}

Note: step_type is either "research" (gather information using tools) or "synthesis" (combine findings).
The final step should typically be "synthesis" to combine all research findings into an answer.
"""


def _parse_plan_response(response_text: str) -> ResearchPlan:
    """Parse the LLM response into a ResearchPlan.

    With structured output, this is a simple Pydantic validation.

    Parameters
    ----------
    response_text : str
        JSON response from the planning LLM (structured output).

    Returns
    -------
    ResearchPlan
        Parsed research plan.
    """
    try:
        return ResearchPlan.model_validate_json(response_text)
    except Exception as e:
        logger.error(f"Failed to parse plan response: {e}\nResponse was: {response_text[:500]}")
        raise ValueError(f"Failed to parse planner response: {e}") from e


class ResearchPlanner:
    """Creates multi-step research plans for complex questions.

    This planner analyzes questions and produces structured plans that
    can be executed and evaluated. Plans include tool recommendations,
    step dependencies, and expected outputs.

    Parameters
    ----------
    config : Configs, optional
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
        config: "Configs | None" = None,
        model: str | None = None,
    ) -> None:
        """Initialize the research planner.

        Parameters
        ----------
        config : Configs, optional
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

        # Temperature for planning (use config if available)
        self._temperature = config.default_temperature if config is not None else 0.3

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
                    temperature=self._temperature,
                    response_mime_type="application/json",
                    response_schema=ResearchPlan,
                ),
            )

            response_text = response.text or ""
            plan = _parse_plan_response(response_text)

            logger.info(f"Created plan with {len(plan.steps)} steps (complexity: {plan.complexity_assessment})")
            return plan

        except Exception as e:
            logger.error(f"Error creating research plan: {e}", exc_info=True)
            raise

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
                    temperature=self._temperature,
                    response_mime_type="application/json",
                    response_schema=ResearchPlan,
                ),
            )

            response_text = response.text or ""
            plan = _parse_plan_response(response_text)

            logger.info(f"Created plan with {len(plan.steps)} steps (complexity: {plan.complexity_assessment})")
            return plan

        except Exception as e:
            logger.error(f"Error creating research plan: {e}", exc_info=True)
            raise

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
                    temperature=self._temperature,
                    response_mime_type="application/json",
                    response_schema=NewStepsResponse,
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
                    temperature=self._temperature,
                    response_mime_type="application/json",
                    response_schema=NewStepsResponse,
                ),
            )

            response_text = response.text or ""
            new_steps = _parse_new_steps_response(response_text, plan.get_next_step_id())

            logger.info(f"Suggested {len(new_steps)} new steps")
            return new_steps

        except Exception as e:
            logger.error(f"Error suggesting new steps: {e}")
            return []

    async def reflect_and_update_plan_async(
        self,
        plan: ResearchPlan,
        completed_step: ResearchStep,
        step_result: str,
        all_findings: list[str],
        has_substantial_content: bool = False,
    ) -> PlanReflection:
        """Reflect on a completed step and decide how to update the plan.

        This is the core of dynamic planning - after each step, the agent
        reflects on what was found and decides whether to continue as planned,
        add new steps, skip steps, or finish early.

        Parameters
        ----------
        plan : ResearchPlan
            The current research plan.
        completed_step : ResearchStep
            The step that was just completed.
        step_result : str
            The result/output from the completed step.
        all_findings : list[str]
            All findings accumulated so far from previous steps.
        has_substantial_content : bool
            Whether substantial content has been gathered. Used to prevent
            premature skipping of steps.

        Returns
        -------
        PlanReflection
            The reflection with decision and any plan modifications.
        """
        prompt = self._build_reflection_prompt(plan, completed_step, step_result, all_findings)

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=types.Content(
                    role="user",
                    parts=[types.Part(text=prompt)],
                ),
                config=types.GenerateContentConfig(
                    system_instruction=REFLECTION_SYSTEM_PROMPT,
                    temperature=self._temperature,
                    response_mime_type="application/json",
                    response_schema=PlanReflection,
                ),
            )

            response_text = response.text or ""
            reflection = self._parse_reflection_response(response_text, plan.get_next_step_id())

            # Apply reflection (has_substantial_content prevents premature skipping)
            self._apply_reflection_to_plan(plan, reflection, has_substantial_content)

            logger.info(f"Reflection: can_answer={reflection.can_answer_now} - {reflection.reasoning[:100]}")
            return reflection

        except Exception as e:
            logger.error(f"Error during reflection: {e}")
            # Default to continuing (can_answer_now=False means continue)
            return PlanReflection(
                can_answer_now=False,
                reasoning=f"Continuing due to reflection error: {e}",
            )

    def _build_reflection_prompt(
        self,
        plan: ResearchPlan,
        completed_step: ResearchStep,
        step_result: str,
        all_findings: list[str],
    ) -> str:
        """Build the prompt for reflection.

        Parameters
        ----------
        plan : ResearchPlan
            The current research plan.
        completed_step : ResearchStep
            The step that was just completed.
        step_result : str
            The result from the completed step.
        all_findings : list[str]
            All findings so far.

        Returns
        -------
        str
            The reflection prompt.
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

            line = f"  {status_icon} Step {step.step_id}: {step.description}"
            if step.actual_output:
                line += f"\n      → {step.actual_output[:150]}..."
            status_lines.append(line)

        findings_summary = "\n".join(f"  - {f[:200]}" for f in all_findings) if all_findings else "  (none yet)"

        return f"""Original Question: {plan.original_question}

Current Plan Status:
{chr(10).join(status_lines)}

Just Completed Step {completed_step.step_id}: {completed_step.description}
Result: {step_result[:500]}

All Findings So Far:
{findings_summary}

Remaining Pending Steps: {[s.step_id for s in plan.get_steps_by_status(StepStatus.PENDING)]}

Based on this, decide how to proceed. Can we answer the question now? Do we need more research? Should we skip any remaining steps?
"""

    def _parse_reflection_response(self, response_text: str, next_step_id: int) -> PlanReflection:
        """Parse the reflection response.

        With structured output, this is a simple Pydantic validation.
        We still need to assign step IDs to any new steps.

        Parameters
        ----------
        response_text : str
            JSON response from the reflection LLM (structured output).
        next_step_id : int
            Starting ID for any new steps.

        Returns
        -------
        PlanReflection
            Parsed reflection.
        """
        try:
            reflection = PlanReflection.model_validate_json(response_text)

            # Assign step IDs to new steps (structured output doesn't know next ID)
            for i, step in enumerate(reflection.steps_to_add):
                step.step_id = next_step_id + i

            return reflection

        except Exception as e:
            logger.warning(f"Failed to parse reflection response: {e}")
            return PlanReflection(
                reasoning=f"Continuing due to parse error: {e}",
            )

    def _apply_reflection_to_plan(
        self,
        plan: ResearchPlan,
        reflection: PlanReflection,
        has_substantial_content: bool = False,
    ) -> None:
        """Apply reflection decisions to the plan.

        Parameters
        ----------
        plan : ResearchPlan
            The plan to modify.
        reflection : PlanReflection
            The reflection with decisions.
        has_substantial_content : bool
            Whether substantial content has been gathered. If False,
            remaining steps will NOT be skipped even if can_answer_now is True.
        """
        # 1. Update pending steps based on what was learned
        for update in reflection.steps_to_update:
            step = plan.get_step(update.step_id)
            if step and step.status == StepStatus.PENDING:
                plan.update_step(
                    update.step_id,
                    description=update.new_description if update.new_description else None,
                    expected_output=update.new_expected_output if update.new_expected_output else None,
                )
                logger.info(f"Updated step {update.step_id}: {update.new_description[:50]}...")

        # 2. Remove steps that are no longer needed
        for step_id in reflection.steps_to_remove:
            step = plan.get_step(step_id)
            if step and step.status == StepStatus.PENDING:
                # Don't remove synthesis steps
                if step.step_type == "synthesis":
                    logger.warning(f"Refusing to remove synthesis step {step_id}")
                    continue
                plan.update_step(step_id, status=StepStatus.SKIPPED)
                logger.info(f"Removed step {step_id}: {step.description[:50]}...")

        # 3. Add new steps if needed
        for new_step in reflection.steps_to_add:
            plan.add_step(new_step)
            logger.info(f"Added step {new_step.step_id}: {new_step.description[:50]}...")

        # If can_answer_now, skip remaining steps (only if we have content)
        if reflection.can_answer_now and has_substantial_content:
            for step in plan.get_steps_by_status(StepStatus.PENDING):
                if step.step_type == "synthesis":
                    # Never skip synthesis
                    continue
                plan.update_step(step.step_id, status=StepStatus.SKIPPED)
                logger.info(f"Skipping step {step.step_id} - can answer now")

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
- Attempts: {failed_step.attempts}
- Failure reason: {failed_step.failure_reason}
"""

        prompt += "\nBased on this, suggest any new steps needed to complete the research."

        return prompt


REPLANNING_SYSTEM_PROMPT = """\
You are a research planning expert helping to dynamically adjust research plans.

Given the current state of a research plan and recent execution results, suggest
new steps if needed.

## CRITICAL: Avoid Redundant Searches
- If we already fetched a useful source, extract MORE from that source instead of searching again
- DO NOT suggest "search for X" if a similar search was already done
- Each new search should have SIGNIFICANTLY different keywords

## When to Add Steps

1. **Better Source Exists**: If current source lacks data, search for a MORE authoritative source
   - Government statistics portals (ONS, Census Bureau, etc.)
   - Official databases and datasets
   - Academic or institutional sources

2. **Extract from Existing**: If we have a good source but need more data from it:
   - Use web_fetch with a different, more specific extraction prompt
   - DO NOT re-search when a follow-up web_fetch on the same URL would suffice

## When NOT to Add Steps

- If we already searched 3+ times for similar information
- If the question can be answered with current findings (even if incomplete)
- If adding more searches is unlikely to find better data

## Output Format

Return a JSON object with a "steps" array (empty array if no new steps needed):
{
    "steps": [
        {
            "description": "What this step does",
            "step_type": "research|synthesis",
            "depends_on": [],
            "expected_output": "What we expect to learn"
        }
    ]
}

PREFER empty steps array over adding low-value research steps.
"""


REFLECTION_SYSTEM_PROMPT = """\
You are a research assistant reflecting on your progress after completing a research step.
Your job is to analyze what was found and decide how to adjust the research plan.

## Plan Modification Options

After each step, you can:
1. **Update steps**: Modify pending steps based on what you learned (PREFERRED)
2. **Remove steps**: Delete steps that are no longer needed
3. **Add steps**: Add new steps if genuinely new work is discovered
4. **Finish early**: Set can_answer_now=true if you have verified the answer

**IMPORTANT**: Prefer UPDATING existing steps over removing and adding new ones.
If step 2 needs a different approach, UPDATE step 2's description rather than removing it and adding step 4.

## CRITICAL: Terminology Discovery

When you discover a SPECIFIC TERM for what you're researching:
- Question asks about "buffs" -> source calls them "Sigils" -> UPDATE next step to search for "Sigils"
- Question asks about "types" -> source calls them "Classes" -> UPDATE next step to search for "Classes"

## Verification Before can_answer_now=true

Before setting can_answer_now=true, verify:

1. **Do we have the EXACT answer?**
   - If asked for "3 categories", do we have exactly 3 named items from fetched content?
   - If asked for a specific name, do we have that name (not a description)?

2. **Is it from actual page content (not just search snippets)?**
   - Search snippets are often incomplete or outdated
   - The answer must be in fetched document content

If ANY answer is "no" -> keep can_answer_now=false.

## When to UPDATE Steps

UPDATE pending steps when:
- You discovered a specific term to search for
- The approach needs refinement based on what you learned
- The expected output needs to be more specific

## When to REMOVE Steps

REMOVE steps when:
- A step is now redundant (we already have that information)
- The step was based on an assumption that proved wrong
- Never remove synthesis steps

## When to ADD Steps

ADD new steps only when:
- Genuinely new work is discovered that doesn't fit existing steps
- Multiple independent sources need to be checked
- Avoid if you could just UPDATE an existing step instead

## Output Format

Return a JSON object:
{
    "can_answer_now": false,
    "key_findings": "Brief summary of what was learned",
    "reasoning": "Why these plan changes make sense",
    "steps_to_update": [
        {
            "step_id": 2,
            "new_description": "Updated task description",
            "new_expected_output": "Updated expected output"
        }
    ],
    "steps_to_remove": [3],
    "steps_to_add": [
        {
            "description": "New task if truly needed",
            "step_type": "research",
            "expected_output": "What we expect to learn"
        }
    ]
}

All arrays can be empty. Prefer updating over removing+adding.
"""


def _parse_new_steps_response(response_text: str, start_id: int) -> list[ResearchStep]:
    """Parse the replanning response into new steps.

    With structured output, this is a simple Pydantic validation.
    We still need to assign step IDs to the new steps.

    Parameters
    ----------
    response_text : str
        JSON response from the replanning LLM (structured output).
    start_id : int
        Starting step ID for new steps.

    Returns
    -------
    list[ResearchStep]
        Parsed new steps.
    """
    try:
        response = NewStepsResponse.model_validate_json(response_text)

        # Assign step IDs (structured output doesn't know the next ID)
        for i, step in enumerate(response.steps):
            step.step_id = start_id + i

        return response.steps

    except Exception as e:
        logger.warning(f"Failed to parse new steps response: {e}")
        return []

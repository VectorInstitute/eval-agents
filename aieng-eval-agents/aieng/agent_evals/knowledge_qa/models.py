"""Data models for the Knowledge-Grounded QA Agent.

This module contains Pydantic models for representing research plans,
execution traces, and agent responses.
"""

from typing import Any

from aieng.agent_evals.tools import GroundingChunk
from pydantic import BaseModel, Field


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
    steps : list[ResearchStep]
        Ordered list of research steps to execute.
    reasoning : str
        Explanation of why this plan was chosen.
    """

    original_question: str
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

    def is_complete(self) -> bool:
        """Check if all steps are either completed, failed, or skipped.

        Returns
        -------
        bool
            True if no steps are pending or in progress.
        """
        terminal_statuses = {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED}
        return all(s.status in terminal_statuses for s in self.steps)


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


class AgentResponse(BaseModel):
    """Response from the knowledge agent with execution trace.

    Contains the answer text along with metadata about how the agent
    arrived at the answer: the research plan, tool calls, sources, and reasoning.

    Attributes
    ----------
    text : str
        The generated response text.
    plan : ResearchPlan
        The research plan created for the question.
    execution_trace : list[StepExecution]
        Record of each step's execution.
    sources : list[GroundingChunk]
        Web sources used in the response.
    search_queries : list[str]
        Search queries executed.
    reasoning_chain : list[str]
        Step-by-step reasoning trace.
    tool_calls : list[dict]
        Raw tool calls made during execution.
    total_duration_ms : int
        Total execution time in milliseconds.
    """

    text: str
    plan: ResearchPlan
    execution_trace: list[StepExecution] = Field(default_factory=list)
    sources: list[GroundingChunk] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    reasoning_chain: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    total_duration_ms: int = 0

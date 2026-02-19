"""Plan parsing utilities for PlanReAct planner output.

This module provides data models and parsing functions for research plans
from the PlanReActPlanner's tagged output format (PLANNING, REPLANNING,
REASONING, etc.).
"""

import re
from enum import Enum

from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    """Status constants for research steps.

    This enum uses string values for easy serialization and comparison.
    """

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
    status : StepStatus
        Current execution status: pending, in_progress, completed, failed, or skipped.
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
    status: StepStatus = StepStatus.PENDING
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
        status: StepStatus | None = None,
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
        status : StepStatus, optional
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

    def get_steps_by_status(self, status: StepStatus) -> list[ResearchStep]:
        """Get all steps with a specific status.

        Parameters
        ----------
        status : StepStatus
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


# PlanReActPlanner tag constants (from google.adk.planners.plan_re_act_planner)
PLANNING_TAG = "/*PLANNING*/"
REPLANNING_TAG = "/*REPLANNING*/"
REASONING_TAG = "/*REASONING*/"
ACTION_TAG = "/*ACTION*/"
FINAL_ANSWER_TAG = "/*FINAL_ANSWER*/"


def extract_plan_text(text: str) -> str | None:
    """Extract plan text from PLANNING or REPLANNING tags.

    Parameters
    ----------
    text : str
        Text that may contain planning tags.

    Returns
    -------
    str | None
        The plan text if found, None otherwise.
    """
    # Check for REPLANNING first (updated plan takes precedence)
    for tag in [REPLANNING_TAG, PLANNING_TAG]:
        if tag in text:
            start = text.find(tag) + len(tag)
            # Find the end - next tag or end of text
            end = len(text)
            for end_tag in [REASONING_TAG, ACTION_TAG, FINAL_ANSWER_TAG, PLANNING_TAG, REPLANNING_TAG]:
                if end_tag in text[start:]:
                    tag_pos = text.find(end_tag, start)
                    if tag_pos != -1 and tag_pos < end:
                        end = tag_pos
            plan_text = text[start:end].strip()
            if plan_text:
                return plan_text
    return None


def parse_plan_steps_from_text(plan_text: str) -> list[ResearchStep]:
    """Parse numbered steps from plan text.

    Parameters
    ----------
    plan_text : str
        Raw plan text, typically with numbered steps.

    Returns
    -------
    list[ResearchStep]
        Parsed research steps.
    """
    steps = []
    # Match numbered steps: "1. Description", "1) Description", or "Step 1: Description"
    patterns = [
        r"^\s*(\d+)[.\)]\s*(.+?)(?=\n\s*\d+[.\)]|\n\s*Step\s+\d+|\Z)",  # "1. desc" or "1) desc"
        r"^\s*Step\s+(\d+)[:\.]?\s*(.+?)(?=\n\s*Step\s+\d+|\n\s*\d+[.\)]|\Z)",  # "Step 1: desc"
        r"^\s*[-*]\s*(.+?)(?=\n\s*[-*]|\Z)",  # Bullet points
    ]

    # Try numbered patterns first
    for pattern in patterns[:2]:
        matches = re.findall(pattern, plan_text, re.MULTILINE | re.DOTALL)
        if matches:
            for i, match in enumerate(matches[:10]):  # Max 10 steps
                step_num = int(match[0]) if len(match) > 1 else i + 1
                description = match[1] if len(match) > 1 else match[0]
                description = description.strip()
                # Clean up description - remove trailing newlines and extra whitespace
                description = " ".join(description.split())
                if description and len(description) > 5:
                    steps.append(
                        ResearchStep(
                            step_id=step_num,
                            description=description[:200],
                            step_type="research",
                            status=StepStatus.PENDING,
                        )
                    )
            if steps:
                return steps

    # Try bullet pattern
    matches = re.findall(patterns[2], plan_text, re.MULTILINE | re.DOTALL)
    if matches:
        for i, desc in enumerate(matches[:10], 1):
            description = " ".join(desc.strip().split())
            if description and len(description) > 5:
                steps.append(
                    ResearchStep(
                        step_id=i,
                        description=description[:200],
                        step_type="research",
                        status=StepStatus.PENDING,
                    )
                )
        if steps:
            return steps

    # Fallback: split by newlines if no pattern matched
    lines = [line.strip() for line in plan_text.split("\n") if line.strip() and len(line.strip()) > 10]
    for i, line in enumerate(lines[:10], 1):
        # Skip lines that look like headers
        if line.endswith(":") or line.startswith("#"):
            continue
        steps.append(
            ResearchStep(
                step_id=i,
                description=line[:200],
                step_type="research",
                status=StepStatus.PENDING,
            )
        )

    return steps


def extract_reasoning_text(text: str) -> str | None:
    """Extract reasoning text from REASONING tag.

    Parameters
    ----------
    text : str
        Text that may contain reasoning tag.

    Returns
    -------
    str | None
        The reasoning text if found, None otherwise.
    """
    if REASONING_TAG not in text:
        return None

    start = text.find(REASONING_TAG) + len(REASONING_TAG)
    end = len(text)
    for end_tag in [ACTION_TAG, FINAL_ANSWER_TAG, PLANNING_TAG, REPLANNING_TAG]:
        if end_tag in text[start:]:
            tag_pos = text.find(end_tag, start)
            if tag_pos != -1 and tag_pos < end:
                end = tag_pos
    return text[start:end].strip() or None


def extract_final_answer_text(text: str) -> str | None:
    """Extract final answer text from FINAL_ANSWER tag.

    Parameters
    ----------
    text : str
        Text that may contain final answer tag.

    Returns
    -------
    str | None
        The final answer text if found, None if tag missing or content empty.
    """
    if not text or FINAL_ANSWER_TAG not in text:
        return None

    start = text.find(FINAL_ANSWER_TAG) + len(FINAL_ANSWER_TAG)

    # Find the end - truncate at the next tag if any
    end = len(text)
    for end_tag in [PLANNING_TAG, REPLANNING_TAG, REASONING_TAG, ACTION_TAG]:
        if end_tag in text[start:]:
            tag_pos = text.find(end_tag, start)
            if tag_pos != -1 and tag_pos < end:
                end = tag_pos

    answer_text = text[start:end].strip()

    # Return None for empty/whitespace-only content
    if not answer_text:
        return None

    return answer_text

"""Plan parsing utilities for PlanReAct planner output.

This module provides functions to parse research plans from the
PlanReActPlanner's tagged output format (PLANNING, REPLANNING, REASONING, etc.).
"""

import re

from .models import ResearchStep, StepStatus


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
    answer_text = text[start:].strip()

    # Return None for empty/whitespace-only content
    if not answer_text:
        return None

    # Handle case where FINAL_ANSWER is followed by another tag with no content between
    for end_tag in [PLANNING_TAG, REPLANNING_TAG, REASONING_TAG, ACTION_TAG]:
        if answer_text.startswith(end_tag):
            return None

    return answer_text

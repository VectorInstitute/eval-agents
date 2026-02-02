"""Notebook display utilities for the Knowledge Agent.

Provides live progress display for Jupyter notebooks, showing plan status
and tool calls while the agent works.

Example
-------
>>> from aieng.agent_evals.knowledge_agent import KnowledgeGroundedAgent
>>> from aieng.agent_evals.knowledge_agent.notebook import run_with_display
>>> agent = KnowledgeGroundedAgent(enable_planning=True)
>>> response = await run_with_display(agent, "What is quantum computing?")
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from IPython.display import HTML, clear_output, display


if TYPE_CHECKING:
    from .agent import KnowledgeGroundedAgent
    from .models import AgentResponse, ResearchPlan


logger = logging.getLogger(__name__)


class ToolCallCapture(logging.Handler):
    """Captures tool calls from agent logs for display."""

    def __init__(self):
        super().__init__()
        self.tool_calls: list[dict] = []

    def emit(self, record):
        """Capture tool call and response log messages."""
        msg = record.getMessage()
        if "Tool call:" in msg:
            try:
                parts = msg.split("Tool call: ", 1)[1]
                paren_idx = parts.find("(")
                if paren_idx > 0:
                    tool_name = parts[:paren_idx]
                    args_str = parts[paren_idx + 1 : -1]
                    if len(args_str) > 60:
                        args_str = args_str[:57] + "..."
                    self.tool_calls.append({"name": tool_name, "args": args_str, "completed": False})
            except Exception:
                pass
        elif "Tool response:" in msg:
            try:
                parts = msg.split("Tool response: ", 1)[1]
                tool_name = parts.split(" ")[0]
                for tc in reversed(self.tool_calls):
                    if tc["name"] == tool_name and not tc["completed"]:
                        tc["completed"] = True
                        break
            except Exception:
                pass

    def clear(self):
        """Clear captured tool calls."""
        self.tool_calls = []


def _format_plan_html(plan: "ResearchPlan") -> str:
    """Format the research plan as HTML."""
    from .models import StepStatus  # noqa: PLC0415

    lines = ['<div style="font-family: monospace; padding: 10px; background: #f8f9fa; border-radius: 8px;">']
    lines.append('<div style="font-weight: bold; margin-bottom: 8px;">📋 Research Plan</div>')

    for step in plan.steps:
        if step.status == StepStatus.COMPLETED:
            icon, color = "✓", "#28a745"
        elif step.status == StepStatus.FAILED:
            icon, color = "✗", "#dc3545"
        elif step.status == StepStatus.IN_PROGRESS:
            icon, color = "→", "#ffc107"
        elif step.status == StepStatus.SKIPPED:
            icon, color = "○", "#6c757d"
        else:
            icon, color = "○", "#adb5bd"

        lines.append(f'<div style="color: {color}; margin: 4px 0;">{icon} {step.step_id}. {step.description}</div>')

    lines.append("</div>")
    return "\n".join(lines)


def _format_tools_html(tool_calls: list[dict]) -> str:
    """Format tool calls as HTML."""
    if not tool_calls:
        return '<div style="color: #6c757d;">Waiting for tool calls...</div>'

    lines = [
        '<div style="font-family: monospace; padding: 10px; background: #e9ecef; border-radius: 8px; margin-top: 8px;">'
    ]
    lines.append(f'<div style="font-weight: bold; margin-bottom: 8px;">🔧 Tool Calls ({len(tool_calls)})</div>')

    # Show last 8 tool calls
    display_calls = tool_calls[-8:]
    if len(tool_calls) > 8:
        lines.append(f'<div style="color: #6c757d;">... ({len(tool_calls) - 8} earlier calls)</div>')

    tool_icons = {
        "google_search": "🔍",
        "google_search_agent": "🔍",
        "fetch_url": "🌐",
        "web_fetch": "🌐",
        "read_pdf": "📄",
        "grep_file": "📑",
        "read_file": "📖",
    }

    for tc in display_calls:
        name = tc["name"]
        if name == "google_search_agent":
            name = "google_search"
        icon = tool_icons.get(name, "🔧")
        status_icon = "✓" if tc.get("completed") else "→"
        status_color = "#28a745" if tc.get("completed") else "#ffc107"

        lines.append(
            f'<div style="margin: 2px 0;">'
            f'<span style="color: {status_color};">{status_icon}</span> '
            f"{icon} <b>{name}</b> "
            f'<span style="color: #6c757d;">{tc["args"]}</span>'
            f"</div>"
        )

    lines.append("</div>")
    return "\n".join(lines)


def _format_display_html(plan: "ResearchPlan | None", tool_calls: list[dict], question: str) -> str:
    """Create the full HTML display."""
    html = ['<div style="max-width: 800px;">']

    # Question
    html.append(
        f'<div style="padding: 10px; background: #cfe2ff; border-radius: 8px; margin-bottom: 8px;">'
        f"<b>Question:</b> {question}</div>"
    )

    # Plan
    if plan and plan.steps:
        html.append(_format_plan_html(plan))

    # Tools
    html.append(_format_tools_html(tool_calls))

    html.append("</div>")
    return "\n".join(html)


async def run_with_display(
    agent: "KnowledgeGroundedAgent",
    question: str,
    refresh_rate: float = 0.5,
) -> "AgentResponse":
    """Run the agent with live progress display in a Jupyter notebook.

    Shows the research plan checklist and tool calls while the agent works,
    updating the display periodically.

    Parameters
    ----------
    agent : KnowledgeGroundedAgent
        The agent to run.
    question : str
        The question to answer.
    refresh_rate : float
        How often to update the display in seconds (default 0.5).

    Returns
    -------
    AgentResponse
        The agent's response.

    Example
    -------
    >>> agent = KnowledgeGroundedAgent(enable_planning=True)
    >>> response = await run_with_display(agent, "What is quantum computing?")
    >>> print(response.text)
    """
    # Suppress verbose logging from external libraries (same as CLI)
    verbose_loggers = ["google.adk", "google.genai", "httpx", "httpcore"]
    original_levels = {}
    for name in verbose_loggers:
        _logger = logging.getLogger(name)
        original_levels[name] = _logger.level
        _logger.setLevel(logging.ERROR)
        _logger.propagate = False

    # Set up tool call capture on the agent logger (same as CLI)
    tool_capture = ToolCallCapture()
    tool_capture.setLevel(logging.INFO)
    agent_logger = logging.getLogger("aieng.agent_evals.knowledge_agent.agent")
    original_agent_level = agent_logger.level
    original_handlers = agent_logger.handlers.copy()
    agent_logger.handlers.clear()
    agent_logger.addHandler(tool_capture)
    agent_logger.setLevel(logging.INFO)
    agent_logger.propagate = False

    try:
        # Create the plan first if planning is enabled
        if agent.enable_planning and hasattr(agent, "create_plan_async"):
            clear_output(wait=True)
            display(HTML('<div style="color: #6c757d;">Creating research plan...</div>'))
            await agent.create_plan_async(question)

        # Start the agent task
        task = asyncio.create_task(agent.answer_async(question))

        # Update display while agent works
        while not task.done():
            clear_output(wait=True)
            display(
                HTML(
                    _format_display_html(
                        plan=agent.current_plan if hasattr(agent, "current_plan") else None,
                        tool_calls=tool_capture.tool_calls,
                        question=question,
                    )
                )
            )
            await asyncio.sleep(refresh_rate)

        # Get the result
        response = await task

        # Final display with completion status
        clear_output(wait=True)
        display(
            HTML(
                _format_display_html(
                    plan=agent.current_plan if hasattr(agent, "current_plan") else None,
                    tool_calls=tool_capture.tool_calls,
                    question=question,
                )
                + f'<div style="margin-top: 12px; padding: 10px; background: #d4edda; border-radius: 8px;">'
                f"✓ Complete in {response.total_duration_ms / 1000:.1f}s | "
                f"{len(response.tool_calls)} tool calls | "
                f"{len(response.sources)} sources</div>"
            )
        )

        return response

    finally:
        # Clean up logging - restore original state
        agent_logger.removeHandler(tool_capture)
        agent_logger.handlers = original_handlers
        agent_logger.setLevel(original_agent_level)
        agent_logger.propagate = True

        # Restore verbose logger levels
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)

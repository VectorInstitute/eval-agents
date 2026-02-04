#!/usr/bin/env python3
"""Knowledge Agent CLI.

Command-line interface for running and evaluating the Knowledge-Grounded QA Agent.

Usage::

    knowledge-qa ask "What is..."
    knowledge-qa eval --samples 3
    knowledge-qa eval --ids 123 456 789
    knowledge-qa sample --ids 123
    knowledge-qa sample --category "Finance & Economics" --count 5
"""

import argparse
import asyncio
import io
import logging
import re
import sys
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text


# Load .env file from current directory or parent directories
def _load_env() -> None:
    """Load environment variables from .env file."""
    # Try current directory first, then walk up
    for parent in [Path.cwd(), *Path.cwd().parents]:
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return
    # Fallback to default dotenv behavior
    load_dotenv()


_load_env()

console = Console()

# Vector Institute cyan color
VECTOR_CYAN = "#00B4D8"


def get_version() -> str:
    """Get the installed version of the package."""
    try:
        return version("aieng-eval-agents")
    except Exception:
        return "dev"


def _get_model_config() -> tuple[str, str]:
    """Get model names from config.

    Returns
    -------
    tuple[str, str]
        The worker model and evaluator model names from config.
    """
    from aieng.agent_evals.configs import Configs  # noqa: PLC0415

    try:
        config = Configs()  # type: ignore[call-arg]
        return config.default_worker_model, config.default_evaluator_model
    except Exception:
        return "gemini-2.5-flash", "gemini-2.5-pro"


def display_banner() -> None:
    """Display the CLI banner with version and model info."""
    ver = get_version()
    worker_model, evaluator_model = _get_model_config()

    # Robot face with magnifying glass
    line0 = Text()
    line0.append("  ◯─◯    ", style=f"{VECTOR_CYAN} bold")
    line0.append("knowledge-qa ", style="white bold")
    line0.append(f"v{ver}", style="bright_black")

    line1 = Text()
    line1.append(" ╱ 🔍 ╲   ", style=f"{VECTOR_CYAN} bold")
    line1.append("Agent: ", style="dim")
    line1.append(worker_model, style="cyan")

    line2 = Text()
    line2.append(" │    │   ", style=f"{VECTOR_CYAN} bold")
    line2.append("Evaluator: ", style="dim")
    line2.append(evaluator_model, style="yellow")

    line3 = Text()
    line3.append("  ╲__╱   ", style=f"{VECTOR_CYAN} bold")
    line3.append("Vector Institute AI Engineering", style="bright_black")

    console.print()
    console.print(line0)
    console.print(line1)
    console.print(line2)
    console.print(line3)
    console.print()


def display_tools_info() -> None:
    """Display information about available tools."""
    console.print("[bold]Available Tools:[/bold]")
    console.print()

    tools = [
        ("google_search", "blue", "Search the web for current information and sources"),
        ("fetch_url", "green", "Fetch webpage content and save locally for searching"),
        ("grep_file", "cyan", "Search within fetched files for matching patterns"),
        ("read_file", "cyan", "Read sections of fetched files"),
        ("read_pdf", "green", "Read and extract text from PDF documents"),
    ]

    for name, color, desc in tools:
        console.print(f"  [{color}]{name:<16}[/{color}] {desc}")

    console.print()


def _parse_structured_answer(text: str) -> dict[str, str] | None:
    """Parse structured answer format (ANSWER/SOURCES/REASONING).

    Parameters
    ----------
    text : str
        The raw response text.

    Returns
    -------
    dict[str, str] | None
        Parsed sections or None if parsing fails.
    """
    if not text:
        return None

    # Check if text contains our structured format
    text_upper = text.upper()
    if "ANSWER:" not in text_upper:
        return None

    result = {"answer": "", "sources": "", "reasoning": ""}

    # Find positions of each section (case-insensitive)
    import re  # noqa: PLC0415

    # Match ANSWER:, SOURCES:, REASONING: with flexible spacing
    answer_match = re.search(r"ANSWER:\s*", text, re.IGNORECASE)
    sources_match = re.search(r"SOURCES:\s*", text, re.IGNORECASE)
    reasoning_match = re.search(r"REASONING:\s*", text, re.IGNORECASE)

    if answer_match:
        start = answer_match.end()
        # Find end - next section or end of text
        end = len(text)
        if sources_match and sources_match.start() > start:
            end = min(end, sources_match.start())
        if reasoning_match and reasoning_match.start() > start:
            end = min(end, reasoning_match.start())
        result["answer"] = text[start:end].strip()

    if sources_match:
        start = sources_match.end()
        end = len(text)
        if reasoning_match and reasoning_match.start() > start:
            end = min(end, reasoning_match.start())
        if answer_match and answer_match.start() > start:
            end = min(end, answer_match.start())
        result["sources"] = text[start:end].strip()

    if reasoning_match:
        start = reasoning_match.end()
        end = len(text)
        if sources_match and sources_match.start() > start:
            end = min(end, sources_match.start())
        if answer_match and answer_match.start() > start:
            end = min(end, answer_match.start())
        result["reasoning"] = text[start:end].strip()

    # Return None if we didn't extract any meaningful content
    if not result["answer"]:
        return None

    return result


class ToolCallHandler(logging.Handler):
    """Custom logging handler that captures tool calls for rich display."""

    def __init__(self):
        super().__init__()
        self.tool_calls: list[dict] = []

    def emit(self, record):
        """Process a log record, capturing tool calls for display."""
        msg = record.getMessage()
        if "Tool call:" in msg:
            try:
                parts = msg.split("Tool call: ", 1)[1]
                paren_idx = parts.find("(")
                if paren_idx > 0:
                    tool_name = parts[:paren_idx]
                    args_str = parts[paren_idx + 1 : -1]
                    if len(args_str) > 80:
                        args_str = args_str[:77] + "..."
                    self.tool_calls.append(
                        {
                            "name": tool_name,
                            "args": args_str,
                            "completed": False,
                            "failed": False,
                            "error": None,
                        }
                    )
            except Exception:
                pass
        elif "Tool error:" in msg:
            # Mark the most recent incomplete tool call as failed
            try:
                parts = msg.split("Tool error: ", 1)[1]
                # Format: "tool_name failed - error message"
                tool_part, error_msg = (
                    parts.split(" failed - ", 1) if " failed - " in parts else (parts, "Unknown error")
                )
                tool_name = tool_part.strip()
                # Find the most recent matching incomplete tool call
                for tc in reversed(self.tool_calls):
                    if tc["name"] == tool_name and not tc["completed"] and not tc["failed"]:
                        tc["failed"] = True
                        tc["error"] = error_msg[:60] + "..." if len(error_msg) > 60 else error_msg
                        break
            except Exception:
                pass
        elif "Tool response:" in msg:
            # Mark the most recent incomplete tool call as completed
            try:
                parts = msg.split("Tool response: ", 1)[1]
                tool_name = parts.split(" ")[0]
                # Find the most recent matching incomplete tool call
                for tc in reversed(self.tool_calls):
                    if tc["name"] == tool_name and not tc["completed"] and not tc["failed"]:
                        tc["completed"] = True
                        break
            except Exception:
                pass

    def clear(self):
        """Reset captured tool calls."""
        self.tool_calls = []


def _parse_markdown_bold(text: str, base_style: str) -> Text:
    """Parse markdown-style **bold** markers and return styled Rich Text.

    Parameters
    ----------
    text : str
        Text that may contain **bold** markers.
    base_style : str
        The base style to apply to non-bold text.

    Returns
    -------
    Text
        Rich Text object with bold sections properly styled.
    """
    result = Text()
    # Match **text** patterns
    pattern = r"\*\*([^*]+)\*\*"
    last_end = 0

    for match in re.finditer(pattern, text):
        # Add text before the match with base style
        if match.start() > last_end:
            result.append(text[last_end : match.start()], style=base_style)
        # Add the bold text (combine bold with base style)
        bold_style = f"bold {base_style}" if base_style else "bold"
        result.append(match.group(1), style=bold_style)
        last_end = match.end()

    # Add remaining text after last match
    if last_end < len(text):
        result.append(text[last_end:], style=base_style)

    return result


def _create_plan_display(plan) -> Panel:
    """Create a rich panel showing the research plan checklist.

    Parameters
    ----------
    plan : ResearchPlan
        The research plan to display. Step statuses are read directly from the plan.

    Returns
    -------
    Panel
        A rich panel with the plan checklist.
    """
    from .models import StepStatus  # noqa: PLC0415

    lines = []

    for step in plan.steps:
        # Use the step's actual status from the plan (updated by the agent in real-time)
        if step.status == StepStatus.COMPLETED:
            icon, icon_style = "✓", "green"
            desc_style = "dim"
        elif step.status == StepStatus.FAILED:
            icon, icon_style = "✗", "red"
            desc_style = "red"
        elif step.status == StepStatus.IN_PROGRESS:
            icon, icon_style = "→", "bold yellow"
            desc_style = "yellow"
        elif step.status == StepStatus.SKIPPED:
            icon, icon_style = "○", "dim"
            desc_style = "strike dim"
        else:
            # PENDING - not yet started
            icon, icon_style = "○", "dim"
            desc_style = "dim"

        line = Text()
        line.append("  ")
        line.append(icon, style=icon_style)
        line.append(f" {step.step_id}. ", style="bold")
        # Parse markdown bold markers in description
        styled_desc = _parse_markdown_bold(step.description, desc_style)
        line.append_text(styled_desc)
        lines.append(line)

    content = Group(*lines) if lines else Text("No plan steps", style="dim")

    return Panel(
        content,
        title="[bold magenta]📋 Research Plan[/bold magenta]",
        subtitle=f"[dim]{len(plan.steps)} steps[/dim]",
        border_style="magenta",
        padding=(0, 1),
    )


def _get_tool_display_info(name: str) -> tuple[str, str, str]:
    """Get display name, icon, and style for a tool.

    Returns (display_name, icon, style).
    """
    # Normalize tool name for display
    display_name = "google_search" if name == "google_search_agent" else name

    # Tool icon and style lookup
    tool_styles = {
        "fetch_url": ("🌐", "green"),
        "read_pdf": ("📄", "green"),
        "grep_file": ("📑", "cyan"),
        "read_file": ("📖", "cyan"),
        "google_search": ("🔍", "blue"),
        "google_search_agent": ("🔍", "blue"),
    }
    icon, style = tool_styles.get(name, ("🔧", "white"))
    return display_name, icon, style


def _create_compact_question_panel(
    question: str, example_id: int | None = None, answer_type: str | None = None
) -> Panel:
    """Create a compact question panel for the live display.

    Parameters
    ----------
    question : str
        The question text.
    example_id : int, optional
        The example ID if in eval mode.
    answer_type : str, optional
        The answer type if in eval mode.

    Returns
    -------
    Panel
        A compact question panel.
    """
    title = "[bold blue]📋 Question[/bold blue]"
    if example_id is not None:
        title = f"[bold blue]📋 Question (ID: {example_id})[/bold blue]"

    subtitle = f"[dim]Answer Type: {answer_type}[/dim]" if answer_type else None

    return Panel(
        question,
        title=title,
        subtitle=subtitle,
        border_style="blue",
        padding=(0, 1),
    )


def _create_compact_ground_truth_panel(ground_truth: str) -> Panel:
    """Create a compact ground truth panel for the live display.

    Parameters
    ----------
    ground_truth : str
        The ground truth answer.

    Returns
    -------
    Panel
        A compact ground truth panel.
    """
    # Truncate long ground truth for display
    display_gt = ground_truth if len(ground_truth) <= 150 else ground_truth[:147] + "..."

    return Panel(
        f"[yellow]{display_gt}[/yellow]",
        title="[bold yellow]🎯 Ground Truth[/bold yellow]",
        border_style="yellow",
        padding=(0, 1),
    )


def create_tool_display(
    tool_calls: list[dict],
    plan=None,
    context_percent: float | None = None,
    question: str | None = None,
    ground_truth: str | None = None,
    example_id: int | None = None,
    answer_type: str | None = None,
) -> Group | Panel:
    """Create a rich display showing tool calls and optionally the plan.

    Parameters
    ----------
    tool_calls : list[dict]
        List of tool calls made so far.
    plan : ResearchPlan, optional
        If provided, shows the plan checklist above tool calls.
    context_percent : float, optional
        Percentage of context window remaining.
    question : str, optional
        The question being answered (for eval mode display).
    ground_truth : str, optional
        The ground truth answer (for eval mode display).
    example_id : int, optional
        The example ID (for eval mode display).
    answer_type : str, optional
        The answer type (for eval mode display).

    Returns
    -------
    Group or Panel
        The display content.
    """
    tool_content = _build_tool_calls_content(tool_calls, plan is not None)

    # Build subtitle with tool calls and context usage
    subtitle_parts = [f"{len(tool_calls)} tool calls"]
    if context_percent is not None:
        # Color code based on remaining context
        if context_percent > 50:
            color = "green"
        elif context_percent > 20:
            color = "yellow"
        else:
            color = "red"
        subtitle_parts.append(f"[{color}]{context_percent:.0f}% context left[/{color}]")

    tool_panel = Panel(
        tool_content,
        title="[bold cyan]🔧 Agent Working[/bold cyan]",
        subtitle=f"[dim]{' | '.join(subtitle_parts)}[/dim]",
        border_style="cyan",
        padding=(0, 1),
    )

    # Build the display components
    components: list[Panel | Text] = []

    # Add question and ground truth panels if in eval mode
    if question is not None:
        components.append(_create_compact_question_panel(question, example_id, answer_type))
        components.append(Text(""))

    if ground_truth is not None:
        components.append(_create_compact_ground_truth_panel(ground_truth))
        components.append(Text(""))

    # Add plan if available
    if plan and plan.steps:
        components.append(_create_plan_display(plan))
        components.append(Text(""))

    # Always add the tool panel
    components.append(tool_panel)

    # Return as group if we have multiple components, otherwise just the tool panel
    if len(components) > 1:
        return Group(*components)
    return tool_panel


def _build_tool_calls_content(tool_calls: list[dict], has_plan: bool) -> Group | Text:
    """Build the content for tool calls display."""
    if not tool_calls:
        return Text("Waiting for tool calls...", style="dim")

    lines = []
    display_calls = tool_calls[-6:] if has_plan else tool_calls[-8:]
    if len(tool_calls) > len(display_calls):
        lines.append(Text(f"  ... ({len(tool_calls) - len(display_calls)} earlier calls)", style="dim"))

    for tc in display_calls:
        is_completed = tc.get("completed", False)
        is_failed = tc.get("failed", False)
        display_name, icon, style = _get_tool_display_info(tc["name"])

        line = Text()
        if is_failed:
            line.append("  ✗ ", style="bold red")
            line.append(f"{icon} ", style="red")
            line.append(display_name, style="bold red")
            line.append(f"  {tc['args']}", style="dim red")
            if tc.get("error"):
                line.append(f"  [{tc['error']}]", style="red")
        elif is_completed:
            line.append("  ✓ ", style="dim green")
            line.append(f"{icon} ", style=style)
            line.append(display_name, style=f"bold {style}")
            line.append(f"  {tc['args']}", style="dim")
        else:
            line.append("  → ", style="bold yellow")
            line.append(f"{icon} ", style=style)
            line.append(display_name, style=f"bold {style}")
            line.append(f"  {tc['args']}", style="dim")
        lines.append(line)

    return Group(*lines) if lines else Text("No tool calls yet", style="dim")


def display_tool_usage(tool_calls: list[dict]) -> dict[str, int]:
    """Display and return tool usage statistics."""
    tool_counts: dict[str, int] = {}
    for tc in tool_calls:
        name = tc.get("name", "unknown")
        # Normalize google_search_agent to google_search for cleaner display
        if name == "google_search_agent":
            name = "google_search"
        tool_counts[name] = tool_counts.get(name, 0) + 1

    if tool_counts:
        table = Table(title="🔧 Tool Usage", show_header=True, header_style="bold magenta", box=None)
        table.add_column("Tool", style="cyan", no_wrap=True)
        table.add_column("Calls", justify="right", style="bold")

        for tool, count in sorted(tool_counts.items()):
            if tool in ("fetch_url", "read_pdf"):
                table.add_row(f"[bold green]✓ {tool}[/bold green]", f"[green]{count}[/green]")
            elif tool == "grep_file":
                table.add_row(f"[bold cyan]✓ {tool}[/bold cyan]", f"[cyan]{count}[/cyan]")
            elif "search" in tool.lower():
                table.add_row(f"[blue]{tool}[/blue]", str(count))
            else:
                table.add_row(tool, str(count))

        console.print(table)

    return tool_counts


def setup_logging() -> ToolCallHandler:
    """Configure logging to capture tool calls without verbose output."""
    logging.basicConfig(level=logging.ERROR, format="%(message)s", force=True)

    # Suppress verbose logging from external libraries and tools
    for logger_name in [
        "google.adk",
        "google.genai",
        "google.generativeai",
        "httpx",
        "httpcore",
        "aieng.agent_evals.tools",
        "aieng.agent_evals.knowledge_qa.web_tools",
    ]:
        _logger = logging.getLogger(logger_name)
        _logger.setLevel(logging.CRITICAL)
        _logger.propagate = False
        # Clear any existing handlers
        _logger.handlers.clear()

    # Set up custom handler for tool call capture
    tool_handler = ToolCallHandler()
    tool_handler.setLevel(logging.INFO)

    # Configure agent logger to only capture tool calls, suppress other messages
    agent_logger = logging.getLogger("aieng.agent_evals.knowledge_qa.agent")
    agent_logger.handlers.clear()
    agent_logger.addHandler(tool_handler)
    agent_logger.setLevel(logging.INFO)
    agent_logger.propagate = False

    # Add a filter to suppress non-tool-call messages
    class ToolCallOnlyFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return "Tool call:" in msg or "Tool response:" in msg or "Tool error:" in msg

    tool_handler.addFilter(ToolCallOnlyFilter())

    return tool_handler


async def run_agent_with_display(
    agent,
    question: str,
    tool_handler: ToolCallHandler,
    show_plan: bool = False,
    ground_truth: str | None = None,
    example_id: int | None = None,
    answer_type: str | None = None,
    example_num: int | None = None,
    total_examples: int | None = None,
):
    """Run the agent with live tool call display.

    Parameters
    ----------
    agent : KnowledgeGroundedAgent
        The agent to run.
    question : str
        The question to answer.
    tool_handler : ToolCallHandler
        Handler for capturing tool calls.
    show_plan : bool
        If True, display the research plan checklist during execution.
    ground_truth : str, optional
        The ground truth answer (for eval mode - shown in live display).
    example_id : int, optional
        The example ID (for eval mode - shown in live display).
    answer_type : str, optional
        The answer type (for eval mode - shown in live display).
    example_num : int, optional
        Current example number (for eval mode spinner display).
    total_examples : int, optional
        Total number of examples (for eval mode spinner display).
    """
    live_console = Console(file=sys.stdout, force_terminal=True)

    # Show spinner while preparing (planning if enabled)
    if example_num is not None and total_examples is not None:
        spinner_text = f"[bold cyan]Example {example_num}/{total_examples}[/bold cyan]"
        with Status(spinner_text, console=console, spinner="dots"):
            if show_plan and hasattr(agent, "create_plan_async"):
                await agent.create_plan_async(question)
    elif show_plan and hasattr(agent, "create_plan_async"):
        await agent.create_plan_async(question)

    # Capture stdout/stderr before Live to suppress agent output
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        with Live(
            create_tool_display(
                [],
                plan=agent.current_plan if show_plan else None,
                question=question,
                ground_truth=ground_truth,
                example_id=example_id,
                answer_type=answer_type,
            ),
            console=live_console,
            screen=True,
            refresh_per_second=10,
        ) as live:
            task = asyncio.create_task(agent.answer_async(question))

            while not task.done():
                current_plan = agent.current_plan if show_plan else None
                # Get context percentage from token tracker if available
                context_pct = None
                if hasattr(agent, "token_tracker"):
                    context_pct = agent.token_tracker.usage.context_remaining_percent
                live.update(
                    create_tool_display(
                        tool_handler.tool_calls,
                        plan=current_plan,
                        context_percent=context_pct,
                        question=question,
                        ground_truth=ground_truth,
                        example_id=example_id,
                        answer_type=answer_type,
                    )
                )
                await asyncio.sleep(0.1)

            return await task
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def _setup_tracing(log_trace: bool) -> bool:
    """Initialize Langfuse tracing if requested.

    Parameters
    ----------
    log_trace : bool
        Whether to enable tracing.

    Returns
    -------
    bool
        True if tracing was successfully enabled, False otherwise.
    """
    if not log_trace:
        return False

    from ..langfuse import init_tracing  # noqa: PLC0415

    enabled = init_tracing()
    if enabled:
        console.print("[green]✓ Langfuse tracing enabled[/green]\n")
    else:
        console.print("[yellow]⚠ Could not initialize Langfuse tracing[/yellow]\n")
    return enabled


def _flush_tracing(tracing_enabled: bool) -> None:
    """Flush traces to Langfuse if tracing was enabled.

    Parameters
    ----------
    tracing_enabled : bool
        Whether tracing is enabled.
    """
    if not tracing_enabled:
        return

    from ..langfuse import flush_traces  # noqa: PLC0415

    flush_traces()
    console.print("\n[dim]Traces flushed to Langfuse[/dim]")


async def cmd_ask(question: str, show_plan: bool = False, log_trace: bool = False) -> int:
    """Ask the agent a question.

    Parameters
    ----------
    question : str
        The question to ask.
    show_plan : bool
        Display the research plan checklist during execution.
    log_trace : bool
        Enable Langfuse tracing for this run.
    """
    from .agent import KnowledgeGroundedAgent  # noqa: PLC0415

    display_banner()
    tracing_enabled = _setup_tracing(log_trace)

    console.print(
        Panel(
            question,
            title="[bold blue]📋 Question[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )
    console.print()

    tool_handler = setup_logging()

    agent = KnowledgeGroundedAgent(enable_planning=True)

    tool_handler.clear()
    response = await run_agent_with_display(agent, question, tool_handler, show_plan=show_plan)

    # Display results
    console.print()
    display_tool_usage(response.tool_calls)

    console.print()

    # Parse structured answer format (ANSWER/SOURCES/REASONING)
    answer_text = response.text
    parsed_answer = _parse_structured_answer(answer_text)

    if parsed_answer:
        # Display formatted answer with sections
        answer_content = Text()
        if parsed_answer.get("answer"):
            answer_content.append(parsed_answer["answer"], style="white")

        console.print(
            Panel(
                answer_content,
                title="[bold cyan]🤖 Answer[/bold cyan]",
                subtitle=f"[dim]Duration: {response.total_duration_ms / 1000:.1f}s[/dim]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        if parsed_answer.get("reasoning"):
            console.print(
                Panel(
                    parsed_answer["reasoning"],
                    title="[bold dim]💭 Reasoning[/bold dim]",
                    border_style="dim",
                    padding=(0, 1),
                )
            )
    else:
        # Fallback to raw display if parsing fails
        console.print(
            Panel(
                answer_text,
                title="[bold cyan]🤖 Answer[/bold cyan]",
                subtitle=f"[dim]Duration: {response.total_duration_ms / 1000:.1f}s[/dim]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    if response.sources:
        console.print("\n[bold]Sources:[/bold]")
        for src in response.sources[:5]:
            if src.uri:
                console.print(f"  • [blue]{src.title or 'Source'}[/blue]: {src.uri}")

    _flush_tracing(tracing_enabled)
    console.print("\n[bold green]✓ Complete[/bold green]")
    return 0


# Outcome display configuration
OUTCOME_COLORS = {
    "fully_correct": "green",
    "correct_with_extraneous": "yellow",
    "partially_correct": "orange1",
    "fully_incorrect": "red",
}
OUTCOME_ICONS = {
    "fully_correct": "✅",
    "correct_with_extraneous": "🟡",
    "partially_correct": "🔶",
    "fully_incorrect": "❌",
}


def _display_example_result(example, response, idx: int, total: int) -> dict[str, int]:
    """Display the full results for an evaluated example.

    Parameters
    ----------
    example : DSQAExample
        The example that was evaluated.
    response : AgentResponse
        The agent's response.
    idx : int
        Current index (1-based).
    total : int
        Total number of examples.

    Returns
    -------
    dict[str, int]
        Tool usage counts.
    """
    console.print(f"\n[bold cyan]━━━ Example {idx}/{total} - Results ━━━[/bold cyan]\n")
    console.print(
        Panel(
            example.problem,
            title=f"[bold blue]📋 Question (ID: {example.example_id})[/bold blue]",
            subtitle=f"[dim]Answer Type: {example.answer_type}[/dim]",
            border_style="blue",
            padding=(1, 2),
        )
    )
    console.print()
    console.print(
        Panel(
            f"[yellow]{example.answer}[/yellow]",
            title="[bold yellow]🎯 Ground Truth[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )
    console.print()
    tool_counts = display_tool_usage(response.tool_calls)
    console.print()

    # Parse structured answer format (ANSWER/SOURCES/REASONING)
    parsed_answer = _parse_structured_answer(response.text)

    if parsed_answer and parsed_answer.get("answer"):
        # Display formatted answer
        console.print(
            Panel(
                parsed_answer["answer"],
                title="[bold cyan]🤖 Answer[/bold cyan]",
                subtitle=f"[dim]Duration: {response.total_duration_ms / 1000:.1f}s[/dim]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        # Display reasoning if present
        if parsed_answer.get("reasoning"):
            console.print()
            console.print(
                Panel(
                    parsed_answer["reasoning"],
                    title="[bold dim]💭 Reasoning[/bold dim]",
                    border_style="dim",
                    padding=(0, 1),
                )
            )
    else:
        # Fallback to raw display if parsing fails
        console.print(
            Panel(
                response.text,
                title="[bold cyan]🤖 Agent Response[/bold cyan]",
                subtitle=f"[dim]Duration: {response.total_duration_ms / 1000:.1f}s[/dim]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    return tool_counts


def _display_eval_result(result) -> None:
    """Display evaluation metrics for a result."""
    color = OUTCOME_COLORS.get(result.outcome, "white")
    icon = OUTCOME_ICONS.get(result.outcome, "•")

    # Main metrics table
    metrics_table = Table(show_header=False, box=None, padding=(0, 2))
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Value", justify="right")
    metrics_table.add_row("Outcome", f"[{color}]{icon} {result.outcome}[/{color}]")
    metrics_table.add_row("Precision", f"[bold]{result.precision:.2f}[/bold]")
    metrics_table.add_row("Recall", f"[bold]{result.recall:.2f}[/bold]")
    metrics_table.add_row("F1 Score", f"[bold]{result.f1_score:.2f}[/bold]")

    console.print(Panel(metrics_table, title="[bold magenta]📊 Evaluation[/bold magenta]", border_style="magenta"))

    # Display judge explanation if available
    if result.explanation:
        console.print()
        console.print(
            Panel(
                result.explanation,
                title="[bold blue]💭 Judge Explanation[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # Display correctness details if available
    if result.correctness_details:
        console.print()
        details_table = Table(
            title="🎯 Correctness Details",
            show_header=True,
            header_style="bold cyan",
            box=None,
        )
        details_table.add_column("Ground Truth Item", style="white")
        details_table.add_column("Found", justify="center", width=8)

        for item, found in result.correctness_details.items():
            found_icon = "[green]✓[/green]" if found else "[red]✗[/red]"
            details_table.add_row(item, found_icon)

        console.print(details_table)

    # Display extraneous items if any
    if result.extraneous_items:
        console.print()
        extra_text = "\n".join(f"  • {item}" for item in result.extraneous_items)
        console.print(
            Panel(
                f"[yellow]{extra_text}[/yellow]",
                title="[bold yellow]⚠️ Extraneous Items[/bold yellow]",
                subtitle=f"[dim]{len(result.extraneous_items)} item(s) not in ground truth[/dim]",
                border_style="yellow",
                padding=(0, 2),
            )
        )


def _display_eval_summary(results: list) -> None:
    """Display comprehensive summary table for multiple evaluation results.

    Shows per-sample results, outcome distribution, and aggregate metrics.
    """
    console.print()

    # Per-sample results table
    sample_table = Table(
        title="[bold cyan]📋 Per-Sample Results[/bold cyan]",
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
        title_justify="left",
    )
    sample_table.add_column("ID", style="dim", width=8)
    sample_table.add_column("Outcome", width=26)
    sample_table.add_column("Precision", justify="right", width=10)
    sample_table.add_column("Recall", justify="right", width=10)
    sample_table.add_column("F1", justify="right", width=10)

    for example_id, result, _ in results:
        color = OUTCOME_COLORS.get(result.outcome, "white")
        icon = OUTCOME_ICONS.get(result.outcome, "•")
        sample_table.add_row(
            str(example_id),
            f"[{color}]{icon} {result.outcome}[/{color}]",
            f"{result.precision:.2f}",
            f"{result.recall:.2f}",
            f"{result.f1_score:.2f}",
        )

    console.print(sample_table)
    console.print()

    # Count outcomes
    outcome_counts = {
        "fully_correct": 0,
        "correct_with_extraneous": 0,
        "partially_correct": 0,
        "fully_incorrect": 0,
    }
    for _, result, _ in results:
        if result.outcome in outcome_counts:
            outcome_counts[result.outcome] += 1

    total = len(results)

    # Outcome distribution table
    outcome_table = Table(
        title="[bold magenta]📊 Outcome Distribution[/bold magenta]",
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
        title_justify="left",
    )
    outcome_table.add_column("Outcome", width=30)
    outcome_table.add_column("Count", justify="right", width=8)
    outcome_table.add_column("Percentage", justify="right", width=12)

    outcome_display = [
        ("fully_correct", "Fully Correct", "green"),
        ("correct_with_extraneous", "Correct with Extraneous", "yellow"),
        ("partially_correct", "Partially Correct", "orange1"),
        ("fully_incorrect", "Fully Incorrect", "red"),
    ]

    for key, label, color in outcome_display:
        count = outcome_counts[key]
        pct = (count / total * 100) if total > 0 else 0
        icon = OUTCOME_ICONS.get(key, "•")
        outcome_table.add_row(
            f"[{color}]{icon} {label}[/{color}]",
            f"[{color}]{count}[/{color}]",
            f"[{color}]{pct:.1f}%[/{color}]",
        )

    console.print(outcome_table)
    console.print()

    # Calculate aggregate metrics
    avg_precision = sum(r.precision for _, r, _ in results) / total if total > 0 else 0
    avg_recall = sum(r.recall for _, r, _ in results) / total if total > 0 else 0
    avg_f1 = sum(r.f1_score for _, r, _ in results) / total if total > 0 else 0

    # Aggregate metrics table
    metrics_table = Table(
        title="[bold green]📈 Aggregate Metrics[/bold green]",
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
        title_justify="left",
    )
    metrics_table.add_column("Metric", width=20)
    metrics_table.add_column("Value", justify="right", width=12)

    # Color code F1 based on performance
    if avg_f1 >= 0.8:
        f1_color = "green"
    elif avg_f1 >= 0.5:
        f1_color = "yellow"
    else:
        f1_color = "red"

    metrics_table.add_row("Samples Evaluated", f"[bold]{total}[/bold]")
    metrics_table.add_row("Avg Precision", f"[bold]{avg_precision:.3f}[/bold]")
    metrics_table.add_row("Avg Recall", f"[bold]{avg_recall:.3f}[/bold]")
    metrics_table.add_row("Avg F1 Score", f"[bold {f1_color}]{avg_f1:.3f}[/bold {f1_color}]")

    console.print(metrics_table)


async def cmd_eval(
    samples: int = 1,
    category: str = "Finance & Economics",
    ids: list[int] | None = None,
    show_plan: bool = False,
    log_trace: bool = False,
) -> int:
    """Run evaluation on DeepSearchQA samples.

    Parameters
    ----------
    samples : int
        Number of samples to evaluate (used when ids not specified).
    category : str
        Dataset category to filter by (used when ids not specified).
    ids : list[int], optional
        Specific example IDs to evaluate. If provided, samples and category are ignored.
    show_plan : bool
        Display the research plan checklist during execution.
    log_trace : bool
        Enable Langfuse tracing for this run.
    """
    from .agent import KnowledgeGroundedAgent  # noqa: PLC0415
    from .evaluation import DeepSearchQADataset  # noqa: PLC0415
    from .judges import DeepSearchQAJudge  # noqa: PLC0415

    display_banner()
    tracing_enabled = _setup_tracing(log_trace)

    # Build info text based on selection mode
    if ids:
        info_text = f"[bold]Evaluation Mode[/bold]\n\nExample IDs: [cyan]{', '.join(map(str, ids))}[/cyan]"
    else:
        info_text = (
            f"[bold]Evaluation Mode[/bold]\n\nCategory: [cyan]{category}[/cyan]\nSamples: [cyan]{samples}[/cyan]"
        )

    if show_plan:
        info_text += "\nPlan Display: [green]enabled[/green]"

    console.print(
        Panel(
            info_text,
            title="📊 DeepSearchQA Evaluation",
            border_style="blue",
        )
    )
    console.print()

    console.print("[bold blue]Loading dataset...[/bold blue]")
    dataset = DeepSearchQADataset()

    # Get examples by ID or by category
    if ids:
        examples = dataset.get_by_ids(ids)
        if len(examples) != len(ids):
            found_ids = {ex.example_id for ex in examples}
            missing_ids = [eid for eid in ids if eid not in found_ids]
            console.print(f"[yellow]Warning: IDs not found: {missing_ids}[/yellow]")
    else:
        examples = dataset.get_by_category(category)[:samples]

    if not examples:
        console.print("[bold red]Error: No examples found matching the criteria.[/bold red]")
        return 1

    console.print(f"[green]✓ Loaded {len(examples)} example(s)[/green]\n")

    console.print("[bold blue]Initializing agent and judge...[/bold blue]")
    agent = KnowledgeGroundedAgent(enable_planning=True)
    judge = DeepSearchQAJudge()
    console.print("[green]✓ Ready[/green]\n")

    tool_handler = setup_logging()
    results = []

    for i, example in enumerate(examples, 1):
        tool_handler.clear()
        agent.reset()  # Clear session state between examples

        try:
            response = await run_agent_with_display(
                agent,
                example.problem,
                tool_handler,
                show_plan=show_plan,
                ground_truth=example.answer,
                example_id=example.example_id,
                answer_type=example.answer_type,
                example_num=i,
                total_examples=len(examples),
            )

            # Display full results after Live display ends
            tool_counts = _display_example_result(example, response, i, len(examples))
            console.print("\n[bold blue]⏳ Evaluating...[/bold blue]\n")
            _, result = judge.evaluate_with_details(
                question=example.problem,
                answer=response.text,
                ground_truth=example.answer,
                answer_type=example.answer_type,
            )
            _display_eval_result(result)
            results.append((example.example_id, result, tool_counts))

        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")

    if results:
        _display_eval_summary(results)

    _flush_tracing(tracing_enabled)
    console.print("\n[bold green]✓ Evaluation complete[/bold green]")
    return 0


def _display_sample_detailed(example, idx: int | None = None, total: int | None = None) -> None:
    """Display a single sample with full details.

    Parameters
    ----------
    example : DSQAExample
        The example to display.
    idx : int, optional
        Current index (1-based) for display in a list.
    total : int, optional
        Total number of examples being displayed.
    """
    # Header with index if provided
    if idx is not None and total is not None:
        console.print(f"\n[bold cyan]━━━ Sample {idx}/{total} ━━━[/bold cyan]\n")
    else:
        console.print()

    # Metadata table
    meta_table = Table(show_header=False, box=None, padding=(0, 2))
    meta_table.add_column("Field", style="bold dim")
    meta_table.add_column("Value")
    meta_table.add_row("ID", f"[cyan]{example.example_id}[/cyan]")
    meta_table.add_row("Category", f"[magenta]{example.problem_category}[/magenta]")
    meta_table.add_row("Answer Type", f"[blue]{example.answer_type}[/blue]")

    console.print(Panel(meta_table, title="[bold]📋 Metadata[/bold]", border_style="dim"))

    # Question
    console.print()
    console.print(
        Panel(
            example.problem,
            title="[bold blue]❓ Question[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Ground truth answer
    console.print()
    console.print(
        Panel(
            f"[yellow]{example.answer}[/yellow]",
            title="[bold yellow]🎯 Ground Truth Answer[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )


def cmd_sample(
    ids: list[int] | None = None,
    category: str | None = None,
    count: int = 5,
    random: bool = False,
    list_categories: bool = False,
) -> int:
    """View samples from the DeepSearchQA dataset.

    Parameters
    ----------
    ids : list[int], optional
        Specific example IDs to view.
    category : str, optional
        Filter by category.
    count : int
        Number of samples to show (default 5).
    random : bool
        If True, select random samples instead of first N.
    list_categories : bool
        If True, list all available categories and exit.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    from .evaluation import DeepSearchQADataset  # noqa: PLC0415

    display_banner()

    console.print("[bold blue]Loading dataset...[/bold blue]")
    dataset = DeepSearchQADataset()
    console.print(f"[green]✓ Loaded {len(dataset)} total examples[/green]\n")

    # List categories mode
    if list_categories:
        categories = dataset.get_categories()
        table = Table(title="📂 Available Categories", show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Category", style="bold")
        table.add_column("Count", justify="right")

        for i, cat in enumerate(sorted(categories), 1):
            cat_count = len(dataset.get_by_category(cat))
            table.add_row(str(i), cat, str(cat_count))

        console.print(table)
        console.print(f"\n[dim]Total: {len(categories)} categories[/dim]")
        return 0

    # Get examples based on selection criteria
    if ids:
        examples = dataset.get_by_ids(ids)
        if len(examples) != len(ids):
            found_ids = {ex.example_id for ex in examples}
            missing_ids = [eid for eid in ids if eid not in found_ids]
            console.print(f"[yellow]Warning: IDs not found: {missing_ids}[/yellow]\n")
        selection_desc = f"IDs: {', '.join(map(str, ids))}"
    elif category:
        all_in_category = dataset.get_by_category(category)
        if not all_in_category:
            console.print(f"[bold red]Error: Category '{category}' not found.[/bold red]")
            console.print("[dim]Use --list-categories to see available categories.[/dim]")
            return 1
        if random:
            import random as rand_module  # noqa: PLC0415

            examples = rand_module.sample(all_in_category, min(count, len(all_in_category)))
        else:
            examples = all_in_category[:count]
        selection_desc = f"Category: {category} ({len(all_in_category)} total)"
    elif random:
        examples = dataset.sample(n=count)
        selection_desc = f"Random {count} samples"
    else:
        examples = dataset.examples[:count]
        selection_desc = f"First {count} samples"

    if not examples:
        console.print("[bold red]No examples found matching the criteria.[/bold red]")
        return 1

    # Display selection info
    console.print(
        Panel(
            f"[bold]Selection:[/bold] {selection_desc}\n[bold]Showing:[/bold] {len(examples)} example(s)",
            title="📊 Dataset View",
            border_style="blue",
        )
    )

    # Display each example
    for i, example in enumerate(examples, 1):
        _display_sample_detailed(example, idx=i, total=len(examples))

    console.print("\n[bold green]✓ Done[/bold green]")
    return 0


def _display_help() -> None:
    """Display colorful help message using Rich."""
    console.print()

    # Commands table
    commands_table = Table(
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 2),
    )
    commands_table.add_column("Command", style="bold green", width=12)
    commands_table.add_column("Description")

    commands_table.add_row("ask", "Ask the agent a question")
    commands_table.add_row("eval", "Run evaluation on DeepSearchQA")
    commands_table.add_row("sample", "View samples from the DeepSearchQA dataset")
    commands_table.add_row("tools", "Display available tools")

    console.print("[bold]Commands:[/bold]")
    console.print(commands_table)
    console.print()

    # Options
    console.print("[bold]Options:[/bold]")
    console.print("  [cyan]-h, --help[/cyan]      Show this help message")
    console.print("  [cyan]--version[/cyan]       Show version number")
    console.print()

    # Usage examples
    console.print("[bold]Examples:[/bold]")
    console.print('  [dim]$[/dim] knowledge-qa [green]ask[/green] [yellow]"What is quantum computing?"[/yellow]')
    console.print(
        '  [dim]$[/dim] knowledge-qa [green]ask[/green] [yellow]"What is AI?"[/yellow] [cyan]--log-trace[/cyan]'
    )
    console.print("  [dim]$[/dim] knowledge-qa [green]eval[/green] [cyan]--samples[/cyan] 3")
    console.print("  [dim]$[/dim] knowledge-qa [green]eval[/green] [cyan]--ids[/cyan] 123 456 [cyan]--show-plan[/cyan]")
    console.print("  [dim]$[/dim] knowledge-qa [green]eval[/green] [cyan]--samples[/cyan] 5 [cyan]--log-trace[/cyan]")
    console.print(
        '  [dim]$[/dim] knowledge-qa [green]sample[/green] [cyan]--category[/cyan] [yellow]"Finance & Economics"[/yellow]'
    )
    console.print()


def main() -> int:
    """Run the Knowledge Agent CLI."""
    parser = argparse.ArgumentParser(
        prog="knowledge-qa",
        description="Knowledge-Grounded QA Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,  # We'll handle help ourselves
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show this help message and exit",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version number and exit",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask the agent a question")
    ask_parser.add_argument("question", type=str, help="The question to ask")
    ask_parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Display the research plan checklist during execution",
    )
    ask_parser.add_argument(
        "--log-trace",
        action="store_true",
        help="Enable Langfuse tracing for this run",
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation on DeepSearchQA")
    eval_parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of samples to evaluate (default: 1, ignored if --ids is used)",
    )
    eval_parser.add_argument(
        "--category",
        type=str,
        default="Finance & Economics",
        help="Dataset category (default: Finance & Economics, ignored if --ids is used)",
    )
    eval_parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        metavar="ID",
        help="Specific example ID(s) to evaluate (overrides --samples and --category)",
    )
    eval_parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Display the research plan checklist during execution",
    )
    eval_parser.add_argument(
        "--log-trace",
        action="store_true",
        help="Enable Langfuse tracing for this run",
    )

    # Sample command
    sample_parser = subparsers.add_parser("sample", help="View samples from the DeepSearchQA dataset")
    sample_parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        metavar="ID",
        help="Specific example ID(s) to view",
    )
    sample_parser.add_argument(
        "--category",
        type=str,
        help="Filter samples by category",
    )
    sample_parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of samples to show (default: 5)",
    )
    sample_parser.add_argument(
        "--random",
        action="store_true",
        help="Select random samples instead of first N",
    )
    sample_parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all available categories and exit",
    )

    # Tools command
    subparsers.add_parser("tools", help="Display available tools")

    args = parser.parse_args()

    if args.command == "ask":
        return asyncio.run(cmd_ask(args.question, args.show_plan, args.log_trace))
    if args.command == "eval":
        return asyncio.run(cmd_eval(args.samples, args.category, args.ids, args.show_plan, args.log_trace))
    if args.command == "sample":
        return cmd_sample(
            ids=args.ids,
            category=args.category,
            count=args.count,
            random=args.random,
            list_categories=args.list_categories,
        )
    if args.command == "tools":
        display_banner()
        display_tools_info()
        return 0

    # Show help for no command or explicit --help
    display_banner()
    if args.version:
        console.print(f"[bold]knowledge-qa[/bold] v{get_version()}")
        return 0
    _display_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

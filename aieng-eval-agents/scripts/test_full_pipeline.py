#!/usr/bin/env python
"""Test the full pipeline: Enhanced Agent with Web Tools + Judges on DeepSearchQA.

This script runs the EnhancedKnowledgeAgent (with google_search, fetch_url, read_pdf)
on actual DeepSearchQA samples, then evaluates the responses using the judges.

Usage:
    uv run --env-file .env python scripts/test_full_pipeline.py
"""

import asyncio
import io
import logging
import sys
from pathlib import Path


# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from aieng.agent_evals.display import create_console, display_metrics_table
from aieng.agent_evals.knowledge_agent.agent import EnhancedKnowledgeAgent
from aieng.agent_evals.knowledge_agent.evaluation import DeepSearchQADataset
from aieng.agent_evals.knowledge_agent.judges import (
    DeepSearchQAJudge,
    DeepSearchQAResult,
)
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class RichToolCallHandler(logging.Handler):
    """Custom logging handler that captures tool calls for rich display."""

    def __init__(self):
        super().__init__()
        self.tool_calls: list[dict] = []
        self.current_status: str = ""

    def emit(self, record):
        """Process a log record, capturing tool calls for display."""
        msg = record.getMessage()
        # Capture tool calls
        if "Tool call:" in msg:
            # Parse: "Tool call: tool_name({...})"
            try:
                parts = msg.split("Tool call: ", 1)[1]
                paren_idx = parts.find("(")
                if paren_idx > 0:
                    tool_name = parts[:paren_idx]
                    args_str = parts[paren_idx + 1 : -1]  # Remove parens
                    # Truncate long args
                    if len(args_str) > 80:
                        args_str = args_str[:77] + "..."
                    self.tool_calls.append({"name": tool_name, "args": args_str})
            except Exception:
                pass
        elif "Answering question" in msg:
            self.current_status = "Thinking..."
        elif "Created plan" in msg:
            self.current_status = "Plan created"

    def clear(self):
        """Reset captured tool calls and status."""
        self.tool_calls = []
        self.current_status = ""


def create_tool_call_display(tool_calls: list[dict], status: str = "") -> Panel:
    """Create a rich panel showing tool calls in progress."""
    content: Group | Text
    if not tool_calls:
        content = Text("Waiting for tool calls...", style="dim")
    else:
        lines = []
        # Show last 8 tool calls to keep it compact
        display_calls = tool_calls[-8:]
        if len(tool_calls) > 8:
            lines.append(Text(f"  ... ({len(tool_calls) - 8} earlier calls)", style="dim"))

        for i, tc in enumerate(display_calls):
            is_last = i == len(display_calls) - 1
            name = tc["name"]
            args = tc["args"]

            # Color based on tool type
            if name == "fetch_url":
                icon = "🌐"
                style = "green"
            elif name == "read_pdf":
                icon = "📄"
                style = "green"
            elif "search" in name.lower():
                icon = "🔍"
                style = "blue"
            else:
                icon = "🔧"
                style = "white"

            # Format the line
            line = Text()
            if is_last:
                line.append("  → ", style="bold yellow")
            else:
                line.append("  ✓ ", style="dim green")

            line.append(f"{icon} ", style=style)
            line.append(f"{name}", style=f"bold {style}")
            line.append(f"  {args}", style="dim")
            lines.append(line)

        content = Group(*lines) if lines else Text("No tool calls yet", style="dim")

    # Add spinner for current status
    title_parts = ["[bold cyan]🔧 Agent Working[/bold cyan]"]
    if status:
        title_parts.append(f" [dim]({status})[/dim]")

    return Panel(
        content,
        title="".join(title_parts),
        subtitle=f"[dim]{len(tool_calls)} tool calls[/dim]",
        border_style="cyan",
        padding=(0, 1),
    )


# Suppress all other logging first
logging.basicConfig(level=logging.ERROR, format="%(message)s")

# Suppress verbose logging from external libraries
for logger_name in [
    "google.adk",
    "google.genai",
    "httpx",
    "httpcore",
    "aieng.agent_evals.knowledge_agent.web_tools",
]:
    _logger = logging.getLogger(logger_name)
    _logger.setLevel(logging.ERROR)
    _logger.propagate = False

# Set up our custom handler for tool call capture
# This needs to capture INFO level to see tool calls, but NOT print them
tool_handler = RichToolCallHandler()
tool_handler.setLevel(logging.INFO)

agent_logger = logging.getLogger("aieng.agent_evals.knowledge_agent.agent")
agent_logger.handlers.clear()  # Remove any existing handlers
agent_logger.addHandler(tool_handler)
agent_logger.setLevel(logging.INFO)
agent_logger.propagate = False  # Don't propagate to root logger

logger = logging.getLogger(__name__)


def display_tool_usage(tool_calls: list[dict], console: Console) -> dict[str, int]:
    """Display and return tool usage statistics."""
    tool_counts: dict[str, int] = {}
    for tc in tool_calls:
        name = tc.get("name", "unknown")
        tool_counts[name] = tool_counts.get(name, 0) + 1

    if tool_counts:
        table = Table(title="🔧 Tool Usage", show_header=True, header_style="bold magenta", box=None)
        table.add_column("Tool", style="cyan", no_wrap=True)
        table.add_column("Calls", justify="right", style="bold")

        for tool, count in sorted(tool_counts.items()):
            if tool in ("fetch_url", "read_pdf"):
                table.add_row(f"[bold green]✓ {tool}[/bold green]", f"[green]{count}[/green]")
            elif "search" in tool.lower():
                table.add_row(f"[blue]{tool}[/blue]", str(count))
            else:
                table.add_row(tool, str(count))

        console.print(table)

    return tool_counts


def display_question(question: str, answer_type: str, example_id: int, console: Console) -> None:
    """Display the question in a colored panel."""
    console.print(
        Panel(
            question,
            title=f"[bold blue]📋 Question (ID: {example_id})[/bold blue]",
            subtitle=f"[dim]Answer Type: {answer_type}[/dim]",
            border_style="blue",
            padding=(1, 2),
        )
    )


def display_ground_truth(ground_truth: str, console: Console) -> None:
    """Display ground truth in a colored panel."""
    console.print(
        Panel(
            f"[yellow]{ground_truth}[/yellow]",
            title="[bold yellow]🎯 Ground Truth[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )


def display_agent_response(prediction: str, sources_count: int, duration_ms: int, console: Console) -> None:
    """Display agent response in a colored panel."""
    console.print(
        Panel(
            prediction,
            title="[bold cyan]🤖 Agent Response[/bold cyan]",
            subtitle=f"[dim]Sources: {sources_count} | Duration: {duration_ms}ms[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def display_evaluation(result: DeepSearchQAResult, console: Console) -> None:
    """Display full evaluation results."""
    outcome_colors = {
        "fully_correct": "green",
        "correct_with_extraneous": "yellow",
        "partially_correct": "orange1",
        "fully_incorrect": "red",
    }
    outcome_icons = {
        "fully_correct": "✅",
        "correct_with_extraneous": "⚠️",
        "partially_correct": "🔶",
        "fully_incorrect": "❌",
    }
    color = outcome_colors.get(result.outcome, "white")
    icon = outcome_icons.get(result.outcome, "•")

    # Metrics table
    metrics_table = Table(show_header=False, box=None, padding=(0, 2))
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Value", justify="right")

    metrics_table.add_row("Outcome", f"[{color}]{icon} {result.outcome}[/{color}]")
    metrics_table.add_row("Precision", f"[bold]{result.precision:.2f}[/bold]")
    metrics_table.add_row("Recall", f"[bold]{result.recall:.2f}[/bold]")
    metrics_table.add_row("F1 Score", f"[bold]{result.f1_score:.2f}[/bold]")

    console.print(
        Panel(
            metrics_table,
            title="[bold magenta]📊 Evaluation Metrics[/bold magenta]",
            border_style="magenta",
        )
    )

    # Correctness details
    if result.correctness_details:
        details_text = Text()
        for item, correct in result.correctness_details.items():
            if correct:
                details_text.append("  ✓ ", style="bold green")
                details_text.append(f"{item}\n", style="green")
            else:
                details_text.append("  ✗ ", style="bold red")
                details_text.append(f"{item}\n", style="red")

        console.print(
            Panel(
                details_text,
                title="[bold]Correctness Details[/bold]",
                border_style="dim",
            )
        )

    # Extraneous items
    if result.extraneous_items:
        console.print(
            Panel(
                f"[yellow]{', '.join(result.extraneous_items)}[/yellow]",
                title="[bold yellow]⚠️ Extraneous Items[/bold yellow]",
                border_style="yellow",
            )
        )

    # Full explanation
    console.print(
        Panel(
            result.explanation,
            title="[bold]📝 Judge Explanation[/bold]",
            border_style=color,
            padding=(1, 2),
        )
    )


def display_summary_table(
    results: list[tuple[int, DeepSearchQAResult, dict[str, int]]],
    console: Console,
) -> None:
    """Display a summary table of all results."""
    table = Table(
        title="📈 Full Pipeline Evaluation Summary",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("ID", style="dim", width=6)
    table.add_column("Outcome", width=24)
    table.add_column("Precision", justify="right", width=10)
    table.add_column("Recall", justify="right", width=10)
    table.add_column("F1", justify="right", width=10)
    table.add_column("Web Tools", width=12, justify="center")

    outcome_colors = {
        "fully_correct": "green",
        "correct_with_extraneous": "yellow",
        "partially_correct": "orange1",
        "fully_incorrect": "red",
    }

    for example_id, result, tool_counts in results:
        color = outcome_colors.get(result.outcome, "white")
        used_web = tool_counts.get("fetch_url", 0) > 0 or tool_counts.get("read_pdf", 0) > 0
        web_status = "[green]✓ Yes[/green]" if used_web else "[dim]No[/dim]"

        table.add_row(
            str(example_id),
            f"[{color}]{result.outcome}[/{color}]",
            f"{result.precision:.2f}",
            f"{result.recall:.2f}",
            f"{result.f1_score:.2f}",
            web_status,
        )

    console.print(table)

    # Calculate aggregates
    if results:
        avg_precision = sum(r.precision for _, r, _ in results) / len(results)
        avg_recall = sum(r.recall for _, r, _ in results) / len(results)
        avg_f1 = sum(r.f1_score for _, r, _ in results) / len(results)

        outcome_counts: dict[str, int] = {}
        web_tool_usage = 0
        for _, r, tc in results:
            outcome_counts[r.outcome] = outcome_counts.get(r.outcome, 0) + 1
            if tc.get("fetch_url", 0) > 0 or tc.get("read_pdf", 0) > 0:
                web_tool_usage += 1

        console.print("\n[bold]Aggregate Metrics:[/bold]")
        display_metrics_table(
            {
                "Average Precision": avg_precision,
                "Average Recall": avg_recall,
                "Average F1": avg_f1,
                "Web Tool Usage": f"{web_tool_usage}/{len(results)}",
                **{f"Count: {k}": v for k, v in outcome_counts.items()},
            },
            title="Overall Performance",
            console=console,
        )


async def _run_agent_with_live_display(agent: EnhancedKnowledgeAgent, example, live_console: Console):
    """Run the agent with live display updates, suppressing verbose output.

    Parameters
    ----------
    agent
        The EnhancedKnowledgeAgent instance.
    example
        The DeepSearchQA example to process.
    live_console
        Console for live display output.

    Returns
    -------
    AgentResponse
        The agent response with answer text, sources, and tool calls.
    """
    with Live(
        create_tool_call_display([], "Starting..."),
        console=live_console,
        refresh_per_second=4,
        transient=True,
    ) as live:
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Redirect stdout/stderr to suppress Google ADK verbose output
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            task = asyncio.create_task(agent.answer_async(example.problem))

            while not task.done():
                live.update(create_tool_call_display(tool_handler.tool_calls, tool_handler.current_status))
                await asyncio.sleep(0.25)

            return await task
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


async def run_pipeline():
    """Run the full pipeline: agent + judges on DeepSearchQA samples."""
    console = create_console(force_jupyter=False)

    console.print(
        Panel(
            "[bold]Full Pipeline Test: Enhanced Agent + Judges[/bold]\n\n"
            "This script runs the EnhancedKnowledgeAgent (with google_search, fetch_url, "
            "read_pdf tools) on DeepSearchQA samples and evaluates responses using the "
            "DeepSearchQAJudge.\n\n"
            "Watch for [green]fetch_url[/green] and [green]read_pdf[/green] tool usage!",
            title="🚀 Full Pipeline Test",
            border_style="blue",
        )
    )

    # Load dataset
    console.print("\n[bold blue]Loading DeepSearchQA dataset...[/bold blue]")
    dataset = DeepSearchQADataset()

    # Get Finance & Economics examples - just 1 for quick testing
    finance_examples = dataset.get_by_category("Finance & Economics")[7:8]
    console.print(f"[green]✓ Loaded {len(finance_examples)} Finance & Economics example(s)[/green]\n")

    # Initialize agent with web tools
    console.print("[bold blue]Initializing EnhancedKnowledgeAgent...[/bold blue]")
    agent = EnhancedKnowledgeAgent(enable_planning=False)
    console.print("[green]✓ Agent initialized with tools: google_search, fetch_url, read_pdf[/green]\n")

    # Initialize judge
    console.print("[bold blue]Initializing DeepSearchQAJudge...[/bold blue]")
    judge = DeepSearchQAJudge()
    console.print("[green]✓ Judge initialized[/green]\n")

    # Run evaluations
    results: list[tuple[int, DeepSearchQAResult, dict[str, int]]] = []

    for i, example in enumerate(finance_examples, 1):
        console.print(f"\n[bold white on blue] Example {i}/{len(finance_examples)} [/bold white on blue]\n")

        try:
            # Display the question
            display_question(example.problem, example.answer_type, example.example_id, console)

            # Display ground truth
            console.print()
            display_ground_truth(example.answer, console)

            # Run the agent with live tool call display
            console.print()
            tool_handler.clear()

            # Create a dedicated console that always writes to original stdout
            live_console = Console(file=sys.stdout, force_terminal=True)
            response = await _run_agent_with_live_display(agent, example, live_console)

            # Show final tool usage summary
            console.print()
            tool_counts = display_tool_usage(response.tool_calls, console)

            # Display agent response
            console.print()
            display_agent_response(response.text, len(response.sources), response.total_duration_ms, console)

            # Evaluate with judge
            console.print("\n[bold blue]⏳ Evaluating with DeepSearchQAJudge...[/bold blue]\n")
            _, detailed_result = judge.evaluate_with_details(
                question=example.problem,
                answer=response.text,
                ground_truth=example.answer,
                answer_type=example.answer_type,
            )

            # Display evaluation
            display_evaluation(detailed_result, console)

            results.append((example.example_id, detailed_result, tool_counts))

        except Exception as e:
            console.print(f"[bold red]Error processing example {example.example_id}: {e}[/bold red]")
            logger.exception(f"Error in example {example.example_id}")

    # Display summary
    if len(results) > 1:
        console.print("\n[bold white on blue] Summary [/bold white on blue]\n")
        display_summary_table(results, console)

    console.print("\n[bold green]✓ Full pipeline evaluation complete![/bold green]")


def main():
    """Entry point."""
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()

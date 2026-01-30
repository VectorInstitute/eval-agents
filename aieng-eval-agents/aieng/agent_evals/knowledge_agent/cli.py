#!/usr/bin/env python3
"""Knowledge Agent CLI.

Command-line interface for running and evaluating the Knowledge-Grounded QA Agent.

Usage::

    knowledge-agent ask "What is..."
    knowledge-agent eval --samples 3
"""

import argparse
import asyncio
import io
import logging
import sys
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
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


def display_banner() -> None:
    """Display the CLI banner with version info."""
    ver = get_version()

    # Compact search/magnifying glass ASCII art
    line0 = Text()
    line0.append("   ◯─◯   ", style=f"{VECTOR_CYAN} bold")
    line0.append("   knowledge-agent ", style="white bold")
    line0.append(f"v{ver}", style="bright_black")

    line1 = Text()
    line1.append("  ╱ 🔍 ╲  ", style=f"{VECTOR_CYAN} bold")

    line2 = Text()
    line2.append(" │     │ ", style=f"{VECTOR_CYAN} bold")
    line2.append("   ", style="")
    line2.append("search → fetch → grep → answer", style="cyan")

    line3 = Text()
    line3.append("  ╲___╱  ", style=f"{VECTOR_CYAN} bold")
    line3.append("   Vector Institute AI Engineering", style="bright_black")

    console.print()
    console.print(line0)
    console.print(line1)
    console.print(line2)
    console.print(line3)
    console.print()


def display_tools_info() -> None:
    """Display information about available tools."""
    table = Table(
        title="Available Tools",
        show_header=True,
        header_style="bold cyan",
        box=None,
    )
    table.add_column("Tool", style="bold", width=24)
    table.add_column("Description")

    table.add_row(
        "[blue]google_search[/blue]",
        "Search the web for current information and sources",
    )
    table.add_row(
        "[green]fetch_url[/green]",
        "Fetch webpage content and save locally for searching",
    )
    table.add_row(
        "[cyan]grep_file[/cyan]",
        "Search any file for matching patterns (like Unix grep)",
    )
    table.add_row(
        "[green]read_pdf[/green]",
        "Read and extract text from PDF documents",
    )

    console.print(table)
    console.print()


class ToolCallHandler(logging.Handler):
    """Custom logging handler that captures tool calls for rich display."""

    def __init__(self):
        super().__init__()
        self.tool_calls: list[dict] = []
        self.current_status: str = ""

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


def create_tool_display(tool_calls: list[dict], status: str = "") -> Panel:
    """Create a rich panel showing tool calls in progress."""
    content: Group | Text
    if not tool_calls:
        content = Text("Waiting for tool calls...", style="dim")
    else:
        lines = []
        display_calls = tool_calls[-8:]
        if len(tool_calls) > 8:
            lines.append(Text(f"  ... ({len(tool_calls) - 8} earlier calls)", style="dim"))

        for i, tc in enumerate(display_calls):
            is_last = i == len(display_calls) - 1
            name = tc["name"]
            args = tc["args"]

            # Icon and style based on tool type
            if name == "fetch_url":
                icon, style = "🌐", "green"
            elif name == "read_pdf":
                icon, style = "📄", "green"
            elif name == "grep_file":
                icon, style = "📑", "cyan"
            elif "search" in name.lower():
                icon, style = "🔍", "blue"
            else:
                icon, style = "🔧", "white"

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


def display_tool_usage(tool_calls: list[dict]) -> dict[str, int]:
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

    # Set up custom handler for tool call capture
    tool_handler = ToolCallHandler()
    tool_handler.setLevel(logging.INFO)

    agent_logger = logging.getLogger("aieng.agent_evals.knowledge_agent.agent")
    agent_logger.handlers.clear()
    agent_logger.addHandler(tool_handler)
    agent_logger.setLevel(logging.INFO)
    agent_logger.propagate = False

    return tool_handler


async def run_agent_with_display(agent, question: str, tool_handler: ToolCallHandler):
    """Run the agent with live tool call display."""
    live_console = Console(file=sys.stdout, force_terminal=True)

    with Live(
        create_tool_display([], "Starting..."),
        console=live_console,
        refresh_per_second=4,
        transient=True,
    ) as live:
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            task = asyncio.create_task(agent.answer_async(question))

            while not task.done():
                live.update(create_tool_display(tool_handler.tool_calls, tool_handler.current_status))
                await asyncio.sleep(0.25)

            return await task
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


async def cmd_ask(question: str, planning: bool = False) -> int:
    """Ask the agent a question."""
    from .agent import EnhancedKnowledgeAgent  # noqa: PLC0415

    display_banner()

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

    console.print("[bold blue]Initializing agent...[/bold blue]")
    agent = EnhancedKnowledgeAgent(enable_planning=planning)
    console.print("[green]✓ Agent ready[/green]\n")

    tool_handler.clear()
    response = await run_agent_with_display(agent, question, tool_handler)

    # Display results
    console.print()
    display_tool_usage(response.tool_calls)

    console.print()
    console.print(
        Panel(
            response.text,
            title="[bold cyan]🤖 Answer[/bold cyan]",
            subtitle=f"[dim]Sources: {len(response.sources)} | Duration: {response.total_duration_ms}ms[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    if response.sources:
        console.print("\n[bold]Sources:[/bold]")
        for src in response.sources[:5]:
            if src.uri:
                console.print(f"  • [blue]{src.title or 'Source'}[/blue]: {src.uri}")

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
    "correct_with_extraneous": "⚠️",
    "partially_correct": "🔶",
    "fully_incorrect": "❌",
}


def _display_example(example, idx: int, total: int) -> None:
    """Display question and ground truth for an example."""
    console.print(f"\n[bold white on blue] Example {idx}/{total} [/bold white on blue]\n")
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


def _display_eval_result(result) -> None:
    """Display evaluation metrics for a result."""
    color = OUTCOME_COLORS.get(result.outcome, "white")
    icon = OUTCOME_ICONS.get(result.outcome, "•")

    metrics_table = Table(show_header=False, box=None, padding=(0, 2))
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Value", justify="right")
    metrics_table.add_row("Outcome", f"[{color}]{icon} {result.outcome}[/{color}]")
    metrics_table.add_row("Precision", f"[bold]{result.precision:.2f}[/bold]")
    metrics_table.add_row("Recall", f"[bold]{result.recall:.2f}[/bold]")
    metrics_table.add_row("F1 Score", f"[bold]{result.f1_score:.2f}[/bold]")

    console.print(Panel(metrics_table, title="[bold magenta]📊 Evaluation[/bold magenta]", border_style="magenta"))


def _display_eval_summary(results: list) -> None:
    """Display summary table for multiple evaluation results."""
    console.print("\n[bold white on blue] Summary [/bold white on blue]\n")

    summary_table = Table(title="📈 Evaluation Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("ID", style="dim", width=6)
    summary_table.add_column("Outcome", width=24)
    summary_table.add_column("F1", justify="right", width=10)

    for example_id, result, _ in results:
        color = OUTCOME_COLORS.get(result.outcome, "white")
        summary_table.add_row(str(example_id), f"[{color}]{result.outcome}[/{color}]", f"{result.f1_score:.2f}")

    console.print(summary_table)

    avg_f1 = sum(r.f1_score for _, r, _ in results) / len(results)
    console.print(f"\n[bold]Average F1:[/bold] {avg_f1:.2f}")


async def cmd_eval(samples: int = 1, category: str = "Finance & Economics") -> int:
    """Run evaluation on DeepSearchQA samples."""
    from .agent import EnhancedKnowledgeAgent  # noqa: PLC0415
    from .evaluation import DeepSearchQADataset  # noqa: PLC0415
    from .judges import DeepSearchQAJudge  # noqa: PLC0415

    display_banner()
    console.print(
        Panel(
            f"[bold]Evaluation Mode[/bold]\n\nCategory: [cyan]{category}[/cyan]\nSamples: [cyan]{samples}[/cyan]",
            title="📊 DeepSearchQA Evaluation",
            border_style="blue",
        )
    )
    console.print()

    console.print("[bold blue]Loading dataset...[/bold blue]")
    dataset = DeepSearchQADataset()
    examples = dataset.get_by_category(category)[:samples]
    console.print(f"[green]✓ Loaded {len(examples)} example(s)[/green]\n")

    console.print("[bold blue]Initializing agent and judge...[/bold blue]")
    agent = EnhancedKnowledgeAgent(enable_planning=False)
    judge = DeepSearchQAJudge()
    console.print("[green]✓ Ready[/green]\n")

    tool_handler = setup_logging()
    results = []

    for i, example in enumerate(examples, 1):
        _display_example(example, i, len(examples))
        console.print()
        tool_handler.clear()

        try:
            response = await run_agent_with_display(agent, example.problem, tool_handler)
            console.print()
            tool_counts = display_tool_usage(response.tool_calls)
            console.print()
            console.print(
                Panel(
                    response.text,
                    title="[bold cyan]🤖 Agent Response[/bold cyan]",
                    subtitle=f"[dim]Duration: {response.total_duration_ms}ms[/dim]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )

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

    if len(results) > 1:
        _display_eval_summary(results)

    console.print("\n[bold green]✓ Evaluation complete[/bold green]")
    return 0


def main() -> int:
    """Run the Knowledge Agent CLI."""
    parser = argparse.ArgumentParser(
        prog="knowledge-agent",
        description="Knowledge-Grounded QA Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {get_version()}",
        help="Show version number and exit",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask the agent a question")
    ask_parser.add_argument("question", type=str, help="The question to ask")
    ask_parser.add_argument(
        "--planning",
        action="store_true",
        help="Enable research planning for complex questions",
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation on DeepSearchQA")
    eval_parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of samples to evaluate (default: 1)",
    )
    eval_parser.add_argument(
        "--category",
        type=str,
        default="Finance & Economics",
        help="Dataset category (default: Finance & Economics)",
    )

    # Tools command
    subparsers.add_parser("tools", help="Display available tools")

    args = parser.parse_args()

    if args.command == "ask":
        return asyncio.run(cmd_ask(args.question, args.planning))
    if args.command == "eval":
        return asyncio.run(cmd_eval(args.samples, args.category))
    if args.command == "tools":
        display_banner()
        display_tools_info()
        return 0
    display_banner()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Rich display utilities for agent evaluation outputs.

This module provides beautifully formatted console output for Jupyter notebooks
using the rich library. It can be used across all agent evaluation modules.
"""

from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from rich.tree import Tree


if TYPE_CHECKING:
    from .tools import GroundedResponse


# Custom theme for consistent styling
KNOWLEDGE_AGENT_THEME = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red bold",
        "highlight": "magenta",
        "muted": "dim",
        "agent": "bold blue",
        "user": "bold green",
    }
)


def create_console(force_jupyter: bool = True) -> Console:
    """Create a configured rich Console for the knowledge agent.

    Parameters
    ----------
    force_jupyter : bool, optional
        Force Jupyter rendering mode, by default True.

    Returns
    -------
    Console
        Configured rich Console instance.
    """
    return Console(theme=KNOWLEDGE_AGENT_THEME, force_jupyter=force_jupyter)


def display_response(
    response: "GroundedResponse",
    console: Console | None = None,
    title: str = "Response",
    show_sources: bool = True,
    show_queries: bool = True,
) -> None:
    """Display a grounded response with rich formatting.

    Parameters
    ----------
    response : GroundedResponse
        The grounded response to display.
    console : Console, optional
        Rich console to use. Creates one if not provided.
    title : str, optional
        Title for the response panel, by default "Response".
    show_sources : bool, optional
        Whether to show sources, by default True.
    show_queries : bool, optional
        Whether to show search queries, by default True.
    """
    if console is None:
        console = create_console()

    # Main response panel
    console.print(Panel(Markdown(response.text), title=f"📖 {title}", border_style="blue"))

    # Stats bar
    stats = f"[success]✓[/success] {len(response.search_queries)} queries | {len(response.sources)} sources"
    console.print(f"\n{stats}\n")

    # Search queries
    if show_queries and response.search_queries:
        query_tree = Tree("[bold cyan]🔍 Search Queries[/bold cyan]")
        for i, sq in enumerate(response.search_queries[:5], 1):
            query_tree.add(f"[info]{i}.[/info] {sq}")
        if len(response.search_queries) > 5:
            query_tree.add(f"[muted]... and {len(response.search_queries) - 5} more[/muted]")
        console.print(query_tree)

    # Sources
    if show_sources and response.sources:
        source_tree = Tree("[bold]📚 Sources[/bold]")
        for i, source in enumerate(response.sources, 1):
            source_tree.add(f"[muted][{i}][/muted] [link={source.uri}]{source.title}[/link]")
        console.print(source_tree)


def display_source_table(
    response: "GroundedResponse",
    console: Console | None = None,
) -> None:
    """Display sources as a formatted table.

    Parameters
    ----------
    response : GroundedResponse
        The grounded response containing sources.
    console : Console, optional
        Rich console to use. Creates one if not provided.
    """
    if console is None:
        console = create_console()

    if not response.sources:
        console.print("[muted]No sources available[/muted]")
        return

    source_table = Table(title="📚 Sources", show_header=True, header_style="bold green")
    source_table.add_column("#", style="muted", width=3)
    source_table.add_column("Title", style="white")
    source_table.add_column("URL", style="blue", overflow="fold")

    for i, source in enumerate(response.sources, 1):
        url_display = source.uri[:60] + "..." if len(source.uri) > 60 else source.uri
        source_table.add_row(str(i), source.title, url_display)

    console.print(source_table)


def display_comparison(
    response_a: str,
    response_b: "GroundedResponse",
    console: Console | None = None,
    title_a: str = "Without Grounding",
    title_b: str = "With Grounding",
) -> None:
    """Display side-by-side comparison of grounded vs non-grounded responses.

    Parameters
    ----------
    response_a : str
        The non-grounded response text.
    response_b : GroundedResponse
        The grounded response.
    console : Console, optional
        Rich console to use. Creates one if not provided.
    title_a : str, optional
        Title for first response, by default "Without Grounding".
    title_b : str, optional
        Title for second response, by default "With Grounding".
    """
    if console is None:
        console = create_console()

    panel_a = Panel(
        Markdown(response_a),
        title=f"❌ {title_a}",
        border_style="yellow",
        subtitle="[muted]May be outdated[/muted]",
    )

    panel_b = Panel(
        Markdown(response_b.text),
        title=f"✅ {title_b}",
        border_style="green",
        subtitle=f"[muted]{len(response_b.sources)} sources[/muted]",
    )

    console.print(panel_a)
    console.print(panel_b)


def display_example(
    example_id: int,
    problem: str,
    category: str,
    answer: str,
    answer_type: str | None = None,
    console: Console | None = None,
) -> None:
    """Display a dataset example with rich formatting.

    Parameters
    ----------
    example_id : int
        The example ID.
    problem : str
        The problem/question text.
    category : str
        The problem category.
    answer : str
        The expected answer.
    answer_type : str, optional
        The answer type if available.
    console : Console, optional
        Rich console to use. Creates one if not provided.
    """
    if console is None:
        console = create_console()

    content = f"[bold]Problem:[/bold]\n{problem}\n\n[bold]Category:[/bold] {category}\n"
    if answer_type:
        content += f"[bold]Answer Type:[/bold] {answer_type}\n"
    content += f"\n[success]Expected Answer:[/success]\n{answer}"

    console.print(Panel(content, title=f"📝 Example {example_id}", border_style="blue"))


def display_evaluation_result(
    example_id: int,
    problem: str,
    ground_truth: str,
    prediction: str,
    sources_used: int,
    search_queries: list[str],
    contains_answer: bool,
    console: Console | None = None,
) -> None:
    """Display an evaluation result with rich formatting.

    Parameters
    ----------
    example_id : int
        The example ID.
    problem : str
        The problem text.
    ground_truth : str
        The expected answer.
    prediction : str
        The model's prediction.
    sources_used : int
        Number of sources used.
    search_queries : list[str]
        Search queries executed.
    contains_answer : bool
        Whether the prediction contains the expected answer.
    console : Console, optional
        Rich console to use. Creates one if not provided.
    """
    if console is None:
        console = create_console()

    status_icon = "[success]✓[/success]" if contains_answer else "[warning]✗[/warning]"
    status_text = "MATCH" if contains_answer else "NO MATCH"
    border_color = "green" if contains_answer else "yellow"

    # Truncate long texts
    problem_display = problem[:200] + "..." if len(problem) > 200 else problem
    pred_display = prediction[:300] + "..." if len(prediction) > 300 else prediction

    content = (
        f"{status_icon} [bold]{status_text}[/bold]\n\n"
        f"[info]Problem:[/info]\n{problem_display}\n\n"
        f"[info]Expected:[/info] {ground_truth}\n\n"
        f"[info]Prediction:[/info]\n{pred_display}\n\n"
        f"[muted]Sources: {sources_used} | Queries: {len(search_queries)}[/muted]"
    )

    console.print(Panel(content, title=f"Example {example_id}", border_style=border_color))


def display_metrics_table(
    metrics: dict[str, float | int | str],
    title: str = "Evaluation Metrics",
    console: Console | None = None,
) -> None:
    """Display evaluation metrics as a formatted table.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric names to values.
    title : str, optional
        Table title, by default "Evaluation Metrics".
    console : Console, optional
        Rich console to use. Creates one if not provided.
    """
    if console is None:
        console = create_console()

    table = Table(title=f"📊 {title}", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="info")
    table.add_column("Value", style="white", justify="right")

    for name, value in metrics.items():
        if isinstance(value, float):
            table.add_row(name, f"{value:.1f}")
        else:
            table.add_row(name, str(value))

    console.print(table)


def display_success(message: str, console: Console | None = None) -> None:
    """Display a success message.

    Parameters
    ----------
    message : str
        The success message.
    console : Console, optional
        Rich console to use.
    """
    if console is None:
        console = create_console()
    console.print(f"[success]✓[/success] {message}")


def display_info(message: str, console: Console | None = None) -> None:
    """Display an info message.

    Parameters
    ----------
    message : str
        The info message.
    console : Console, optional
        Rich console to use.
    """
    if console is None:
        console = create_console()
    console.print(f"[info]ℹ[/info] {message}")


def display_warning(message: str, console: Console | None = None) -> None:
    """Display a warning message.

    Parameters
    ----------
    message : str
        The warning message.
    console : Console, optional
        Rich console to use.
    """
    if console is None:
        console = create_console()
    console.print(f"[warning]⚠[/warning] {message}")

"""Demo UI for the legislative content extraction agent (Google ADK + Gemini).

Example
-------
$ python -m implementations.legislative_content_extraction.demo
$ python -m implementations.legislative_content_extraction.demo --enable-public-link
"""

import asyncio
import json
import logging
import os
import threading
import urllib.request
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import click
import gradio as gr
from aieng.agent_evals.async_client_manager import AsyncClientManager
from dotenv import load_dotenv

load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FILES_DIR = Path(__file__).resolve().parent / "files"
DATASET_JSON = Path(__file__).resolve().parent / "legislative_content_extraction_dataset.json"

GRADIO_STATE = gr.State(value={"trace_id": None})


def load_input_data() -> list[dict[str, str]]:
    """Load the legislative docs input JSON."""
    if DATASET_JSON.exists():
        with open(DATASET_JSON) as f:
            return json.load(f)
    return []


def get_record_choices() -> list[str]:
    """Return record IDs from the dataset JSON."""
    return [r["record_id"] for r in load_input_data() if r.get("record_id")]


def get_record(record_id: str) -> dict[str, str]:
    """Look up the dataset record for a given record ID."""
    for record in load_input_data():
        if record.get("record_id") == record_id:
            return record
    return {}


def ensure_pdf(record_id: str) -> tuple[str, str]:
    """Return the local path to a PDF and its content directory, downloading if needed.

    Parameters
    ----------
    record_id : str
        The record ID (e.g. ``"ID_H0004"``).

    Returns
    -------
    tuple[str, str]
        (pdf_path, content_dir) — absolute path to the PDF and its per-ID directory.

    Raises
    ------
    FileNotFoundError
        If the PDF cannot be found locally and cannot be downloaded.
    """
    record = get_record(record_id)
    content_dir = FILES_DIR / record_id
    pdf_name = record.get("pdf_file_name", f"{record_id}.pdf")
    local_path = content_dir / pdf_name

    if local_path.exists():
        return str(local_path), str(content_dir)

    pdf_link = record.get("pdf_file_link", "")
    if not pdf_link:
        msg = f"PDF not found locally and no pdf_file_link available: {pdf_name}"
        raise FileNotFoundError(msg)

    logger.info("Downloading %s from %s", pdf_name, pdf_link)
    content_dir.mkdir(parents=True, exist_ok=True)

    try:
        req = urllib.request.Request(
            pdf_link,
            headers={"User-Agent": "Mozilla/5.0 (legislative-content-extraction-agent)"},
        )
        with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310
            local_path.write_bytes(response.read())
    except (HTTPError, URLError, OSError) as e:
        msg = f"Failed to download PDF from {pdf_link}: {e}"
        raise FileNotFoundError(msg) from e

    logger.info("Downloaded %s to %s", pdf_name, local_path)
    return str(local_path), str(content_dir)


def calculate_and_send_scores(callback_context):
    """Calculate token usage and latency scores and submit them to Langfuse.

    This is a callback function to be called after the agent has run.

    Parameters
    ----------
    callback_context : CallbackContext
        The callback context at the end of the agent run.
    """
    from aieng.agent_evals.async_client_manager import AsyncClientManager
    from aieng.agent_evals.langfuse import report_usage_scores

    for event in callback_context.session.events:
        if event.is_final_response() and event.content and event.content.role == "model":
            langfuse_client = AsyncClientManager.get_instance().langfuse_client
            trace_id = langfuse_client.get_current_trace_id()

            GRADIO_STATE.value["trace_id"] = trace_id

            thread = threading.Thread(
                target=report_usage_scores,
                kwargs={
                    "trace_id": trace_id,
                    "token_threshold": 15000,
                    "latency_threshold": 60,
                },
                daemon=True,
            )
            thread.start()

            return

    logger.error("No final response found in the callback context. Will not report scores to Langfuse.")


def on_feedback(vote: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Handle user feedback and send the score to Langfuse.

    Parameters
    ----------
    vote : str
        The feedback vote: "full_success", "partial_success", or "fail".

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Updated visibility states for the feedback row and thank-you row.
    """
    trace_id = GRADIO_STATE.value.get("trace_id")
    if trace_id:
        score = 1 if vote == "full_success" else (0.5 if vote == "partial_success" else 0)
        logger.info("Reporting user feedback score for trace %s with value %s", trace_id, score)
        try:
            langfuse_client = AsyncClientManager.get_instance().langfuse_client
            langfuse_client.create_score(
                value=score,
                name="User Feedback",
                comment=f"User voted: {vote}",
                trace_id=trace_id,
            )
            langfuse_client.flush()
        except Exception as e:
            logger.warning("Failed to report feedback to Langfuse: %s", e)
        GRADIO_STATE.value["trace_id"] = None
    else:
        logger.warning("No trace ID found. Skipping Langfuse feedback.")

    return gr.update(visible=False), gr.update(visible=True)


def toggle_feedback_row() -> tuple[dict[str, Any], dict[str, Any]]:
    """Show feedback row and hide thank-you row after metadata is updated.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Updated visibility states for the feedback row and thank-you row.
    """
    return gr.update(visible=True), gr.update(visible=False)


async def run_adk_agent(record_id: str) -> tuple[str, str, str, str, str]:
    """Run the Google ADK agent on the selected PDF."""
    if not record_id:
        return "No record selected.", "", "", "", ""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "GOOGLE_API_KEY or GEMINI_API_KEY not set in .env file.", "", "", "", ""

    from aieng.agent_evals.legislative_content_extraction import LegislativeContentExtractionAgent

    try:
        pdf_path, content_dir = ensure_pdf(record_id)
    except FileNotFoundError as e:
        return str(e), "", "", "", ""

    record = get_record(record_id)
    html_page_link = record.get("html_page_link", "")
    agent = LegislativeContentExtractionAgent(
        after_agent_callback=calculate_and_send_scores,
        files_dir=content_dir,
    )

    response = await agent.answer_async(
        pdf_path=pdf_path,
        prompt="Extract legislative metadata from this bill.",
        html_page_link=html_page_link,
    )

    # Extract summary from JSON response (strip markdown fences if present)
    summary = ""
    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        metadata = json.loads(raw_text)
        summary = metadata.pop("summary", "")
        metadata_json = json.dumps(metadata, indent=2)
    except (json.JSONDecodeError, TypeError):
        metadata_json = response.text

    reasoning_text = "\n\n---\n\n".join(response.reasoning_chain) if response.reasoning_chain else "(no reasoning captured)"

    return summary, metadata_json, f"{response.total_duration_ms}ms", json.dumps(response.tool_calls, indent=2), reasoning_text


def run_agent(record_id: str) -> tuple[str, str, str, str, str]:
    """Run the ADK agent."""
    return asyncio.run(run_adk_agent(record_id))


def start_gradio_app(enable_public_link: bool = False) -> None:
    """Start the Gradio app.

    Parameters
    ----------
    enable_public_link : bool
        Whether to create a public Gradio share link.
    """
    record_choices = get_record_choices()

    with gr.Blocks(title="Legislative Content Extraction Agent (Google ADK + Gemini)") as demo:
        gr.Markdown("# Legislative Content Extraction Agent (Google ADK + Gemini)")
        gr.Markdown("Select a legislative record and run the agent to extract structured metadata.")

        record_dropdown = gr.Dropdown(
            choices=record_choices,
            label="Select Record",
            value=record_choices[0] if record_choices else None,
        )

        run_btn = gr.Button("Extract Metadata", variant="primary")

        with gr.Row():
            duration_output = gr.Textbox(label="Duration", interactive=False)

        summary_output = gr.Textbox(label="Summary", interactive=False, lines=4)
        metadata_output = gr.Code(label="Extracted Metadata (JSON)", language="json")
        reasoning_output = gr.Textbox(label="Reasoning", interactive=False, lines=8)
        tool_calls_output = gr.Code(label="Tool Calls", language="json")

        with gr.Row(visible=False) as thank_you_row:
            gr.Markdown("Thank you for your feedback!")

        with gr.Row(visible=False) as feedback_row:
            gr.Markdown("### Feedback")
            full_success_btn = gr.Button("Full Success", variant="primary")
            partial_success_btn = gr.Button("Partial Success", variant="secondary")
            fail_btn = gr.Button("Fail", variant="stop")

        run_btn.click(
            fn=run_agent,
            inputs=[record_dropdown],
            outputs=[summary_output, metadata_output, duration_output, tool_calls_output, reasoning_output],
        ).then(fn=toggle_feedback_row, outputs=[feedback_row, thank_you_row])

        full_success_btn.click(
            fn=lambda: on_feedback("full_success"),
            outputs=[feedback_row, thank_you_row],
        )

        partial_success_btn.click(
            fn=lambda: on_feedback("partial_success"),
            outputs=[feedback_row, thank_you_row],
        )

        fail_btn.click(
            fn=lambda: on_feedback("fail"),
            outputs=[feedback_row, thank_you_row],
        )

    demo.launch(server_name="0.0.0.0", share=enable_public_link)


@click.command()
@click.option(
    "--enable-public-link",
    is_flag=True,
    default=False,
    help="Create a public Gradio share link.",
)
def cli(enable_public_link: bool = False) -> None:
    """Launch the legislative content extraction demo UI."""
    start_gradio_app(enable_public_link=enable_public_link)


if __name__ == "__main__":
    cli()

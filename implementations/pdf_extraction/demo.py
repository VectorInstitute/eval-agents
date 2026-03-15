"""Demo UI for the PDF extraction agent (Google ADK + Gemini).

Example
-------
$ python -m implementations.pdf_extraction.demo
$ python -m implementations.pdf_extraction.demo --enable-public-link
"""

import asyncio
import glob
import json
import logging
import os
from pathlib import Path

import click
import gradio as gr
from dotenv import load_dotenv

load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PDF_DIR = Path(__file__).resolve().parent / "pdfs"


def find_pdfs() -> list[str]:
    """Find all PDF files in the pdfs directory."""
    return sorted(glob.glob(str(PDF_DIR / "*.pdf")))


def get_pdf_choices() -> list[str]:
    """Return display names for available PDFs."""
    return [Path(p).name for p in find_pdfs()]


def get_pdf_path(name: str) -> str:
    """Get full path for a PDF filename."""
    return str(PDF_DIR / name)


def log_feedback(
    vote: str,
    pdf_name: str,
    jurisdiction: str,
    metadata_json: str,
) -> str:
    """Log feedback to the console."""
    logger.info("Feedback: vote=%s pdf=%s jurisdiction=%s", vote, pdf_name, jurisdiction)
    logger.debug("Metadata: %s", metadata_json)
    return f"Feedback recorded: {vote}"


async def run_adk_agent(pdf_name: str, jurisdiction: str) -> tuple[str, str, str]:
    """Run the Google ADK agent on the selected PDF."""
    if not pdf_name:
        return "No PDF selected.", "", ""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "GOOGLE_API_KEY or GEMINI_API_KEY not set in .env file.", "", ""

    from aieng.agent_evals.pdf_extraction import PdfExtractionAgent

    pdf_path = get_pdf_path(pdf_name)
    agent = PdfExtractionAgent()

    response = await agent.answer_async(
        pdf_path=pdf_path,
        jurisdiction=jurisdiction,
        prompt="Extract legislative metadata from this bill.",
    )

    return response.text, f"{response.total_duration_ms}ms", json.dumps(response.tool_calls, indent=2)


def run_agent(pdf_name: str, jurisdiction: str) -> tuple[str, str, str]:
    """Run the ADK agent."""
    return asyncio.run(run_adk_agent(pdf_name, jurisdiction))


def start_gradio_app(enable_public_link: bool = False) -> None:
    """Start the Gradio app.

    Parameters
    ----------
    enable_public_link : bool
        Whether to create a public Gradio share link.
    """
    pdf_choices = get_pdf_choices()

    with gr.Blocks(title="PDF Extraction Agent (Google ADK + Gemini)") as demo:
        gr.Markdown("# PDF Extraction Agent (Google ADK + Gemini)")
        gr.Markdown("Select a PDF and run the agent to extract structured legislative metadata.")

        with gr.Row():
            pdf_dropdown = gr.Dropdown(
                choices=pdf_choices,
                label="Select PDF",
                value=pdf_choices[0] if pdf_choices else None,
            )
            jurisdiction_input = gr.Dropdown(
                choices=["Idaho"],
                value="Idaho",
                label="Jurisdiction",
            )

        run_btn = gr.Button("Extract Metadata", variant="primary")

        with gr.Row():
            duration_output = gr.Textbox(label="Duration", interactive=False)

        metadata_output = gr.Code(label="Extracted Metadata (JSON)", language="json")
        tool_calls_output = gr.Code(label="Tool Calls", language="json")

        gr.Markdown("### Feedback")
        with gr.Row():
            full_success_btn = gr.Button("Full Success", variant="primary")
            partial_success_btn = gr.Button("Partial Success", variant="secondary")
            fail_btn = gr.Button("Fail", variant="stop")
        feedback_status = gr.Textbox(label="Feedback Status", interactive=False)

        run_btn.click(
            fn=run_agent,
            inputs=[pdf_dropdown, jurisdiction_input],
            outputs=[metadata_output, duration_output, tool_calls_output],
        )

        full_success_btn.click(
            fn=lambda pdf, jur, meta: log_feedback("full_success", pdf, jur, meta),
            inputs=[pdf_dropdown, jurisdiction_input, metadata_output],
            outputs=[feedback_status],
        )

        partial_success_btn.click(
            fn=lambda pdf, jur, meta: log_feedback("partial_success", pdf, jur, meta),
            inputs=[pdf_dropdown, jurisdiction_input, metadata_output],
            outputs=[feedback_status],
        )

        fail_btn.click(
            fn=lambda pdf, jur, meta: log_feedback("fail", pdf, jur, meta),
            inputs=[pdf_dropdown, jurisdiction_input, metadata_output],
            outputs=[feedback_status],
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
    """Launch the PDF extraction demo UI."""
    start_gradio_app(enable_public_link=enable_public_link)


if __name__ == "__main__":
    cli()

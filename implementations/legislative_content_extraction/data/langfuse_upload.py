"""Transform and/or upload legislative content extraction dataset to Langfuse.

Usage:
    # Transform + upload (default)
    uv run implementations/legislative_content_extraction/data/langfuse_upload.py

    # Transform only, no upload
    uv run implementations/legislative_content_extraction/data/langfuse_upload.py \
        --no-upload

    # Upload only (skip transform, use existing eval dataset)
    uv run implementations/legislative_content_extraction/data/langfuse_upload.py \
        --no-transform

    # Upload only Idaho records
    uv run implementations/legislative_content_extraction/data/langfuse_upload.py \
        --jurisdiction ID
"""

import asyncio
import json
import logging
from pathlib import Path

import click
from aieng.agent_evals.langfuse import upload_dataset_to_langfuse
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv(verbose=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FIXED_PROMPT = "Extract legislative metadata from this bill."

DEFAULT_INPUT = (
    "implementations/legislative_content_extraction/"
    "legislative_content_extraction_dataset.json"
)
DEFAULT_OUTPUT = (
    "implementations/legislative_content_extraction/"
    "data/legislative_eval_dataset.json"
)
DEFAULT_DATASET_NAME = "legislative-docs"


# ---------------------------------------------------------------------------
# Raw dataset models (LegislativeContentExtractionDataset)
# ---------------------------------------------------------------------------


class RawSectionAffected(BaseModel):
    """A section-affected entry as it appears in the raw dataset."""

    raw_section: str
    action: str
    code_title: int | str | None = None
    code_chapter: int | str | None = None
    code_section: int | str | None = None
    code_subdivision: str | None = None
    note: str | None = None


class LegislativeContentExtractionDataset(BaseModel):
    """A single record from the raw legislative content extraction dataset."""

    record_id: str
    pdf_file_link: str
    pdf_file_name: str
    html_page_link: str
    jurisdiction_code: str
    session_code: str
    chamber_code: str
    measure_type_code: str
    measure_number: str
    title: str
    summary: str
    sponsors: list[str]
    sections_affected: list[RawSectionAffected] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Langfuse-format models (LegislativeContentLangfuseDataset)
# ---------------------------------------------------------------------------


class LangfuseInput(BaseModel):
    """Input payload for a Langfuse dataset item.

    Fields map to agent.answer_async arguments:
    - record_id + pdf_file_name -> pdf_path (resolved by the task function)
    - html_page_link -> html_page_link
    - prompt -> prompt
    """

    record_id: str
    pdf_file_name: str
    html_page_link: str
    prompt: str


class CleanedSectionAffected(BaseModel):
    """A section-affected entry with code_title/chapter/section/subdivision stripped."""

    raw_section: str
    action: str
    note: str | None = None


class LangfuseExpectedOutput(BaseModel):
    """Expected output payload for a Langfuse dataset item."""

    jurisdiction_code: str
    session_code: str
    chamber_code: str
    measure_type_code: str
    measure_number: str
    title: str
    summary: str
    sponsors: list[str]
    sections_affected: list[CleanedSectionAffected]


class LegislativeContentLangfuseDataset(BaseModel):
    """A single Langfuse-format dataset item for legislative content extraction."""

    id: str
    input: LangfuseInput
    expected_output: LangfuseExpectedOutput


# ---------------------------------------------------------------------------
# Transform logic
# ---------------------------------------------------------------------------


def transform_record(raw: LegislativeContentExtractionDataset) -> LegislativeContentLangfuseDataset:
    """Transform a single raw dataset record to Langfuse format."""
    return LegislativeContentLangfuseDataset(
        id=raw.record_id,
        input=LangfuseInput(
            record_id=raw.record_id,
            pdf_file_name=raw.pdf_file_name,
            html_page_link=raw.html_page_link,
            prompt=FIXED_PROMPT,
        ),
        expected_output=LangfuseExpectedOutput(
            jurisdiction_code=raw.jurisdiction_code,
            session_code=raw.session_code,
            chamber_code=raw.chamber_code,
            measure_type_code=raw.measure_type_code,
            measure_number=raw.measure_number,
            title=raw.title,
            summary=raw.summary,
            sponsors=raw.sponsors,
            sections_affected=[
                CleanedSectionAffected(
                    raw_section=s.raw_section,
                    action=s.action,
                    note=s.note,
                )
                for s in raw.sections_affected
            ],
        ),
    )


def transform_dataset(
    input_path: Path, output_path: Path, jurisdiction: str | None = None
) -> None:
    """Read raw dataset, transform records, write Langfuse-format JSON.

    Parameters
    ----------
    input_path : Path
        Raw dataset JSON.
    output_path : Path
        Output eval dataset JSON.
    jurisdiction : str, optional
        If set, only include records matching this jurisdiction_code (e.g. "ID", "WI").
    """
    logger.info("Reading raw dataset from '%s'", input_path)
    with input_path.open("r", encoding="utf-8") as f:
        raw_json = json.load(f)

    raw_records = [LegislativeContentExtractionDataset.model_validate(r) for r in raw_json]

    if jurisdiction:
        raw_records = [r for r in raw_records if r.jurisdiction_code.upper() == jurisdiction.upper()]
        logger.info("Filtered to %d records for jurisdiction '%s'", len(raw_records), jurisdiction)
    else:
        logger.info("Transforming all %d records", len(raw_records))

    langfuse_records = [transform_record(r) for r in raw_records]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump(exclude_none=True) for r in langfuse_records],
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("Wrote %d Langfuse-format records to '%s'", len(langfuse_records), output_path)


@click.command()
@click.option(
    "--input",
    "input_path",
    default=DEFAULT_INPUT,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the raw legislative content extraction dataset JSON.",
)
@click.option(
    "--output",
    "output_path",
    default=DEFAULT_OUTPUT,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the Langfuse-format eval dataset JSON.",
)
@click.option(
    "--dataset-name",
    default=DEFAULT_DATASET_NAME,
    help="Langfuse dataset name for upload.",
)
@click.option(
    "--jurisdiction",
    default=None,
    help="Filter to a single jurisdiction code (e.g. ID, WI, MN, DE, MI). "
    "Appended to --dataset-name as '<name>-<jurisdiction>'.",
)
@click.option(
    "--transform/--no-transform",
    default=True,
    help="Whether to transform the raw dataset before uploading.",
)
@click.option(
    "--upload/--no-upload",
    default=True,
    help="Whether to upload the dataset to Langfuse.",
)
def cli(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    jurisdiction: str | None,
    transform: bool,
    upload: bool,
) -> None:
    """Transform and/or upload legislative dataset to Langfuse."""
    if jurisdiction:
        dataset_name = f"{dataset_name}-{jurisdiction.upper()}"

    if transform:
        transform_dataset(input_path, output_path, jurisdiction=jurisdiction)
    else:
        logger.info("Skipping transform (--no-transform).")

    if upload:
        logger.info("Uploading '%s' to Langfuse as '%s'", output_path, dataset_name)
        asyncio.run(upload_dataset_to_langfuse(str(output_path), dataset_name))
    else:
        logger.info("Skipping upload (--no-upload).")


if __name__ == "__main__":
    cli()

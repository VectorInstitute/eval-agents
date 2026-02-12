r"""Run AML cases from JSONL.

This module provides a CLI for running the AML investigation agent over cases
defined in a JSONL file. The workflow is:

1. Read input cases from JSONL.
2. Run the AML task over pending cases. Save intermediate results and show progress
   while cases run.
3. Write results back to JSONL.
4. Print a simple confusion matrix.

Examples
--------
Run with defaults:
    uv run --env-file .env implementations/aml_investigation/cli.py

Run with custom settings:
    uv run --env-file .env implementations/aml_investigation/cli.py \
      --input-path implementations/aml_investigation/data/aml_cases.jsonl \
      --output-path implementations/aml_investigation/data/aml_cases_with_output.jsonl \
      --max-concurrent-cases 8 \
      --resume
"""

import asyncio
import logging
import os
from pathlib import Path

import click
from aieng.agent_evals.aml_investigation.agent import create_aml_investigation_agent
from aieng.agent_evals.aml_investigation.data import AnalystOutput, CaseRecord
from aieng.agent_evals.aml_investigation.task import AmlInvestigationTask
from aieng.agent_evals.progress import create_progress
from langfuse.experiment import LocalExperimentItem
from rich.logging import RichHandler


logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(show_path=False)], force=True)
logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = Path("implementations/aml_investigation/data/aml_cases.jsonl")
DEFAULT_OUTPUT_FILENAME = "aml_cases_with_output.jsonl"
DEFAULT_MAX_CONCURRENT_CASES = 10


def _load_case_records(path: Path) -> list[CaseRecord]:
    """Load case records from a JSONL file.

    Invalid rows are skipped with a warning.
    """
    if not path.exists():
        return []

    records: list[CaseRecord] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(CaseRecord.model_validate_json(stripped))
            except Exception as exc:
                logger.warning("Skipping invalid JSONL row at %s:%d (%s)", path, line_number, exc)
    return records


def _write_case_records(path: Path, records: list[CaseRecord]) -> None:
    """Write case records to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(record.model_dump_json() + "\n")


def _merge_records_in_input_order(
    input_records: list[CaseRecord], updates_by_id: dict[str, CaseRecord]
) -> list[CaseRecord]:
    """Merge updates back into the original input order.

    If duplicate ``case_id`` values exist in input, only the first instance is kept.
    """
    merged: list[CaseRecord] = []
    seen: set[str] = set()
    for record in input_records:
        case_id = record.input.case_id
        if case_id in seen:
            continue
        seen.add(case_id)
        merged.append(updates_by_id.get(case_id, record))
    return merged


def _log_is_laundering_confusion_matrix(records: list[CaseRecord]) -> None:
    """Log a simple confusion matrix using records that have predictions."""
    scored = [record for record in records if record.output is not None]
    if not scored:
        logger.info("Metrics: N/A (no analyzed cases)")
        return

    tp = fp = fn = tn = 0
    for record in scored:
        gt = record.expected_output.is_laundering
        assert record.output is not None
        pred = record.output.is_laundering
        if gt and pred:
            tp += 1
        elif (not gt) and pred:
            fp += 1
        elif gt and (not pred):
            fn += 1
        else:
            tn += 1

    logger.info("is_laundering confusion matrix:")
    logger.info("  TP=%d  FP=%d", tp, fp)
    logger.info("  FN=%d  TN=%d", fn, tn)


async def _run_case(task: AmlInvestigationTask, record: CaseRecord, semaphore: asyncio.Semaphore) -> CaseRecord:
    """Run one case.

    This function is intentionally defensive: it logs errors and always returns
    a ``CaseRecord`` so the full batch can continue.
    """
    try:
        async with semaphore:
            item: LocalExperimentItem = {"input": record.input.model_dump(), "metadata": {"id": record.input.case_id}}
            output = await task(item=item)

        if output is None:
            logger.warning("No analyst output produced for case_id=%s", record.input.case_id)
            return record

        record.output = AnalystOutput.model_validate(output)
        return record
    except Exception as exc:
        logger.exception("Case failed (case_id=%s): %s", record.input.case_id, exc)
        return record


async def run_cases(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path | None = None,
    max_concurrent_cases: int = DEFAULT_MAX_CONCURRENT_CASES,
    resume: bool = True,
) -> Path:
    """Run AML investigations for cases from an input JSONL file.

    Parameters
    ----------
    input_path : Path, optional
        Input case JSONL path.
    output_path : Path | None, optional
        Output JSONL path. If ``None``, a default filename is used in the input
        directory.
    max_concurrent_cases : int, optional
        Maximum number of cases to process at the same time.
    resume : bool, optional
        If ``True``, cases already analyzed in the output file are skipped.

    Returns
    -------
    Path
        Final output path.
    """
    if max_concurrent_cases <= 0:
        raise ValueError("max_concurrent_cases must be > 0")
    if not input_path.exists():
        raise FileNotFoundError(f"Case JSONL file not found at {input_path.resolve()}")

    resolved_output_path = output_path or input_path.with_name(DEFAULT_OUTPUT_FILENAME)
    if not resume and resolved_output_path.exists():
        resolved_output_path.unlink()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    input_records = _load_case_records(input_path)
    existing_by_id: dict[str, CaseRecord] = {}
    if resume:
        for record in _load_case_records(resolved_output_path):
            existing_by_id[record.input.case_id] = record

    if resume:
        pending = [
            record for record in input_records if existing_by_id.get(record.input.case_id, record).output is None
        ]
        logger.info(
            "Resume: %d/%d done; %d remaining.", len(input_records) - len(pending), len(input_records), len(pending)
        )
    else:
        pending = input_records
        logger.info("Running %d/%d cases from scratch.", len(pending), len(input_records))

    semaphore = asyncio.Semaphore(max_concurrent_cases)
    agent = create_aml_investigation_agent(enable_tracing=True)
    task_runner = AmlInvestigationTask(agent=agent)
    tasks = [asyncio.create_task(_run_case(task_runner, record, semaphore)) for record in pending]
    try:
        with resolved_output_path.open("a", encoding="utf-8") as checkpoint_file, create_progress() as progress:
            progress_task = progress.add_task("Analyzing AML cases", total=len(tasks))
            for finished in asyncio.as_completed(tasks):
                record = await finished
                existing_by_id[record.input.case_id] = record

                # Save each completed case immediately so resume works after
                # cancellation/crash and we do not waste API calls
                checkpoint_file.write(record.model_dump_json() + "\n")
                checkpoint_file.flush()
                os.fsync(checkpoint_file.fileno())

                progress.update(progress_task, advance=1)
    except asyncio.CancelledError:
        logger.warning("Run cancelled. Partial results are saved in %s", resolved_output_path)
        raise
    finally:
        await task_runner.close()

    final_records = _merge_records_in_input_order(input_records, existing_by_id)
    _write_case_records(resolved_output_path, final_records)
    logger.info("Wrote %d analyzed cases to %s", sum(r.output is not None for r in final_records), resolved_output_path)

    _log_is_laundering_confusion_matrix(final_records)
    return resolved_output_path


@click.command()
@click.option(
    "--input-path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    default=DEFAULT_INPUT_PATH,
    show_default=True,
    help="Input case JSONL file.",
)
@click.option(
    "--output-path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Output JSONL file. Defaults to aml_cases_with_output.jsonl next to input.",
)
@click.option(
    "--max-concurrent-cases",
    type=int,
    default=DEFAULT_MAX_CONCURRENT_CASES,
    show_default=True,
    help="Maximum number of cases to run at once.",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Skip cases that already have analysis in the output file.",
)
def cli(input_path: Path, output_path: Path | None, max_concurrent_cases: int, resume: bool) -> None:
    """Run the AML demo workflow over example cases."""
    if max_concurrent_cases <= 0:
        raise click.BadParameter("max-concurrent-cases must be > 0", param_hint="--max-concurrent-cases")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    final_path = asyncio.run(
        run_cases(
            input_path=input_path,
            output_path=output_path,
            max_concurrent_cases=max_concurrent_cases,
            resume=resume,
        )
    )
    click.echo(f"Done. Output written to {final_path}")


if __name__ == "__main__":
    cli()

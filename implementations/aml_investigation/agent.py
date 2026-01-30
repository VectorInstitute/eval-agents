"""AML Investigation Agent Implementation.

Run with: uv run --env-file .env implementations/aml_investigation/agent.py
"""

import asyncio
import getpass
import json
import logging
import os
import uuid
from pathlib import Path

import google.genai.types
from aieng.agent_evals.aml_investigation.data import AnalystOutput, CaseRecord
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.async_utils import rate_limited
from aieng.agent_evals.tools import ReadOnlySqlDatabase
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn


logger = logging.getLogger(__name__)

ANALYST_PROMPT = """\
You are a newly hired Anti‑Money Laundering (AML) Investigation Analyst at a financial institution.
Your job is to investigate a single AML case by reviewing transaction activity in our database and documenting
whether the activity is consistent with money laundering or a benign explanation.

You have access to database query tools. Use them. Do not guess or invent transactions.

## Core Principle: Falsification (Default to Benign)
Start with the hypothesis that the case is benign (a false positive). Your primary goal is to find positive,
transaction-level evidence that supports a legitimate explanation. Only conclude laundering when the transaction
pattern supports it.

## Your Input (CaseFile JSON)
You will be given a JSON object with these fields:
- `case_id`: unique case identifier.
- `seed_transaction_id`: the primary transaction that triggered the case.
- `seed_timestamp`: timestamp of the seed transaction (end of the investigation window).
- `window_start`: timestamp of the beginning of the investigation window.
- `suspected_pattern_type`: a preliminary label/hint (may be wrong or intentionally misleading).

### Time Scope Rule (Strict)
Only analyze transactions with `timestamp` between `window_start` and `seed_timestamp` (inclusive).
Do not use transactions after `seed_timestamp`.

## What to Investigate (Standard Workflow)
1) **Seed review**
   - Pull the seed transaction details and identify parties (from/to accounts and banks), currencies, amounts, and format.
2) **Account history within the window**
   - Pull inbound/outbound activity for the involved accounts within the window.
3) **Pattern detection**
   Look for transaction patterns consistent with laundering typologies, such as:
   - **FAN-IN**: many sources → one account (aggregation).
   - **FAN-OUT**: one account → many destinations (dispersion).
   - **GATHER-SCATTER / SCATTER-GATHER**: aggregation then dispersion (or vice‑versa), often over short time windows.
   - **STACK / LAYERING**: sequential hops meant to obscure origin.
   - **RANDOM / CYCLE**: complex paths or circular movement.
   - **BIPARTITE**: two groups of accounts with structured flows between them.
4) **Legitimate explanations**
   - If activity is consistent with normal behavior (e.g., payroll, vendor payments, routine transfers),
     state what makes it consistent and what would have been suspicious but is not present.

## Your Output (AnalystOutput JSON)
Return a single JSON object that matches the `AnalystOutput` schema exactly. Fill every field.

- `summary_narrative` (str):
  2–3 paragraphs in plain text summarizing what you checked and what you found.
  Cite specific evidence (transaction IDs, dates/times, amounts, counterparties) and explain your reasoning.
- `is_laundering` (bool):
  `true` only when the observed pattern within the window supports laundering; otherwise `false`.
- `pattern_type` (LaunderingPattern):
  Your best classification of the dominant pattern observed. Use:
  - `NONE` if you find no laundering pattern,
  - `UNKNOWN` if you believe laundering is present but does not match a known pattern well.
- `pattern_description` (str):
  A short, human-readable description of the observed pattern (or `"N/A"` when `pattern_type` is `NONE`).
- `flagged_transaction_ids` (list[str]):
  The transaction IDs you consider most relevant/suspicious for this case.
  If `is_laundering` is `false`, this can be an empty list.
"""

client_manager = AsyncClientManager().get_instance()

if client_manager.configs.aml_db is None:
    raise ValueError("AML database configuration is missing.")

db = ReadOnlySqlDatabase(
    connection_uri=client_manager.configs.aml_db.build_uri(),
    agent_name="FraudInvestigationAnalyst",
)

root_agent = Agent(
    name="AmlInvestigationAnalyst",
    description="Conducts detailed multi-step AML investigations with database queries.",
    tools=[db.execute, db.get_schema_info],
    model="gemini-3-flash-preview",
    instruction=ANALYST_PROMPT,
    output_schema=AnalystOutput,
)


def _load_records(path: Path) -> list[CaseRecord]:
    if not path.exists():
        return []

    records: list[CaseRecord] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                records.append(CaseRecord.model_validate_json(stripped_line))
            except Exception as exc:
                logger.warning("Skipping invalid JSONL record at %s:%d (%s)", path, line_number, exc)
    return records


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _write_results(output_path: Path, input_records: list[CaseRecord], results_by_id: dict[str, CaseRecord]) -> int:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    written: set[str] = set()
    analyzed = 0

    with tmp_path.open("w", encoding="utf-8") as outfile:
        for record in input_records:
            case_id = record.case.case_id
            if case_id in written:
                continue
            written.add(case_id)
            out_record = results_by_id.get(case_id, record)
            analyzed += int(out_record.analysis is not None)
            outfile.write(out_record.model_dump_json() + "\n")

    tmp_path.replace(output_path)
    return analyzed


async def _analyze_case(runner: Runner, record: CaseRecord) -> CaseRecord:
    message = google.genai.types.Content(
        role="user", parts=[google.genai.types.Part(text=record.case.model_dump_json())]
    )
    events_async = runner.run_async(session_id=str(uuid.uuid4()), user_id=getpass.getuser(), new_message=message)

    final_text: str | None = None
    async for event in events_async:
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(part.text or "" for part in event.content.parts if part.text)

    if not final_text:
        logger.warning("No analyst output produced for case_id=%s", record.case.case_id)
        return record

    record.analysis = AnalystOutput.model_validate(_extract_json(final_text.strip()))
    return record


async def _safe_analyze_case(runner: Runner, record: CaseRecord) -> CaseRecord:
    try:
        return await _analyze_case(runner, record)
    except Exception as exc:
        logger.exception("Case failed (case_id=%s): %s", record.case.case_id, exc)
        return record


async def _analyze_cases_to_jsonl(
    runner: Runner,
    cases: list[CaseRecord],
    semaphore: asyncio.Semaphore,
    output_path: Path,
) -> dict[str, CaseRecord]:
    if not cases:
        return {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = [
        asyncio.create_task(rate_limited(lambda r=record: _safe_analyze_case(runner, r), semaphore)) for record in cases
    ]

    analyzed_by_id: dict[str, CaseRecord] = {}
    with (
        output_path.open("a", encoding="utf-8") as outfile,
        Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
        ) as progress,
    ):
        progress_task = progress.add_task("Analyzing AML cases", total=len(tasks))

        for finished in asyncio.as_completed(tasks):
            record = await finished
            analyzed_by_id[record.case.case_id] = record
            outfile.write(record.model_dump_json() + "\n")
            outfile.flush()
            os.fsync(outfile.fileno())
            progress.update(progress_task, advance=1)

    return analyzed_by_id


async def main() -> None:
    """Run the AML investigation agent on cases from JSONL."""
    input_path = Path("implementations/aml_investigation/data/aml_cases.jsonl")
    if not input_path.exists():
        raise FileNotFoundError(f"Case JSONL not found at {input_path.resolve()}")

    output_path = input_path.with_name("aml_cases_with_analysis.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_records = _load_records(input_path)
    existing_results = {record.case.case_id: record for record in _load_records(output_path)}
    to_run = [r for r in input_records if existing_results.get(r.case.case_id, r).analysis is None]

    logger.info("Resume: %d/%d done; %d remaining.", len(input_records) - len(to_run), len(input_records), len(to_run))

    runner = Runner(
        app_name="aml_investigation",
        agent=root_agent,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )

    try:
        analyzed_by_id = await _analyze_cases_to_jsonl(runner, to_run, asyncio.Semaphore(5), output_path)
        existing_results.update(analyzed_by_id)
        analyzed_count = _write_results(output_path, input_records, existing_results)
        logger.info("Wrote %d analyzed cases to %s", analyzed_count, output_path)

        final_records = [existing_results.get(r.case.case_id, r) for r in input_records]
        scored = [r for r in final_records if r.analysis is not None]
        if not scored:
            logger.info("Metrics: N/A (no analyzed cases)")
        else:
            tp = fp = fn = tn = 0
            for r in scored:
                gt = r.groundtruth.is_laundering
                pred = r.analysis.is_laundering
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
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()

    asyncio.run(main())

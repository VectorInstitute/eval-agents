"""Fraud Investigation Agent Implementation.

Run with: uv run --env-file .env implementations/fraud_investigation/agent.py
"""

import asyncio
import getpass
import logging
import uuid
from pathlib import Path
from typing import Any

import google.genai.types
import pandas as pd
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.async_utils import gather_with_progress, rate_limited
from aieng.agent_evals.fraud_investigation.policy import RuleBasedTriage, TriageOutcome, TriagePolicyConfig
from aieng.agent_evals.fraud_investigation.types import AnalystResult
from aieng.agent_evals.fraud_investigation.workflow import AmlInvestigationWorkflow
from aieng.agent_evals.tools import ReadOnlySqlDatabase
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


logger = logging.getLogger(__name__)

ANALYST_PROMPT = """\
You are a Financial Crimes Investigation Analyst. Your primary role is to conduct a focused investigation into \
transactions flagged by our automated monitoring system. Your objective is to determine if the activity is suspicious \
and requires reporting, or if it is a false positive that can be closed.

## Core Principle: Falsification
Begin with the hypothesis that the automated alert is a false positive. \
Your primary goal is to find positive evidence to prove the transaction is legitimate.

## Standard Operating Procedure
### Initial Assessment
- Review the provided transaction data. Understand why it was flagged (e.g., high amount, unusual pattern).

### Evidence Gathering
- **Customer Profile:** Examine the historical activity of the involved parties. Is this transaction consistent with their typical behavior?
- **Counterparty Analysis:** Investigate the transaction's source or destination. What is the relationship between the parties?
- **Pattern Analysis:** Search for broader patterns that could indicate illicit activity, such as:
    - Structuring: multiple transactions just below reporting thresholds
    - Layering: complex fund movements to obscure origins
    - Rapid Movement: high volume or velocity of transfers
    - Unusual Volume: sudden spikes in transaction amounts or frequency
    - Terrorist Financing Indicators
    - Other red flags as per FinCEN guidelines

REMEMBER: The risk scores and triggered rules in the database are imperfect. Take them with a grain of salt. The actual \
transaction patterns are the hard evidence.

### Analysis and Verdict
Synthesize your findings and determine the appropriate verdict:
- BENIGN: Clear evidence supports legitimacy. Recommend closing the alert.
- SUSPICIOUS: Strong indicators of illicit activity. Recommend filing a SAR.
- NEEDS_REVIEW: Inconclusive evidence to confirm nor deny the suspicion after thorough investigation. Recommend escalation to a senior analyst.

### Case Documentation
Document your findings comprehensively, citing specific evidence and rationale for your verdict. \
The documentation must include the following:
- Summary Narrative: A concise explanation (2-3 paragraphs) of the fraud scheme. Include the reasoning behind your verdict and enough information for filing a SAR (the 5 "Ws") if needed.
- Risk Score: A value between 0-100 indicating the level of risk based on your findings.
- Verdict: One of BENIGN, SUSPICIOUS, or NEEDS_REVIEW. Applies to ONLY the transaction under investigation.
- Suspicious Activity Type: Classify the dominant type of suspicious activity identified.
- Suspicious Transactions: Tabulate the transactions deemed suspicious, including the transaction IDs, timestamp, amount, currencies, banks, entities involved.

NOTE: For any given transaction, focus the investigation on transactions that happenned before or on the same day as the \
transaction date. Do not consider transactions that occurred after the transaction date.
"""

SAR_REPORT_PROMPT = """\
You are a Compliance Officer responsible for drafting a formal Suspicious Activity Report (SAR) narrative for submission \
to the Financial Crimes Enforcement Network (FinCEN). Your work is critical for law enforcement and regulatory oversight.

You will receive a completed case file. This file contains the investigation summary, verdict, and key evidence. \
Your task is to transform this information into a compliant and comprehensive SAR narrative.

## SAR Narrative Structure
The narrative must be written in the third person, be clear, concise, and chronological. Adhere strictly to the following "Five Ws" structure:

- **Who** is conducting the suspicious activity? Introduce the subject(s) and any associated entities.
- **What** instruments or mechanisms are being used? Describe the types of transactions (e.g., wire transfers, cash deposits) and accounts involved.
- **When** did the suspicious activity take place? Reference specific dates from `suspicious_transactions`.
- **Where** did the suspicious activity take place?
- **Why** do you deem the activity suspicious?
  - Explain the "red flags" that led to the suspicion.
  - Clearly state *why* the observed behavior deviates from the expected norm for the customer.

## Output Format
Generate a single, cohesive narrative text. Do not use markdown or headers. \
The text should be ready for direct insertion into Part V of the FinCEN SAR form. \
Begin the report with an opening sentence that summarizes the filing, for example: "[Your Financial Institution] is filing \
this report on [Subject Name] due to suspicious [activity type, e.g., wire transfers] intended to obscure the source of funds."
"""


client_manager = AsyncClientManager().get_instance()

# Define Agents
if client_manager.configs.aml_db is None:
    raise ValueError("AML database configuration is missing.")

db = ReadOnlySqlDatabase(
    connection_uri=client_manager.configs.aml_db.build_uri(),
    agent_name="FraudInvestigationAnalyst",
)

# NOTE: these are left here so that `adk web` command can pick up the `root_agent` for
# launching a web interface
analyst = Agent(
    name="FraudInvestigationAnalyst",
    description="Conducts detailed multi-step fraud investigations with database queries.",
    tools=[db.execute, db.get_schema_info],
    model="gemini-3-flash-preview",
    instruction=ANALYST_PROMPT,
    output_schema=AnalystResult,
    generate_content_config=google.genai.types.GenerateContentConfig(
        thinking_config=google.genai.types.ThinkingConfig(include_thoughts=True)
    ),
)

sar_report_agent = Agent(
    name="SARReportGenerator",
    description="Generates comprehensive Suspicious Activity Reports from investigation findings.",
    model="gemini-2.5-flash",
    instruction=SAR_REPORT_PROMPT,
)

root_agent = AmlInvestigationWorkflow(
    triage_strategy=RuleBasedTriage(config=TriagePolicyConfig()), analyst=analyst, reporter=sar_report_agent
)


async def _investigate_transaction(row: pd.Series) -> dict[str, Any]:
    """Investigate a single transaction and return structured results."""
    # Initialize a session service to store conversation state
    session_service = InMemorySessionService()

    analyst = Agent(
        name="FraudInvestigationAnalyst",
        description="Conducts detailed multi-step fraud investigations with database queries.",
        tools=[db.execute, db.get_schema_info],
        model="gemini-3-flash-preview",
        instruction=ANALYST_PROMPT,
        output_schema=AnalystResult,
    )

    sar_report_agent = Agent(
        name="SARReportGenerator",
        description="Generates comprehensive Suspicious Activity Reports from investigation findings.",
        model="gemini-2.5-flash",
        instruction=SAR_REPORT_PROMPT,
    )

    root_agent = AmlInvestigationWorkflow(
        triage_strategy=RuleBasedTriage(config=TriagePolicyConfig()), analyst=analyst, reporter=sar_report_agent
    )

    # Create a runner to manage the execution loop
    runner = Runner(
        app_name="fraud_investigation", agent=root_agent, session_service=session_service, auto_create_session=True
    )

    transaction_id = row.get("transaction_id", "unknown")

    # Collect output in memory instead of printing
    output_buffer: list[str] = []

    message = google.genai.types.Content(
        role="user",
        parts=[google.genai.types.Part(text=row.drop(labels=["is_laundering", "triggered_rules"]).to_json())],
    )
    events_async = runner.run_async(session_id=str(uuid.uuid4()), user_id=getpass.getuser(), new_message=message)

    # Extract structured results from the workflow state
    triage_decision: TriageOutcome | None = None
    analyst_findings: dict[str, Any] | None = None
    final_report = None

    async for event in events_async:
        # Capture text output
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    output_buffer.append(part.text)

        # Extract workflow state when investigation completes
        if event.actions.agent_state:
            state_data = event.actions.agent_state
            triage_decision = state_data.get("triage_decision")
            analyst_findings = state_data.get("analysis")
            final_report = state_data.get("final_report")

    logger.info(f"✅ Investigation for transaction {transaction_id} completed.")
    logger.debug("".join(output_buffer))

    return {
        "transaction_id": transaction_id,
        "is_laundering": row.get("is_laundering"),
        "system_risk_score": row.get("risk_score"),
        "triggered_rules": row.get("triggered_rules"),
        "triage_decision": triage_decision,
        "analyst_verdict": analyst_findings.get("verdict") if analyst_findings else None,
        "agent_risk_score": analyst_findings.get("risk_score") if analyst_findings else None,
        "suspicious_activity_type": analyst_findings.get("suspicious_activity_type") if analyst_findings else None,
        "summary_narrative": analyst_findings.get("summary_narrative") if analyst_findings else None,
        "suspicious_transactions": analyst_findings.get("suspicious_transactions") if analyst_findings else None,
        "suspicious_entities": analyst_findings.get("suspicious_entities") if analyst_findings else None,
        "final_report": final_report,
    }


async def main() -> None:
    """Run fraud investigations on a batch of transactions."""
    # Prepare your input
    path_to_transactions = Path("implementations/fraud_investigation/data/test_transactions.csv")
    if not path_to_transactions.exists():
        raise FileNotFoundError(f"Transaction file not found at {path_to_transactions.resolve()}")

    transactions = pd.read_csv(path_to_transactions, dtype_backend="pyarrow")

    # Investigate each transaction with limited concurrency
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent investigations
    results: list[dict[str, Any]] = []

    # TODO: remove or change this line for full run
    transactions = transactions.sample(n=100, random_state=42)

    # Gather all results
    tasks = [
        rate_limited(lambda _row=_row: _investigate_transaction(_row), semaphore) for _, _row in transactions.iterrows()
    ]
    results = list(await gather_with_progress(tasks, description="Investigating transactions..."))

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_path = path_to_transactions.parent / "investigation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Completed {len(results)} investigations. Results saved to {results_path}")

    # Print summary statistics
    print("\nSummary:")
    print(f"  Triage decisions: {results_df['triage_decision'].value_counts().to_dict()}")
    print(f"  Analyst verdicts: {results_df['analyst_verdict'].value_counts().to_dict()}")
    print(f"  Suspicious activity types: {results_df['suspicious_activity_type'].value_counts().to_dict()}")
    print(f"  Average agent risk score: {results_df['agent_risk_score'].dropna().mean():.2f}")

    db.close()


if __name__ == "__main__":
    load_dotenv()

    asyncio.run(main())

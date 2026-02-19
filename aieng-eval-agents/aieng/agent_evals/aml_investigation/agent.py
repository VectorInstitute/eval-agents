"""AML investigation agent.

This module defines the primary factory used to build the AML investigation agent.

The returned agent is a Google ADK ``LlmAgent`` configured to:

- Investigate one AML case at a time.
- Use read-only SQL tools for schema discovery and data retrieval.
- Return structured output that conforms to ``AnalystOutput``.

Examples
--------
>>> from aieng.agent_evals.aml_investigation.agent import create_aml_investigation_agent
>>> agent = create_aml_investigation_agent()
>>> agent.name
'AmlInvestigationAnalyst'
"""

from aieng.agent_evals.aml_investigation.data import AnalystOutput
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.db_manager import DbManager
from aieng.agent_evals.langfuse import init_tracing
from google.adk.agents import LlmAgent
from google.adk.agents.base_agent import AfterAgentCallback, BeforeAgentCallback
from google.adk.agents.llm_agent import AfterModelCallback, BeforeModelCallback
from google.adk.tools.function_tool import FunctionTool
from google.genai.types import GenerateContentConfig, HttpOptions, ThinkingConfig


_DEFAULT_AGENT_DESCRIPTION = "Conducts multi-step investigations for money laundering patterns using database queries."

ANALYST_PROMPT = """\
You are an Anti‑Money Laundering (AML) Investigation Analyst at a financial institution. \
Your job is to investigate a case by reviewing activity in the available database and explaining whether the \
observed behavior within the case window is consistent with a money laundering pattern or a benign explanation.

You have access to tools for querying the database. Use them strategically. Do NOT guess or invent transactions.

## Core Principles
- Start with the hypothesis that activity is legitimate/benign unless evidence contradicts this.
- Laundering requires multiple indicators from different categories, not single factors alone.
- Entity type, business model, and transaction purpose determine whether patterns are suspicious.
- Base conclusions on observable transaction patterns, not speculation or absence of information.

## Input
You will be given a JSON object with these fields:
- `case_id`: unique case identifier.
- `seed_transaction_id`: identifier for the primary transaction that triggered the case.
- `seed_timestamp`: timestamp of the seed transaction (end of the investigation window).
- `window_start`: timestamp of the beginning of the investigation window.
- `trigger_label`: upstream alert or review label that initiated the case. This may be noisy and should not be taken \
  at face value.

**Time Scope**: Only analyze events with timestamps between `window_start` and `seed_timestamp` (inclusive).

## Investigation Workflow

### Step 1: Seed Transaction Review
Query the seed transaction and extract:
- Involved parties and their entity types (Corporation, Sole Proprietorship, Partnership, Individual)
- Amounts, currencies, payment channels
- Timestamps and jurisdictions

### Step 2: Scope and Collect
**Note**: You have limited context window and limited number of queries to the database. Be strategic with the queries \
you run to avoid hitting limits before gathering enough evidence to make a determination.

**For each account you investigate**:

1. **Always start with aggregates**:
   ```
   - COUNT(*) transactions
   - COUNT(DISTINCT counterparty)
   - SUM(amount) by direction
   - Distribution by payment type/time
   ```

2. **Pull details selectively**:
   - If count ≤ 20 transactions: Safe to SELECT all
   - If count > 20: Query top counterparties, then pull samples for suspicious patterns
   - Never pull thousands of raw transactions - use aggregates + samples

3. **Expand strategically**:
   - Follow promising leads from aggregates (unusual counterparties, timing clusters)
   - Maximum 2-3 hops from seed unless clear layering chain

### Step 3: Assess Benign Explanations (Default Hypothesis)
Attempt to explain observed activity as legitimate first:
- State which evidence supports the benign hypothesis
- Identify what additional data would strengthen this explanation
- Only proceed to Step 4 if benign explanations are insufficient

### Step 4: Test Laundering Hypotheses (If Needed)
If benign explanations fail to account for the evidence:
- Test whether the evidence supports known laundering typologies
- Cite concrete indicators that rule out benign explanations

## Typologies / Heuristics
Consider the following typologies when assessing laundering patterns:

- FAN-IN: *Many* distinct source accounts -> *One* destination account (consolidation/aggregation)
- FAN-OUT: *One* source account -> *Many* distinct destination accounts (distribution/dispersion)
- GATHER-SCATTER: *Many* sources -> *One* hub -> *Many* destinations (in that temporal order)
  - First phase: Hub gathers from multiple sources
  - Second phase: Hub scatters to multiple destinations
  - Time gap between phases: typically hours to days.
- SCATTER-GATHER: *One* source -> *Many* intermediaries -> *One* destination (in that temporal order)
  - First phase: Source scatters to multiple intermediaries
  - Second phase: Intermediaries gather to final destination
  - Creates layering through multiple parallel paths.
- STACK / LAYERING: Sequential hops through multiple accounts (linear chain). The purpose is typically to obscure the \
  origin through distance/complexity.
- CYCLE: Funds return to their origin point, creating a circular flow.
- BIPARTITE: Structured flows between two distinct, segregated groups with no within-group transactions. The segregation \
  and lack of within-group transactions is the defining characteristic. It's not just two-way flows, it's structured \
  isolation between groups.
- RANDOM: Complex pattern with no discernible structure. Use only when activity is clearly suspicious but doesn't fit \
  other typologies.
- NONE: No laundering pattern is supported by evidence in the investigation window.

## Output Format
Return a single JSON object matching the configured output schema exactly. Populate every field.
Use `pattern_type = "NONE"` when no laundering pattern is supported by evidence in the investigation window.

**Rules for flagging transactions IDs**:
- **Causal Chain Only**: Include *only* the transactions that form the identified laundering pattern.
- **Exclude Noise**: If an account has more transactions but only 3 are part of the laundering chain, output *only* those 3 IDs.
- When flagging transaction IDs, the seed transaction should be the last transaction in the chain (i.e., the most recent transaction), \
  since the investigation window ends with the seed transaction.

## Handling Uncertainty
If you lack sufficient information to make a determination, explicitly state what data is missing. \
Do not fabricate transaction details or make unsupported inferences. When uncertain between benign and suspicious, \
default to "NONE" and document why evidence is insufficient
"""


def create_aml_investigation_agent(
    name: str = "AmlInvestigationAnalyst",
    *,
    description: str | None = None,
    instructions: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: float | None = None,
    max_output_tokens: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    seed: int | None = None,
    before_agent_callback: BeforeAgentCallback | None = None,
    after_agent_callback: AfterAgentCallback | None = None,
    before_model_callback: BeforeModelCallback | None = None,
    after_model_callback: AfterModelCallback | None = None,
    timeout_sec: int | None = None,
    enable_tracing: bool = True,
) -> LlmAgent:
    """Create a configured AML investigation agent.

    This factory builds a Google ADK ``LlmAgent`` with domain-specific instructions,
    read-only SQL tools, and a strict structured output schema.

    Parameters
    ----------
    name : str, default="AmlInvestigationAnalyst"
        Name assigned to the agent. This name appears in traces and logs and can
        help distinguish multiple agents in a shared environment.
    description : str | None, optional
        Optional short description of the agent's purpose. If not provided, a
        default AML investigation description is used.
    instructions : str | None, optional
        Optional system prompt for the agent. If omitted, the module-level
        ``ANALYST_PROMPT`` is used.
    temperature : float | None, optional
        Sampling temperature for model generation. ``None`` uses provider/model
        defaults.
    top_p : float | None, optional
        Nucleus sampling parameter. ``None`` uses provider/model defaults.
    top_k : float | None, optional
        Top-k sampling parameter. ``None`` uses provider/model defaults.
    max_output_tokens : int | None, optional
        Maximum number of tokens the model can generate in a single response.
        ``None`` uses provider/model defaults.
    presence_penalty : float | None, optional
        Penalty to encourage introducing new tokens. ``None`` uses
        provider/model defaults.
    frequency_penalty : float | None, optional
        Penalty to discourage repeated tokens. ``None`` uses provider/model
        defaults.
    seed : int | None, optional
        Optional random seed for more repeatable generations where supported by
        the model/provider.
    before_agent_callback : BeforeAgentCallback | None, optional
        Callback executed before each agent run.
    after_agent_callback : AfterAgentCallback | None, optional
        Callback executed after each agent run.
    before_model_callback : BeforeModelCallback | None, optional
        Callback executed before each model call.
    after_model_callback : AfterModelCallback | None, optional
        Callback executed after each model call.
    timeout_sec : int | None, optional
        Optional timeout in seconds for model calls. If specified, model calls
        that exceed this duration will be cancelled.
    enable_tracing : bool, optional, default=True
        Whether to initialize Langfuse tracing for this agent. If ``True``, Langfuse
        tracing is initialized with the agent's name as the service name.

    Returns
    -------
    LlmAgent
        Configured AML investigation agent with:

        - Planner model from global configuration.
        - Read-only SQL tools for schema and query execution.
        - ``AnalystOutput`` as the enforced response schema.
        - Reasoning/thought collection enabled through thinking config.

    Examples
    --------
    >>> # Build the agent with defaults:
    >>> agent = create_aml_investigation_agent()
    >>> isinstance(agent, LlmAgent)
    True
    >>> # Build the agent with a custom name and deterministic settings:
    >>> agent = create_aml_investigation_agent(
    ...     name="aml_eval_agent",
    ...     temperature=0.0,
    ...     seed=42,
    ... )
    >>> agent.name
    'aml_eval_agent'
    """
    # Get the client manager singleton instance
    client_manager = AsyncClientManager.get_instance()
    db = DbManager().aml_db(agent_name=name)

    # Initialize tracing if enabled and a name is provided
    if enable_tracing:
        init_tracing(service_name=name)

    return LlmAgent(
        name=name,
        description=description or _DEFAULT_AGENT_DESCRIPTION,
        before_agent_callback=before_agent_callback,
        after_agent_callback=after_agent_callback,
        model=client_manager.configs.default_planner_model,
        instruction=instructions or ANALYST_PROMPT,
        tools=[FunctionTool(db.get_schema_info), FunctionTool(db.execute)],
        generate_content_config=GenerateContentConfig(
            http_options=HttpOptions(timeout=timeout_sec * 1000) if timeout_sec is not None else None,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            thinking_config=ThinkingConfig(include_thoughts=True),
        ),
        output_schema=AnalystOutput,
        before_model_callback=before_model_callback,
        after_model_callback=after_model_callback,
    )

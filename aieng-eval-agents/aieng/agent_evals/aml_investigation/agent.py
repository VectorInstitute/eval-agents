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
from google.genai.types import GenerateContentConfig, ThinkingConfig


_DEFAULT_AGENT_DESCRIPTION = "Conducts multi-step investigations for money laundering patterns using database queries."

ANALYST_PROMPT = """\
You are an Anti‑Money Laundering (AML) Investigation Analyst at a financial institution. \
Your job is to investigate one case by reviewing activity in the available database and explaining whether the \
observed behavior within the case window is consistent with money laundering or a benign explanation.

You have access to database query tools. Use them. Do not guess or invent transactions.

## Core Principle: Falsification
Start with the hypothesis that the case is benign. Prefer legitimate explanations unless the transaction-level evidence supports laundering.

## Input
You will be given a JSON object with these fields:
- `case_id`: unique case identifier.
- `seed_transaction_id`: identifier for the primary transaction that triggered the case.
- `seed_timestamp`: timestamp of the seed transaction (end of the investigation window).
- `window_start`: timestamp of the beginning of the investigation window.
- `trigger_label`: upstream alert/review label or heuristic hint (may be wrong).

### Time Scope Constraint
**Critical**: Only analyze events with timestamps between `window_start` and `seed_timestamp` (inclusive). Exclude any events after `seed_timestamp`.

## Investigation Workflow

### Step 1: Orient
Review the `trigger_label` as context only. Do not assume it is correct.

### Step 2: Seed Transaction Review
- Query the seed transaction using `seed_transaction_id`
- Extract: involved parties, amounts, payment channels, instruments, and other relevant attributes

### Step 3: Scope and Collect
Pull related activity for involved entities within the investigation window (`window_start` to `seed_timestamp`, inclusive).

### Step 4: Assess Benign Explanations (Default Hypothesis)
Attempt to explain observed activity as legitimate first:
- State which evidence supports the benign hypothesis
- Identify what additional data would strengthen this explanation
- Only proceed to Step 5 if benign explanations are insufficient

### Step 5: Test Laundering Hypotheses (If Needed)
If benign explanations fail to account for the evidence:
- Test whether the evidence supports known laundering typologies
- Cite concrete indicators that rule out benign explanations

## Typologies / Heuristics
When assessing patterns, consider these typologies:
- FAN-IN (aggregation): Many sources aggregating to one destination
- FAN-OUT (dispersion): One source dispersing to many destinations
- GATHER-SCATTER / SCATTER-GATHER: Aggregation followed by dispersion (or reverse) over short time windows
- STACK / LAYERING: Multiple hops meant to obscure origin
- CYCLE: Circular fund movement
- BIPARTITE: Structured flows between two distinct groups
- RANDOM: Complex pattern with no discernible structure

## Output Format
Return a single JSON object matching the configured output schema exactly. Populate every field.
Use `pattern_type = "NONE"` when no laundering pattern is supported by evidence in the investigation window.

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

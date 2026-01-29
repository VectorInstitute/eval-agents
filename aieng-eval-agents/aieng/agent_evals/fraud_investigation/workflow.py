"""Anti-Money Laundering workflow."""

import json
import logging
from contextlib import aclosing
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

import google.genai.types
from aieng.agent_evals.fraud_investigation.policy import TriageOutcome, TriageStrategy
from aieng.agent_evals.fraud_investigation.types import AnalystResult, AnalystVerdict
from google.adk.agents.base_agent import BaseAgent, BaseAgentState
from google.adk.events.event import Event
from pydantic import Field
from pydantic.fields import PrivateAttr


if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext

logger = logging.getLogger(__name__)


class _WorkflowStep(str, Enum):
    """Steps in the AML investigation workflow."""

    START = "START"
    ANALYZE = "ANALYZE"
    REPORT = "REPORT"
    DONE = "DONE"


class _WorkflowState(BaseAgentState):
    """State for the AML investigation workflow."""

    input_dict: dict[str, Any]
    step: _WorkflowStep = _WorkflowStep.START
    triage_decision: TriageOutcome | None = None
    analysis: AnalystResult | None = None
    final_report: Optional[str] = None


class AmlInvestigationWorkflow(BaseAgent):
    """An AML investigation workflow coordinating triage, analysis, and reporting."""

    name: str = Field(default="AMLWorkflowManager", description="Coordinates AML investigation and reporting agents.")
    description: str = Field(
        default="Oversees the end-to-end AML investigation process, delegating tasks to specialized agents."
    )

    # PrivateAttr are used here to hide complex sub-agents from Pydantic validation
    _triage: TriageStrategy = PrivateAttr()
    _analyst: BaseAgent = PrivateAttr()
    _reporter: BaseAgent = PrivateAttr()

    def __init__(self, triage_strategy: TriageStrategy, analyst: BaseAgent, reporter: BaseAgent, **kwargs: Any) -> None:
        """Initialize the AML investigation workflow.

        Parameters
        ----------
        triage_strategy : TriageStrategy
            The triage strategy to use for initial alert assessment.
        analyst : BaseAgent
            The agent responsible for in-depth fraud analysis. This agent must
            produce an ``AnalystResult``, which will be passed to the reporter.
        reporter : BaseAgent
            The agent responsible for generating the final SAR report.
        **kwargs : Any
            Additional keyword arguments for the ``BaseAgent`` class.
        """
        super().__init__(**kwargs)
        self._triage = triage_strategy
        self._analyst = analyst
        self._reporter = reporter
        self.sub_agents = [analyst, reporter]

    async def _initialize_state(self, ctx: "InvocationContext") -> _WorkflowState:
        """Initialize workflow state from context."""
        state = self._load_agent_state(ctx, _WorkflowState)

        if state:
            return state

        input_text = ""
        if ctx.user_content and ctx.user_content.parts:
            input_text = ctx.user_content.parts[0].text or ""

        try:
            alert_dict = json.loads(input_text.strip())
        except json.JSONDecodeError:
            logger.warning("Failed to parse alert payload as JSON. Using raw text.")
            alert_dict = {"description": input_text}

        return _WorkflowState(input_dict=alert_dict)

    async def _update_step(
        self, ctx: "InvocationContext", state: _WorkflowState, new_step: _WorkflowStep
    ) -> AsyncGenerator["Event", None]:
        """Update workflow step and persist state."""
        state.step = new_step
        ctx.set_agent_state(self.name, agent_state=state)
        logger.info(ctx)
        yield self._create_agent_state_event(ctx)

    async def _handle_triage(self, ctx: "InvocationContext", state: _WorkflowState) -> AsyncGenerator["Event", None]:
        """Handle triage step."""
        decision = await self._triage.triage(state.input_dict)
        state.triage_decision = decision

        triage_event = _create_text_event(f"Triage decision: {decision.value}", ctx)
        if decision == TriageOutcome.CLOSE:
            state.final_report = "CASE CLOSED: No further action required after triage."
            yield _create_text_event(state.final_report, ctx)

            async for event in self._update_step(ctx, state, _WorkflowStep.DONE):
                yield event
        elif decision == TriageOutcome.ESCALATE:
            triage_event.actions.escalate = True
            yield triage_event
            async for event in self._update_step(ctx, state, _WorkflowStep.DONE):
                yield event
        else:
            triage_event.actions.transfer_to_agent = self._analyst.name
            yield triage_event
            async for event in self._update_step(ctx, state, _WorkflowStep.ANALYZE):
                yield event

    async def _handle_analysis(self, ctx: "InvocationContext", state: _WorkflowState) -> AsyncGenerator["Event", None]:
        """Handle analysis step by delegating to analyst agent."""
        analyst_ctx = ctx.model_copy()
        analyst_ctx.branch = f"{ctx.branch}.{self._analyst.name}" if ctx.branch else self._analyst.name

        async with aclosing(self._analyst.run_async(analyst_ctx)) as agen:
            async for event in agen:
                yield event

                if event.is_final_response() and event.content and event.content.parts:
                    analysis_text = event.content.parts[0].text or ""
                    try:
                        analysis_dict = json.loads(analysis_text.strip())
                        state.analysis = AnalystResult.model_validate(analysis_dict)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse analyst result as JSON.")

                if state.analysis or ctx.should_pause_invocation(event):
                    break

        if state.analysis:
            async for e in self._route_analysis_result(ctx, state):
                yield e
        else:
            async for e in self._update_step(ctx, state, _WorkflowStep.DONE):
                yield e

    async def _route_analysis_result(
        self, ctx: "InvocationContext", state: _WorkflowState
    ) -> AsyncGenerator["Event", None]:
        """Route based on analysis verdict."""
        if not state.analysis:
            return

        verdict = state.analysis.verdict

        if verdict == AnalystVerdict.SUSPICIOUS:
            reporter_msg = f"""\
            # Analyst Findings
            **Analyst Risk Score**: {state.analysis.risk_score}
            **Analyst Verdict**: {state.analysis.verdict.value}
            ## Summary Narrative
            {state.analysis.summary_narrative}
            ## Key Findings
            **Suspicious Activity Type**: {state.analysis.suspicious_activity_type.value}
            ### Suspicious Transactions
            {state.analysis.suspicious_transactions}
            """
            analyst_event = _create_text_event(reporter_msg, ctx)
            analyst_event.actions.transfer_to_agent = self._reporter.name
            yield analyst_event

            async for e in self._update_step(ctx, state, _WorkflowStep.REPORT):
                yield e
        elif verdict == AnalystVerdict.NEEDS_REVIEW:
            state.final_report = (
                f"ESCALATED TO HUMAN: Further review required.\n"
                f"Reason: {state.analysis.summary_narrative}\n"
                f"Evidence: Identified {len(state.analysis.suspicious_transactions)} suspicious transactions."
            )
            analyst_event = _create_text_event(state.final_report, ctx)
            analyst_event.actions.escalate = True
            yield analyst_event

            async for e in self._update_step(ctx, state, _WorkflowStep.DONE):
                yield e
        else:
            state.final_report = (
                f"CASE CLOSED (False Positive).\n"
                f"Reason: {state.analysis.summary_narrative}\n"
                f"Evidence: Investigated {len(state.analysis.suspicious_transactions)} transactions and found no illicit pattern."
            )
            yield _create_text_event(state.final_report, ctx)

            async for e in self._update_step(ctx, state, _WorkflowStep.DONE):
                yield e

    async def _handle_reporting(self, ctx: "InvocationContext", state: _WorkflowState) -> AsyncGenerator["Event", None]:
        """Handle reporting step by delegating to reporter agent."""
        if not state.analysis:
            return

        reporter_ctx = ctx.model_copy()
        reporter_ctx.branch = f"{ctx.branch}.{self._reporter.name}" if ctx.branch else self._reporter.name

        async with aclosing(self._reporter.run_async(reporter_ctx)) as agen:
            async for event in agen:
                yield event

                if event.is_final_response() and event.content and event.content.parts:
                    state.final_report = event.content.parts[0].text
                    break

                if ctx.should_pause_invocation(event):
                    return

        async for e in self._update_step(ctx, state, _WorkflowStep.DONE):
            yield e

    async def _run_async_impl(self, ctx: "InvocationContext") -> AsyncGenerator["Event", None]:
        """Run the AML investigation workflow."""
        state = await self._initialize_state(ctx)

        if state.step == _WorkflowStep.START:
            async for event in self._handle_triage(ctx, state):
                yield event

        if state.step == _WorkflowStep.ANALYZE:
            async for event in self._handle_analysis(ctx, state):
                yield event

        if state.step == _WorkflowStep.REPORT and state.analysis:
            async for event in self._handle_reporting(ctx, state):
                yield event

        if state.step == _WorkflowStep.DONE:
            ctx.set_agent_state(self.name, end_of_agent=True)
            yield self._create_agent_state_event(ctx)


def _create_text_event(text: str, ctx: "InvocationContext") -> Event:
    """Create a text event to be sent back to the user."""
    return Event(
        invocation_id=ctx.invocation_id,
        author=ctx.agent.name,
        content=google.genai.types.Content(role="model", parts=[google.genai.types.Part(text=text)]),
    )

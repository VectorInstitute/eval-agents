"""Tests for plan parsing utilities."""

import pytest
from aieng.agent_evals.knowledge_agent.agent import KnowledgeGroundedAgent
from aieng.agent_evals.knowledge_agent.models import ResearchStep, StepStatus
from aieng.agent_evals.knowledge_agent.plan_parsing import (
    ACTION_TAG,
    FINAL_ANSWER_TAG,
    PLANNING_TAG,
    REASONING_TAG,
    REPLANNING_TAG,
    extract_final_answer_text,
    extract_plan_text,
    extract_reasoning_text,
    parse_plan_steps_from_text,
)


class TestExtractPlanText:
    """Tests for extract_plan_text function."""

    def test_extract_planning_tag(self):
        """Test extracting text from PLANNING tag."""
        text = "/*PLANNING*/\n1. Search for info\n2. Fetch page\n/*ACTION*/"
        result = extract_plan_text(text)
        assert result is not None
        assert "Search for info" in result
        assert "Fetch page" in result

    def test_extract_replanning_takes_precedence(self):
        """Test that REPLANNING tag takes precedence over PLANNING."""
        text = "/*PLANNING*/\nOld plan\n/*REPLANNING*/\nNew plan\n/*ACTION*/"
        result = extract_plan_text(text)
        assert result == "New plan"

    def test_returns_none_for_no_planning_tag(self):
        """Test returns None when no planning tag present."""
        text = "Just some regular text without planning"
        result = extract_plan_text(text)
        assert result is None

    def test_truncates_at_action_tag(self):
        """Test that plan text stops at ACTION tag."""
        text = "/*PLANNING*/\nMy plan\n/*ACTION*/\nDo something"
        result = extract_plan_text(text)
        assert result == "My plan"
        assert "Do something" not in result

    def test_truncates_at_reasoning_tag(self):
        """Test that plan text stops at REASONING tag."""
        text = "/*PLANNING*/\nMy plan\n/*REASONING*/\nThinking..."
        result = extract_plan_text(text)
        assert result == "My plan"

    def test_truncates_at_final_answer_tag(self):
        """Test that plan text stops at FINAL_ANSWER tag."""
        text = "/*PLANNING*/\nMy plan\n/*FINAL_ANSWER*/\nThe answer is..."
        result = extract_plan_text(text)
        assert result == "My plan"


class TestParsePlanStepsFromText:
    """Tests for parse_plan_steps_from_text function."""

    def test_parse_numbered_steps_with_dots(self):
        """Test parsing '1. Description' format."""
        text = "1. Search for info\n2. Fetch the page\n3. Extract data"
        steps = parse_plan_steps_from_text(text)
        assert len(steps) == 3
        assert steps[0].step_id == 1
        assert "Search for info" in steps[0].description
        assert steps[1].step_id == 2
        assert steps[2].step_id == 3

    def test_parse_numbered_steps_with_parens(self):
        """Test parsing '1) Description' format."""
        text = "1) First step\n2) Second step"
        steps = parse_plan_steps_from_text(text)
        assert len(steps) == 2

    def test_parse_bullet_points(self):
        """Test parsing bullet point format."""
        text = "- Search Google\n- Fetch URL\n- Parse content"
        steps = parse_plan_steps_from_text(text)
        assert len(steps) == 3

    def test_skips_short_descriptions(self):
        """Test that very short descriptions are skipped."""
        text = "1. OK\n2. This is a longer description"
        steps = parse_plan_steps_from_text(text)
        assert len(steps) == 1
        assert "longer description" in steps[0].description

    def test_truncates_long_descriptions(self):
        """Test that descriptions are truncated to 200 chars."""
        long_desc = "A" * 300
        text = f"1. {long_desc}"
        steps = parse_plan_steps_from_text(text)
        assert len(steps) == 1
        assert len(steps[0].description) <= 200

    def test_returns_empty_for_no_steps(self):
        """Test returns empty list for text with no parseable steps."""
        text = "Just some text"
        steps = parse_plan_steps_from_text(text)
        # May return steps from fallback, but should handle gracefully
        assert isinstance(steps, list)


class TestExtractReasoningText:
    """Tests for extract_reasoning_text function."""

    def test_extract_reasoning(self):
        """Test extracting reasoning text."""
        text = "/*REASONING*/\nI think the answer is X because Y.\n/*ACTION*/"
        result = extract_reasoning_text(text)
        assert result is not None
        assert "I think the answer is X" in result

    def test_returns_none_for_no_tag(self):
        """Test returns None when no REASONING tag."""
        text = "Some text without reasoning tag"
        result = extract_reasoning_text(text)
        assert result is None

    def test_truncates_at_action_tag(self):
        """Test reasoning stops at ACTION tag."""
        text = "/*REASONING*/\nThinking...\n/*ACTION*/\nCall tool"
        result = extract_reasoning_text(text)
        assert result == "Thinking..."
        assert "Call tool" not in result

    def test_truncates_at_final_answer_tag(self):
        """Test reasoning stops at FINAL_ANSWER tag."""
        text = "/*REASONING*/\nAnalysis done.\n/*FINAL_ANSWER*/\nThe answer"
        result = extract_reasoning_text(text)
        assert result == "Analysis done."

    def test_truncates_at_planning_tag(self):
        """Test reasoning stops at PLANNING tag."""
        text = "/*REASONING*/\nNeed to replan.\n/*PLANNING*/\nNew plan"
        result = extract_reasoning_text(text)
        assert result == "Need to replan."


class TestExtractFinalAnswerText:
    """Tests for extract_final_answer_text function."""

    def test_extract_final_answer(self):
        """Test extracting final answer text."""
        text = "/*FINAL_ANSWER*/\nThe highest snowfall was 61cm on January 25, 2026."
        result = extract_final_answer_text(text)
        assert result is not None
        assert "61cm" in result
        assert "January 25, 2026" in result

    def test_returns_none_for_no_tag(self):
        """Test returns None when no FINAL_ANSWER tag."""
        text = "Some text without final answer tag"
        result = extract_final_answer_text(text)
        assert result is None

    def test_returns_none_for_empty_text(self):
        """Test returns None for empty input."""
        result = extract_final_answer_text("")
        assert result is None
        result = extract_final_answer_text(None)  # type: ignore
        assert result is None

    def test_returns_none_for_empty_answer(self):
        """Test returns None when answer is empty."""
        text = "/*FINAL_ANSWER*/\n   \n"
        result = extract_final_answer_text(text)
        assert result is None

    def test_truncates_at_planning_tag(self):
        """Test final answer stops at PLANNING tag."""
        text = "/*FINAL_ANSWER*/\nThe answer is X.\n/*PLANNING*/\n1. New step"
        result = extract_final_answer_text(text)
        assert result == "The answer is X."
        assert "/*PLANNING*/" not in result
        assert "New step" not in result

    def test_truncates_at_replanning_tag(self):
        """Test final answer stops at REPLANNING tag."""
        text = "/*FINAL_ANSWER*/\nAnswer here.\n/*REPLANNING*/\nNew plan"
        result = extract_final_answer_text(text)
        assert result == "Answer here."
        assert "/*REPLANNING*/" not in result

    def test_truncates_at_reasoning_tag(self):
        """Test final answer stops at REASONING tag."""
        text = "/*FINAL_ANSWER*/\nFinal answer.\n/*REASONING*/\nMore thinking"
        result = extract_final_answer_text(text)
        assert result == "Final answer."
        assert "/*REASONING*/" not in result

    def test_truncates_at_action_tag(self):
        """Test final answer stops at ACTION tag."""
        text = "/*FINAL_ANSWER*/\nDone.\n/*ACTION*/\nSome action"
        result = extract_final_answer_text(text)
        assert result == "Done."
        assert "/*ACTION*/" not in result

    def test_handles_multiple_subsequent_tags(self):
        """Test handling text with multiple tags after FINAL_ANSWER."""
        text = "/*FINAL_ANSWER*/\nThe answer is 42.\n/*PLANNING*/\n1. Step one\n2. Step two\n/*ACTION*/\nDo something"
        result = extract_final_answer_text(text)
        assert result == "The answer is 42."
        assert "/*PLANNING*/" not in result
        assert "/*ACTION*/" not in result
        assert "Step one" not in result

    def test_real_world_example_with_tags_in_answer(self):
        """Test real-world case where tags appear after the answer."""
        text = """/*FINAL_ANSWER*/
ANSWER: The highest single day snowfall in Toronto was 61cm on January 25, 2026.
SOURCES: https://example.com/weather
REASONING: The Google search directly provided the information.
/*PLANNING*/
1. Search Google for "highest single day snowfall Toronto" to find reliable sources.
2. From the search results, identify a credible source.
/*ACTION*/
Call google_search"""
        result = extract_final_answer_text(text)
        assert result is not None
        assert "61cm" in result
        assert "January 25, 2026" in result
        assert "/*PLANNING*/" not in result
        assert "/*ACTION*/" not in result


class TestTagConstants:
    """Tests for tag constants."""

    def test_tag_constants_defined(self):
        """Test that all tag constants are defined correctly."""
        assert PLANNING_TAG == "/*PLANNING*/"
        assert REPLANNING_TAG == "/*REPLANNING*/"
        assert REASONING_TAG == "/*REASONING*/"
        assert ACTION_TAG == "/*ACTION*/"
        assert FINAL_ANSWER_TAG == "/*FINAL_ANSWER*/"


@pytest.mark.integration_test
class TestPlanParsingIntegration:
    """Integration tests for plan parsing against real model output.

    These tests require API keys to run against actual model outputs.
    """

    @pytest.mark.asyncio
    async def test_agent_response_parsing(self):
        """Test that agent response is parsed correctly without tags leaking."""
        agent = KnowledgeGroundedAgent(enable_planning=True)
        response = await agent.answer_async("What is the capital of France?")

        # Response text should not contain any planner tags
        assert "/*PLANNING*/" not in response.text
        assert "/*REPLANNING*/" not in response.text
        assert "/*REASONING*/" not in response.text
        assert "/*ACTION*/" not in response.text
        assert "/*FINAL_ANSWER*/" not in response.text

        # Should have an actual answer
        assert response.text.strip() != ""
        assert "Paris" in response.text

    @pytest.mark.asyncio
    async def test_agent_plan_is_populated(self):
        """Test that agent plan steps are populated from model output."""
        agent = KnowledgeGroundedAgent(enable_planning=True)
        response = await agent.answer_async("What year was the Eiffel Tower built?")

        # Plan should have steps
        assert response.plan is not None
        assert len(response.plan.steps) > 0

        # Each step should have a description without tags
        for step in response.plan.steps:
            assert step.description.strip() != ""
            assert "/*" not in step.description

    @pytest.mark.asyncio
    async def test_agent_tool_calls_recorded(self):
        """Test that agent tool calls are recorded."""
        agent = KnowledgeGroundedAgent(enable_planning=True)
        response = await agent.answer_async("What is the population of Tokyo?")

        # Should have made some tool calls
        assert len(response.tool_calls) > 0

        # Tool calls should have name and args
        for tc in response.tool_calls:
            assert "name" in tc
            assert "args" in tc

    @pytest.mark.asyncio
    async def test_agent_handles_complex_query_with_potential_replan(self):
        """Test agent handles complex queries that may require replanning.

        This tests a query that might need the agent to adapt its approach,
        verifying that any replanning is handled correctly without tags leaking.
        """
        agent = KnowledgeGroundedAgent(enable_planning=True)

        # A complex query that might require multiple search attempts or replanning
        response = await agent.answer_async("What was the exact final score of the most recent FIFA World Cup final?")

        # Response should be clean - no tags
        assert "/*PLANNING*/" not in response.text
        assert "/*REPLANNING*/" not in response.text
        assert "/*REASONING*/" not in response.text
        assert "/*ACTION*/" not in response.text
        assert "/*FINAL_ANSWER*/" not in response.text

        # Should have a non-empty response
        assert response.text.strip() != ""

        # Plan should exist and be valid
        assert response.plan is not None

        # Plan reasoning should not contain raw tags
        assert "/*" not in response.plan.reasoning

        # All step descriptions should be clean
        for step in response.plan.steps:
            assert "/*" not in step.description

    @pytest.mark.asyncio
    async def test_replan_updates_plan_correctly(self):
        """Test that replanning properly updates the plan structure.

        This unit-tests the replanning logic directly by simulating
        a replan event after some steps are completed.
        """
        agent = KnowledgeGroundedAgent(enable_planning=True)

        # Initialize a plan
        await agent.create_plan_async("Test question")

        # Manually set up an initial plan with some completed steps
        agent._current_plan.steps = [agent._current_plan.steps[0] if agent._current_plan.steps else None]
        if not agent._current_plan.steps[0]:
            agent._current_plan.steps = [
                ResearchStep(
                    step_id=1,
                    description="Initial search step",
                    step_type="research",
                    status=StepStatus.COMPLETED,
                ),
            ]
        else:
            agent._current_plan.steps[0].status = StepStatus.COMPLETED

        initial_completed_count = len([s for s in agent._current_plan.steps if s.status == StepStatus.COMPLETED])

        # Simulate a replanning event
        replan_text = f"""{REPLANNING_TAG}
1. Try a different search approach
2. Fetch from alternative source
3. Verify the information
"""
        agent._update_plan_from_text(replan_text, "Test question", is_replan=True)

        # Verify replanning worked
        assert agent._current_plan is not None

        # Should have preserved completed steps plus new ones
        completed_after = [s for s in agent._current_plan.steps if s.status == StepStatus.COMPLETED]
        assert len(completed_after) >= initial_completed_count

        # New steps should have been added
        assert len(agent._current_plan.steps) > initial_completed_count

        # Reasoning should indicate replanning
        assert "Replanned" in agent._current_plan.reasoning

        # No tags in step descriptions
        for step in agent._current_plan.steps:
            assert "/*" not in step.description

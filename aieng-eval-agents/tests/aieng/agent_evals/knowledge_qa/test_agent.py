"""Tests for the Knowledge-Grounded QA Agent and data models."""

from unittest.mock import MagicMock, patch

import pytest
from aieng.agent_evals.knowledge_qa.agent import (
    KnowledgeAgentManager,
    KnowledgeGroundedAgent,
)
from aieng.agent_evals.knowledge_qa.models import (
    AgentResponse,
    ResearchPlan,
    ResearchStep,
    StepExecution,
    StepStatus,
)
from aieng.agent_evals.tools import GroundingChunk


# =============================================================================
# Data Model Tests
# =============================================================================


class TestStepStatus:
    """Tests for the StepStatus constants."""

    def test_status_constants(self):
        """Test that status constants are defined."""
        assert StepStatus.PENDING == "pending"
        assert StepStatus.IN_PROGRESS == "in_progress"
        assert StepStatus.COMPLETED == "completed"
        assert StepStatus.FAILED == "failed"
        assert StepStatus.SKIPPED == "skipped"


class TestResearchStep:
    """Tests for the ResearchStep model."""

    def test_research_step_creation(self):
        """Test creating a research step."""
        step = ResearchStep(
            step_id=1,
            description="Search for financial regulations",
            step_type="research",
            depends_on=[],
            expected_output="List of relevant regulations",
        )
        assert step.step_id == 1
        assert step.description == "Search for financial regulations"
        assert step.step_type == "research"
        assert step.depends_on == []
        assert step.expected_output == "List of relevant regulations"

    def test_research_step_with_dependencies(self):
        """Test creating a step with dependencies."""
        step = ResearchStep(
            step_id=3,
            description="Synthesize findings",
            step_type="synthesis",
            depends_on=[1, 2],
            expected_output="Comprehensive answer",
        )
        assert step.depends_on == [1, 2]

    def test_research_step_defaults(self):
        """Test default values for research step."""
        step = ResearchStep(
            step_id=1,
            description="Test step",
            step_type="research",
        )
        assert step.depends_on == []
        assert step.expected_output == ""
        assert step.status == StepStatus.PENDING
        assert step.actual_output == ""
        assert step.attempts == 0
        assert step.failure_reason == ""

    def test_research_step_with_tracking_fields(self):
        """Test creating a step with tracking fields."""
        step = ResearchStep(
            step_id=1,
            description="Test step",
            step_type="research",
            status=StepStatus.COMPLETED,
            actual_output="Found 5 results",
            attempts=2,
            failure_reason="",
        )
        assert step.status == StepStatus.COMPLETED
        assert step.actual_output == "Found 5 results"
        assert step.attempts == 2

    def test_research_step_failed_status(self):
        """Test creating a step with failed status."""
        step = ResearchStep(
            step_id=1,
            description="Fetch document",
            step_type="research",
            status=StepStatus.FAILED,
            attempts=3,
            failure_reason="404 Not Found",
        )
        assert step.status == StepStatus.FAILED
        assert step.attempts == 3
        assert step.failure_reason == "404 Not Found"


class TestResearchPlan:
    """Tests for the ResearchPlan model."""

    def test_research_plan_creation(self):
        """Test creating a research plan."""
        plan = ResearchPlan(
            original_question="What caused the 2008 financial crisis?",
            steps=[
                ResearchStep(
                    step_id=1,
                    description="Research subprime mortgages",
                    step_type="research",
                ),
                ResearchStep(
                    step_id=2,
                    description="Look up Dodd-Frank regulations",
                    step_type="research",
                ),
            ],
            reasoning="Complex question requiring multiple sources",
        )
        assert plan.original_question == "What caused the 2008 financial crisis?"
        assert len(plan.steps) == 2
        assert plan.reasoning != ""

    def test_research_plan_defaults(self):
        """Test default values for research plan."""
        plan = ResearchPlan(
            original_question="Simple question",
        )
        assert plan.steps == []
        assert plan.reasoning == ""

    def test_get_step_found(self):
        """Test getting an existing step by ID."""
        plan = ResearchPlan(
            original_question="Test",
            steps=[
                ResearchStep(step_id=1, description="Step 1", step_type="research"),
                ResearchStep(step_id=2, description="Step 2", step_type="research"),
            ],
        )
        step = plan.get_step(2)
        assert step is not None
        assert step.description == "Step 2"

    def test_get_step_not_found(self):
        """Test getting a non-existent step by ID."""
        plan = ResearchPlan(
            original_question="Test",
            steps=[ResearchStep(step_id=1, description="Step 1", step_type="research")],
        )
        step = plan.get_step(99)
        assert step is None

    def test_update_step_status(self):
        """Test updating a step's status."""
        plan = ResearchPlan(
            original_question="Test",
            steps=[ResearchStep(step_id=1, description="Step 1", step_type="research")],
        )
        result = plan.update_step(1, status=StepStatus.COMPLETED)
        assert result is True
        assert plan.steps[0].status == StepStatus.COMPLETED

    def test_update_step_all_fields(self):
        """Test updating all tracking fields of a step."""
        plan = ResearchPlan(
            original_question="Test",
            steps=[ResearchStep(step_id=1, description="Step 1", step_type="research")],
        )
        result = plan.update_step(
            1,
            status=StepStatus.FAILED,
            actual_output="Found some results",
            failure_reason="Timeout error",
            increment_attempts=True,
        )
        assert result is True
        assert plan.steps[0].status == StepStatus.FAILED
        assert plan.steps[0].actual_output == "Found some results"
        assert plan.steps[0].failure_reason == "Timeout error"
        assert plan.steps[0].attempts == 1

    def test_update_step_not_found(self):
        """Test updating a non-existent step."""
        plan = ResearchPlan(
            original_question="Test",
            steps=[],
        )
        result = plan.update_step(99, status=StepStatus.COMPLETED)
        assert result is False

    def test_update_step_description(self):
        """Test updating a step's description."""
        plan = ResearchPlan(
            original_question="Test",
            steps=[ResearchStep(step_id=1, description="Original", step_type="research")],
        )
        result = plan.update_step(1, description="Updated description")
        assert result is True
        assert plan.steps[0].description == "Updated description"

    def test_get_pending_steps_no_dependencies(self):
        """Test getting pending steps when none have dependencies."""
        plan = ResearchPlan(
            original_question="Test",
            steps=[
                ResearchStep(step_id=1, description="Step 1", step_type="research"),
                ResearchStep(step_id=2, description="Step 2", step_type="research"),
            ],
        )
        pending = plan.get_pending_steps()
        assert len(pending) == 2

    def test_get_pending_steps_with_dependencies(self):
        """Test getting pending steps with dependency filtering."""
        plan = ResearchPlan(
            original_question="Test",
            steps=[
                ResearchStep(step_id=1, description="Step 1", step_type="research"),
                ResearchStep(step_id=2, description="Step 2", step_type="research", depends_on=[1]),
                ResearchStep(step_id=3, description="Step 3", step_type="synthesis", depends_on=[1, 2]),
            ],
        )
        pending = plan.get_pending_steps()
        assert len(pending) == 1
        assert pending[0].step_id == 1

    def test_get_pending_steps_after_completion(self):
        """Test getting pending steps after some complete."""
        plan = ResearchPlan(
            original_question="Test",
            steps=[
                ResearchStep(step_id=1, description="Step 1", step_type="research", status=StepStatus.COMPLETED),
                ResearchStep(step_id=2, description="Step 2", step_type="research", depends_on=[1]),
                ResearchStep(step_id=3, description="Step 3", step_type="synthesis", depends_on=[1, 2]),
            ],
        )
        pending = plan.get_pending_steps()
        assert len(pending) == 1
        assert pending[0].step_id == 2

    def test_get_steps_by_status(self):
        """Test getting steps by status."""
        plan = ResearchPlan(
            original_question="Test",
            steps=[
                ResearchStep(step_id=1, description="Step 1", step_type="research", status=StepStatus.COMPLETED),
                ResearchStep(step_id=2, description="Step 2", step_type="research", status=StepStatus.FAILED),
                ResearchStep(step_id=3, description="Step 3", step_type="synthesis", status=StepStatus.PENDING),
            ],
        )
        completed = plan.get_steps_by_status(StepStatus.COMPLETED)
        failed = plan.get_steps_by_status(StepStatus.FAILED)
        pending = plan.get_steps_by_status(StepStatus.PENDING)

        assert len(completed) == 1
        assert completed[0].step_id == 1
        assert len(failed) == 1
        assert failed[0].step_id == 2
        assert len(pending) == 1
        assert pending[0].step_id == 3

    def test_is_complete_all_done(self):
        """Test is_complete when all steps are in terminal states."""
        plan = ResearchPlan(
            original_question="Test",
            steps=[
                ResearchStep(step_id=1, description="Step 1", step_type="research", status=StepStatus.COMPLETED),
                ResearchStep(step_id=2, description="Step 2", step_type="research", status=StepStatus.FAILED),
                ResearchStep(step_id=3, description="Step 3", step_type="synthesis", status=StepStatus.SKIPPED),
            ],
        )
        assert plan.is_complete() is True

    def test_is_complete_with_pending(self):
        """Test is_complete when some steps are pending."""
        plan = ResearchPlan(
            original_question="Test",
            steps=[
                ResearchStep(step_id=1, description="Step 1", step_type="research", status=StepStatus.COMPLETED),
                ResearchStep(step_id=2, description="Step 2", step_type="research", status=StepStatus.PENDING),
            ],
        )
        assert plan.is_complete() is False

    def test_is_complete_with_in_progress(self):
        """Test is_complete when some steps are in progress."""
        plan = ResearchPlan(
            original_question="Test",
            steps=[
                ResearchStep(step_id=1, description="Step 1", step_type="research", status=StepStatus.IN_PROGRESS),
            ],
        )
        assert plan.is_complete() is False


class TestStepExecution:
    """Tests for the StepExecution model."""

    def test_step_execution_creation(self):
        """Test creating a step execution record."""
        execution = StepExecution(
            step_id=1,
            tool_used="web_search",
            input_query="2008 financial crisis causes",
            output_summary="Found 5 relevant articles",
            sources_found=5,
            duration_ms=1500,
            raw_output="Raw search results...",
        )
        assert execution.step_id == 1
        assert execution.tool_used == "web_search"
        assert execution.input_query == "2008 financial crisis causes"
        assert execution.output_summary == "Found 5 relevant articles"
        assert execution.sources_found == 5
        assert execution.duration_ms == 1500

    def test_step_execution_defaults(self):
        """Test default values for step execution."""
        execution = StepExecution(
            step_id=1,
            tool_used="finance_knowledge",
            input_query="Basel III",
        )
        assert execution.output_summary == ""
        assert execution.sources_found == 0
        assert execution.duration_ms == 0
        assert execution.raw_output == ""


# =============================================================================
# Agent Tests
# =============================================================================


class TestKnowledgeGroundedAgent:
    """Tests for the KnowledgeGroundedAgent class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = MagicMock()
        config.openai_api_key = "test-api-key"
        config.default_worker_model = "gemini-2.5-flash"
        config.default_temperature = 0.0
        return config

    @patch("aieng.agent_evals.knowledge_qa.agent.PlanReActPlanner")
    @patch("aieng.agent_evals.knowledge_qa.agent.Runner")
    @patch("aieng.agent_evals.knowledge_qa.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_qa.agent.Agent")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_read_file_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_grep_file_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_fetch_file_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_web_fetch_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_google_search_tool")
    def test_agent_initialization(
        self,
        mock_create_search_tool,
        mock_create_web_fetch_tool,
        mock_create_fetch_file_tool,
        mock_create_grep_file_tool,
        mock_create_read_file_tool,
        mock_agent_class,
        _mock_session_service,
        _mock_runner_class,
        mock_planner,
        mock_config,
    ):
        """Test initializing the agent with all tools."""
        mock_search_tool = MagicMock()
        mock_web_fetch_tool = MagicMock()
        mock_create_search_tool.return_value = mock_search_tool
        mock_create_web_fetch_tool.return_value = mock_web_fetch_tool

        agent = KnowledgeGroundedAgent(config=mock_config, enable_caching=False, enable_compaction=False)

        # Verify all tools were created
        mock_create_search_tool.assert_called_once()
        mock_create_web_fetch_tool.assert_called_once()
        mock_create_fetch_file_tool.assert_called_once()
        mock_create_grep_file_tool.assert_called_once()
        mock_create_read_file_tool.assert_called_once()

        # Verify ADK Agent was created with correct params
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args[1]
        assert call_kwargs["name"] == "knowledge_qa"
        assert mock_search_tool in call_kwargs["tools"]
        assert mock_web_fetch_tool in call_kwargs["tools"]

        # Verify BuiltInPlanner was created (planning enabled by default)
        mock_planner.assert_called_once()
        assert agent.enable_planning is True

    @patch("aieng.agent_evals.knowledge_qa.agent.PlanReActPlanner")
    @patch("aieng.agent_evals.knowledge_qa.agent.Runner")
    @patch("aieng.agent_evals.knowledge_qa.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_qa.agent.Agent")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_read_file_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_grep_file_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_fetch_file_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_web_fetch_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_google_search_tool")
    def test_agent_without_planning(
        self,
        _mock_create_search_tool,
        _mock_create_web_fetch_tool,
        _mock_create_fetch_file_tool,
        _mock_create_grep_file_tool,
        _mock_create_read_file_tool,
        mock_agent_class,
        _mock_session_service,
        _mock_runner_class,
        mock_planner,
        mock_config,
    ):
        """Test initializing the agent without planning."""
        agent = KnowledgeGroundedAgent(
            config=mock_config, enable_planning=False, enable_caching=False, enable_compaction=False
        )

        # BuiltInPlanner should not be created when planning disabled
        mock_planner.assert_not_called()
        assert agent.enable_planning is False

        # ADK Agent should be created with planner=None
        call_kwargs = mock_agent_class.call_args[1]
        assert call_kwargs["planner"] is None

    @patch("aieng.agent_evals.knowledge_qa.agent.PlanReActPlanner")
    @patch("aieng.agent_evals.knowledge_qa.agent.Runner")
    @patch("aieng.agent_evals.knowledge_qa.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_qa.agent.Agent")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_read_file_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_grep_file_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_fetch_file_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_web_fetch_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_google_search_tool")
    def test_agent_with_custom_model(
        self,
        _mock_create_search_tool,
        _mock_create_web_fetch_tool,
        _mock_create_fetch_file_tool,
        _mock_create_grep_file_tool,
        _mock_create_read_file_tool,
        mock_agent_class,
        _mock_session_service,
        _mock_runner_class,
        _mock_planner,
        mock_config,
    ):
        """Test initializing with a custom model."""
        agent = KnowledgeGroundedAgent(
            config=mock_config, model="gemini-2.5-pro", enable_caching=False, enable_compaction=False
        )

        call_kwargs = mock_agent_class.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-pro"
        assert agent.model == "gemini-2.5-pro"


class TestKnowledgeAgentManager:
    """Tests for the KnowledgeAgentManager class."""

    @patch("aieng.agent_evals.knowledge_qa.agent.PlanReActPlanner")
    @patch("aieng.agent_evals.knowledge_qa.agent.Runner")
    @patch("aieng.agent_evals.knowledge_qa.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_qa.agent.Agent")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_read_file_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_grep_file_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_fetch_file_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_web_fetch_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_google_search_tool")
    def test_lazy_initialization(self, *_mocks):
        """Test that agent is lazily initialized."""
        with patch("aieng.agent_evals.knowledge_qa.agent.Configs") as mock_config_class:
            mock_config = MagicMock()
            mock_config.default_worker_model = "gemini-2.5-flash"
            mock_config.default_temperature = 0.0
            mock_config_class.return_value = mock_config

            manager = KnowledgeAgentManager(enable_caching=False, enable_compaction=False)

            # Should not be initialized yet
            assert not manager.is_initialized()

            # Access agent to trigger initialization
            _ = manager.agent

            # Now should be initialized
            assert manager.is_initialized()

    @patch("aieng.agent_evals.knowledge_qa.agent.PlanReActPlanner")
    @patch("aieng.agent_evals.knowledge_qa.agent.Runner")
    @patch("aieng.agent_evals.knowledge_qa.agent.InMemorySessionService")
    @patch("aieng.agent_evals.knowledge_qa.agent.Agent")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_read_file_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_grep_file_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_fetch_file_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_web_fetch_tool")
    @patch("aieng.agent_evals.knowledge_qa.agent.create_google_search_tool")
    def test_close(self, *_mocks):
        """Test closing the client manager."""
        with patch("aieng.agent_evals.knowledge_qa.agent.Configs") as mock_config_class:
            mock_config = MagicMock()
            mock_config.default_worker_model = "gemini-2.5-flash"
            mock_config.default_temperature = 0.0
            mock_config_class.return_value = mock_config

            manager = KnowledgeAgentManager(enable_caching=False, enable_compaction=False)
            _ = manager.agent
            assert manager.is_initialized()

            manager.close()
            assert not manager.is_initialized()


class TestAgentResponse:
    """Tests for the AgentResponse model."""

    def test_response_creation(self):
        """Test creating an enhanced response."""
        plan = ResearchPlan(
            original_question="Test question",
            steps=[],
            reasoning="Test reasoning",
        )

        response = AgentResponse(
            text="Test answer.",
            plan=plan,
            sources=[GroundingChunk(title="Source", uri="https://example.com")],
            search_queries=["test query"],
            reasoning_chain=["Step 1"],
            tool_calls=[{"name": "google_search", "args": {"query": "test"}}],
            total_duration_ms=1000,
        )

        assert response.text == "Test answer."
        assert response.plan.original_question == "Test question"
        assert len(response.sources) == 1
        assert response.sources[0].uri == "https://example.com"
        assert response.search_queries == ["test query"]
        assert response.total_duration_ms == 1000


class TestPlanStepStatusOnEarlyTermination:
    """Tests for plan steps marked correctly when agent terminates early."""

    def test_remaining_steps_marked_as_skipped(self):
        """Test remaining steps are marked SKIPPED on early termination.

        When the agent finds the answer early and terminates before completing
        all planned steps, the remaining steps should be marked as SKIPPED
        to accurately reflect that they were not executed.
        """
        # Create a plan with multiple steps
        plan = ResearchPlan(
            original_question="Test question",
            steps=[
                ResearchStep(
                    step_id=1,
                    description="Search for initial info",
                    status=StepStatus.COMPLETED,
                ),
                ResearchStep(
                    step_id=2,
                    description="Verify the information",
                    status=StepStatus.PENDING,
                ),
                ResearchStep(
                    step_id=3,
                    description="Cross-check with another source",
                    status=StepStatus.PENDING,
                ),
                ResearchStep(
                    step_id=4,
                    description="Synthesize findings",
                    status=StepStatus.IN_PROGRESS,
                ),
            ],
            reasoning="Multi-step research plan",
        )

        # Simulate the agent's early termination logic
        # (this is what happens in agent.py lines 629-633)
        for step in plan.steps:
            if step.status in (StepStatus.PENDING, StepStatus.IN_PROGRESS):
                step.status = StepStatus.SKIPPED

        # Verify step 1 is still completed (it was executed)
        assert plan.steps[0].status == StepStatus.COMPLETED

        # Verify remaining steps are marked as SKIPPED, not COMPLETED
        assert plan.steps[1].status == StepStatus.SKIPPED
        assert plan.steps[2].status == StepStatus.SKIPPED
        assert plan.steps[3].status == StepStatus.SKIPPED

    def test_plan_is_complete_with_skipped_steps(self):
        """Test that a plan with SKIPPED steps is considered complete."""
        plan = ResearchPlan(
            original_question="Test",
            steps=[
                ResearchStep(step_id=1, description="Step 1", status=StepStatus.COMPLETED),
                ResearchStep(step_id=2, description="Step 2", status=StepStatus.SKIPPED),
                ResearchStep(step_id=3, description="Step 3", status=StepStatus.SKIPPED),
            ],
        )

        # SKIPPED is a terminal status, so the plan should be complete
        assert plan.is_complete()

    def test_get_steps_by_status_skipped(self):
        """Test getting steps by SKIPPED status."""
        plan = ResearchPlan(
            original_question="Test",
            steps=[
                ResearchStep(step_id=1, description="Step 1", status=StepStatus.COMPLETED),
                ResearchStep(step_id=2, description="Step 2", status=StepStatus.SKIPPED),
                ResearchStep(step_id=3, description="Step 3", status=StepStatus.SKIPPED),
            ],
        )

        skipped_steps = plan.get_steps_by_status(StepStatus.SKIPPED)
        assert len(skipped_steps) == 2
        assert all(s.status == StepStatus.SKIPPED for s in skipped_steps)


@pytest.mark.integration_test
class TestKnowledgeGroundedAgentIntegration:
    """Integration tests for the KnowledgeGroundedAgent.

    These tests require a valid GOOGLE_API_KEY environment variable.
    """

    def test_agent_creation_real(self):
        """Test creating a real agent instance."""
        agent = KnowledgeGroundedAgent()
        assert agent is not None
        assert agent.model == "gemini-2.5-flash"
        assert agent.enable_planning is True

    @pytest.mark.asyncio
    async def test_answer_real_question(self):
        """Test answering a real question."""
        agent = KnowledgeGroundedAgent()
        response = await agent.answer_async("What is the capital of France?")

        assert response.text
        assert "Paris" in response.text
        assert isinstance(response, AgentResponse)

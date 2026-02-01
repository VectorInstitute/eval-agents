"""Tests for the research planner module."""

import json
from unittest.mock import MagicMock, patch

import pytest
from aieng.agent_evals.knowledge_agent.planner import (
    PlanReflection,
    ResearchPlan,
    ResearchPlanner,
    ResearchStep,
    StepExecution,
    StepStatus,
    StepUpdate,
    _parse_new_steps_response,
    _parse_plan_response,
)


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
        # New dynamic tracking defaults
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


class TestStepUpdate:
    """Tests for the StepUpdate model."""

    def test_step_update_creation(self):
        """Test creating a step update."""
        update = StepUpdate(
            step_id=1,
            new_description="Updated search query",
            new_expected_output="More specific results",
        )
        assert update.step_id == 1
        assert update.new_description == "Updated search query"
        assert update.new_expected_output == "More specific results"

    def test_step_update_defaults(self):
        """Test default values for step update."""
        update = StepUpdate(
            step_id=1,
            new_description="Updated description",
        )
        assert update.new_expected_output == ""


class TestPlanReflection:
    """Tests for the PlanReflection model."""

    def test_plan_reflection_creation(self):
        """Test creating a plan reflection with all fields."""
        reflection = PlanReflection(
            can_answer_now=True,
            key_findings="Found GDP data for 2023",
            reasoning="Have all necessary information",
            steps_to_update=[StepUpdate(step_id=2, new_description="Updated step")],
            steps_to_remove=[3],
            steps_to_add=[ResearchStep(step_id=4, description="New step", step_type="research")],
        )
        assert reflection.can_answer_now is True
        assert reflection.key_findings == "Found GDP data for 2023"
        assert reflection.reasoning == "Have all necessary information"
        assert len(reflection.steps_to_update) == 1
        assert reflection.steps_to_update[0].step_id == 2
        assert reflection.steps_to_remove == [3]
        assert len(reflection.steps_to_add) == 1

    def test_plan_reflection_defaults(self):
        """Test default values for plan reflection."""
        reflection = PlanReflection()
        assert reflection.can_answer_now is False
        assert reflection.key_findings == ""
        assert reflection.reasoning == ""
        assert reflection.steps_to_update == []
        assert reflection.steps_to_remove == []
        assert reflection.steps_to_add == []

    def test_plan_reflection_with_only_reasoning(self):
        """Test creating reflection with only reasoning."""
        reflection = PlanReflection(reasoning="Need more information")
        assert reflection.can_answer_now is False
        assert reflection.reasoning == "Need more information"


class TestResearchPlan:
    """Tests for the ResearchPlan model."""

    def test_research_plan_creation(self):
        """Test creating a research plan."""
        plan = ResearchPlan(
            original_question="What caused the 2008 financial crisis?",
            complexity_assessment="complex",
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
        assert plan.complexity_assessment == "complex"
        assert len(plan.steps) == 2
        assert plan.reasoning != ""

    def test_research_plan_defaults(self):
        """Test default values for research plan."""
        plan = ResearchPlan(
            original_question="Simple question",
            complexity_assessment="simple",
        )
        assert plan.steps == []
        assert plan.reasoning == ""

    def test_get_step_found(self):
        """Test getting an existing step by ID."""
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="simple",
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
            complexity_assessment="simple",
            steps=[ResearchStep(step_id=1, description="Step 1", step_type="research")],
        )
        step = plan.get_step(99)
        assert step is None

    def test_update_step_status(self):
        """Test updating a step's status."""
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="simple",
            steps=[ResearchStep(step_id=1, description="Step 1", step_type="research")],
        )
        result = plan.update_step(1, status=StepStatus.COMPLETED)
        assert result is True
        assert plan.steps[0].status == StepStatus.COMPLETED

    def test_update_step_all_fields(self):
        """Test updating all tracking fields of a step."""
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="simple",
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

    def test_update_step_increment_attempts_multiple(self):
        """Test incrementing attempts multiple times."""
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="simple",
            steps=[ResearchStep(step_id=1, description="Step 1", step_type="research")],
        )
        plan.update_step(1, increment_attempts=True)
        plan.update_step(1, increment_attempts=True)
        plan.update_step(1, increment_attempts=True)
        assert plan.steps[0].attempts == 3

    def test_update_step_not_found(self):
        """Test updating a non-existent step."""
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="simple",
            steps=[],
        )
        result = plan.update_step(99, status=StepStatus.COMPLETED)
        assert result is False

    def test_update_step_description(self):
        """Test updating a step's description."""
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="simple",
            steps=[ResearchStep(step_id=1, description="Original", step_type="research")],
        )
        result = plan.update_step(1, description="Updated description")
        assert result is True
        assert plan.steps[0].description == "Updated description"

    def test_update_step_expected_output(self):
        """Test updating a step's expected output."""
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="simple",
            steps=[
                ResearchStep(
                    step_id=1,
                    description="Step 1",
                    step_type="research",
                    expected_output="Original output",
                )
            ],
        )
        result = plan.update_step(1, expected_output="Updated expected output")
        assert result is True
        assert plan.steps[0].expected_output == "Updated expected output"

    def test_update_step_multiple_fields(self):
        """Test updating multiple fields at once including description."""
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="simple",
            steps=[
                ResearchStep(
                    step_id=1,
                    description="Original description",
                    step_type="research",
                    expected_output="Original output",
                )
            ],
        )
        result = plan.update_step(
            1,
            status=StepStatus.IN_PROGRESS,
            description="New description",
            expected_output="New output",
        )
        assert result is True
        assert plan.steps[0].status == StepStatus.IN_PROGRESS
        assert plan.steps[0].description == "New description"
        assert plan.steps[0].expected_output == "New output"

    def test_get_pending_steps_no_dependencies(self):
        """Test getting pending steps when none have dependencies."""
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="moderate",
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
            complexity_assessment="moderate",
            steps=[
                ResearchStep(step_id=1, description="Step 1", step_type="research"),
                ResearchStep(step_id=2, description="Step 2", step_type="research", depends_on=[1]),
                ResearchStep(step_id=3, description="Step 3", step_type="synthesis", depends_on=[1, 2]),
            ],
        )
        # Only step 1 should be pending (no dependencies)
        pending = plan.get_pending_steps()
        assert len(pending) == 1
        assert pending[0].step_id == 1

    def test_get_pending_steps_after_completion(self):
        """Test getting pending steps after some complete."""
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="moderate",
            steps=[
                ResearchStep(step_id=1, description="Step 1", step_type="research", status=StepStatus.COMPLETED),
                ResearchStep(step_id=2, description="Step 2", step_type="research", depends_on=[1]),
                ResearchStep(step_id=3, description="Step 3", step_type="synthesis", depends_on=[1, 2]),
            ],
        )
        # Step 1 is done, step 2 should now be pending
        pending = plan.get_pending_steps()
        assert len(pending) == 1
        assert pending[0].step_id == 2

    def test_get_steps_by_status(self):
        """Test getting steps by status."""
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="moderate",
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

    def test_add_step(self):
        """Test adding a new step to the plan."""
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="simple",
            steps=[ResearchStep(step_id=1, description="Step 1", step_type="research")],
        )
        new_step = ResearchStep(step_id=2, description="Step 2", step_type="research")
        plan.add_step(new_step)

        assert len(plan.steps) == 2
        assert plan.steps[1].step_id == 2

    def test_is_complete_all_done(self):
        """Test is_complete when all steps are in terminal states."""
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="moderate",
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
            complexity_assessment="moderate",
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
            complexity_assessment="simple",
            steps=[
                ResearchStep(step_id=1, description="Step 1", step_type="research", status=StepStatus.IN_PROGRESS),
            ],
        )
        assert plan.is_complete() is False

    def test_get_next_step_id_empty(self):
        """Test getting next step ID when no steps exist."""
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="simple",
            steps=[],
        )
        assert plan.get_next_step_id() == 1

    def test_get_next_step_id_with_steps(self):
        """Test getting next step ID when steps exist."""
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="moderate",
            steps=[
                ResearchStep(step_id=1, description="Step 1", step_type="research"),
                ResearchStep(step_id=3, description="Step 3", step_type="research"),  # Gap in IDs
            ],
        )
        assert plan.get_next_step_id() == 4

    def test_get_progress_summary(self):
        """Test getting progress summary."""
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="complex",
            steps=[
                ResearchStep(step_id=1, description="Step 1", step_type="research", status=StepStatus.COMPLETED),
                ResearchStep(step_id=2, description="Step 2", step_type="research", status=StepStatus.COMPLETED),
                ResearchStep(step_id=3, description="Step 3", step_type="research", status=StepStatus.FAILED),
                ResearchStep(step_id=4, description="Step 4", step_type="synthesis", status=StepStatus.PENDING),
                ResearchStep(step_id=5, description="Step 5", step_type="research", status=StepStatus.IN_PROGRESS),
            ],
        )
        summary = plan.get_progress_summary()

        assert summary[StepStatus.COMPLETED] == 2
        assert summary[StepStatus.FAILED] == 1
        assert summary[StepStatus.PENDING] == 1
        assert summary[StepStatus.IN_PROGRESS] == 1
        assert summary[StepStatus.SKIPPED] == 0


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


class TestParsePlanResponse:
    """Tests for the _parse_plan_response function.

    With structured output, the function expects clean JSON matching
    the ResearchPlan schema.
    """

    def test_parses_valid_json(self):
        """Test parsing valid JSON response."""
        response = json.dumps(
            {
                "original_question": "Test question",
                "complexity_assessment": "moderate",
                "steps": [
                    {
                        "step_id": 1,
                        "description": "Search web",
                        "step_type": "research",
                        "depends_on": [],
                        "expected_output": "Results",
                    }
                ],
                "reasoning": "Simple plan",
            }
        )

        plan = _parse_plan_response(response)

        assert plan.original_question == "Test question"
        assert plan.complexity_assessment == "moderate"
        assert len(plan.steps) == 1
        assert plan.steps[0].step_type == "research"

    def test_parses_minimal_valid_json(self):
        """Test parsing minimal valid JSON with required fields."""
        response = json.dumps(
            {
                "original_question": "Test",
                "complexity_assessment": "simple",
                "steps": [
                    {
                        "step_id": 1,
                        "description": "Search",
                        "step_type": "research",
                    }
                ],
                "reasoning": "Quick lookup",
            }
        )
        plan = _parse_plan_response(response)

        assert plan.complexity_assessment == "simple"
        assert len(plan.steps) == 1

    def test_parses_empty_steps(self):
        """Test parsing JSON with empty steps array."""
        response = json.dumps(
            {
                "original_question": "Test",
                "complexity_assessment": "complex",
                "steps": [],
                "reasoning": "Empty plan",
            }
        )
        plan = _parse_plan_response(response)

        assert plan.complexity_assessment == "complex"
        assert len(plan.steps) == 0

    def test_raises_error_for_invalid_json(self):
        """Test that invalid JSON raises ValueError."""
        response = "This is not valid JSON at all"

        with pytest.raises(ValueError, match="Failed to parse planner response"):
            _parse_plan_response(response)

    def test_raises_error_for_missing_required_fields(self):
        """Test that missing required fields raise ValueError."""
        # Missing original_question and complexity_assessment
        response = json.dumps(
            {
                "steps": [
                    {
                        "step_id": 1,
                        "description": "Search",
                        "step_type": "research",
                    }
                ]
            }
        )

        with pytest.raises(ValueError, match="Failed to parse planner response"):
            _parse_plan_response(response)


class TestResearchPlanner:
    """Tests for the ResearchPlanner class."""

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_planner_initialization_without_config(self, mock_client_class):
        """Test planner initialization without config."""
        planner = ResearchPlanner()
        assert planner._model == "gemini-2.5-flash"

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_planner_initialization_with_model(self, mock_client_class):
        """Test planner initialization with explicit model."""
        planner = ResearchPlanner(model="gemini-2.5-pro")
        assert planner._model == "gemini-2.5-pro"

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_planner_initialization_with_config(self, mock_client_class):
        """Test planner initialization with config."""
        mock_config = MagicMock()
        mock_config.default_planner_model = "gemini-2.5-pro"

        planner = ResearchPlanner(config=mock_config)
        assert planner._model == "gemini-2.5-pro"

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_planner_model_parameter_overrides_config(self, mock_client_class):
        """Test that explicit model parameter overrides config."""
        mock_config = MagicMock()
        mock_config.default_planner_model = "gemini-2.5-pro"

        planner = ResearchPlanner(config=mock_config, model="gemini-2.5-flash")
        assert planner._model == "gemini-2.5-flash"

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_create_plan_calls_genai(self, mock_client_class):
        """Test that create_plan calls the genai client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "original_question": "Test",
                "complexity_assessment": "simple",
                "steps": [
                    {
                        "step_id": 1,
                        "description": "Search",
                        "step_type": "research",
                    }
                ],
                "reasoning": "Simple lookup",
            }
        )
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        planner = ResearchPlanner()
        plan = planner.create_plan("Test question")

        mock_client.models.generate_content.assert_called_once()
        assert plan.original_question == "Test"

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_create_plan_raises_on_api_error(self, mock_client_class):
        """Test that create_plan raises exception on API error."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        planner = ResearchPlanner()

        with pytest.raises(Exception, match="API Error"):
            planner.create_plan("Test question")

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    async def test_create_plan_async(self, mock_client_class):
        """Test async version of create_plan."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "original_question": "Async test",
                "complexity_assessment": "moderate",
                "steps": [],
                "reasoning": "Async plan",
            }
        )

        # Setup async mock
        async def mock_generate(*args, **kwargs):
            return mock_response

        mock_client.aio.models.generate_content = mock_generate
        mock_client_class.return_value = mock_client

        planner = ResearchPlanner()
        plan = await planner.create_plan_async("Async test question")

        assert plan.original_question == "Async test"
        assert plan.complexity_assessment == "moderate"


class TestParseNewStepsResponse:
    """Tests for the _parse_new_steps_response function.

    With structured output, the function expects NewStepsResponse format:
    {"steps": [...]}.
    """

    def test_parses_valid_json(self):
        """Test parsing valid JSON with steps array."""
        response = json.dumps(
            {
                "steps": [
                    {
                        "step_id": 1,
                        "description": "Search alternative",
                        "step_type": "research",
                        "depends_on": [],
                        "expected_output": "Alternative results",
                    }
                ]
            }
        )

        steps = _parse_new_steps_response(response, start_id=5)

        assert len(steps) == 1
        assert steps[0].step_id == 5
        assert steps[0].description == "Search alternative"
        assert steps[0].step_type == "research"

    def test_parses_multiple_steps(self):
        """Test parsing multiple steps with sequential IDs."""
        response = json.dumps(
            {
                "steps": [
                    {"step_id": 1, "description": "Step A", "step_type": "research"},
                    {"step_id": 2, "description": "Step B", "step_type": "research"},
                    {"step_id": 3, "description": "Step C", "step_type": "research"},
                ]
            }
        )

        steps = _parse_new_steps_response(response, start_id=10)

        assert len(steps) == 3
        assert steps[0].step_id == 10
        assert steps[1].step_id == 11
        assert steps[2].step_id == 12

    def test_parses_steps_with_dependencies(self):
        """Test parsing steps with depends_on field."""
        response = json.dumps(
            {
                "steps": [
                    {
                        "step_id": 1,
                        "description": "Try alternative source",
                        "step_type": "research",
                        "depends_on": [1],
                        "expected_output": "Document content",
                    }
                ]
            }
        )
        steps = _parse_new_steps_response(response, start_id=2)

        assert len(steps) == 1
        assert steps[0].step_id == 2
        assert steps[0].depends_on == [1]

    def test_returns_empty_for_invalid_json(self):
        """Test that invalid JSON returns empty list."""
        response = "This is not valid JSON"

        steps = _parse_new_steps_response(response, start_id=1)

        assert steps == []

    def test_returns_empty_for_empty_steps(self):
        """Test parsing empty steps array response."""
        response = json.dumps({"steps": []})

        steps = _parse_new_steps_response(response, start_id=1)

        assert steps == []


class TestParseReflectionResponse:
    """Tests for the _parse_reflection_response method of ResearchPlanner.

    With structured output, the function expects clean JSON matching
    the PlanReflection schema.
    """

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_parses_valid_json(self, mock_client_class):
        """Test parsing valid JSON reflection response."""
        planner = ResearchPlanner()
        response = json.dumps(
            {
                "can_answer_now": True,
                "key_findings": "Found key information",
                "reasoning": "Have enough data to answer",
                "steps_to_update": [],
                "steps_to_remove": [],
                "steps_to_add": [],
            }
        )

        reflection = planner._parse_reflection_response(response, next_step_id=5)

        assert reflection.can_answer_now is True
        assert reflection.key_findings == "Found key information"
        assert reflection.reasoning == "Have enough data to answer"

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_parses_steps_to_update(self, mock_client_class):
        """Test parsing steps_to_update field."""
        planner = ResearchPlanner()
        response = json.dumps(
            {
                "can_answer_now": False,
                "key_findings": "",
                "reasoning": "Need to refine search",
                "steps_to_update": [
                    {
                        "step_id": 2,
                        "new_description": "Updated search query",
                        "new_expected_output": "More specific results",
                    }
                ],
                "steps_to_remove": [],
                "steps_to_add": [],
            }
        )

        reflection = planner._parse_reflection_response(response, next_step_id=5)

        assert len(reflection.steps_to_update) == 1
        assert reflection.steps_to_update[0].step_id == 2
        assert reflection.steps_to_update[0].new_description == "Updated search query"
        assert reflection.steps_to_update[0].new_expected_output == "More specific results"

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_parses_steps_to_remove(self, mock_client_class):
        """Test parsing steps_to_remove field."""
        planner = ResearchPlanner()
        response = json.dumps(
            {
                "can_answer_now": False,
                "key_findings": "",
                "reasoning": "Some steps no longer needed",
                "steps_to_update": [],
                "steps_to_remove": [2, 3],
                "steps_to_add": [],
            }
        )

        reflection = planner._parse_reflection_response(response, next_step_id=5)

        assert reflection.steps_to_remove == [2, 3]

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_parses_steps_to_add(self, mock_client_class):
        """Test parsing steps_to_add field with correct IDs."""
        planner = ResearchPlanner()
        response = json.dumps(
            {
                "can_answer_now": False,
                "key_findings": "",
                "reasoning": "Need additional research",
                "steps_to_update": [],
                "steps_to_remove": [],
                "steps_to_add": [
                    {
                        "step_id": 0,
                        "description": "New step 1",
                        "step_type": "research",
                        "depends_on": [1],
                        "expected_output": "Expected output 1",
                    },
                    {
                        "step_id": 0,
                        "description": "New step 2",
                        "step_type": "synthesis",
                        "depends_on": [1, 5],
                        "expected_output": "Expected output 2",
                    },
                ],
            }
        )

        reflection = planner._parse_reflection_response(response, next_step_id=5)

        assert len(reflection.steps_to_add) == 2
        # Verify IDs are assigned starting from next_step_id
        assert reflection.steps_to_add[0].step_id == 5
        assert reflection.steps_to_add[1].step_id == 6
        assert reflection.steps_to_add[0].description == "New step 1"
        assert reflection.steps_to_add[0].depends_on == [1]
        assert reflection.steps_to_add[1].step_type == "synthesis"

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_parses_minimal_valid_json(self, mock_client_class):
        """Test parsing minimal valid JSON with defaults."""
        planner = ResearchPlanner()
        response = json.dumps(
            {
                "can_answer_now": False,
                "key_findings": "Found partial data",
                "reasoning": "Need more sources",
                "steps_to_update": [],
                "steps_to_remove": [],
                "steps_to_add": [],
            }
        )
        reflection = planner._parse_reflection_response(response, next_step_id=5)

        assert reflection.can_answer_now is False
        assert reflection.key_findings == "Found partial data"

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_returns_default_for_invalid_json(self, mock_client_class):
        """Test that invalid JSON returns default reflection."""
        planner = ResearchPlanner()
        response = "This is not valid JSON at all"

        reflection = planner._parse_reflection_response(response, next_step_id=5)

        # Should return default values with parse error reasoning
        assert reflection.can_answer_now is False
        assert "parse error" in reflection.reasoning.lower()

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_handles_optional_fields(self, mock_client_class):
        """Test handling of optional fields with defaults."""
        planner = ResearchPlanner()
        response = json.dumps(
            {
                "can_answer_now": True,
                "key_findings": "",
                "reasoning": "",
                "steps_to_update": [],
                "steps_to_remove": [],
                "steps_to_add": [],
            }
        )

        reflection = planner._parse_reflection_response(response, next_step_id=5)

        assert reflection.can_answer_now is True
        assert reflection.key_findings == ""
        assert reflection.reasoning == ""
        assert reflection.steps_to_update == []
        assert reflection.steps_to_remove == []
        assert reflection.steps_to_add == []


class TestApplyReflectionToPlan:
    """Tests for the _apply_reflection_to_plan method of ResearchPlanner."""

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_applies_step_updates(self, mock_client_class):
        """Test that step updates are applied correctly."""
        planner = ResearchPlanner()
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="moderate",
            steps=[
                ResearchStep(
                    step_id=1,
                    description="Step 1",
                    step_type="research",
                    status=StepStatus.COMPLETED,
                ),
                ResearchStep(
                    step_id=2,
                    description="Original description",
                    step_type="research",
                    status=StepStatus.PENDING,
                    expected_output="Original output",
                ),
            ],
        )
        reflection = PlanReflection(
            can_answer_now=False,
            steps_to_update=[
                StepUpdate(
                    step_id=2,
                    new_description="Updated description",
                    new_expected_output="Updated output",
                )
            ],
        )

        planner._apply_reflection_to_plan(plan, reflection)

        step2 = plan.get_step(2)
        assert step2.description == "Updated description"
        assert step2.expected_output == "Updated output"

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_applies_step_removals(self, mock_client_class):
        """Test that step removals mark steps as skipped."""
        planner = ResearchPlanner()
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="moderate",
            steps=[
                ResearchStep(step_id=1, description="Step 1", step_type="research"),
                ResearchStep(step_id=2, description="Step 2", step_type="research"),
                ResearchStep(step_id=3, description="Step 3", step_type="research"),
            ],
        )
        reflection = PlanReflection(
            can_answer_now=False,
            steps_to_remove=[2],
        )

        planner._apply_reflection_to_plan(plan, reflection)

        assert plan.get_step(1).status == StepStatus.PENDING  # Unchanged
        assert plan.get_step(2).status == StepStatus.SKIPPED  # Removed
        assert plan.get_step(3).status == StepStatus.PENDING  # Unchanged

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_does_not_remove_synthesis_steps(self, mock_client_class):
        """Test that synthesis steps cannot be removed."""
        planner = ResearchPlanner()
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="moderate",
            steps=[
                ResearchStep(step_id=1, description="Research", step_type="research"),
                ResearchStep(step_id=2, description="Synthesize", step_type="synthesis"),
            ],
        )
        reflection = PlanReflection(
            can_answer_now=False,
            steps_to_remove=[2],
        )

        planner._apply_reflection_to_plan(plan, reflection)

        # Synthesis step should NOT be skipped
        assert plan.get_step(2).status == StepStatus.PENDING

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_applies_step_additions(self, mock_client_class):
        """Test that new steps are added to the plan."""
        planner = ResearchPlanner()
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="moderate",
            steps=[
                ResearchStep(step_id=1, description="Step 1", step_type="research"),
            ],
        )
        new_step = ResearchStep(
            step_id=5,
            description="New step",
            step_type="research",
        )
        reflection = PlanReflection(
            can_answer_now=False,
            steps_to_add=[new_step],
        )

        planner._apply_reflection_to_plan(plan, reflection)

        assert len(plan.steps) == 2
        assert plan.get_step(5) is not None
        assert plan.get_step(5).description == "New step"

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_can_answer_now_skips_pending_with_content(self, mock_client_class):
        """Test that can_answer_now skips pending steps when content exists."""
        planner = ResearchPlanner()
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="moderate",
            steps=[
                ResearchStep(step_id=1, description="Step 1", step_type="research", status=StepStatus.COMPLETED),
                ResearchStep(step_id=2, description="Step 2", step_type="research"),
                ResearchStep(step_id=3, description="Synthesize", step_type="synthesis"),
            ],
        )
        reflection = PlanReflection(
            can_answer_now=True,
            reasoning="Have enough information",
        )

        planner._apply_reflection_to_plan(plan, reflection, has_substantial_content=True)

        # Research step should be skipped
        assert plan.get_step(2).status == StepStatus.SKIPPED
        # Synthesis step should NOT be skipped
        assert plan.get_step(3).status == StepStatus.PENDING

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_can_answer_now_does_not_skip_without_content(self, mock_client_class):
        """Test that can_answer_now doesn't skip when no substantial content."""
        planner = ResearchPlanner()
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="moderate",
            steps=[
                ResearchStep(step_id=1, description="Step 1", step_type="research"),
                ResearchStep(step_id=2, description="Step 2", step_type="research"),
            ],
        )
        reflection = PlanReflection(
            can_answer_now=True,
            reasoning="Have enough information",
        )

        # has_substantial_content defaults to False
        planner._apply_reflection_to_plan(plan, reflection)

        # Steps should NOT be skipped
        assert plan.get_step(1).status == StepStatus.PENDING
        assert plan.get_step(2).status == StepStatus.PENDING

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_does_not_update_completed_steps(self, mock_client_class):
        """Test that completed steps are not updated."""
        planner = ResearchPlanner()
        plan = ResearchPlan(
            original_question="Test",
            complexity_assessment="simple",
            steps=[
                ResearchStep(
                    step_id=1,
                    description="Original",
                    step_type="research",
                    status=StepStatus.COMPLETED,
                ),
            ],
        )
        reflection = PlanReflection(
            can_answer_now=False,
            steps_to_update=[StepUpdate(step_id=1, new_description="Should not change")],
        )

        planner._apply_reflection_to_plan(plan, reflection)

        # Completed step should not be updated
        assert plan.get_step(1).description == "Original"


class TestResearchPlannerReplanning:
    """Tests for the replanning methods of ResearchPlanner."""

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_suggest_new_steps_calls_genai(self, mock_client_class):
        """Test that suggest_new_steps calls the genai client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "steps": [
                    {
                        "step_id": 0,
                        "description": "Try alternative search",
                        "step_type": "research",
                        "depends_on": [],
                        "expected_output": "Alternative results",
                    }
                ]
            }
        )
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        planner = ResearchPlanner()
        plan = ResearchPlan(
            original_question="Test question",
            complexity_assessment="moderate",
            steps=[
                ResearchStep(
                    step_id=1,
                    description="Initial search",
                    step_type="research",
                    status=StepStatus.FAILED,
                    failure_reason="No results found",
                ),
            ],
        )

        new_steps = planner.suggest_new_steps(plan, "Search returned no results")

        mock_client.models.generate_content.assert_called_once()
        assert len(new_steps) == 1
        assert new_steps[0].step_id == 2  # Next ID after existing step

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_suggest_new_steps_with_failed_step(self, mock_client_class):
        """Test suggesting new steps when a specific step failed."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({"steps": []})  # No new steps suggested
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        planner = ResearchPlanner()
        plan = ResearchPlan(
            original_question="Test question",
            complexity_assessment="simple",
            steps=[
                ResearchStep(
                    step_id=1,
                    description="Fetch document",
                    step_type="research",
                    status=StepStatus.FAILED,
                    attempts=3,
                    failure_reason="404 Not Found",
                ),
            ],
        )

        failed_step = plan.steps[0]
        planner.suggest_new_steps(plan, "URL returned 404", failed_step=failed_step)

        # Should have called genai
        mock_client.models.generate_content.assert_called_once()
        # Check that the prompt includes failed step info
        call_args = mock_client.models.generate_content.call_args
        prompt_content = call_args[1]["contents"].parts[0].text
        assert "Failed Step Details" in prompt_content
        assert "404 Not Found" in prompt_content
        assert "Attempts: 3" in prompt_content

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_suggest_new_steps_returns_empty_on_error(self, mock_client_class):
        """Test that suggest_new_steps returns empty list on error."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        planner = ResearchPlanner()
        plan = ResearchPlan(
            original_question="Test question",
            complexity_assessment="simple",
            steps=[],
        )

        new_steps = planner.suggest_new_steps(plan, "Some result")

        assert new_steps == []

    @pytest.mark.asyncio
    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    async def test_suggest_new_steps_async(self, mock_client_class):
        """Test async version of suggest_new_steps."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "steps": [
                    {
                        "step_id": 0,
                        "description": "Alternative search",
                        "step_type": "research",
                    }
                ]
            }
        )

        async def mock_generate(*args, **kwargs):
            return mock_response

        mock_client.aio.models.generate_content = mock_generate
        mock_client_class.return_value = mock_client

        planner = ResearchPlanner()
        plan = ResearchPlan(
            original_question="Async test",
            complexity_assessment="simple",
            steps=[ResearchStep(step_id=1, description="Step 1", step_type="research")],
        )

        new_steps = await planner.suggest_new_steps_async(plan, "Initial result")

        assert len(new_steps) == 1
        assert new_steps[0].step_id == 2

    @patch("aieng.agent_evals.knowledge_agent.planner.genai.Client")
    def test_build_replanning_prompt(self, mock_client_class):
        """Test that _build_replanning_prompt includes all relevant info."""
        planner = ResearchPlanner()
        plan = ResearchPlan(
            original_question="What is the GDP of France?",
            complexity_assessment="moderate",
            steps=[
                ResearchStep(
                    step_id=1,
                    description="Search for GDP data",
                    step_type="research",
                    status=StepStatus.COMPLETED,
                    actual_output="Found several sources",
                ),
                ResearchStep(
                    step_id=2,
                    description="Fetch World Bank data",
                    step_type="research",
                    depends_on=[1],
                    status=StepStatus.FAILED,
                    failure_reason="Timeout",
                ),
                ResearchStep(
                    step_id=3,
                    description="Synthesize findings",
                    step_type="synthesis",
                    depends_on=[1, 2],
                    status=StepStatus.PENDING,
                ),
            ],
        )

        failed_step = plan.get_step(2)
        prompt = planner._build_replanning_prompt(plan, "Fetch timed out", failed_step=failed_step)

        # Check that the prompt contains essential information
        assert "What is the GDP of France?" in prompt
        assert "Step 1" in prompt
        assert "Step 2" in prompt
        assert "completed" in prompt.lower()
        assert "failed" in prompt.lower()
        assert "pending" in prompt.lower()
        assert "Fetch timed out" in prompt
        assert "Failed Step Details" in prompt
        assert "Timeout" in prompt


@pytest.mark.integration_test
class TestResearchPlannerIntegration:
    """Integration tests for the research planner.

    These tests require a valid GOOGLE_API_KEY environment variable.
    """

    def test_create_plan_real(self):
        """Test creating a real research plan."""
        planner = ResearchPlanner()
        plan = planner.create_plan("What are the main provisions of Basel III?")

        assert plan.original_question is not None
        assert plan.complexity_assessment in ["simple", "moderate", "complex"]
        assert len(plan.steps) > 0

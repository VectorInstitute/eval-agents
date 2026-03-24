"""Tests for Vertex AI Search tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aieng.agent_evals.configs import Configs
from aieng.agent_evals.tools import create_vertex_search_tool, vertex_search
from aieng.agent_evals.tools.vertex_search import (
    _extract_datastore_sources,
    _parse_project_from_datastore_id,
    _vertex_search_async,
)
from dotenv import load_dotenv
from google.adk.tools.function_tool import FunctionTool


SAMPLE_DATASTORE_ID = "projects/my-project/locations/global/collections/default_collection/dataStores/my-store"


class TestParseProjectFromDatastoreId:
    """Tests for _parse_project_from_datastore_id."""

    def test_standard_resource_name(self):
        """Test parsing project from a well-formed resource name."""
        assert _parse_project_from_datastore_id(SAMPLE_DATASTORE_ID) == "my-project"

    def test_numeric_project_id(self):
        """Test parsing a numeric project number."""
        datastore_id = "projects/123456789/locations/global/collections/default_collection/dataStores/my-store"
        assert _parse_project_from_datastore_id(datastore_id) == "123456789"

    def test_returns_none_for_invalid_format(self):
        """Test that a non-resource-name string returns None."""
        assert _parse_project_from_datastore_id("not-a-resource-name") is None

    def test_returns_none_for_empty_string(self):
        """Test that an empty string returns None."""
        assert _parse_project_from_datastore_id("") is None

    def test_returns_none_when_no_projects_prefix(self):
        """Test that a string without 'projects' prefix returns None."""
        assert _parse_project_from_datastore_id("locations/global/collections/default_collection") is None


class TestExtractDatastoreSources:
    """Tests for _extract_datastore_sources."""

    def test_no_candidates_returns_empty(self):
        """Test that an empty candidates list yields no sources."""
        response = MagicMock()
        response.candidates = []
        assert _extract_datastore_sources(response) == []

    def test_no_grounding_metadata_returns_empty(self):
        """Test that a candidate with no grounding_metadata yields no sources."""
        candidate = MagicMock()
        candidate.grounding_metadata = None
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_datastore_sources(response) == []

    def test_grounding_chunks_attribute_missing_returns_empty(self):
        """Test that grounding_metadata without grounding_chunks yields no sources."""
        gm = MagicMock(spec=[])  # hasattr(gm, "grounding_chunks") → False
        candidate = MagicMock()
        candidate.grounding_metadata = gm
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_datastore_sources(response) == []

    def test_none_grounding_chunks_returns_empty(self):
        """Test that grounding_chunks=None yields no sources."""
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = None
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_datastore_sources(response) == []

    def test_empty_grounding_chunks_returns_empty(self):
        """Test that an empty grounding_chunks list yields no sources."""
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = []
        response = MagicMock()
        response.candidates = [candidate]
        assert _extract_datastore_sources(response) == []

    def test_single_valid_retrieved_context_chunk(self):
        """Test that a single retrieved_context chunk with document_name is returned."""
        chunk = MagicMock()
        chunk.retrieved_context.document_name = "projects/my-project/locations/global/collections/default_collection/dataStores/my-store/branches/0/documents/doc-1"
        chunk.retrieved_context.title = "Company Policy"
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk]
        response = MagicMock()
        response.candidates = [candidate]

        result = _extract_datastore_sources(response)

        assert result == [
            {
                "title": "Company Policy",
                "uri": "projects/my-project/locations/global/collections/default_collection/dataStores/my-store/branches/0/documents/doc-1",
            }
        ]

    def test_multiple_chunks_preserved_in_order(self):
        """Test that multiple retrieved_context chunks are returned in order."""
        chunk1 = MagicMock()
        chunk1.retrieved_context.document_name = (
            "projects/p/locations/global/collections/c/dataStores/s/branches/0/documents/doc1"
        )
        chunk1.retrieved_context.title = "Document 1"
        chunk2 = MagicMock()
        chunk2.retrieved_context.document_name = (
            "projects/p/locations/global/collections/c/dataStores/s/branches/0/documents/doc2"
        )
        chunk2.retrieved_context.title = "Document 2"
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk1, chunk2]
        response = MagicMock()
        response.candidates = [candidate]

        result = _extract_datastore_sources(response)

        assert result == [
            {
                "title": "Document 1",
                "uri": "projects/p/locations/global/collections/c/dataStores/s/branches/0/documents/doc1",
            },
            {
                "title": "Document 2",
                "uri": "projects/p/locations/global/collections/c/dataStores/s/branches/0/documents/doc2",
            },
        ]

    def test_chunk_without_retrieved_context_is_skipped(self):
        """Test that chunks with no retrieved_context attribute are ignored."""
        chunk_no_rc = MagicMock(spec=[])  # getattr returns None fallback
        chunk_valid = MagicMock()
        chunk_valid.retrieved_context.document_name = (
            "projects/p/locations/global/collections/c/dataStores/s/branches/0/documents/valid"
        )
        chunk_valid.retrieved_context.title = "Valid Doc"
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk_no_rc, chunk_valid]
        response = MagicMock()
        response.candidates = [candidate]

        result = _extract_datastore_sources(response)

        assert result == [
            {
                "title": "Valid Doc",
                "uri": "projects/p/locations/global/collections/c/dataStores/s/branches/0/documents/valid",
            }
        ]

    def test_chunk_with_none_retrieved_context_is_skipped(self):
        """Test that chunks whose retrieved_context is falsy are skipped."""
        chunk = MagicMock()
        chunk.retrieved_context = None
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk]
        response = MagicMock()
        response.candidates = [candidate]

        assert _extract_datastore_sources(response) == []

    def test_empty_document_name_is_excluded(self):
        """Test that a retrieved_context with an empty document_name is excluded."""
        chunk = MagicMock()
        chunk.retrieved_context.document_name = ""
        chunk.retrieved_context.title = "No Name"
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk]
        response = MagicMock()
        response.candidates = [candidate]

        assert _extract_datastore_sources(response) == []

    def test_none_document_name_is_excluded(self):
        """Test that a retrieved_context with a None document_name is excluded."""
        chunk = MagicMock()
        chunk.retrieved_context.document_name = None
        chunk.retrieved_context.title = "None Name"
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk]
        response = MagicMock()
        response.candidates = [candidate]

        assert _extract_datastore_sources(response) == []

    def test_empty_title_is_preserved(self):
        """Test that a chunk with an empty title is returned with empty title string."""
        chunk = MagicMock()
        chunk.retrieved_context.document_name = (
            "projects/p/locations/global/collections/c/dataStores/s/branches/0/documents/doc"
        )
        chunk.retrieved_context.title = ""
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk]
        response = MagicMock()
        response.candidates = [candidate]

        result = _extract_datastore_sources(response)

        assert result == [
            {"title": "", "uri": "projects/p/locations/global/collections/c/dataStores/s/branches/0/documents/doc"}
        ]

    def test_none_title_normalised_to_empty_string(self):
        """Test that a None title is coerced to an empty string."""
        chunk = MagicMock()
        chunk.retrieved_context.document_name = (
            "projects/p/locations/global/collections/c/dataStores/s/branches/0/documents/doc"
        )
        chunk.retrieved_context.title = None
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk]
        response = MagicMock()
        response.candidates = [candidate]

        result = _extract_datastore_sources(response)

        assert result == [
            {"title": "", "uri": "projects/p/locations/global/collections/c/dataStores/s/branches/0/documents/doc"}
        ]

    def test_only_first_candidate_is_used(self):
        """Test that only the first candidate's grounding chunks are considered."""
        chunk1 = MagicMock()
        chunk1.retrieved_context.document_name = (
            "projects/p/locations/global/collections/c/dataStores/s/branches/0/documents/first"
        )
        chunk1.retrieved_context.title = "First"
        chunk2 = MagicMock()
        chunk2.retrieved_context.document_name = (
            "projects/p/locations/global/collections/c/dataStores/s/branches/0/documents/second"
        )
        chunk2.retrieved_context.title = "Second"
        candidate1 = MagicMock()
        candidate1.grounding_metadata.grounding_chunks = [chunk1]
        candidate2 = MagicMock()
        candidate2.grounding_metadata.grounding_chunks = [chunk2]
        response = MagicMock()
        response.candidates = [candidate1, candidate2]

        result = _extract_datastore_sources(response)

        assert result == [
            {
                "title": "First",
                "uri": "projects/p/locations/global/collections/c/dataStores/s/branches/0/documents/first",
            }
        ]

    def test_web_chunks_are_not_retrieved_context_and_skipped(self):
        """Test that web-type grounding chunks are skipped (not datastore sources)."""
        chunk = MagicMock(spec=["web"])  # has 'web' but not 'retrieved_context'
        chunk.web.uri = "https://example.com"
        chunk.web.title = "Web Result"
        candidate = MagicMock()
        candidate.grounding_metadata.grounding_chunks = [chunk]
        response = MagicMock()
        response.candidates = [candidate]

        # getattr(chunk, "retrieved_context", None) returns None for spec=["web"]
        assert _extract_datastore_sources(response) == []


class TestVertexSearchAsync:
    """Tests for _vertex_search_async."""

    @pytest.mark.asyncio
    async def test_success_returns_expected_structure(self):
        """Test that a successful call returns the correct response dict."""
        mock_part = MagicMock()
        mock_part.text = "The leave policy allows 20 days per year."
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_chunk = MagicMock()
        mock_chunk.retrieved_context.document_name = "projects/my-project/locations/global/collections/default_collection/dataStores/my-store/branches/0/documents/policy"
        mock_chunk.retrieved_context.title = "Leave Policy"
        mock_candidate.grounding_metadata.grounding_chunks = [mock_chunk]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client.close = MagicMock()

        with patch("aieng.agent_evals.tools.vertex_search.Client", return_value=mock_client):
            result = await _vertex_search_async(
                query="What is the leave policy?",
                model="gemini-2.5-flash",
                datastore_id=SAMPLE_DATASTORE_ID,
                location="us-central1",
            )

        assert result["status"] == "success"
        assert result["summary"] == "The leave policy allows 20 days per year."
        assert result["sources"] == [
            {
                "title": "Leave Policy",
                "uri": "projects/my-project/locations/global/collections/default_collection/dataStores/my-store/branches/0/documents/policy",
            }
        ]
        assert result["source_count"] == 1

    @pytest.mark.asyncio
    async def test_client_created_with_correct_project_and_location(self):
        """Test that the client is created with the correct project and location."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(candidates=[])
        mock_client.close = MagicMock()

        with patch("aieng.agent_evals.tools.vertex_search.Client", return_value=mock_client) as mock_cls:
            await _vertex_search_async(
                query="test",
                model="gemini-2.5-flash",
                datastore_id=SAMPLE_DATASTORE_ID,
                location="us-central1",
            )

        mock_cls.assert_called_once_with(vertexai=True, project="my-project", location="us-central1")

    @pytest.mark.asyncio
    async def test_client_closed_on_success(self):
        """Test that the client is always closed after a successful call."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(candidates=[])
        mock_client.close = MagicMock()

        with patch("aieng.agent_evals.tools.vertex_search.Client", return_value=mock_client):
            await _vertex_search_async("test", "gemini-2.5-flash", SAMPLE_DATASTORE_ID, "us-central1")

        mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_closed_on_exception(self):
        """Test that the client is closed even when generate_content raises."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API error")
        mock_client.close = MagicMock()

        with patch("aieng.agent_evals.tools.vertex_search.Client", return_value=mock_client):
            result = await _vertex_search_async("test", "gemini-2.5-flash", SAMPLE_DATASTORE_ID, "us-central1")

        mock_client.close.assert_called_once()
        assert result["status"] == "error"
        assert "API error" in result["error"]

    @pytest.mark.asyncio
    async def test_exception_returns_error_structure(self):
        """Test that an exception produces a well-formed error response."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("Connection timeout")
        mock_client.close = MagicMock()

        with patch("aieng.agent_evals.tools.vertex_search.Client", return_value=mock_client):
            result = await _vertex_search_async("test", "gemini-2.5-flash", SAMPLE_DATASTORE_ID, "us-central1")

        assert result["status"] == "error"
        assert "Connection timeout" in result["error"]
        assert result["summary"] == ""
        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_uses_vertex_ai_search_tool_in_config(self):
        """Test that generate_content is called with a VertexAISearch retrieval tool."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(candidates=[])
        mock_client.close = MagicMock()

        with patch("aieng.agent_evals.tools.vertex_search.Client", return_value=mock_client):
            await _vertex_search_async("test", "gemini-2.5-flash", SAMPLE_DATASTORE_ID, "us-central1")

        call_kwargs = mock_client.models.generate_content.call_args
        config_arg = call_kwargs.kwargs["config"]
        assert config_arg.tools is not None
        tool = config_arg.tools[0]
        assert tool.retrieval is not None
        assert tool.retrieval.vertex_ai_search.datastore == SAMPLE_DATASTORE_ID

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty_summary_and_sources(self):
        """Test that a response with no candidates returns empty summary and sources."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(candidates=[])
        mock_client.close = MagicMock()

        with patch("aieng.agent_evals.tools.vertex_search.Client", return_value=mock_client):
            result = await _vertex_search_async("test", "gemini-2.5-flash", SAMPLE_DATASTORE_ID, "us-central1")

        assert result["status"] == "success"
        assert result["summary"] == ""
        assert result["sources"] == []
        assert result["source_count"] == 0


class TestCreateVertexSearchTool:
    """Tests for create_vertex_search_tool."""

    def test_raises_if_no_datastore_id(self):
        """Test that ValueError is raised when vertex_datastore_id is not configured."""
        mock_config = MagicMock()
        mock_config.vertex_datastore_id = None

        with pytest.raises(ValueError, match="VERTEX_AI_DATASTORE_ID"):
            create_vertex_search_tool(config=mock_config)

    def test_raises_if_empty_datastore_id(self):
        """Test that ValueError is raised for an empty vertex_datastore_id."""
        mock_config = MagicMock()
        mock_config.vertex_datastore_id = ""

        with pytest.raises(ValueError, match="VERTEX_AI_DATASTORE_ID"):
            create_vertex_search_tool(config=mock_config)

    def test_returns_function_tool(self):
        """Test that a FunctionTool is returned when datastore_id is set."""
        mock_config = MagicMock()
        mock_config.vertex_datastore_id = SAMPLE_DATASTORE_ID

        result = create_vertex_search_tool(config=mock_config)

        assert isinstance(result, FunctionTool)

    def test_tool_func_named_vertex_search(self):
        """Test that the wrapped function is named 'vertex_search' for ADK discovery."""
        mock_config = MagicMock()
        mock_config.vertex_datastore_id = SAMPLE_DATASTORE_ID

        result = create_vertex_search_tool(config=mock_config)

        assert result.func.__name__ == "vertex_search"

    @pytest.mark.asyncio
    async def test_tool_calls_vertex_search_async(self):
        """Test that the tool delegates to _vertex_search_async with config values."""
        mock_config = MagicMock()
        mock_config.vertex_datastore_id = SAMPLE_DATASTORE_ID
        mock_config.default_worker_model = "gemini-2.5-flash"
        mock_config.default_temperature = 1.0
        mock_config.google_cloud_location = "us-central1"

        tool = create_vertex_search_tool(config=mock_config)

        expected = {"status": "success", "summary": "Answer.", "sources": [], "source_count": 0}
        with (
            patch("aieng.agent_evals.tools.vertex_search.Configs", return_value=mock_config),
            patch(
                "aieng.agent_evals.tools.vertex_search._vertex_search_async",
                new=AsyncMock(return_value=expected),
            ) as mock_async,
        ):
            result = await tool.func("leave policy?")

        mock_async.assert_called_once_with(
            "leave policy?",
            model="gemini-2.5-flash",
            datastore_id=SAMPLE_DATASTORE_ID,
            location="us-central1",
            temperature=1.0,
        )
        assert result == expected


class TestVertexSearchPublicFunction:
    """Tests for the standalone vertex_search public function."""

    @pytest.mark.asyncio
    async def test_raises_if_datastore_id_not_configured(self):
        """Test that ValueError is raised when VERTEX_AI_DATASTORE_ID is not set."""
        mock_config = MagicMock()
        mock_config.vertex_datastore_id = None

        with (
            patch("aieng.agent_evals.tools.vertex_search.Configs", return_value=mock_config),
            pytest.raises(ValueError, match="VERTEX_AI_DATASTORE_ID"),
        ):
            await vertex_search("test query")

    @pytest.mark.asyncio
    async def test_uses_default_model_from_config(self):
        """Test that the worker model defaults to config when none is specified."""
        mock_config = MagicMock()
        mock_config.vertex_datastore_id = SAMPLE_DATASTORE_ID
        mock_config.default_worker_model = "gemini-2.5-flash"
        mock_config.default_temperature = 1.0
        mock_config.google_cloud_location = "us-central1"

        expected = {"status": "success", "summary": "ok", "sources": [], "source_count": 0}
        with (
            patch("aieng.agent_evals.tools.vertex_search.Configs", return_value=mock_config),
            patch(
                "aieng.agent_evals.tools.vertex_search._vertex_search_async",
                new=AsyncMock(return_value=expected),
            ) as mock_async,
        ):
            await vertex_search("test query")

        mock_async.assert_called_once_with(
            "test query",
            model="gemini-2.5-flash",
            datastore_id=SAMPLE_DATASTORE_ID,
            location="us-central1",
            temperature=1.0,
        )

    @pytest.mark.asyncio
    async def test_explicit_model_overrides_config(self):
        """Test that passing a model explicitly overrides the config default."""
        mock_config = MagicMock()
        mock_config.vertex_datastore_id = SAMPLE_DATASTORE_ID
        mock_config.default_worker_model = "gemini-2.5-flash"
        mock_config.default_temperature = 1.0
        mock_config.google_cloud_location = "us-central1"

        expected = {"status": "success", "summary": "ok", "sources": [], "source_count": 0}
        with (
            patch("aieng.agent_evals.tools.vertex_search.Configs", return_value=mock_config),
            patch(
                "aieng.agent_evals.tools.vertex_search._vertex_search_async",
                new=AsyncMock(return_value=expected),
            ) as mock_async,
        ):
            await vertex_search("test query", model="gemini-2.5-pro")

        mock_async.assert_called_once_with(
            "test query",
            model="gemini-2.5-pro",
            datastore_id=SAMPLE_DATASTORE_ID,
            location="us-central1",
            temperature=1.0,
        )


@pytest.mark.integration_test
class TestVertexSearchIntegration:
    """Integration tests for the Vertex AI Search tool.

    These tests run against a real Vertex AI Search data store loaded with
    ``aieng-eval-agents/tests/fixtures/vertex_test_data.jsonl`` (synthetic
    Northstar Analytics content). Provision the store once before running:

        uv run python -m scripts.create_test_datastore --bucket <your-bucket>

    Then set in .env:
        VERTEX_AI_DATASTORE_ID="projects/agentic-ai-evaluation-bootcamp/locations/global/..."
        GOOGLE_CLOUD_LOCATION="us-central1"

    Authentication is handled automatically via ADC / the GCE service account.
    """

    @pytest.fixture(autouse=True)
    def skip_if_not_configured(self):
        """Skip the entire class if VERTEX_AI_DATASTORE_ID is not set."""
        load_dotenv(verbose=False)
        config = Configs()  # type: ignore[call-arg]
        if not config.vertex_datastore_id:
            pytest.skip("VERTEX_AI_DATASTORE_ID not set — run scripts/create_test_datastore.py first")

    def test_create_vertex_search_tool_real(self):
        """Test that a FunctionTool can be created against the real data store."""
        tool = create_vertex_search_tool()
        assert isinstance(tool, FunctionTool)
        assert tool.func.__name__ == "vertex_search"

    @pytest.mark.asyncio
    async def test_response_structure(self):
        """Test that vertex_search returns a well-formed response dict."""
        result = await vertex_search("What does Northstar Analytics do?")

        assert result["status"] == "success", f"Unexpected error: {result.get('error')}"
        assert isinstance(result["summary"], str)
        assert result["summary"], "Expected a non-empty summary"
        assert isinstance(result["sources"], list)
        assert isinstance(result["source_count"], int)
        assert result["source_count"] == len(result["sources"])

    @pytest.mark.asyncio
    async def test_sources_use_uri_not_url(self):
        """Test that sources contain 'uri' (GCS / resource name) not 'url'."""
        result = await vertex_search("What does Northstar Analytics do?")

        assert result["status"] == "success"
        if result["sources"]:
            source = result["sources"][0]
            assert "uri" in source, "Sources must have a 'uri' key"
            assert "title" in source, "Sources must have a 'title' key"
            assert "url" not in source, "Sources must not use 'url' — this is not a web search tool"

    @pytest.mark.asyncio
    async def test_grounding_professional_tier_price(self):
        """Test that grounding retrieves the Professional tier price from the datastore.

        The fixture contains: Professional at $899/month.
        This number does not exist in model training data (it is synthetic),
        so its presence in the summary confirms the data store was consulted.
        """
        result = await vertex_search("What is the monthly price for the Professional tier at Northstar Analytics?")

        assert result["status"] == "success", f"Search failed: {result.get('error')}"
        assert result["source_count"] > 0, "Expected at least one grounding source"
        assert "899" in result["summary"], (
            f"Expected '899' (Professional tier price) in summary, got: {result['summary']}"
        )

    @pytest.mark.asyncio
    async def test_grounding_enterprise_sla(self):
        """Test that grounding retrieves the Enterprise SLA uptime figure.

        The fixture contains: Enterprise tier 99.95% uptime guarantee.
        """
        result = await vertex_search("What uptime does Northstar Analytics guarantee for Enterprise customers?")

        assert result["status"] == "success", f"Search failed: {result.get('error')}"
        assert result["source_count"] > 0, "Expected at least one grounding source"
        assert "99.95" in result["summary"], f"Expected '99.95' (Enterprise SLA) in summary, got: {result['summary']}"

    @pytest.mark.asyncio
    async def test_grounding_api_rate_limit(self):
        """Test that grounding retrieves the Enterprise API rate limit.

        The fixture contains: Enterprise tier 8,000 requests per minute.
        """
        result = await vertex_search(
            "How many API requests per minute does the Enterprise tier allow at Northstar Analytics?"
        )

        assert result["status"] == "success", f"Search failed: {result.get('error')}"
        assert result["source_count"] > 0, "Expected at least one grounding source"
        # Allow for comma-formatted (8,000) or plain (8000)
        assert "8,000" in result["summary"] or "8000" in result["summary"], (
            f"Expected '8,000' or '8000' (Enterprise API limit) in summary, got: {result['summary']}"
        )

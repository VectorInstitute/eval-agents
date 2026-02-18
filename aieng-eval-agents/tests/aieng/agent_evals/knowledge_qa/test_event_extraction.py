"""Tests for event_extraction utilities."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aieng.agent_evals.knowledge_qa.event_extraction import (
    extract_event_text,
    extract_final_response,
    extract_grounding_queries,
    extract_grounding_sources,
    extract_search_queries_from_tool_calls,
    extract_sources_from_responses,
    extract_thoughts_from_event,
    extract_tool_calls,
    resolve_source_urls,
)
from aieng.agent_evals.tools import GroundingChunk


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_RESOLVE_REDIRECT_PATCH = "aieng.agent_evals.knowledge_qa.event_extraction.resolve_redirect_urls_async"


# ---------------------------------------------------------------------------
# Helpers: lightweight ADK event stubs
# ---------------------------------------------------------------------------


def _make_function_call(name="tool_name", args=None):
    return SimpleNamespace(name=name, args=args or {})


def _make_function_response(name="tool_name", response=None):
    return SimpleNamespace(name=name, id=name, response=response or {})


def _make_part(text=None, thought=False):
    ns = SimpleNamespace(thought=thought)
    if text is not None:
        ns.text = text
    return ns


def _make_content(parts=None, grounding_metadata=None):
    return SimpleNamespace(parts=parts or [], grounding_metadata=grounding_metadata)


def _make_web(title="", uri=""):
    return SimpleNamespace(title=title, uri=uri)


def _make_grounding_chunk(web=None):
    return SimpleNamespace(web=web)


def _make_grounding_metadata(grounding_chunks=None, web_search_queries=None):
    return SimpleNamespace(
        grounding_chunks=grounding_chunks or [],
        web_search_queries=web_search_queries or [],
    )


def _make_event(
    function_calls=None,
    function_responses=None,
    content=None,
    grounding_metadata=None,
    is_final=False,
):
    """Build a minimal ADK-style event stub."""

    def get_function_calls():
        return function_calls

    def get_function_responses():
        return function_responses

    def is_final_response():
        return is_final

    return SimpleNamespace(
        get_function_calls=get_function_calls,
        get_function_responses=get_function_responses,
        is_final_response=is_final_response,
        content=content,
        grounding_metadata=grounding_metadata,
    )


# ---------------------------------------------------------------------------
# extract_tool_calls
# ---------------------------------------------------------------------------


class TestExtractToolCalls:
    """Tests for extract_tool_calls."""

    @pytest.mark.parametrize(
        "event",
        [
            SimpleNamespace(),  # no get_function_calls attr
            _make_event(function_calls=None),  # returns None
            _make_event(function_calls=[]),  # returns []
        ],
        ids=["no_attr", "returns_none", "returns_empty"],
    )
    def test_returns_empty_list(self, event):
        """Returns empty list when function calls are absent, None, or empty."""
        assert extract_tool_calls(event) == []

    def test_single_function_call_extracted(self):
        """Single function call is extracted with correct name and args."""
        fc = _make_function_call(name="google_search", args={"query": "test"})
        event = _make_event(function_calls=[fc])
        result = extract_tool_calls(event)
        assert len(result) == 1
        assert result[0]["name"] == "google_search"
        assert result[0]["args"] == {"query": "test"}

    def test_multiple_function_calls_extracted_in_order(self):
        """Multiple function calls are all extracted in order."""
        fcs = [
            _make_function_call(name="google_search", args={"query": "q1"}),
            _make_function_call(name="web_fetch", args={"url": "https://example.com"}),
        ]
        result = extract_tool_calls(_make_event(function_calls=fcs))
        assert [tc["name"] for tc in result] == ["google_search", "web_fetch"]

    def test_missing_name_defaults_to_unknown(self):
        """Function call without a name attribute defaults to 'unknown'."""
        fc = SimpleNamespace(args={"query": "q"})  # no 'name' attr
        result = extract_tool_calls(_make_event(function_calls=[fc]))
        assert result[0]["name"] == "unknown"

    def test_missing_args_defaults_to_empty_dict(self):
        """Function call without an args attribute defaults to empty dict."""
        fc = SimpleNamespace(name="web_fetch")  # no 'args' attr
        result = extract_tool_calls(_make_event(function_calls=[fc]))
        assert result[0]["args"] == {}

    def test_each_item_has_name_and_args_keys(self):
        """Each returned dict has both 'name' and 'args' keys."""
        fc = _make_function_call(name="fetch_file", args={"url": "https://data.csv"})
        result = extract_tool_calls(_make_event(function_calls=[fc]))
        assert all("name" in tc and "args" in tc for tc in result)


# ---------------------------------------------------------------------------
# extract_search_queries_from_tool_calls
# ---------------------------------------------------------------------------


class TestExtractSearchQueriesFromToolCalls:
    """Tests for extract_search_queries_from_tool_calls."""

    @pytest.mark.parametrize(
        "tool_calls",
        [
            [],
            [{"name": "web_fetch", "args": {"url": "https://example.com"}}],  # non-search tool
            [{"name": "google_search", "args": {"request": "", "query": ""}}],  # empty query values
            [{"name": "google_search", "args": "not a dict"}],  # non-dict args
            [{"args": {"query": "q"}}],  # missing 'name' key
            [{"name": "google_search"}],  # missing 'args' key
        ],
        ids=["empty", "non_search", "empty_query", "non_dict_args", "no_name", "no_args"],
    )
    def test_returns_empty(self, tool_calls):
        """Returns empty list when no extractable search queries are present."""
        assert extract_search_queries_from_tool_calls(tool_calls) == []

    @pytest.mark.parametrize(
        "args,expected",
        [
            ({"request": "population of Canada"}, ["population of Canada"]),
            ({"query": "Python async"}, ["Python async"]),
            ({"request": "primary", "query": "secondary"}, ["primary"]),  # request wins
        ],
        ids=["request_arg", "query_arg", "request_over_query"],
    )
    def test_extracts_query(self, args, expected):
        """Extracts the correct query string from search tool args."""
        tool_calls = [{"name": "google_search", "args": args}]
        assert extract_search_queries_from_tool_calls(tool_calls) == expected

    def test_mixed_tools_only_search_queries_returned(self):
        """Only queries from search-named tools are returned."""
        tool_calls = [
            {"name": "google_search", "args": {"request": "first query"}},
            {"name": "web_fetch", "args": {"url": "https://example.com"}},
            {"name": "search_agent", "args": {"request": "second query"}},
        ]
        assert extract_search_queries_from_tool_calls(tool_calls) == ["first query", "second query"]

    def test_search_name_match_is_case_insensitive(self):
        """'search' substring match in tool name is case-insensitive."""
        tool_calls = [{"name": "GOOGLE_SEARCH", "args": {"query": "test"}}]
        assert extract_search_queries_from_tool_calls(tool_calls) == ["test"]


# ---------------------------------------------------------------------------
# extract_sources_from_responses
# ---------------------------------------------------------------------------


class TestExtractSourcesFromResponses:
    """Tests for extract_sources_from_responses."""

    @pytest.mark.parametrize(
        "event",
        [
            SimpleNamespace(),  # no get_function_responses attr
            _make_event(function_responses=None),
            _make_event(function_responses=[]),
        ],
        ids=["no_attr", "returns_none", "returns_empty"],
    )
    def test_returns_empty_list(self, event):
        """Returns empty list when function responses are absent, None, or empty."""
        assert extract_sources_from_responses(event) == []

    def test_non_dict_response_data_skipped(self):
        """Function response with non-dict response data is skipped entirely."""
        fr = _make_function_response(name="tool", response="plain string")
        assert extract_sources_from_responses(_make_event(function_responses=[fr])) == []

    def test_sources_list_produces_grounding_chunks(self):
        """Sources list in response data is converted to GroundingChunks."""
        sources_data = [{"title": "Page A", "uri": "https://a.com"}, {"title": "Page B", "uri": "https://b.com"}]
        fr = _make_function_response(response={"sources": sources_data})
        result = extract_sources_from_responses(_make_event(function_responses=[fr]))
        assert len(result) == 2
        assert result[0].title == "Page A" and result[0].uri == "https://a.com"
        assert result[1].title == "Page B" and result[1].uri == "https://b.com"

    def test_url_key_accepted_in_place_of_uri(self):
        """Source entry using 'url' key instead of 'uri' is handled correctly."""
        fr = _make_function_response(response={"sources": [{"title": "Site", "url": "https://site.com"}]})
        result = extract_sources_from_responses(_make_event(function_responses=[fr]))
        assert result[0].uri == "https://site.com"

    @pytest.mark.parametrize(
        "source_entry,expected_title,expected_uri",
        [
            ({"uri": "https://notitle.com"}, "", "https://notitle.com"),  # no title
            ({"title": "No URL"}, "No URL", ""),  # no uri/url
        ],
        ids=["no_title", "no_uri"],
    )
    def test_missing_source_fields_default_to_empty_string(self, source_entry, expected_title, expected_uri):
        """Missing title or URI in a source entry defaults to empty string."""
        fr = _make_function_response(response={"sources": [source_entry]})
        result = extract_sources_from_responses(_make_event(function_responses=[fr]))
        assert result[0].title == expected_title
        assert result[0].uri == expected_uri

    def test_grounding_chunks_in_response_extracted(self):
        """grounding_chunks in response data are converted to GroundingChunks."""
        chunks = [{"web": {"title": "GChunk", "uri": "https://gchunk.com"}}]
        fr = _make_function_response(response={"grounding_chunks": chunks})
        result = extract_sources_from_responses(_make_event(function_responses=[fr]))
        assert len(result) == 1
        assert result[0].title == "GChunk" and result[0].uri == "https://gchunk.com"

    def test_grounding_chunk_without_web_key_skipped(self):
        """grounding_chunk entry missing the 'web' key is skipped."""
        fr = _make_function_response(response={"grounding_chunks": [{"other": {}}]})
        assert extract_sources_from_responses(_make_event(function_responses=[fr])) == []

    def test_sources_and_grounding_chunks_aggregated(self):
        """Both sources and grounding_chunks within the same response are aggregated."""
        data = {
            "sources": [{"title": "S1", "uri": "https://s1.com"}],
            "grounding_chunks": [{"web": {"title": "G1", "uri": "https://g1.com"}}],
        }
        result = extract_sources_from_responses(
            _make_event(function_responses=[_make_function_response(response=data)])
        )
        assert len(result) == 2

    @pytest.mark.parametrize(
        "response_data",
        [{"error": "Something went wrong"}, {"status": "error"}],
        ids=["error_key", "status_error"],
    )
    def test_error_response_returns_empty(self, response_data):
        """Response indicating an error produces no sources."""
        fr = _make_function_response(response=response_data)
        assert extract_sources_from_responses(_make_event(function_responses=[fr])) == []

    def test_multiple_responses_aggregate_sources(self):
        """Sources from multiple function responses are all collected."""
        fr1 = _make_function_response(name="s1", response={"sources": [{"title": "T1", "uri": "https://t1.com"}]})
        fr2 = _make_function_response(name="s2", response={"sources": [{"title": "T2", "uri": "https://t2.com"}]})
        assert len(extract_sources_from_responses(_make_event(function_responses=[fr1, fr2]))) == 2

    def test_non_dict_source_entry_skipped(self):
        """Non-dict entries inside the 'sources' list are ignored."""
        sources_data = ["not a dict", {"title": "Valid", "uri": "https://valid.com"}]
        fr = _make_function_response(response={"sources": sources_data})
        result = extract_sources_from_responses(_make_event(function_responses=[fr]))
        assert len(result) == 1 and result[0].title == "Valid"

    def test_function_response_without_name_falls_back_to_id(self):
        """Function response without 'name' attr falls back to 'id' for logging."""
        fr = SimpleNamespace(response={"sources": []}, id="fallback_id")
        assert extract_sources_from_responses(_make_event(function_responses=[fr])) == []


# ---------------------------------------------------------------------------
# extract_grounding_sources
# ---------------------------------------------------------------------------


class TestExtractGroundingSources:
    """Tests for extract_grounding_sources."""

    @pytest.mark.parametrize(
        "event",
        [
            SimpleNamespace(grounding_metadata=None, content=None),  # no metadata at all
            SimpleNamespace(grounding_metadata=SimpleNamespace(), content=None),  # no grounding_chunks attr
            SimpleNamespace(grounding_metadata=_make_grounding_metadata(grounding_chunks=[]), content=None),
            SimpleNamespace(
                grounding_metadata=_make_grounding_metadata(grounding_chunks=[SimpleNamespace()]),
                content=None,
            ),  # chunk has no 'web' attr
            SimpleNamespace(
                grounding_metadata=_make_grounding_metadata(grounding_chunks=[_make_grounding_chunk(web=None)]),
                content=None,
            ),  # web is None
        ],
        ids=["no_metadata", "no_chunks_attr", "empty_chunks", "chunk_no_web", "web_is_none"],
    )
    def test_returns_empty_list(self, event):
        """Returns empty list when grounding metadata or chunks are absent/empty."""
        assert extract_grounding_sources(event) == []

    def test_grounding_metadata_on_event(self):
        """Grounding metadata directly on the event is used."""
        chunk = _make_grounding_chunk(web=_make_web(title="Direct", uri="https://direct.com"))
        gm = _make_grounding_metadata(grounding_chunks=[chunk])
        result = extract_grounding_sources(SimpleNamespace(grounding_metadata=gm, content=None))
        assert len(result) == 1
        assert result[0].title == "Direct" and result[0].uri == "https://direct.com"

    def test_grounding_metadata_on_event_content(self):
        """Grounding metadata on event.content is used when event has none."""
        chunk = _make_grounding_chunk(web=_make_web(title="ContentGM", uri="https://content.com"))
        gm = _make_grounding_metadata(grounding_chunks=[chunk])
        event = SimpleNamespace(grounding_metadata=None, content=SimpleNamespace(grounding_metadata=gm))
        result = extract_grounding_sources(event)
        assert len(result) == 1 and result[0].title == "ContentGM"

    @pytest.mark.parametrize(
        "web,expected_title,expected_uri",
        [
            (SimpleNamespace(uri="https://notitle.com"), "", "https://notitle.com"),  # no title attr
            (SimpleNamespace(title="No URI"), "No URI", ""),  # no uri attr
        ],
        ids=["no_title", "no_uri"],
    )
    def test_missing_web_fields_default_to_empty_string(self, web, expected_title, expected_uri):
        """Missing web title or URI defaults to empty string."""
        gm = _make_grounding_metadata(grounding_chunks=[_make_grounding_chunk(web=web)])
        result = extract_grounding_sources(SimpleNamespace(grounding_metadata=gm, content=None))
        assert result[0].title == expected_title
        assert result[0].uri == expected_uri

    def test_multiple_chunks_returned_in_order(self):
        """Multiple grounding chunks are all returned in order."""
        chunks = [_make_grounding_chunk(web=_make_web(t, f"https://{t}.com")) for t in ("a", "b", "c")]
        gm = _make_grounding_metadata(grounding_chunks=chunks)
        result = extract_grounding_sources(SimpleNamespace(grounding_metadata=gm, content=None))
        assert [r.title for r in result] == ["a", "b", "c"]

    def test_returns_grounding_chunk_instances(self):
        """All returned items are GroundingChunk instances."""
        gm = _make_grounding_metadata(grounding_chunks=[_make_grounding_chunk(web=_make_web("T", "https://t.com"))])
        result = extract_grounding_sources(SimpleNamespace(grounding_metadata=gm, content=None))
        assert all(isinstance(r, GroundingChunk) for r in result)


# ---------------------------------------------------------------------------
# resolve_source_urls
# ---------------------------------------------------------------------------


class TestResolveSourceUrls:
    """Tests for resolve_source_urls (async)."""

    @pytest.mark.asyncio
    async def test_empty_sources_returns_empty(self):
        """Empty source list is returned immediately without calling the resolver."""
        assert await resolve_source_urls([]) == []

    @pytest.mark.asyncio
    async def test_uris_replaced_with_resolved_values(self):
        """URIs are replaced with their resolved counterparts."""
        sources = [
            GroundingChunk(title="Page A", uri="https://a.com"),
            GroundingChunk(title="Page B", uri="https://b.com"),
        ]
        resolved = ["https://a-resolved.com", "https://b-resolved.com"]
        with patch(_RESOLVE_REDIRECT_PATCH, new_callable=AsyncMock, return_value=resolved):
            result = await resolve_source_urls(sources)
        assert [r.uri for r in result] == resolved

    @pytest.mark.asyncio
    async def test_titles_preserved_after_resolution(self):
        """Titles are preserved unchanged after URI resolution."""
        sources = [GroundingChunk(title="Keep This Title", uri="https://redirect.com")]
        with patch(_RESOLVE_REDIRECT_PATCH, new_callable=AsyncMock, return_value=["https://final.com"]):
            result = await resolve_source_urls(sources)
        assert result[0].title == "Keep This Title"

    @pytest.mark.asyncio
    async def test_order_preserved(self):
        """Resolved sources are returned in the same order as the input."""
        sources = [GroundingChunk(title=f"T{i}", uri=f"https://src{i}.com") for i in range(5)]
        resolved = [f"https://resolved{i}.com" for i in range(5)]
        with patch(_RESOLVE_REDIRECT_PATCH, new_callable=AsyncMock, return_value=resolved):
            result = await resolve_source_urls(sources)
        assert [r.uri for r in result] == resolved
        assert [r.title for r in result] == [s.title for s in sources]

    @pytest.mark.asyncio
    async def test_returns_grounding_chunk_instances(self):
        """All returned items are GroundingChunk instances."""
        sources = [GroundingChunk(title="T", uri="https://t.com")]
        with patch(_RESOLVE_REDIRECT_PATCH, new_callable=AsyncMock, return_value=["https://t.com"]):
            result = await resolve_source_urls(sources)
        assert all(isinstance(r, GroundingChunk) for r in result)


# ---------------------------------------------------------------------------
# extract_grounding_queries
# ---------------------------------------------------------------------------


class TestExtractGroundingQueries:
    """Tests for extract_grounding_queries."""

    @pytest.mark.parametrize(
        "event",
        [
            SimpleNamespace(grounding_metadata=None, content=None),  # no metadata
            SimpleNamespace(grounding_metadata=SimpleNamespace(), content=None),  # no web_search_queries attr
            SimpleNamespace(grounding_metadata=SimpleNamespace(web_search_queries=None), content=None),
            SimpleNamespace(grounding_metadata=_make_grounding_metadata(web_search_queries=[]), content=None),
        ],
        ids=["no_metadata", "no_queries_attr", "queries_is_none", "empty_list"],
    )
    def test_returns_empty_list(self, event):
        """Returns empty list when grounding metadata or queries are absent/empty."""
        assert extract_grounding_queries(event) == []

    def test_queries_from_event_grounding_metadata(self):
        """Queries from event-level grounding metadata are returned."""
        gm = _make_grounding_metadata(web_search_queries=["q1", "q2"])
        assert extract_grounding_queries(SimpleNamespace(grounding_metadata=gm, content=None)) == ["q1", "q2"]

    def test_queries_from_content_grounding_metadata(self):
        """Content-level grounding metadata queries are used when event has none."""
        gm = _make_grounding_metadata(web_search_queries=["content query"])
        event = SimpleNamespace(grounding_metadata=None, content=SimpleNamespace(grounding_metadata=gm))
        assert extract_grounding_queries(event) == ["content query"]

    def test_empty_strings_filtered_out(self):
        """Empty string entries in web_search_queries are excluded."""
        gm = _make_grounding_metadata(web_search_queries=["valid", "", "another", ""])
        event = SimpleNamespace(grounding_metadata=gm, content=None)
        assert extract_grounding_queries(event) == ["valid", "another"]

    def test_event_level_metadata_takes_precedence_over_content(self):
        """Event-level grounding metadata is used before content-level."""
        gm_event = _make_grounding_metadata(web_search_queries=["direct"])
        gm_content = _make_grounding_metadata(web_search_queries=["from content"])
        event = SimpleNamespace(
            grounding_metadata=gm_event,
            content=SimpleNamespace(grounding_metadata=gm_content),
        )
        assert extract_grounding_queries(event) == ["direct"]


# ---------------------------------------------------------------------------
# extract_final_response
# ---------------------------------------------------------------------------


class TestExtractFinalResponse:
    """Tests for extract_final_response."""

    @pytest.mark.parametrize(
        "event",
        [
            SimpleNamespace(),  # no is_final_response attr
            _make_event(is_final=False),
            _make_event(is_final=True, content=None),
            _make_event(is_final=True, content=SimpleNamespace()),  # content has no 'parts' attr
            _make_event(is_final=True, content=_make_content(parts=[])),
            _make_event(
                is_final=True,
                content=_make_content(parts=[_make_part(text="reasoning", thought=True)]),
            ),  # only thoughts
        ],
        ids=["no_attr", "not_final", "no_content", "no_parts_attr", "empty_parts", "only_thoughts"],
    )
    def test_returns_none(self, event):
        """Returns None when the event is not final or has no non-thought text."""
        assert extract_final_response(event) is None

    def test_non_thought_text_returned(self):
        """Non-thought text part is returned as the final response."""
        parts = [_make_part(text="The answer is 42.", thought=False)]
        result = extract_final_response(_make_event(is_final=True, content=_make_content(parts=parts)))
        assert result == "The answer is 42."

    def test_thought_parts_filtered_out(self):
        """Thought parts are excluded; only non-thought text is returned."""
        parts = [
            _make_part(text="hidden reasoning", thought=True),
            _make_part(text="visible answer", thought=False),
        ]
        result = extract_final_response(_make_event(is_final=True, content=_make_content(parts=parts)))
        assert result == "visible answer"

    def test_multiple_non_thought_parts_joined_with_newline(self):
        """Multiple non-thought parts are joined with newlines."""
        parts = [_make_part(text="Part one.", thought=False), _make_part(text="Part two.", thought=False)]
        result = extract_final_response(_make_event(is_final=True, content=_make_content(parts=parts)))
        assert result == "Part one.\nPart two."

    def test_empty_text_parts_excluded(self):
        """Parts with empty text are excluded from the response."""
        parts = [_make_part(text="", thought=False), _make_part(text="Actual response", thought=False)]
        result = extract_final_response(_make_event(is_final=True, content=_make_content(parts=parts)))
        assert result == "Actual response"

    def test_parts_without_text_attr_skipped(self):
        """Parts lacking a 'text' attribute are skipped."""
        parts = [SimpleNamespace(thought=False), _make_part(text="Has text", thought=False)]
        result = extract_final_response(_make_event(is_final=True, content=_make_content(parts=parts)))
        assert result == "Has text"


# ---------------------------------------------------------------------------
# extract_thoughts_from_event
# ---------------------------------------------------------------------------


class TestExtractThoughtsFromEvent:
    """Tests for extract_thoughts_from_event."""

    @pytest.mark.parametrize(
        "event",
        [
            SimpleNamespace(),  # no 'content' attr
            SimpleNamespace(content=None),
            SimpleNamespace(content=SimpleNamespace()),  # content has no 'parts' attr
            SimpleNamespace(content=_make_content(parts=[])),
            SimpleNamespace(content=_make_content(parts=[_make_part(text="visible", thought=False)])),
        ],
        ids=["no_content_attr", "none_content", "no_parts_attr", "empty_parts", "only_non_thought"],
    )
    def test_returns_empty_string(self, event):
        """Returns empty string when no thought parts are present."""
        assert extract_thoughts_from_event(event) == ""

    def test_single_thought_part_returned(self):
        """A single thought part's text is returned."""
        parts = [_make_part(text="internal reasoning", thought=True)]
        result = extract_thoughts_from_event(SimpleNamespace(content=_make_content(parts=parts)))
        assert result == "internal reasoning"

    def test_multiple_thought_parts_joined_with_newline(self):
        """Multiple thought parts are joined with newlines."""
        parts = [_make_part(text="step 1", thought=True), _make_part(text="step 2", thought=True)]
        result = extract_thoughts_from_event(SimpleNamespace(content=_make_content(parts=parts)))
        assert result == "step 1\nstep 2"

    def test_non_thought_parts_excluded(self):
        """Non-thought parts do not appear in the thoughts output."""
        parts = [_make_part(text="thinking...", thought=True), _make_part(text="response text", thought=False)]
        result = extract_thoughts_from_event(SimpleNamespace(content=_make_content(parts=parts)))
        assert result == "thinking..."

    def test_thought_parts_with_empty_text_excluded(self):
        """Thought parts with empty text are excluded."""
        parts = [_make_part(text="", thought=True), _make_part(text="real thought", thought=True)]
        result = extract_thoughts_from_event(SimpleNamespace(content=_make_content(parts=parts)))
        assert result == "real thought"

    def test_thought_parts_without_text_attr_skipped(self):
        """Thought parts lacking a 'text' attribute are skipped."""
        parts = [SimpleNamespace(thought=True), _make_part(text="valid thought", thought=True)]
        result = extract_thoughts_from_event(SimpleNamespace(content=_make_content(parts=parts)))
        assert result == "valid thought"


# ---------------------------------------------------------------------------
# extract_event_text
# ---------------------------------------------------------------------------


class TestExtractEventText:
    """Tests for extract_event_text."""

    @pytest.mark.parametrize(
        "event",
        [
            SimpleNamespace(),  # no 'content' attr
            SimpleNamespace(content=None),
            SimpleNamespace(content=SimpleNamespace()),  # content has no 'parts' attr
            SimpleNamespace(content=SimpleNamespace(parts=None)),
            SimpleNamespace(content=_make_content(parts=[])),
        ],
        ids=["no_content_attr", "none_content", "no_parts_attr", "parts_is_none", "empty_parts"],
    )
    def test_returns_empty_string(self, event):
        """Returns empty string when content or parts are absent, None, or empty."""
        assert extract_event_text(event) == ""

    def test_single_text_part_returned(self):
        """Single text part is returned as-is."""
        event = SimpleNamespace(content=_make_content(parts=[_make_part(text="Hello world")]))
        assert extract_event_text(event) == "Hello world"

    def test_multiple_text_parts_joined_with_newline(self):
        """Multiple text parts are joined with newlines."""
        parts = [_make_part(text="Line 1"), _make_part(text="Line 2")]
        assert extract_event_text(SimpleNamespace(content=_make_content(parts=parts))) == "Line 1\nLine 2"

    def test_includes_thought_and_non_thought_parts(self):
        """Both thought and non-thought parts contribute to the text output."""
        parts = [_make_part(text="thinking", thought=True), _make_part(text="responding", thought=False)]
        result = extract_event_text(SimpleNamespace(content=_make_content(parts=parts)))
        assert "thinking" in result and "responding" in result

    def test_parts_without_text_attr_skipped(self):
        """Parts lacking a 'text' attribute are skipped."""
        parts = [SimpleNamespace(), _make_part(text="valid")]
        assert extract_event_text(SimpleNamespace(content=_make_content(parts=parts))) == "valid"

    def test_parts_with_empty_text_skipped(self):
        """Parts whose text is an empty string are skipped."""
        parts = [_make_part(text=""), _make_part(text="non-empty")]
        assert extract_event_text(SimpleNamespace(content=_make_content(parts=parts))) == "non-empty"

    def test_all_empty_text_parts_returns_empty_string(self):
        """All parts having empty text results in an empty string."""
        parts = [_make_part(text=""), _make_part(text="")]
        assert extract_event_text(SimpleNamespace(content=_make_content(parts=parts))) == ""

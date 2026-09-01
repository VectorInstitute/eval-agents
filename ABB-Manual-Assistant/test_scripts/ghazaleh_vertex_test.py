"""Integration tests for querying a real Vertex AI Search vector store.

These tests exercise the `vertex_search` helper against the datastore configured
by `VERTEX_AI_DATASTORE_ID`. If the variable is not set, a known default
datastore ID is used.
"""

import asyncio
import base64
import os
from pprint import pprint
from typing import Any

import google.auth
import google.auth.transport.requests
import pytest
from aieng.agent_evals.tools.vertex_search import vertex_search
from dotenv import load_dotenv


DEFAULT_VERTEX_DATASTORE_ID = (
	"projects/agentic-ai-evaluation-bootcamp/locations/global/collections/default_collection/"
	"dataStores/linamar-vector-bootcamp"
)

KNOWN_GOOD_ABB_SUMMARY_QUERY = "What is ArcC used for in ABB arc welding manuals?"
KNOWN_GOOD_ABB_CITATION_QUERY = "What is ArcC used for in ABB arc welding manuals? Cite sources."


def configure_vertex_datastore() -> None:
	"""Ensure the datastore ID is available for integration tests and direct runs."""
	load_dotenv(verbose=False)
	os.environ.setdefault("VERTEX_AI_DATASTORE_ID", DEFAULT_VERTEX_DATASTORE_ID)
	os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")


@pytest.fixture(autouse=True)
def configure_vertex_datastore_fixture() -> None:
	"""Pytest fixture wrapper around the shared datastore setup."""
	configure_vertex_datastore()


async def _print_vertex_search_result(query: str = KNOWN_GOOD_ABB_SUMMARY_QUERY) -> None:
	"""Run a sample Vertex Search query and print the structured result."""
	configure_vertex_datastore()
	result = await vertex_search(query)
	print("Query:", query)
	print("Status:", result.get("status"))
	print("Summary:")
	print((result.get("summary") or "").strip())
	print("Source count:", result.get("source_count"))
	print("Sources:")
	pprint(result.get("sources", []))
	print("Raw result:")
	pprint(result)


if __name__ == "__main__":
	asyncio.run(_print_vertex_search_result())


def _get_datastore_document_count(datastore_resource: str) -> int:
	"""Return the indexed document count from Discovery Engine documents list API."""
	credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
	session = google.auth.transport.requests.AuthorizedSession(credentials)
	base_url = f"https://discoveryengine.googleapis.com/v1/{datastore_resource}/branches/default_branch/documents"

	count = 0
	next_token = ""
	while True:
		params = {"pageSize": 100}
		if next_token:
			params["pageToken"] = next_token

		response = session.get(base_url, params=params, timeout=30)
		response.raise_for_status()
		payload = response.json()
		documents = payload.get("documents", [])
		count += len(documents)

		next_token = payload.get("nextPageToken", "")
		if not next_token:
			break

	return count


def _list_datastore_documents(datastore_resource: str, max_docs: int = 10) -> list[dict[str, Any]]:
	"""List up to ``max_docs`` documents from the configured data store."""
	credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
	session = google.auth.transport.requests.AuthorizedSession(credentials)
	base_url = f"https://discoveryengine.googleapis.com/v1/{datastore_resource}/branches/default_branch/documents"

	documents: list[dict[str, Any]] = []
	next_token = ""
	while len(documents) < max_docs:
		params = {"pageSize": min(100, max_docs - len(documents))}
		if next_token:
			params["pageToken"] = next_token

		response = session.get(base_url, params=params, timeout=30)
		response.raise_for_status()
		payload = response.json()
		documents.extend(payload.get("documents", []))

		next_token = payload.get("nextPageToken", "")
		if not next_token:
			break

	return documents[:max_docs]


def _extract_text_preview(document: dict[str, Any], max_chars: int = 200) -> str:
	"""Extract a human-readable content preview from a Discovery Engine document."""
	content = document.get("content", {}) or {}

	raw_bytes = content.get("rawBytes")
	if isinstance(raw_bytes, str) and raw_bytes:
		try:
			decoded = base64.b64decode(raw_bytes).decode("utf-8", errors="ignore").strip()
			if decoded:
				return decoded[:max_chars]
		except Exception:
			pass

	uri = content.get("uri")
	if isinstance(uri, str) and uri.strip():
		return uri[:max_chars]

	struct_data = document.get("structData")
	if isinstance(struct_data, dict):
		for key in ("text", "content", "body", "chunk_text", "chunk"):
			value = struct_data.get(key)
			if isinstance(value, str) and value.strip():
				return value.strip()[:max_chars]

	title = document.get("title")
	if isinstance(title, str) and title.strip():
		return title[:max_chars]

	return ""


@pytest.mark.integration_test
def test_vertex_datastore_has_indexed_documents() -> None:
	"""Ensure the configured data store contains at least one indexed document."""
	datastore = os.environ["VERTEX_AI_DATASTORE_ID"]
	doc_count = _get_datastore_document_count(datastore)
	assert doc_count > 0, f"No indexed documents found in datastore: {datastore}"


@pytest.mark.integration_test
def test_vertex_datastore_stats_and_sample_documents() -> None:
	"""Print and validate basic datastore stats with sample document identities."""
	datastore = os.environ["VERTEX_AI_DATASTORE_ID"]
	doc_count = _get_datastore_document_count(datastore)
	docs = _list_datastore_documents(datastore, max_docs=5)

	print(f"Datastore: {datastore}")
	print(f"Document count: {doc_count}")
	print("Sample document names:")
	for doc in docs:
		print(f"- {doc.get('name', '<missing-name>')}")

	assert doc_count > 0, "Expected at least one document in datastore"
	assert docs, "Expected at least one sample document to be returned"


@pytest.mark.integration_test
def test_vertex_datastore_chunk_previews() -> None:
	"""Read and print content previews from stored documents/chunks."""
	datastore = os.environ["VERTEX_AI_DATASTORE_ID"]
	docs = _list_datastore_documents(datastore, max_docs=10)
	previews: list[str] = []

	for doc in docs:
		preview = _extract_text_preview(doc)
		if preview:
			previews.append(preview)

	print(f"Found {len(previews)} previews from {len(docs)} sampled documents")
	for idx, preview in enumerate(previews[:5], start=1):
		print(f"Preview {idx}: {preview}")

	assert docs, "No documents returned from datastore"
	assert previews, "Could not extract any text/URI preview from sampled datastore documents"


@pytest.mark.integration_test
@pytest.mark.asyncio
async def test_vertex_search_returns_success_and_schema() -> None:
	"""Query ABB content and validate the response shape."""
	result = await vertex_search(KNOWN_GOOD_ABB_SUMMARY_QUERY)

	print("Search query:", KNOWN_GOOD_ABB_SUMMARY_QUERY)
	print("Search status:", result.get("status"))
	print("Search summary:")
	print((result.get("summary") or "").strip())
	print("Search source_count:", result.get("source_count"))
	print("Search sources:")
	for idx, source in enumerate(result.get("sources", []), start=1):
		print(f"- Source {idx}: title={source.get('title', '')} | uri={source.get('uri', '')}")

	assert result["status"] == "success", f"Vertex search failed: {result.get('error')}"
	assert isinstance(result.get("summary"), str)
	assert result["summary"].strip(), "Expected a non-empty summary"
	assert isinstance(result.get("sources"), list)
	assert isinstance(result.get("source_count"), int)
	assert result["source_count"] == len(result["sources"])


@pytest.mark.integration_test
@pytest.mark.asyncio
async def test_vertex_search_returns_uri_citations() -> None:
	"""Ensure a known ABB query returns citations in Vertex format."""
	result = await vertex_search(KNOWN_GOOD_ABB_CITATION_QUERY)

	print("Citation query:", KNOWN_GOOD_ABB_CITATION_QUERY)
	print("Citation status:", result.get("status"))
	print("Citation summary:")
	print((result.get("summary") or "").strip())
	print("Citation source_count:", result.get("source_count"))
	print("Citation sources:")
	for idx, source in enumerate(result.get("sources", []), start=1):
		print(f"- Source {idx}: title={source.get('title', '')} | uri={source.get('uri', '')}")

	assert result["status"] == "success", f"Vertex search failed: {result.get('error')}"
	assert isinstance(result.get("sources"), list)
	assert result["source_count"] > 0, "Expected at least one citation for known in-domain ABB query"
	assert result["source_count"] == len(result["sources"])
	for source in result["sources"]:
		assert "uri" in source and isinstance(source["uri"], str) and source["uri"].strip()
		assert "title" in source and isinstance(source["title"], str)
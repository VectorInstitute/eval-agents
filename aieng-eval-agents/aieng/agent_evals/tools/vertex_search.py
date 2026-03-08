"""Vertex AI Search tool for knowledge-grounded QA using a custom data store.

This module provides a search tool that queries a Vertex AI Search data store,
returning grounded summaries with document citations. Unlike the Google Search
tool, content is retrieved by the grounding mechanism — no separate fetch step
is required and no API key is needed (authentication uses ADC).
"""

import logging
from typing import Any

from aieng.agent_evals.configs import Configs
from google.adk.tools.function_tool import FunctionTool
from google.genai import Client, types


logger = logging.getLogger(__name__)


def _parse_project_from_datastore_id(datastore_id: str) -> str | None:
    """Parse GCP project ID from a Vertex AI Search data store resource name.

    Parameters
    ----------
    datastore_id : str
        Full resource name, e.g.
        ``projects/my-project/locations/global/collections/default_collection/dataStores/my-store``.

    Returns
    -------
    str or None
        The project ID, or None if the resource name is not in the expected format.
    """
    parts = datastore_id.split("/")
    if len(parts) >= 2 and parts[0] == "projects":
        return parts[1]
    return None


def _extract_datastore_sources(response: Any) -> list[dict[str, str]]:
    """Extract grounding sources from a Vertex AI Search grounded response.

    Vertex AI Search returns ``retrieved_context`` chunks (not ``web`` chunks).
    Each chunk has a ``uri`` (GCS path or document resource name) and an
    optional ``title``.

    Parameters
    ----------
    response : Any
        The Gemini API response object from a Vertex AI Search grounded call.

    Returns
    -------
    list[dict[str, str]]
        List of source dictionaries with ``'title'`` and ``'uri'`` keys.
        Sources with an empty URI are excluded.
    """
    sources: list[dict[str, str]] = []
    if not response.candidates:
        return sources

    gm = getattr(response.candidates[0], "grounding_metadata", None)
    if not gm or not hasattr(gm, "grounding_chunks") or not gm.grounding_chunks:
        return sources

    for chunk in gm.grounding_chunks:
        rc = getattr(chunk, "retrieved_context", None)
        if rc:
            # Vertex AI Search returns 'document_name' (full resource path), not 'uri'
            uri = getattr(rc, "document_name", "") or ""
            title = getattr(rc, "title", "") or ""
            if uri:
                sources.append({"title": title, "uri": uri})

    return sources


async def _vertex_search_async(
    query: str,
    model: str,
    datastore_id: str,
    location: str,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Query a Vertex AI Search data store with grounding enabled.

    Parameters
    ----------
    query : str
        The search query.
    model : str
        The Gemini model to use (accessed via the Vertex AI endpoint).
    datastore_id : str
        Full resource name of the Vertex AI Search data store.
    location : str
        GCP region for the Vertex AI model call (e.g. ``'us-central1'``).
        This is the *compute* region and may differ from the data store's
        ``global`` location.
    temperature : float, default=1.0
        Temperature for generation.

    Returns
    -------
    dict
        Search results with the following keys:

        - **status** (str): ``"success"`` or ``"error"``
        - **summary** (str): Grounded text answer drawn from the data store
        - **sources** (list[dict]): Each entry has:
            - **title** (str): Document title
            - **uri** (str): GCS path or Vertex AI document resource name
        - **source_count** (int): Number of sources cited (success case only)
        - **error** (str): Error message (error case only)
    """
    project = _parse_project_from_datastore_id(datastore_id)
    client = Client(vertexai=True, project=project, location=location)
    try:
        response = client.models.generate_content(
            model=model,
            contents=query,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(retrieval=types.Retrieval(vertex_ai_search=types.VertexAISearch(datastore=datastore_id)))
                ],
                temperature=temperature,
            ),
        )

        summary = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    summary += part.text

        sources = _extract_datastore_sources(response)
        return {
            "status": "success",
            "summary": summary,
            "sources": sources,
            "source_count": len(sources),
        }

    except Exception as e:
        logger.exception("Vertex AI Search failed: %s", e)
        return {
            "status": "error",
            "error": str(e),
            "summary": "",
            "sources": [],
        }
    finally:
        client.close()


async def vertex_search(query: str, model: str | None = None) -> dict[str, Any]:
    """Search the custom knowledge base and return grounded results with citations.

    Use this tool to find information from internal documents and knowledge bases.
    Results are grounded directly from retrieved document content — the summary
    is more reliable than web search snippets and no separate fetch step is needed.

    Authentication uses Application Default Credentials (ADC) — no API key is
    required. On GCE/Coder workspaces the attached service account is used
    automatically.

    Parameters
    ----------
    query : str
        The search query. Be specific and include key terms.
    model : str, optional
        The Gemini model to use. Defaults to ``config.default_worker_model``.

    Returns
    -------
    dict
        Search results with the following keys:

        - **status** (str): ``"success"`` or ``"error"``
        - **summary** (str): Grounded answer from the knowledge base
        - **sources** (list[dict]): Each with ``'title'`` and ``'uri'``
        - **source_count** (int): Number of sources cited (success case only)
        - **error** (str): Error message (error case only)

    Raises
    ------
    ValueError
        If ``VERTEX_AI_DATASTORE_ID`` is not set in config.

    Examples
    --------
    >>> result = await vertex_search("What is the company leave policy?")
    >>> print(result["summary"])
    >>> for source in result["sources"]:
    ...     print(f"{source['title']}: {source['uri']}")
    """
    config = Configs()  # type: ignore[call-arg]
    if not config.vertex_datastore_id:
        raise ValueError(
            "VERTEX_AI_DATASTORE_ID must be set to use vertex_search. "
            "Set it in your .env file or as an environment variable."
        )
    if model is None:
        model = config.default_worker_model

    return await _vertex_search_async(
        query,
        model=model,
        datastore_id=config.vertex_datastore_id,
        location=config.google_cloud_location,
        temperature=config.default_temperature,
    )


def create_vertex_search_tool(config: Configs | None = None) -> FunctionTool:
    """Create a search tool backed by a custom Vertex AI Search data store.

    Authentication uses Application Default Credentials (ADC) — no API key is
    needed. On GCE/Coder workspaces the attached service account handles auth
    automatically.

    Parameters
    ----------
    config : Configs, optional
        Configuration settings. If not provided, creates default config.
        Must have ``vertex_datastore_id`` set.

    Returns
    -------
    FunctionTool
        An ADK-compatible tool that returns grounded summaries with citations.

    Raises
    ------
    ValueError
        If ``VERTEX_AI_DATASTORE_ID`` is not set in config.

    Examples
    --------
    >>> from aieng.agent_evals.tools import create_vertex_search_tool
    >>> tool = create_vertex_search_tool()
    >>> agent = Agent(tools=[tool])
    """
    if config is None:
        config = Configs()  # type: ignore[call-arg]

    if not config.vertex_datastore_id:
        raise ValueError(
            "VERTEX_AI_DATASTORE_ID must be set to use create_vertex_search_tool. "
            "Set it in your .env file or as an environment variable."
        )

    return FunctionTool(func=vertex_search)

"""Reusable tools for ADK agents.

This package provides modular tools for:
- Google Search (search.py)
- Vertex AI Search / custom knowledge base (vertex_search.py)
- Web content fetching - HTML and PDF (web.py)
- File downloading and searching - CSV, XLSX, text (file.py)
- SQL Database access (sql_database.py)

"""

from .file import (
    create_fetch_file_tool,
    create_grep_file_tool,
    create_read_file_tool,
    fetch_file,
    grep_file,
    read_file,
)
from .search import (
    GroundedResponse,
    GroundingChunk,
    create_google_search_tool,
    format_response_with_citations,
    google_search,
)
from .sql_database import ReadOnlySqlDatabase, ReadOnlySqlPolicy
from .vertex_search import create_vertex_search_tool, vertex_search
from .web import (
    create_web_fetch_tool,
    web_fetch,
)


__all__ = [
    # Google Search tools
    "create_google_search_tool",
    "google_search",
    "format_response_with_citations",
    "GroundedResponse",
    "GroundingChunk",
    # Vertex AI Search tools (custom knowledge base)
    "create_vertex_search_tool",
    "vertex_search",
    # Web tools (HTML pages and PDFs)
    "web_fetch",
    "create_web_fetch_tool",
    # File tools (data files - CSV, XLSX, text)
    "fetch_file",
    "grep_file",
    "read_file",
    "create_fetch_file_tool",
    "create_grep_file_tool",
    "create_read_file_tool",
    # SQL Database tools
    "ReadOnlySqlDatabase",
    "ReadOnlySqlPolicy",
]

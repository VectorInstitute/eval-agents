"""Reusable tools for ADK agents.

This package provides modular tools for:
- Google Search (search.py)
- Web content fetching - HTML and PDF (web.py)
- File downloading and searching - CSV, XLSX, text (file.py)

Tool Selection Guide:
- web_fetch(url): HTML pages and PDFs - returns content for agent to analyze
- fetch_file(url) + grep_file + read_file: Data files (CSV, XLSX) - download and search
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
)
from .web import create_web_fetch_tool, web_fetch


__all__ = [
    # Search tools
    "create_google_search_tool",
    "format_response_with_citations",
    "GroundedResponse",
    "GroundingChunk",
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
]

"""Reusable tools for ADK agents.

This package provides modular tools for:
- Web content fetching (web.py)
- File searching and reading (file.py)
- PDF document reading (pdf.py)
"""

from .file import create_grep_file_tool, create_read_file_tool, grep_file, read_file
from .pdf import create_read_pdf_tool, read_pdf
from .web import create_fetch_url_tool, fetch_url


__all__ = [
    # Web tools
    "fetch_url",
    "create_fetch_url_tool",
    # File tools
    "grep_file",
    "read_file",
    "create_grep_file_tool",
    "create_read_file_tool",
    # PDF tools
    "read_pdf",
    "create_read_pdf_tool",
]

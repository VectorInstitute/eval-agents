"""PDF reading tool for extracting text from local PDF files.

Provides the read_pdf tool which reads a local PDF file and extracts
its text content page by page using pypdf.
"""

import logging
import os
from typing import Any

from google.adk.tools.function_tool import FunctionTool
from pypdf import PdfReader


logger = logging.getLogger(__name__)

# Default maximum pages to extract from a PDF
MAX_PDF_PAGES = 50


def read_pdf(file_path: str, max_pages: int = MAX_PDF_PAGES) -> dict[str, Any]:
    """Read a local PDF file and extract its text content.

    Extracts text from each page of the PDF, returning the full text
    with page markers for easy reference.

    Parameters
    ----------
    file_path : str
        Absolute path to the local PDF file.
    max_pages : int, optional
        Maximum number of pages to extract (default 50).

    Returns
    -------
    dict
        On success: 'status', 'content', 'num_pages', 'pages_extracted'.
        On error: 'status', 'error'.

    Examples
    --------
    >>> result = read_pdf("/path/to/document.pdf")
    >>> print(result["content"])
    """
    if not file_path:
        return {"status": "error", "error": "file_path is required."}

    if file_path.startswith(("http://", "https://", "ftp://")):
        return {
            "status": "error",
            "error": "read_pdf only works with local file paths, not URLs.",
        }

    if not os.path.exists(file_path):
        return {
            "status": "error",
            "error": f"File not found: {file_path}",
        }

    if not file_path.lower().endswith(".pdf"):
        return {
            "status": "error",
            "error": f"Not a PDF file: {file_path}",
        }

    try:
        reader = PdfReader(file_path)
        num_pages = len(reader.pages)
        pages_to_read = min(num_pages, max_pages)

        text_parts = []
        for i in range(pages_to_read):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text_parts.append(f"--- Page {i + 1} ---\n{page_text}")

        if pages_to_read < num_pages:
            text_parts.append(
                f"\n[Document has {num_pages} pages. Showing first {pages_to_read}.]"
            )

        content = "\n\n".join(text_parts)

        return {
            "status": "success",
            "content": content,
            "num_pages": num_pages,
            "pages_extracted": pages_to_read,
        }

    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        return {
            "status": "error",
            "error": f"Failed to read PDF: {e!s}",
        }


def create_read_pdf_tool() -> FunctionTool:
    """Create an ADK FunctionTool for reading local PDF files."""
    return FunctionTool(func=read_pdf)

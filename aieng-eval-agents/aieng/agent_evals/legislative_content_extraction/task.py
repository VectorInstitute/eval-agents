"""Task function for legislative content extraction experiment execution."""

import json
import logging
from collections.abc import Mapping
from typing import Any

from aieng.agent_evals.legislative_content_extraction import LegislativeContentExtractionAgent
from langfuse.experiment import ExperimentItem

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = "Extract legislative metadata from this document."


class LegislativeExtractionTask:
    """Langfuse-compatible task wrapper for legislative content extraction.

    Implements ``__call__(*, item, **kwargs)`` as required by the
    evaluation harness.

    Parameters
    ----------
    agent : LegislativeContentExtractionAgent | None, optional
        Pre-configured agent. If ``None``, a default instance is created.
    files_dir : str | None, optional
        Local directory where PDF files are cached.
    """

    def __init__(
        self,
        *,
        agent: LegislativeContentExtractionAgent | None = None,
        files_dir: str | None = None,
    ) -> None:
        self._agent = agent or LegislativeContentExtractionAgent(files_dir=files_dir)

    async def __call__(self, *, item: ExperimentItem, **kwargs: Any) -> dict[str, Any] | None:
        """Run extraction on one dataset item and return a parsed dict.

        Parameters
        ----------
        item : ExperimentItem
            Langfuse dataset item with ``input`` containing ``pdf_path``,
            optional ``html_page_link``, and optional ``prompt``.

        Returns
        -------
        dict[str, Any] | None
            Parsed extracted fields, or ``None`` if extraction failed.
        """
        item_input = item["input"] if isinstance(item, Mapping) else item.input

        pdf_path = item_input.get("pdf_path", "")
        html_page_link = item_input.get("html_page_link", "")
        prompt = item_input.get("prompt", EXTRACTION_PROMPT)

        if not pdf_path:
            logger.warning("No pdf_path in item input, skipping.")
            return None

        try:
            response = await self._agent.answer_async(
                pdf_path=pdf_path,
                prompt=prompt,
                html_page_link=html_page_link,
            )
        except Exception as exc:
            logger.error("Agent failed for pdf_path=%s: %s", pdf_path, exc)
            return None

        raw_text = (response.text or "").strip()
        if not raw_text:
            logger.warning("Empty response for pdf_path=%s", pdf_path)
            return None

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON response for pdf_path=%s: %s", pdf_path, raw_text[:200])
            return None
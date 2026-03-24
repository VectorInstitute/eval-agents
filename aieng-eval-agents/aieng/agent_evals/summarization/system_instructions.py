"""System instructions for the Summarization Agent.

This module contains the system prompt and builder function
for the financial news summarization agent.
"""


SYSTEM_INSTRUCTIONS = """\
Given a news article title and body text, produce a concise and accurate summary.

Your summary must:
- Be 2-4 sentences long
- Capture the main event or announcement
- Include key companies or people involved
- Mention any significant financial figures or outcomes if present
- Be based only on the provided article content — do not add outside information

Return only the summary text. Do not include headings, labels, or preamble.
"""


def build_system_instructions() -> str:
    """Build system instructions for the summarization agent.

    Returns
    -------
    str
        The complete system instructions string.
    """
    return SYSTEM_INSTRUCTIONS

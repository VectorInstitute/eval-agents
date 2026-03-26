# Legislative Content Extraction Agent

An agent that reads US legislative PDF documents and extracts structured metadata as JSON,
built on Google ADK with Gemini.

## Overview

The `LegislativeContentExtractionAgent` takes a PDF file path and an extraction prompt as input. It reads the
PDF using the `read_pdf` tool, auto-detects the jurisdiction (US state) from the document
content, and returns structured legislative metadata as a JSON object.

## Architecture

```
User Input (pdf_path, prompt, html_page_link)
    │
    ▼
┌──────────────────────────────────────────┐
│  LegislativeContentExtractionAgent       │
│  (Google ADK + Gemini)                   │
│                                          │
│  1. Receives input                       │
│  2. Calls read_pdf             ◄── pypdf │
│  3. Analyzes text                        │
│  4. Calls fetch_html_page                │
│  5. Validates JSON with validate_json    │
│  6. Returns JSON                         │
└──────────────────────────────────────────┘
    │
    ▼
AgentResponse (text, reasoning_chain, tool_calls, total_duration_ms)
```

## Extracted Fields

The agent extracts the following metadata from legislative PDFs:

| Field | Type | Description |
|---|---|---|
| `jurisdiction_code` | `string` | Two-letter US state code, auto-detected from document content (e.g. `"ID"`, `"MI"`) |
| `session_code` | `string` | Standardized session code (e.g. `"ID_2025_2025_R1"`, `"WI_2025_2026_R1"`) |
| `chamber_code` | `string` | `HOUSE` or `SENATE` |
| `measure_type_code` | `string` | `BILL`, `CONCURRENT_RESOLUTION`, `JOINT_RESOLUTION`, `JOINT_MEMORIAL`, or `CONCURRENT_MEMORIAL` |
| `measure_number` | `integer` | Integer portion of the measure number (e.g. `4`, `1234`) |
| `title` | `string` | Official title or short description |
| `summary` | `string` | Concise summary of what the measure does |
| `sponsors` | `list[string]` | List of sponsors (legislators or committees) |
| `is_adopted_into_law` | `boolean` | Whether the measure was signed/enacted into law |
| `sections_affected` | `list[object]` | List of objects with `raw_section` and `action` (e.g. `AMEND`, `ADD`, `REPEAL`) |

Fields that cannot be determined from the PDF and HTML content are set to `null`.

## Package Structure

```
legislative_content_extraction/
├── __init__.py                # Package exports
├── agent.py                   # LegislativeContentExtractionAgent (Google ADK)
├── system_instructions.py     # Shared system prompt
├── README.md
└── tools/
    ├── __init__.py
    └── doc_extraction_tools.py  # read_pdf & fetch_html_page tools
```

## Usage

### Prerequisites

Set `GOOGLE_API_KEY` or `GEMINI_API_KEY` in your `.env` file.

### Async

```python
from aieng.agent_evals.legislative_content_extraction import LegislativeContentExtractionAgent

agent = LegislativeContentExtractionAgent()

response = await agent.answer_async(
    pdf_path="/path/to/document.pdf",
    prompt="Extract legislative metadata from this bill.",
)

print(response.text)           # JSON string
print(response.tool_calls)     # Tool calls made
print(response.total_duration_ms)
```

### Sync

```python
response = agent.answer(
    pdf_path="/path/to/document.pdf",
    prompt="Extract legislative metadata from this bill.",
)
```

### Configuration

```python
from aieng.agent_evals.configs import Configs

# Use a specific model
agent = LegislativeContentExtractionAgent(model="gemini-2.5-pro")

# Adjust thinking budget
agent = LegislativeContentExtractionAgent(thinking_budget=4096)

# Pass custom config
config = Configs()
agent = LegislativeContentExtractionAgent(config=config)

# With Langfuse scoring callback (see Langfuse Integration below)
agent = LegislativeContentExtractionAgent(after_agent_callback=calculate_and_send_scores)
```

## Example Output

```json
{
  "jurisdiction_code": "ID",
  "session_code": "ID_2025_2025_R1",
  "chamber_code": "HOUSE",
  "measure_type_code": "BILL",
  "measure_number": 4,
  "title": "AN ACT RELATING TO INFORMATION TECHNOLOGY SERVICES AND CYBERSECURITY",
  "summary": "This bill amends Idaho Code sections 67-827A and 67-831 and adds a new section 67-2362 to enhance cybersecurity practices across state government...",
  "sponsors": ["STATE AFFAIRS COMMITTEE"],
  "is_adopted_into_law": false,
  "sections_affected": [
    {"raw_section": "Section 67-827A, Idaho Code", "action": "AMEND"},
    {"raw_section": "Section 67-831, Idaho Code", "action": "AMEND"},
    {"raw_section": "Section 67-2362, Idaho Code", "action": "ADD"}
  ]
}
```

## Tools

### `read_pdf`

Reads a local PDF file and extracts text content page by page using `pypdf`.

- **Input**: `file_path` (absolute path to PDF), `max_pages` (default 50)
- **Output**: `{status, content, num_pages, pages_extracted}` on success; `{status, error}` on failure
- Validates that the path is local (not a URL), the file exists, and has a `.pdf` extension

### `fetch_html_page`

Fetches an HTML page and extracts its text content. Always called to retrieve supplementary data from the legislative web page.

- **Input**: `url` (URL of the legislative HTML page)
- **Output**: `{status, content, url}` on success; `{status, error}` on failure
- Caches fetched HTML to `content_cache_dir` to avoid repeated downloads

## Langfuse Integration

The agent supports Langfuse tracing and scoring via an `after_agent_callback` parameter.

- **Tracing**: Initialized automatically via `init_tracing()` in the agent constructor using OpenTelemetry + OpenInference instrumentation for Google ADK.
- **Usage scoring**: Pass a callback (e.g. `calculate_and_send_scores`) to report token usage and latency scores to Langfuse after each run.
- **User feedback**: The demo UI collects feedback (Full Success = 1, Partial Success = 0.5, Fail = 0) and sends it as a Langfuse score tied to the trace.
- **Feedback visibility**: Feedback buttons appear after the agent produces output and are replaced by a confirmation message once submitted.

## Error Handling

- **API rate limits**: Automatic retry with exponential backoff and jitter (via `tenacity`)
- **Empty responses**: Retries with a fresh session (up to 3 attempts)
- **Context overflow**: Detects overflow errors and retries with a new session
- **Token tracking**: Built-in `TokenTracker` for monitoring context usage

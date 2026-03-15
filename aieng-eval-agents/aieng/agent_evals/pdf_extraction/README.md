# PDF Extraction Agent

An agent that reads US legislative PDF documents and extracts structured metadata as JSON,
built on Google ADK with Gemini.

## Overview

The `PdfExtractionAgent` takes a PDF file path, a jurisdiction (US state), and an extraction
prompt as input. It reads the PDF using the `read_pdf` tool, analyzes the text in the context
of the given jurisdiction, and returns structured legislative metadata as a JSON object.

## Architecture

```
User Input (pdf_path, jurisdiction, prompt)
    │
    ▼
┌──────────────────────┐
│  PdfExtractionAgent   │
│  (Google ADK + Gemini)│
│                       │
│  1. Receives input    │
│  2. Calls read_pdf    │◄── tools/pdf.py (pypdf)
│  3. Analyzes text     │
│  4. Returns JSON      │
└──────────────────────┘
    │
    ▼
AgentResponse (text, reasoning_chain, tool_calls, total_duration_ms)
```

## Extracted Fields

The agent extracts the following metadata from legislative PDFs:

| Field | Type | Description |
|---|---|---|
| `jurisdiction_code` | `string` | Two-letter US state code (e.g. `"ID"`, `"MI"`) |
| `session` | `string` | Legislative session (e.g. `"68th Legislature, First Regular Session - 2025"`) |
| `chamber_code` | `string` | `HOUSE` or `SENATE` |
| `measure_type_code` | `string` | `BILL`, `CONCURRENT_RESOLUTION`, `JOINT_RESOLUTION`, `JOINT_MEMORIAL`, or `CONCURRENT_MEMORIAL` |
| `measure_number` | `string` | Measure number (e.g. `"H0004"`, `"S1234"`) |
| `title` | `string` | Official title or short description |
| `summary` | `string` | Concise summary of what the measure does |
| `sponsors` | `list[string]` | List of sponsors (legislators or committees) |
| `sections_affected` | `list[object]` | List of objects with `raw_section` (e.g. `"Section 67-827A, Idaho Code"`) |

Fields that cannot be determined from the PDF content are set to `null`.

## Package Structure

```
pdf_extraction/
├── __init__.py                # Package exports
├── agent.py                   # PdfExtractionAgent (Google ADK)
├── system_instructions.py     # Shared system prompt
├── README.md
└── tools/
    ├── __init__.py
    └── pdf.py                 # read_pdf tool (pypdf)
```

## Usage

### Prerequisites

Set `GOOGLE_API_KEY` or `GEMINI_API_KEY` in your `.env` file.

### Async

```python
from aieng.agent_evals.pdf_extraction import PdfExtractionAgent

agent = PdfExtractionAgent()

response = await agent.answer_async(
    pdf_path="/path/to/document.pdf",
    jurisdiction="Idaho",
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
    jurisdiction="Idaho",
    prompt="Extract legislative metadata from this bill.",
)
```

### Configuration

```python
from aieng.agent_evals.configs import Configs

# Use a specific model
agent = PdfExtractionAgent(model="gemini-2.5-pro")

# Adjust thinking budget
agent = PdfExtractionAgent(thinking_budget=4096)

# Pass custom config
config = Configs()
agent = PdfExtractionAgent(config=config)
```

## Example Output

```json
{
  "jurisdiction_code": "ID",
  "session": "Sixty-eighth Legislature, First Regular Session - 2025",
  "chamber_code": "HOUSE",
  "measure_type_code": "BILL",
  "measure_number": "H0004",
  "title": "AN ACT RELATING TO INFORMATION TECHNOLOGY SERVICES AND CYBERSECURITY",
  "summary": "This bill amends Idaho Code sections 67-827A and 67-831 and adds a new section 67-2362 to enhance cybersecurity practices across state government...",
  "sponsors": ["STATE AFFAIRS COMMITTEE"],
  "sections_affected": [
    {"raw_section": "Section 67-827A, Idaho Code"},
    {"raw_section": "Section 67-831, Idaho Code"},
    {"raw_section": "Section 67-2362, Idaho Code"}
  ]
}
```

## Tools

### `read_pdf`

Reads a local PDF file and extracts text content page by page using `pypdf`.

- **Input**: `file_path` (absolute path to PDF), `max_pages` (default 50)
- **Output**: `{status, content, num_pages, pages_extracted}` on success; `{status, error}` on failure
- Validates that the path is local (not a URL), the file exists, and has a `.pdf` extension

## Error Handling

- **API rate limits**: Automatic retry with exponential backoff and jitter (via `tenacity`)
- **Empty responses**: Retries with a fresh session (up to 3 attempts)
- **Context overflow**: Detects overflow errors and retries with a new session
- **Token tracking**: Built-in `TokenTracker` for monitoring context usage

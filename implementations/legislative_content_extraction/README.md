# Legislative Content Extraction (Google ADK)

Evaluation implementation for the legislative content extraction agent using Google ADK.

## Directory Layout

The agent spans two directory trees: `implementations/` (data, notebooks, scripts) and `aieng-eval-agents/` (agent source, graders, tools).

### `implementations/legislative_content_extraction/`

```
legislative_content_extraction/
  data/
    langfuse_upload.py                  # Transform raw dataset + upload to Langfuse
    legislative_eval_dataset.json       # Transformed eval dataset (generated)
  files/                                # PDF + HTML files organized as files/{record_id}/
    ID_H0004/
      ID_H0004.pdf
      ID_H0004.html
    ...                                 # 26 record directories (DE, ID, MI, MN, WI)
  rubrics/
    legislative_summary_quality.md      # LLM-as-judge rubric for summary quality
  legislative_content_extraction_dataset.json   # Raw ground-truth dataset (26 records)
  evaluate.py                           # CLI for running evaluations against Langfuse
  demo.py                               # Quick demo script
  fetch_htmls.py                        # Download HTML pages for all records
  fetch_pdfs.py                         # Download PDF files for all records
  01_legislative_content_extraction_agent.ipynb  # Agent demo notebook
  02_langfuse_dataset_upload.ipynb              # Dataset upload notebook
  03_evaluation.ipynb                           # Evaluation notebook
```

### `aieng-eval-agents/.../legislative_content_extraction/`

```
aieng/agent_evals/legislative_content_extraction/
  graders/
    _common.py          # Shared helpers: normalize_str, normalize_sponsors, normalize_sections
    item.py             # Item-level deterministic grader (exact match + precision/recall)
    run.py              # Run-level aggregation
    trace.py            # Trace-level grading
  tools/
    doc_extraction_tools.py   # read_pdf(), fetch_html_page()
    json_validator.py         # validate_json()
  evaluation/
    exact_match_grader.py     # Exact-match evaluation logic
    offline.py                # Offline evaluation runner
  agent.py              # LegislativeContentExtractionAgent class
  system_instructions.py  # PDF_SYSTEM_INSTRUCTIONS prompt
  task.py               # Task function for Langfuse eval runs
```

## Ground-Truth Dataset

The raw dataset (`legislative_content_extraction_dataset.json`) contains 26 records across 5 jurisdictions:

| Jurisdiction | Code | Records |
|--------------|------|---------|
| Delaware     | DE   | 5       |
| Idaho        | ID   | 6       |
| Michigan     | MI   | 5       |
| Minnesota    | MN   | 5       |
| Wisconsin    | WI   | 5       |

### Fields per record

| Field | Type | Description |
|-------|------|-------------|
| `record_id` | string | Unique identifier (e.g. `ID_H0004`) |
| `pdf_file_link` | string | Source URL for the PDF |
| `pdf_file_name` | string | Local PDF filename |
| `html_page_link` | string | Legislative web page URL |
| `jurisdiction_code` | string | Two-letter state code |
| `session_code` | string | Standardized session identifier |
| `chamber_code` | string | `HOUSE` or `SENATE` |
| `measure_type_code` | string | `BILL`, `JOINT_RESOLUTION`, `JOINT_MEMORIAL`, etc. |
| `measure_number` | string | Integer portion of the measure number |
| `title` | string | Official title |
| `summary` | string | Concise summary of the measure |
| `sponsors` | list[string] | Sponsors (legislators or committees) |
| `is_adopted_into_law` | boolean | Whether the measure was signed/enacted into law |
| `sections_affected` | list[object] | Sections with `raw_section`, `action`, and optional code details |

### Adopted into law

3 of the 26 records have `is_adopted_into_law: true`:

| Record | Evidence |
|--------|----------|
| `ID_H0504` | Signed by Governor 02/26/2026 (Session Law Chapter 6) |
| `ID_HJM010` | Adopted by both chambers, delivered to Secretary of State |
| `ID_S1331` | Signed by Governor 03/16/2026 (Session Law Chapter 45) |

## Extracted Fields & Evaluation Metrics

The agent extracts 10 fields. The deterministic grader (`graders/item.py`) evaluates them as:

| Metric | Type | Fields |
|--------|------|--------|
| Exact match (0/1) | Scalar | `jurisdiction_code`, `session_code`, `chamber_code`, `measure_type_code`, `measure_number`, `is_adopted_into_law` |
| Precision/Recall | Set-based | `sponsors`, `sections_affected` |
| LLM-as-judge | Rubric | `title`, `summary` |

When `is_adopted_into_law` is missing/null in the expected output, the grader returns 1.0 (automatic pass).

## Dataset Transform & Upload

Transform the raw dataset into Langfuse evaluation format and upload:

```bash
# Transform + upload (default)
uv run implementations/legislative_content_extraction/data/langfuse_upload.py

# Transform only, no upload
uv run implementations/legislative_content_extraction/data/langfuse_upload.py \
    --no-upload

# Upload only (skip transform, use existing eval dataset)
uv run implementations/legislative_content_extraction/data/langfuse_upload.py \
    --no-transform
```

### Jurisdiction-Specific Datasets

Use the `--jurisdiction` flag to filter to a single jurisdiction. This appends the jurisdiction code to the Langfuse dataset name (e.g. `legislative-docs-ID`).

```bash
# Transform + upload only Idaho records
uv run implementations/legislative_content_extraction/data/langfuse_upload.py \
    --jurisdiction ID

# Transform only Wisconsin records, no upload
uv run implementations/legislative_content_extraction/data/langfuse_upload.py \
    --jurisdiction WI --no-upload
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | `implementations/.../legislative_content_extraction_dataset.json` | Raw dataset path |
| `--output` | `implementations/.../data/legislative_eval_dataset.json` | Output eval dataset path |
| `--dataset-name` | `legislative-docs` | Langfuse dataset name |
| `--jurisdiction` | None | Filter to a single jurisdiction code |
| `--transform/--no-transform` | `--transform` | Whether to run the transform step |
| `--upload/--no-upload` | `--upload` | Whether to upload to Langfuse |

### Transform Logic

For each raw record, the transform produces:

- **`input`**: `record_id`, `pdf_file_name`, `html_page_link`, `prompt`
- **`expected_output`**: `jurisdiction_code`, `session_code`, `chamber_code`, `measure_type_code`, `measure_number`, `title`, `summary`, `sponsors`, `is_adopted_into_law`, `sections_affected` (with `code_title`/`code_chapter`/`code_section`/`code_subdivision` stripped)
- **`id`**: Set to `record_id` (used as stable Langfuse item ID for upserts)

### Langfuse Upsert Behavior

The legislative upload uses `langfuse_upsert.py` which generates item IDs from the record's `id` field rather than a content hash. This means re-uploading after changing `expected_output` fields (e.g. adding `is_adopted_into_law`) **updates** existing items instead of creating duplicates.

## Agent

The agent (`LegislativeContentExtractionAgent`) is defined in `aieng-eval-agents/.../agent.py`. It accepts:

- `pdf_path` — resolved by the task function from `record_id` + `pdf_file_name`
- `prompt` — extraction prompt
- `html_page_link` — optional supplementary HTML page URL

### System Instructions

The system prompt (`system_instructions.py`) instructs the agent to:

1. Read the PDF using `read_pdf`
2. Determine the jurisdiction from document content
3. Extract all required metadata fields
4. Always fetch the HTML page for supplementary data
5. Return a valid JSON object (validated with `validate_json` tool)

Includes Gemini-specific constraints to ensure raw JSON output without markdown fences or conversational text.

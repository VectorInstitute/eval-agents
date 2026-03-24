# Legislative Content Extraction (Google ADK)

Evaluation implementation for the legislative content extraction agent using Google ADK.

## Directory Layout

```
legislative_content_extraction/
  data/
    langfuse_upload.py                  # Transform + upload to Langfuse
    legislative_eval_dataset.json       # Transformed eval dataset (generated)
  files/                                # PDF/HTML files organized as files/{record_id}/
  legislative_content_extraction_dataset.json   # Raw dataset (26 records)
  demo.py
  fetch_htmls.py
  fetch_pdfs.py
  01_legislative_content_extraction_agent.ipynb
```

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

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | `implementations/legislative_content_extraction/legislative_content_extraction_dataset.json` | Raw dataset path |
| `--output` | `implementations/legislative_content_extraction/data/legislative_eval_dataset.json` | Output eval dataset path |
| `--dataset-name` | `legislative-content-extraction` | Langfuse dataset name |
| `--transform/--no-transform` | `--transform` | Whether to transform the raw dataset |
| `--upload/--no-upload` | `--upload` | Whether to upload to Langfuse |

### Transform Logic

For each of the 26 raw records, the transform produces:

- **`input`**: `record_id`, `pdf_file_name`, `html_page_link`, `prompt`
- **`expected_output`**: All metadata fields (`jurisdiction_code`, `session_code`, `chamber_code`, `measure_type_code`, `measure_number`, `title`, `summary`, `sponsors`, `sections_affected`) with `code_title`, `code_chapter`, `code_section`, and `code_subdivision` stripped from each `sections_affected` entry
- **`id`**: Set to `record_id`

### Agent

The agent (`LegislativeContentExtractionAgent`) is defined in `aieng-eval-agents/aieng/agent_evals/legislative_content_extraction/agent.py`. It accepts:

- `pdf_path` — resolved by the task function from `record_id` + `pdf_file_name`
- `prompt` — extraction prompt
- `html_page_link` — optional supplementary HTML page URL

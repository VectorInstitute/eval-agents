# Report Generation Agent

## Dataset

https://archive.ics.uci.edu/dataset/352/online+retail

To import it into weaviate:

```bash
uv run --env-file .env python -m aieng.agent_evals.report_generation.data_import
```

## Running

```bash
uv run --env-file .env python -m aieng.agent_evals.report_generation.main
```

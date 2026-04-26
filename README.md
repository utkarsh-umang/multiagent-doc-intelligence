# multiagent-doc-intelligence

Scaffolded project structure:

- `agents/`: agent implementations (extract/validate/audit)
- `pipeline/`: graph + shared state schema
- `storage/`: DuckDB + query layer
- `ui/`: Streamlit app
- `rules/`: customer-specific rules
- `samples/`: sample documents (placeholders)
- `docs/`: product + technical docs (placeholders)

## Run With Docker

Create/update `.env` with your model and tracing keys:

```bash
OPENAI_API_KEY=...
LITELLM_VISION_MODEL=gpt-4o
ANTHROPIC_API_KEY=...
VALIDATOR_MODEL=anthropic/claude-3-5-sonnet-20241022
AUDITOR_MODEL=gpt-4o
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Then run:

```bash
docker compose up --build
```

Open the Streamlit app at:

```text
http://localhost:8501
```

DuckDB data is persisted under `./data/app.duckdb` by Docker Compose.


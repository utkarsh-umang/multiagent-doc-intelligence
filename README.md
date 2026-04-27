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

Copy `env.example` to `.env`, then set your model + tracing keys:

```bash
cp env.example .env
```

Edit `.env` and fill in at least `OPENAI_API_KEY` (and/or `ANTHROPIC_API_KEY`, `GOOGLE_AI_API_KEY`).
Optional: override models via `LITELLM_VISION_MODEL`, `VALIDATOR_MODEL`, `AUDITOR_MODEL`, and configure Langfuse via `LANGFUSE_*`.

Then run:

```bash
docker compose up --build
```

Open the Streamlit app at:

```text
http://localhost:8501
```

DuckDB data is persisted under `./data/app.duckdb` by Docker Compose.


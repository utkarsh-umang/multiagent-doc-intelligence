# Nova — Multi-Agent Trade Document Intelligence

A two-phase multi-agent system that extracts, validates, and routes trade documents (Bills of Lading, Commercial Invoices, Packing Lists), then wires the pipeline into a real CG validation workflow.

---

## Project structure

```
agents/          Extractor, Validator, Auditor agents
pipeline/        LangGraph graph, state schema, cross-document validator
storage/         DuckDB store + natural-language query layer
rules/           Customer rule sets, master port/HS data, compliance rules
inbox/           Phase 2 inbox watcher, bundle processor, sample bundles
  samples/       Pre-built sample email bundle (BOL + Invoice + Packing List)
ui/
  app.py         Phase 1 — single-document pipeline UI
  pages/
    2_CG_Dashboard.py   Phase 2 — CG verification dashboard (live pipeline)
samples/         Raw sample trade documents
docs/            PRD, Phase 2 PRD, Technical Writeup
data/            DuckDB database (created on first run, persisted by Docker)
```

---

## Prerequisites

- Docker + Docker Compose **or** Python 3.11+ with a virtual environment
- `OPENAI_API_KEY` — required for the Extractor and Auditor agents (GPT-4o)
- `ANTHROPIC_API_KEY` — required for the Validator agent (Claude)

---

## Quickstart — Docker (recommended)

**1. Configure environment**

```bash
cp env.example .env
```

Open `.env` and fill in at minimum:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**2. Build and start**

```bash
docker compose up --build
```

**3. Open the app**

```
http://localhost:8501
```

DuckDB data is persisted to `./data/app.duckdb` via a Docker volume, so it survives container restarts.

---

## Quickstart — Local (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp env.example .env
# edit .env and fill in API keys

streamlit run ui/app.py
```

---

## Phase 1 — Single-document pipeline (`ui/app.py`)

Accessible at `http://localhost:8501` (the default Streamlit page).

**What it does:**

1. Upload any trade document (PDF, JPG, PNG, WEBP)
2. The LangGraph pipeline runs: **Extractor → Validator → Auditor → DuckDB**
3. Results show extracted fields with per-field confidence, validation status (match / mismatch / uncertain), the agent's decision, and the audit report
4. A draft amendment email is shown if the Auditor decides discrepancies need correction

**Ask stored runs:**

The page includes a natural-language query box backed by DuckDB:

```
how many shipments were flagged this week?
what is the straight-through processing rate this month?
show me all amendment requests today
```

---

## Phase 2 — CG Verification Dashboard (`ui/pages/2_CG_Dashboard.py`)

Accessible at `http://localhost:8501` → **CG Dashboard** in the sidebar.

This page simulates the real CG validation workflow: SU emails documents → agent validates → CG reviews and sends a draft reply.

### Two entry paths

**Simulate Sample**

Runs the live pipeline on a pre-bundled 3-document shipment (no setup beyond API keys):

| File | Document type |
|------|---------------|
| `BOL-SU-2026-001.pdf` | Bill of Lading (`samples/clean_bill_of_lading.pdf`) |
| `INV-SU-2026-001.jpg` | Commercial Invoice (`samples/messy_commercial_invoice.jpg`) |
| `PACK-SU-2026-001.pdf` | Packing List (`samples/packing_list_sample.pdf`) |

Click **▶ Simulate Sample** on the landing screen. The pipeline runs end-to-end and the dashboard transitions through its states automatically.

**Create Custom Email**

Fill in a supplier email form, upload your own trade documents (BOL, Invoice, Packing List), and run the live pipeline against them.

### Dashboard states

```
idle → incoming → [Run Pipeline] → verified → discrepancy detail → draft reply
```

| State | What the CG operator sees |
|-------|--------------------------|
| **Incoming** | Email card with sender, subject, attachments, and the agent pipeline that will run |
| **Verified** | Field-by-field table: status (match / mismatch / uncertain), found vs expected, confidence bar, per-doc source |
| **Discrepancy detail** | Drill-down on one flagged field — found value, expected value, source snippet from the document |
| **Draft reply** | Agent-drafted amendment email, fully editable. CG reviews and clicks **Send to Supplier** — the agent never sends on its own |

### Cross-document consistency check

When a shipment has multiple documents, the pipeline cross-validates shared fields (`consignee_name`, `hs_code`, `invoice_number`) across all of them. Any inconsistency is injected into the validation report as a mismatch or uncertain field before the Auditor routes — preventing the silent cross-document approval failure mode described in the Phase 2 PRD.

### Query stored runs (sidebar)

The **Query Stored Runs** expander in the sidebar is available at any time. Every pipeline run is persisted to DuckDB, so CG can query across all historical runs:

```
show me everything pending review
show me everything pending review for customer ACME
how many shipments were flagged this week?
what is the straight-through processing rate?
```

---

## Phase 2 — Inbox watcher (headless trigger)

The CG Dashboard runs the pipeline on demand. For an always-on trigger that processes bundles automatically, use the inbox watcher:

```bash
# Copy the sample bundle into the pending queue
cp -r inbox/samples/SU-2026-001 inbox/pending/
cp samples/clean_bill_of_lading.pdf    inbox/pending/SU-2026-001/BOL-SU-2026-001.pdf
cp samples/messy_commercial_invoice.jpg inbox/pending/SU-2026-001/INV-SU-2026-001.jpg
cp samples/packing_list_sample.pdf     inbox/pending/SU-2026-001/PACK-SU-2026-001.pdf

# Process the current queue and exit
python inbox/watcher.py --once

# Or watch continuously (polls every 5 seconds until Ctrl-C)
python inbox/watcher.py
```

Results are written to `inbox/results/<bundle-name>.json`. Processed bundles move to `inbox/done/`; failed ones move to `inbox/failed/` with an `error.json`.

See `inbox/samples/README.md` for the full bundle schema and multi-customer setup instructions.

---

## Model defaults

| Agent | Default model | Override env var |
|-------|--------------|-----------------|
| Extractor | `gpt-4o` (vision) | `EXTRACTOR_MODEL` |
| Extractor retry | same as extractor | `EXTRACTOR_RETRY_MODEL` |
| Validator | `anthropic/claude-sonnet-4-6` | `VALIDATOR_MODEL` |
| Auditor | `gpt-4o` | `AUDITOR_MODEL` |
| NL query | falls back to validator model | `QUERY_SQL_MODEL` |

---

## Optional: Langfuse tracing

Add to `.env`:

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

LiteLLM will automatically emit traces for every LLM call via the `langfuse_otel` callback.

---

## Docs

| File | Contents |
|------|----------|
| `docs/PRD.pdf` | Phase 1 product requirements document |
| `docs/Phase 2 PRD.pdf` | Phase 2 CG workflow PRD |
| `docs/Technical Writeup.pdf` | Architecture, failure modes, observability, cost, latency |

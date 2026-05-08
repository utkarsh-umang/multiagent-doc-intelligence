# Inbox Sample Bundles

Sample email bundles you can drop into `inbox/pending/` to test the Phase 2
watcher pipeline without sending real emails.

## What is a bundle?

A bundle is a **directory** inside `inbox/pending/`. Its name becomes the
`shipment_id` in the database. It must contain:

| File | Purpose |
|------|---------|
| `email.json` | Sender metadata, `customer_id`, subject, body |
| `*.pdf` / `*.jpg` / `*.png` | Trade documents (BOL, Invoice, Packing List, …) |

The watcher ignores `email.json` when discovering attachments — it processes
every other supported file as a trade document.

## email.json schema

```json
{
  "from":        "supplier@example.com",
  "to":          "cg@gocomet.com",
  "subject":     "Shipment docs – <ref>",
  "received_at": "2026-05-07T08:31:00+05:30",
  "customer_id": "gocomet_demo_customer",
  "body":        "free-text email body"
}
```

`customer_id` maps to a rule set in `inbox/processor.py`. The built-in value
`"gocomet_demo_customer"` uses the demo rules from `rules/customer_rules.py`.

## Quick start

### 1. Copy a sample bundle and add documents

```bash
# Copy the sample bundle
cp -r inbox/samples/SU-2026-001 inbox/pending/

# Add your own PDFs — the sample contains email.json only
cp samples/clean_bill_of_lading.pdf   inbox/pending/SU-2026-001/BOL-SU-2026-001.pdf
cp samples/messy_commercial_invoice.jpg inbox/pending/SU-2026-001/INV-SU-2026-001.jpg
```

### 2. Run the watcher once (process queue and exit)

```bash
# from the project root with the virtual environment active
python inbox/watcher.py --once
```

Or run continuously (watches until Ctrl-C):

```bash
python inbox/watcher.py
```

### 3. Check the result

```bash
# Result JSON written here after processing
cat inbox/results/SU-2026-001.json | python -m json.tool | head -40
```

If processing fails, the bundle moves to `inbox/failed/SU-2026-001/` and an
`error.json` is written inside it with the full traceback.

## Re-running a bundle

The watcher archives processed bundles to `inbox/done/`. To re-run:

```bash
cp -r inbox/done/SU-2026-001 inbox/pending/
python inbox/watcher.py --once
```

## Adding a second customer

1. Add a new rule set to `rules/customer_rules.py` (copy `CUSTOMER_RULE_SET`
   as a template).
2. Register it in `inbox/processor.py` `_RULE_REGISTRY`:
   ```python
   from rules.customer_rules import MY_CUSTOMER_RULE_SET
   _RULE_REGISTRY["my_customer_id"] = MY_CUSTOMER_RULE_SET
   ```
3. Set `"customer_id": "my_customer_id"` in your `email.json`.

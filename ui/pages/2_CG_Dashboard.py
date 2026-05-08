"""
Phase 2 — CG Verification Dashboard.

Live pipeline integration. Two entry paths:
  1. Simulate Sample  — runs the real pipeline on the pre-bundled sample docs
  2. Create Custom Email — user fills a form and uploads their own documents

Both paths call process_bundle() in-process and display real results across the
4 validation states (incoming → verified → discrepancy detail → draft reply).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st  # type: ignore[import-not-found]

# ── project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from inbox.processor import process_bundle  # noqa: E402
from storage.db import answer_question  # noqa: E402

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nova · CG Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────────────────
_SAMPLES_EMAIL = _PROJECT_ROOT / "inbox" / "samples" / "SU-2026-001" / "email.json"
_SAMPLE_DOCS: list[tuple[Path, str]] = [
    (_PROJECT_ROOT / "samples" / "clean_bill_of_lading.pdf",  "BOL-SU-2026-001.pdf"),
    (_PROJECT_ROOT / "samples" / "messy_commercial_invoice.jpg", "INV-SU-2026-001.jpg"),
    (_PROJECT_ROOT / "samples" / "packing_list_sample.pdf",   "PACK-SU-2026-001.pdf"),
]

_FIELD_LABELS: dict[str, str] = {
    "consignee_name": "Consignee Name",
    "hs_code": "HS Code",
    "port_of_loading": "Port of Loading",
    "port_of_discharge": "Port of Discharge",
    "incoterms": "Incoterms",
    "description_of_goods": "Description of Goods",
    "gross_weight": "Gross Weight",
    "invoice_number": "Invoice Number",
}

_STATUS_BADGE: dict[str, str] = {
    "match": "🟢 Match",
    "mismatch": "🔴 Mismatch",
    "uncertain": "🟠 Uncertain",
}

_OUTCOME_LABEL: dict[str, str] = {
    "auto_approve_and_store": "Auto-approved",
    "flag_for_human_review": "Flagged for review",
    "draft_amendment_request": "Amendment drafted",
}

_OUTCOME_DOT: dict[str, str] = {
    "auto_approve_and_store": "🟢",
    "flag_for_human_review": "🟠",
    "draft_amendment_request": "🔴",
}

_DEFAULT_STATE: dict[str, Any] = {
    "cg_view": "idle",
    "selected_field": None,
    "pipeline_result": None,
    "pending_email": None,
    "pending_attachments": [],  # list of {"name": str, "bytes": bytes} or {"name": str, "path": str}
    "cg_source": None,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _db_path() -> str:
    return os.getenv("NOVA_DB_PATH", str(_PROJECT_ROOT / "app.duckdb"))


def _load_local_env() -> None:
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _reset() -> None:
    for k, v in _DEFAULT_STATE.items():
        st.session_state[k] = v


def _confidence_label(conf: float) -> str:
    if conf >= 0.85:
        return "High"
    if conf >= 0.65:
        return "Medium"
    return "Low"


def _counts(fields: list[dict[str, Any]]) -> tuple[int, int, int]:
    matched = sum(1 for f in fields if f["status"] == "match")
    mismatched = sum(1 for f in fields if f["status"] == "mismatch")
    uncertain = sum(1 for f in fields if f["status"] == "uncertain")
    return matched, mismatched, uncertain


# ── adapters — convert real pipeline output to render format ──────────────────

def _adapt_fields(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert pipeline result to the field list the render states expect."""
    merged = result.get("merged_fields", {})
    fields_raw = result.get("validation_report", {}).get("fields", {})
    rows: list[dict[str, Any]] = []
    for field_name, payload in fields_raw.items():
        doc_src = merged.get(field_name, {}).get("doc_source", "")
        rows.append(
            {
                "field": field_name,
                "label": _FIELD_LABELS.get(
                    field_name, field_name.replace("_", " ").title()
                ),
                "status": payload.get("status", "uncertain"),
                "confidence": float(payload.get("confidence", 0.0)),
                "found": payload.get("found"),
                "expected": payload.get("expected"),
                "reason": payload.get("reason", ""),
                "source_snippet": payload.get("source_snippet") or "",
                "doc_source": doc_src,
            }
        )
    return rows


def _adapt_email(result: dict[str, Any]) -> dict[str, Any]:
    """Build a display-friendly email dict from the pipeline result."""
    email = result.get("email", {})
    from_raw = email.get("from", "")
    if "<" in from_raw and ">" in from_raw:
        from_name = from_raw.split("<")[0].strip()
        from_addr = from_raw.split("<")[1].rstrip(">").strip()
    else:
        from_name = (
            email.get("from_name")
            or from_raw.split("@")[0].replace(".", " ").title()
        )
        from_addr = from_raw

    customer_name = result.get("validation_report", {}).get(
        "customer_name", email.get("customer_id", "")
    )
    return {
        "from_name": from_name,
        "from_addr": from_addr,
        "to_addr": email.get("to", "cg-team@gocomet.com"),
        "subject": email.get("subject", "Shipment documents"),
        "body": email.get("body", ""),
        "received_at": email.get("received_at", ""),
        "attachments": result.get("attachments", []),
        "shipment_ref": result.get("shipment_id", ""),
        "customer": customer_name,
    }


def _adapt_cross_doc(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert cross_doc_report into table rows for the verified state."""
    cross = result.get("cross_doc_report", {})
    rows: list[dict[str, Any]] = []

    for disc in cross.get("discrepancies", []):
        field_name = disc.get("field", "")
        values_by_doc: dict[str, str] = disc.get("values_by_doc", {})
        docs_str = " vs ".join(values_by_doc.keys()) if values_by_doc else "Multiple docs"
        rows.append(
            {
                "field": _FIELD_LABELS.get(field_name, field_name.replace("_", " ").title()),
                "docs": docs_str,
                "status": disc.get("status", "mismatch"),
                "note": disc.get("reason", ""),
            }
        )

    for field_name in cross.get("consistent_fields", []):
        rows.append(
            {
                "field": _FIELD_LABELS.get(field_name, field_name.replace("_", " ").title()),
                "docs": "All docs",
                "status": "match",
                "note": "Consistent across all documents.",
            }
        )

    return rows


def _adapt_amendment(result: dict[str, Any]) -> str:
    """Return the amendment email text from the pipeline decision."""
    decision = result.get("decision", {})
    amendment = decision.get("amendment_request")
    if amendment:
        return amendment

    # flag_for_human_review — no amendment drafted; build a review-request template.
    human_review_reasons: list[dict[str, Any]] = decision.get("human_review_reasons", [])
    shipment_id = result.get("shipment_id", "this shipment")
    if human_review_reasons:
        lines = [
            f"Subject: Documents Require Confirmation – {shipment_id}",
            "",
            "Dear Supplier,",
            "",
            "We have reviewed the submitted documents and require your confirmation "
            "on the following fields before we can proceed:",
        ]
        for issue in human_review_reasons:
            field_label = issue.get("field", "").replace("_", " ").title()
            reason = issue.get("reason", "")
            lines.append(f"\n- {field_label}: {reason}")
        lines += [
            "",
            "Please confirm or resubmit corrected documents at your earliest convenience.",
            "",
            "Regards,",
            "Cargo Validation Team, GoComet Nova",
        ]
        return "\n".join(lines)

    return decision.get("explanation", "No amendment required.")


# ── pipeline runner ───────────────────────────────────────────────────────────

def _run_pipeline() -> None:
    """Create a temp bundle and call process_bundle(). Updates session state."""
    email_meta: dict[str, Any] = st.session_state["pending_email"] or {}
    attachments: list[dict[str, Any]] = st.session_state["pending_attachments"] or []

    with st.status("Running Nova pipeline…", expanded=True) as _status:
        try:
            st.write("Assembling bundle from email + attachments…")
            with tempfile.TemporaryDirectory() as bundle_dir:
                bundle_path = Path(bundle_dir)

                email_json: dict[str, Any] = {
                    "from": email_meta.get("from_addr", ""),
                    "from_name": email_meta.get("from_name", ""),
                    "to": email_meta.get("to_addr", "cg-team@gocomet.com"),
                    "subject": email_meta.get("subject", ""),
                    "received_at": email_meta.get("received_at", ""),
                    "customer_id": email_meta.get("customer_id", "gocomet_demo_customer"),
                    "body": email_meta.get("body", ""),
                }
                (bundle_path / "email.json").write_text(
                    json.dumps(email_json, ensure_ascii=False), encoding="utf-8"
                )

                for att in attachments:
                    if "path" in att:
                        shutil.copy(att["path"], bundle_path / att["name"])
                    else:
                        (bundle_path / att["name"]).write_bytes(att["bytes"])

                n_docs = len(attachments)
                st.write(
                    f"Running Extractor → Validator → Router on {n_docs} "
                    f"document{'s' if n_docs != 1 else ''}…"
                )
                result = process_bundle(bundle_path, db_path=_db_path())

            st.session_state["pipeline_result"] = result
            st.session_state["cg_view"] = "verified"
            _status.update(label="Pipeline complete", state="complete")

        except Exception as exc:
            _status.update(label="Pipeline failed", state="error")
            st.error(f"Pipeline error: {exc}")
            return

    st.rerun()


# ── session state init ────────────────────────────────────────────────────────

for _k, _v in _DEFAULT_STATE.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_load_local_env()

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Nova CG Dashboard")
    st.caption("GoComet · Cargo Validation")
    st.divider()

    _result_for_sidebar = st.session_state.get("pipeline_result")
    if _result_for_sidebar:
        _outcome_key = _result_for_sidebar.get("decision", {}).get("outcome", "")
        _dot = _OUTCOME_DOT.get(_outcome_key, "⚪")
        _label = _OUTCOME_LABEL.get(_outcome_key, _outcome_key or "Unknown")
        _src_label = (
            "Sample run"
            if st.session_state.get("cg_source") == "sample"
            else "Custom run"
        )
        with st.container(border=True):
            st.markdown(f"{_dot} **{_result_for_sidebar.get('shipment_id', '—')}**")
            st.caption(_label)
            st.caption(_src_label)
        st.divider()

    if st.button("＋  New Run", use_container_width=True):
        _reset()
        st.rerun()

    st.divider()

    _current_view = st.session_state.get("cg_view", "idle")
    _steps = [
        ("idle", "Choose mode"),
        ("form", "Build email"),
        ("incoming", "Incoming email"),
        ("verified", "Verification result"),
        ("draft", "Draft reply"),
    ]
    for _step_id, _step_label in _steps:
        if _step_id == _current_view:
            st.markdown(f"**→ {_step_label}**")
        else:
            st.caption(_step_label)

    st.divider()
    with st.expander("🔍  Query Stored Runs"):
        st.caption("Ask natural-language questions over every validated shipment in DuckDB.")
        _nl_question = st.text_input(
            "Question",
            value="show me everything pending review",
            key="sidebar_nl_question",
            label_visibility="collapsed",
        )
        if st.button("Ask", key="sidebar_ask_btn", use_container_width=True):
            if _nl_question.strip():
                try:
                    _qr = answer_question(_nl_question.strip(), db_path=_db_path())
                    st.success(_qr["answer"])
                    with st.expander("SQL + rows", expanded=False):
                        st.caption(f"Planner: {_qr['query_source']}")
                        st.code(_qr["sql"], language="sql")
                        if _qr["rows"]:
                            st.dataframe(_qr["rows"])
                        else:
                            st.caption("No matching rows.")
                except Exception as _qe:
                    st.error(f"Query failed: {_qe}")

# ── state router ──────────────────────────────────────────────────────────────

_view = st.session_state["cg_view"]

# ──────────────────────────────────────────────────────────────────────────────
# State 0 — Idle (landing / choose mode)
# ──────────────────────────────────────────────────────────────────────────────

if _view == "idle":
    st.markdown("## Nova CG Dashboard")
    st.caption(
        "Choose how to run the validation pipeline. "
        "You can use the pre-bundled sample shipment, "
        "or build your own email with custom documents."
    )
    st.divider()

    _col_a, _col_b = st.columns(2)

    with _col_a:
        with st.container(border=True):
            st.markdown("### Simulate Sample")
            st.markdown(
                "Run the live pipeline on the pre-bundled sample shipment:\n"
                "- `BOL-SU-2026-001.pdf` — Bill of Lading\n"
                "- `INV-SU-2026-001.jpg` — Commercial Invoice\n"
                "- `PACK-SU-2026-001.pdf` — Packing List"
            )
            st.caption("No setup required. Uses the demo customer rule set.")

            _missing_samples = [p for p, _ in _SAMPLE_DOCS if not p.exists()]
            if _missing_samples:
                st.warning(
                    "Sample document(s) not found:\n"
                    + "\n".join(f"- `{p.name}`" for p in _missing_samples)
                )
            else:
                if st.button(
                    "▶  Simulate Sample", type="primary", use_container_width=True
                ):
                    _email_data = json.loads(
                        _SAMPLES_EMAIL.read_text(encoding="utf-8")
                    )
                    _from_raw = _email_data.get("from", "")
                    st.session_state["pending_email"] = {
                        "from_name": (
                            _email_data.get("from_name")
                            or _from_raw.split("@")[0].replace(".", " ").title()
                        ),
                        "from_addr": _from_raw,
                        "to_addr": _email_data.get("to", "cg-team@gocomet.com"),
                        "subject": _email_data.get("subject", ""),
                        "received_at": _email_data.get("received_at", ""),
                        "customer_id": _email_data.get(
                            "customer_id", "gocomet_demo_customer"
                        ),
                        "body": _email_data.get("body", ""),
                    }
                    st.session_state["pending_attachments"] = [
                        {"name": bundle_name, "path": str(src_path)}
                        for src_path, bundle_name in _SAMPLE_DOCS
                    ]
                    st.session_state["cg_source"] = "sample"
                    st.session_state["cg_view"] = "incoming"
                    st.rerun()

    with _col_b:
        with st.container(border=True):
            st.markdown("### Create Custom Email")
            st.markdown(
                "Fill in a supplier email, attach your own trade documents "
                "(BOL, Invoice, Packing List, etc.), and run the live pipeline."
            )
            st.caption(
                "Requires `OPENAI_API_KEY` (and `ANTHROPIC_API_KEY` for the validator)."
            )
            if st.button("✏  Create Custom Email", use_container_width=True):
                st.session_state["cg_source"] = "custom"
                st.session_state["cg_view"] = "form"
                st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# State 0b — Form (custom email builder)
# ──────────────────────────────────────────────────────────────────────────────

elif _view == "form":
    if st.button("← Back"):
        st.session_state["cg_view"] = "idle"
        st.rerun()

    st.markdown("## Build a Supplier Email")
    st.caption(
        "Describe the incoming shipment email and attach the trade documents to validate."
    )

    with st.form("custom_email_form"):
        st.markdown("**Sender details**")
        _fc1, _fc2 = st.columns(2)
        _from_name = _fc1.text_input("Sender name", value="Raj Malhotra")
        _from_addr = _fc2.text_input(
            "Sender email", value="raj.logistics@greenfield-exports.com"
        )

        st.markdown("**Email metadata**")
        _subject = st.text_input(
            "Subject", value="Shipment Documents – SU-2026-001 | Greenfield Exports"
        )
        _customer_id = st.selectbox(
            "Customer rule set",
            options=["gocomet_demo_customer"],
            help="The rule set used to validate this shipment's documents.",
        )
        _body = st.text_area(
            "Email body",
            value=(
                "Hi team,\n\nPlease find attached the shipment documents for our latest "
                "consignment.\nKindly validate and confirm at your earliest convenience."
                "\n\nRegards,\nRaj Malhotra\nExport Coordinator, Greenfield Exports"
            ),
            height=140,
        )

        st.markdown("**Attachments**")
        _uploaded_files = st.file_uploader(
            "Trade documents",
            type=["pdf", "jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help="Upload the Bill of Lading, Commercial Invoice, Packing List, etc.",
        )

        _submitted = st.form_submit_button(
            "Continue →", type="primary", use_container_width=True
        )

    if _submitted:
        if not _uploaded_files:
            st.error("Please attach at least one trade document before continuing.")
        else:
            st.session_state["pending_email"] = {
                "from_name": _from_name,
                "from_addr": _from_addr,
                "to_addr": "cg-team@gocomet.com",
                "subject": _subject,
                "received_at": "",
                "customer_id": _customer_id,
                "body": _body,
            }
            st.session_state["pending_attachments"] = [
                {"name": f.name, "bytes": f.getvalue()} for f in _uploaded_files
            ]
            st.session_state["cg_view"] = "incoming"
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# State 1 — Incoming
# ──────────────────────────────────────────────────────────────────────────────

elif _view == "incoming":
    _email = st.session_state.get("pending_email") or {}
    _attachments = st.session_state.get("pending_attachments") or []

    st.markdown("## Incoming Shipment")
    st.caption(
        "Review the supplier email below, then click **Run Pipeline** to start validation."
    )

    with st.container(border=True):
        _c1, _c2 = st.columns([1, 6])
        _c1.markdown("### 📧")
        with _c2:
            st.markdown(
                f"**From:** {_email.get('from_name', '')} "
                f"&lt;{_email.get('from_addr', '')}&gt;"
            )
            st.markdown(f"**To:** {_email.get('to_addr', '')}")
            st.markdown(f"**Subject:** {_email.get('subject', '')}")
            if _email.get("received_at"):
                st.caption(f"Received: {_email['received_at']}")

    if _attachments:
        st.markdown("**Attachments**")
        _att_cols = st.columns(min(len(_attachments), 4))
        for _col, _att in zip(_att_cols, _attachments):
            with _col:
                with st.container(border=True):
                    st.markdown(f"📄 **{_att['name']}**")

    if _email.get("body"):
        with st.expander("Email body"):
            st.text(_email["body"])

    st.divider()

    st.markdown("**Agent pipeline — what will run:**")
    for _step, _desc in [
        (
            "🔍 Extractor Agent",
            "Reads each document with a vision LLM, extracts 8 trade fields with confidence scores",
        ),
        (
            "✅ Validator Agent",
            "Compares extracted fields against customer rules field-by-field",
        ),
        (
            "⚡ Router Agent",
            "Decides: auto-approve, flag for review, or draft amendment request",
        ),
        (
            "💾 Storage",
            "Stores verified output to DuckDB — queryable by CG team",
        ),
    ]:
        _sc1, _sc2 = st.columns([2, 5])
        _sc1.markdown(f"**{_step}**")
        _sc2.caption(_desc)

    st.divider()

    if not os.getenv("OPENAI_API_KEY"):
        st.warning(
            "OPENAI_API_KEY is not set. Add it to `.env` or your shell before running."
        )

    if st.button("▶  Run Pipeline", type="primary", use_container_width=True):
        _run_pipeline()

# ──────────────────────────────────────────────────────────────────────────────
# State 2 — Verification Result
# ──────────────────────────────────────────────────────────────────────────────

elif _view == "verified":
    _result = st.session_state.get("pipeline_result")
    if not _result:
        st.error("No pipeline result found. Start a new run.")
        if st.button("← Start over"):
            _reset()
            st.rerun()
        st.stop()

    _fields = _adapt_fields(_result)
    _email_info = _adapt_email(_result)
    _matched, _mismatched, _uncertain = _counts(_fields)

    st.markdown("## Verification Result")
    st.caption(
        f"Shipment: **{_email_info['shipment_ref']}** · "
        f"Customer: **{_email_info['customer']}**"
    )

    _m1, _m2, _m3 = st.columns(3)
    _m1.metric("Matched", _matched)
    _m2.metric(
        "Mismatches",
        _mismatched,
        delta=f"-{_mismatched} issues" if _mismatched else None,
        delta_color="inverse",
    )
    _m3.metric(
        "Uncertain",
        _uncertain,
        delta=f"-{_uncertain} need review" if _uncertain else None,
        delta_color="inverse",
    )

    if _mismatched > 0 or _uncertain > 0:
        st.error(
            f"**{_mismatched} mismatch(es) and {_uncertain} uncertain field(s) found.** "
            "Review the flagged rows below, then send the draft reply to the supplier."
        )
    else:
        st.success("All fields matched. This shipment can be auto-approved.")

    st.divider()

    _hcols = st.columns([3, 2, 3, 3, 2, 1])
    for _hcol, _hdr in zip(
        _hcols, ["Field", "Status", "Found", "Expected", "Confidence", "Detail"]
    ):
        _hcol.markdown(f"**{_hdr}**")
    st.divider()

    for _fd in _fields:
        _st = _fd["status"]
        _row = st.columns([3, 2, 3, 3, 2, 1])
        _row[0].markdown(_fd["label"])

        _badge = _STATUS_BADGE.get(_st, _st)
        if _st == "mismatch":
            _row[1].markdown(f":red[{_badge}]")
        elif _st == "uncertain":
            _row[1].markdown(f":orange[{_badge}]")
        else:
            _row[1].markdown(f":green[{_badge}]")

        _row[2].markdown(str(_fd["found"] or "—"))
        _exp = _fd["expected"]
        _row[3].markdown(
            ", ".join(_exp) if isinstance(_exp, list) else str(_exp or "—")
        )
        _row[4].progress(
            min(max(_fd["confidence"], 0.0), 1.0),
            text=f"{_fd['confidence']:.0%}",
        )

        if _st in ("mismatch", "uncertain"):
            if _row[5].button("View →", key=f"detail_{_fd['field']}"):
                st.session_state["selected_field"] = _fd["field"]
                st.session_state["cg_view"] = "discrepancy"
                st.rerun()
        else:
            _row[5].markdown("—")

    st.divider()

    _cross_rows = _adapt_cross_doc(_result)
    if _cross_rows:
        _n_docs = len(_result.get("attachments", []))
        with st.expander(f"Cross-document consistency check ({_n_docs} document(s))"):
            _att_names = ", ".join(_result.get("attachments", []))
            if _att_names:
                st.caption(f"Checked across: {_att_names}")
            for _cr in _cross_rows:
                _xc1, _xc2, _xc3 = st.columns([2, 3, 4])
                _xc1.markdown(f"**{_cr['field']}**")
                _xc2.caption(_cr["docs"])
                _xb = _STATUS_BADGE.get(_cr["status"], _cr["status"])
                if _cr["status"] == "mismatch":
                    _xc3.markdown(f":red[{_xb}] — {_cr['note']}")
                elif _cr["status"] == "uncertain":
                    _xc3.markdown(f":orange[{_xb}] — {_cr['note']}")
                else:
                    _xc3.markdown(f":green[{_xb}] — {_cr['note']}")

    _audit_text = _result.get("decision", {}).get("decision_audit_report", "")
    if _audit_text:
        with st.expander("Decision audit report"):
            st.write(_audit_text)

    st.divider()

    if _mismatched > 0 or _uncertain > 0:
        if st.button(
            "📝  Review Draft Reply to Supplier",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["cg_view"] = "draft"
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# State 3 — Discrepancy Detail
# ──────────────────────────────────────────────────────────────────────────────

elif _view == "discrepancy":
    _result = st.session_state.get("pipeline_result")
    if not _result:
        st.error("No pipeline result found. Start a new run.")
        if st.button("← Start over"):
            _reset()
            st.rerun()
        st.stop()

    _fields = _adapt_fields(_result)
    _email_info = _adapt_email(_result)

    _fkey = st.session_state.get("selected_field")
    _fd = next((f for f in _fields if f["field"] == _fkey), None)

    if _fd is None:
        st.error("No field selected.")
        if st.button("← Back to Verification"):
            st.session_state["cg_view"] = "verified"
            st.rerun()
    else:
        _st = _fd["status"]

        if st.button("← Back to Verification Result"):
            st.session_state["cg_view"] = "verified"
            st.rerun()

        st.markdown(f"## Field: {_fd['label']}")
        _badge = _STATUS_BADGE.get(_st, _st)
        if _st == "mismatch":
            st.markdown(f":red[**{_badge}**] · confidence {_fd['confidence']:.0%}")
        else:
            st.markdown(f":orange[**{_badge}**] · confidence {_fd['confidence']:.0%}")

        st.divider()

        _cf, _ce = st.columns(2)
        with _cf:
            with st.container(border=True):
                st.markdown("#### Found in document")
                st.markdown(f"**{_fd['found'] or '— not extracted —'}**")
                if _fd.get("doc_source"):
                    st.caption(f"Source: `{_fd['doc_source']}`")
                st.caption(
                    f"Confidence: {_fd['confidence']:.0%} — "
                    f"{_confidence_label(_fd['confidence'])}"
                )

        with _ce:
            with st.container(border=True):
                st.markdown("#### Expected by customer rule")
                _exp = _fd["expected"]
                if isinstance(_exp, list):
                    for _ei in _exp:
                        st.markdown(f"• **{_ei}**")
                elif _exp:
                    st.markdown(f"**{_exp}**")
                else:
                    st.markdown("_Not specified_")
                st.caption(f"Customer: {_email_info['customer']}")

        st.divider()

        if _st == "mismatch":
            st.error(f"**Validation reason:** {_fd['reason']}")
        else:
            st.warning(f"**Validation reason:** {_fd['reason']}")

        if _fd.get("source_snippet"):
            st.divider()
            st.markdown("#### Source snippet from document")
            st.caption("Exact text region used by the extractor for this field:")
            st.code(_fd["source_snippet"], language=None)

        st.divider()

        _nc1, _nc2 = st.columns(2)
        with _nc1:
            if st.button("← Back to Verification Result", use_container_width=True):
                st.session_state["cg_view"] = "verified"
                st.rerun()
        with _nc2:
            if st.button(
                "📝  Go to Draft Reply", type="primary", use_container_width=True
            ):
                st.session_state["cg_view"] = "draft"
                st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# State 4 — Draft Reply
# ──────────────────────────────────────────────────────────────────────────────

elif _view == "draft":
    _result = st.session_state.get("pipeline_result")
    if not _result:
        st.error("No pipeline result found. Start a new run.")
        if st.button("← Start over"):
            _reset()
            st.rerun()
        st.stop()

    _fields = _adapt_fields(_result)
    _email_info = _adapt_email(_result)
    _amendment = _adapt_amendment(_result)
    _matched, _mismatched, _uncertain = _counts(_fields)

    st.markdown("## Draft Reply to Supplier")
    st.caption(
        f"Shipment: **{_email_info['shipment_ref']}** · "
        f"To: **{_email_info['from_name']}** &lt;{_email_info['from_addr']}&gt;"
    )

    st.info(
        f"The agent drafted this reply based on **{_mismatched} mismatch(es)** "
        f"and **{_uncertain} uncertain field(s)**. "
        "Review and edit before sending — the agent never sends on its own."
    )

    _flagged = [f for f in _fields if f["status"] in ("mismatch", "uncertain")]
    if _flagged:
        st.markdown("**Issues included in this reply:**")
        _chip_cols = st.columns(min(len(_flagged), 4))
        for _col, _f in zip(_chip_cols, _flagged):
            _color = "red" if _f["status"] == "mismatch" else "orange"
            _col.markdown(
                f":{_color}[{_STATUS_BADGE[_f['status']]}  **{_f['label']}**]"
            )

    st.divider()

    st.text_area(
        "Draft email to supplier (fully editable)",
        value=_amendment,
        height=400,
        help="Edit this email before sending. The agent never sends without your approval.",
    )

    st.divider()

    st.warning(
        "**The agent never sends on its own.** "
        "Review the draft above, make any edits, then click **Send to Supplier**."
    )

    _cs, _cb = st.columns([2, 1])
    with _cs:
        if st.button("📤  Send to Supplier", type="primary", use_container_width=True):
            st.toast(
                f"Amendment email sent to {_email_info['from_name']} "
                f"({_email_info['from_addr']})",
                icon="✅",
            )
            time.sleep(0.4)
            st.rerun()
    with _cb:
        if st.button("← Back to Verification", use_container_width=True):
            st.session_state["cg_view"] = "verified"
            st.rerun()

    st.divider()
    with st.expander("What happens after you send?"):
        st.markdown(
            "1. The amendment email is delivered to the supplier.\n"
            "2. This shipment moves to **Amendment Sent** in the queue.\n"
            "3. When the supplier resubmits corrected documents, a new email "
            "arrives and the pipeline runs again from scratch.\n"
            "4. The full audit trail is stored in DuckDB and queryable via the "
            "Nova query layer."
        )

else:
    st.error(f"Unknown view: {_view!r}")
    if st.button("Reset"):
        _reset()
        st.rerun()

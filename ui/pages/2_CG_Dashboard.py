"""
Phase 2 — CG Verification Dashboard.

A 4-state Streamlit screen showing the full CG validation workflow:
  State 1: Incoming          — new SU email arrived, agent processing
  State 2: Verified          — field-by-field verification result
  State 3: Discrepancy Detail — drill-down on a flagged field
  State 4: Draft Reply       — editable amendment email before CG sends

Runs entirely on mock data — no API keys, no database required.
Accessible at http://localhost:8501/ via the "CG Dashboard" page in the sidebar.
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

st.set_page_config(
    page_title="Nova · CG Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MOCK_EMAIL: dict[str, Any] = {
    "id": "email-001",
    "from_name": "Raj Malhotra",
    "from_addr": "raj.logistics@greenfield-exports.com",
    "to_addr": "cg-team@gocomet.com",
    "subject": "Shipment Documents – BOL-2026-0412 | Greenfield Exports",
    "body": (
        "Hi team,\n\n"
        "Please find attached the shipment documents for our latest consignment "
        "destined for ACME Corporation Limited, Chicago.\n\n"
        "Attachments:\n"
        "  • BOL-2026-0412.pdf — Bill of Lading\n"
        "  • INVOICE-2026-0412.pdf — Commercial Invoice\n"
        "  • PACKLIST-2026-0412.pdf — Packing List\n\n"
        "Please confirm once validated.\n\nRegards,\nRaj Malhotra\n"
        "Export Coordinator, Greenfield Exports"
    ),
    "received_at": "2026-05-07  08:31 IST",
    "attachments": [
        "BOL-2026-0412.pdf",
        "INVOICE-2026-0412.pdf",
        "PACKLIST-2026-0412.pdf",
    ],
    "shipment_ref": "BOL-2026-0412",
    "customer": "ACME Corporation Limited",
}

MOCK_FIELDS: list[dict[str, Any]] = [
    {
        "field": "consignee_name",
        "label": "Consignee Name",
        "status": "mismatch",
        "confidence": 0.92,
        "found": "ACME Corp Ltd",
        "expected": "ACME Corporation Limited",
        "reason": (
            "Extracted value 'ACME Corp Ltd' does not match the customer-required "
            "legal name 'ACME Corporation Limited'. Abbreviated names are rejected "
            "by customs for this destination."
        ),
        "source_snippet": (
            "CONSIGNEE\n"
            "ACME Corp Ltd\n"
            "45 Trade Street, Suite 900\n"
            "Chicago, IL 60601, USA\n"
            "Tel: +1-312-555-0198"
        ),
        "doc_source": "BOL-2026-0412.pdf",
    },
    {
        "field": "hs_code",
        "label": "HS Code",
        "status": "mismatch",
        "confidence": 0.88,
        "found": "8471.30",
        "expected": "8471.30.00",
        "reason": (
            "HS code '8471.30' is missing the two-digit national sub-heading '00'. "
            "Customer requires the 8-digit format for US customs filing."
        ),
        "source_snippet": (
            "Description of Goods:\n"
            "Laptop Computers – 120 units\n"
            "HS Code: 8471.30\n"
            "Net Weight: 420.00 KG"
        ),
        "doc_source": "INVOICE-2026-0412.pdf",
    },
    {
        "field": "incoterms",
        "label": "Incoterms",
        "status": "uncertain",
        "confidence": 0.51,
        "found": "CIF",
        "expected": "FOB or CIF",
        "reason": (
            "Value 'CIF' is within the allowed set, but extractor confidence is 0.51 "
            "(below threshold 0.60). The term appears in a footer section with low "
            "scan quality. Manual confirmation recommended."
        ),
        "source_snippet": (
            "[Low quality scan — footer region]\n"
            "...terms of delivery: C|F Chicago...\n"
            "...payment terms: 30 days net..."
        ),
        "doc_source": "BOL-2026-0412.pdf",
    },
    {
        "field": "port_of_loading",
        "label": "Port of Loading",
        "status": "match",
        "confidence": 0.97,
        "found": "Shanghai",
        "expected": "Shanghai",
        "reason": "Extracted value matches the required port of loading.",
        "source_snippet": "Port of Loading: SHANGHAI, CHINA (SHA)",
        "doc_source": "BOL-2026-0412.pdf",
    },
    {
        "field": "port_of_discharge",
        "label": "Port of Discharge",
        "status": "match",
        "confidence": 0.95,
        "found": "Chicago O'Hare",
        "expected": "Chicago",
        "reason": "Extracted value resolved to canonical port 'Chicago' via master data.",
        "source_snippet": "Port of Discharge: CHICAGO O'HARE ICD, USA",
        "doc_source": "BOL-2026-0412.pdf",
    },
    {
        "field": "description_of_goods",
        "label": "Description of Goods",
        "status": "match",
        "confidence": 0.91,
        "found": "Laptop Computers",
        "expected": "Laptop Computers or Notebooks",
        "reason": "Extracted text contains an expected goods keyword.",
        "source_snippet": "Commodity: Laptop Computers, 120 Units, Brand: TechPro",
        "doc_source": "INVOICE-2026-0412.pdf",
    },
    {
        "field": "gross_weight",
        "label": "Gross Weight",
        "status": "match",
        "confidence": 0.94,
        "found": "468.00 KG",
        "expected": "between 400 and 600 KG",
        "reason": "Extracted numeric value is within the allowed range.",
        "source_snippet": "Gross Weight: 468.00 KGS  /  Net Weight: 420.00 KGS",
        "doc_source": "PACKLIST-2026-0412.pdf",
    },
    {
        "field": "invoice_number",
        "label": "Invoice Number",
        "status": "match",
        "confidence": 0.99,
        "found": "INV-GFE-2026-0412",
        "expected": "INV-GFE-.*",
        "reason": "Extracted value matches the required invoice number format.",
        "source_snippet": "Invoice No.: INV-GFE-2026-0412   Date: 05-May-2026",
        "doc_source": "INVOICE-2026-0412.pdf",
    },
]

MOCK_AMENDMENT_EMAIL: str = """\
Subject: Amendment Required – Shipment BOL-2026-0412 | Greenfield Exports

Dear Raj,

Thank you for submitting the shipment documents for BOL-2026-0412. \
We have completed our review and found the following discrepancies that \
require correction before we can approve the consignment.

─────────────────────────────────────────────
DISCREPANCY 1 — Consignee Name
  Document:  BOL-2026-0412.pdf
  Found:     ACME Corp Ltd
  Expected:  ACME Corporation Limited
  Reason:    The consignee must be listed using their full legal registered name. \
Abbreviated forms are rejected at US customs. \
Please update the Bill of Lading to reflect 'ACME Corporation Limited'.

─────────────────────────────────────────────
DISCREPANCY 2 — HS Code
  Document:  INVOICE-2026-0412.pdf
  Found:     8471.30
  Expected:  8471.30.00
  Reason:    The US customs filing requires the 8-digit HS code including the \
national sub-heading. Please update to 8471.30.00.

─────────────────────────────────────────────
FIELD REQUIRING CONFIRMATION — Incoterms
  Document:  BOL-2026-0412.pdf
  Found:     CIF (low confidence — scan quality issue in footer)
  Required:  FOB or CIF
  Action:    Please confirm the agreed Incoterm in a clean, legible section of \
the revised Bill of Lading.

─────────────────────────────────────────────

Please amend the above documents and resubmit at your earliest convenience. \
All other fields — port of loading, port of discharge, description of goods, \
gross weight, and invoice number — have been verified and are correct.

Regards,
Priya Sharma
Cargo Validation Team, GoComet Nova
"""

MOCK_SHIPMENT_QUEUE: list[dict[str, Any]] = [
    {
        "id": "email-001",
        "ref": "BOL-2026-0412",
        "from_name": "Raj Malhotra",
        "customer": "ACME Corporation Limited",
        "received_at": "08:31 IST",
        "queue_status": "needs_attention",
        "summary": "2 mismatches · 1 uncertain",
    },
    {
        "id": "email-002",
        "ref": "BOL-2026-0408",
        "from_name": "Mei Lin",
        "customer": "GlobalTrade GmbH",
        "received_at": "07:15 IST",
        "queue_status": "cleared",
        "summary": "Auto-approved",
    },
    {
        "id": "email-003",
        "ref": "BOL-2026-0401",
        "from_name": "Carlos Mendes",
        "customer": "Pacific Freight Co.",
        "received_at": "Yesterday 21:44",
        "queue_status": "amendment_sent",
        "summary": "Amendment sent",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATUS_BADGE: dict[str, str] = {
    "match": "🟢 Match",
    "mismatch": "🔴 Mismatch",
    "uncertain": "🟠 Uncertain",
}

_QUEUE_DOT: dict[str, str] = {
    "needs_attention": "🔴",
    "cleared": "🟢",
    "amendment_sent": "🔵",
}

_QUEUE_LABEL: dict[str, str] = {
    "needs_attention": "Needs attention",
    "cleared": "Cleared",
    "amendment_sent": "Amendment sent",
}


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


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "cg_view" not in st.session_state:
    st.session_state["cg_view"] = "incoming"
if "selected_field" not in st.session_state:
    st.session_state["selected_field"] = None
if "active_shipment_id" not in st.session_state:
    st.session_state["active_shipment_id"] = "email-001"
if "sent_shipments" not in st.session_state:
    st.session_state["sent_shipments"] = set()

# ---------------------------------------------------------------------------
# Sidebar — shipment queue
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Nova CG Dashboard")
    st.caption("GoComet · Cargo Validation")
    st.divider()
    st.markdown("**Incoming Shipments**")

    for _item in MOCK_SHIPMENT_QUEUE:
        _dot = _QUEUE_DOT[_item["queue_status"]]
        _label = _QUEUE_LABEL[_item["queue_status"]]
        _is_active = st.session_state["active_shipment_id"] == _item["id"]

        with st.container(border=True):
            _dc, _db = st.columns([1, 8])
            _dc.markdown(f"## {_dot}")
            with _db:
                st.markdown(f"**{_item['ref']}**")
                st.caption(f"{_item['from_name']} · {_item['received_at']}")
                st.caption(_item["customer"])
                st.caption(f"_{_item['summary']}_")

            if _is_active:
                st.caption(f"✓ Currently viewing — {_label}")
            elif _item["id"] == "email-001":
                if st.button("Open", key=f"open_{_item['id']}"):
                    st.session_state["active_shipment_id"] = _item["id"]
                    st.session_state["cg_view"] = "incoming"
                    st.session_state["selected_field"] = None
                    st.rerun()
            else:
                st.caption(f"Status: {_label}")

    st.divider()
    st.caption("Phase 2 · Mock demo — no live agents")

# ---------------------------------------------------------------------------
# State router
# ---------------------------------------------------------------------------

_view = st.session_state["cg_view"]

# ---------------------------------------------------------------------------
# State 1 — Incoming
# ---------------------------------------------------------------------------

if _view == "incoming":
    st.markdown("## Incoming Shipment")
    st.caption("A new email from the supplier has arrived. The agent is ready to process the attached documents.")

    with st.container(border=True):
        _c1, _c2 = st.columns([1, 6])
        _c1.markdown("### 📧")
        with _c2:
            st.markdown(f"**From:** {MOCK_EMAIL['from_name']} &lt;{MOCK_EMAIL['from_addr']}&gt;")
            st.markdown(f"**To:** {MOCK_EMAIL['to_addr']}")
            st.markdown(f"**Subject:** {MOCK_EMAIL['subject']}")
            st.caption(f"Received: {MOCK_EMAIL['received_at']}")

    st.markdown("**Attachments**")
    _att_cols = st.columns(3)
    _att_meta = [
        ("📄", "Bill of Lading"),
        ("🧾", "Commercial Invoice"),
        ("📦", "Packing List"),
    ]
    for _col, _att, (_icon, _desc) in zip(_att_cols, MOCK_EMAIL["attachments"], _att_meta):
        with _col:
            with st.container(border=True):
                st.markdown(f"{_icon} **{_att}**")
                st.caption(_desc)

    st.divider()

    with st.expander("Email body"):
        st.text(MOCK_EMAIL["body"])

    st.markdown("**Agent pipeline — what will run:**")
    for _step, _desc in [
        ("🔍 Extractor Agent", "Reads each PDF with a vision LLM, extracts 8 trade fields with confidence scores"),
        ("✅ Validator Agent", "Compares extracted fields against customer rules field-by-field"),
        ("⚡ Router Agent",   "Decides: auto-approve, flag for review, or draft amendment request"),
        ("💾 Storage",        "Stores verified output to DuckDB — queryable by CG team"),
    ]:
        _sc1, _sc2 = st.columns([2, 5])
        _sc1.markdown(f"**{_step}**")
        _sc2.caption(_desc)

    st.divider()
    st.info("Click **Simulate Processing** to run the pipeline on the 3 attached documents and see the verification result.")

    if st.button("▶  Simulate Processing", type="primary", use_container_width=True):
        with st.spinner("Agent processing 3 documents — Extractor → Validator → Router..."):
            time.sleep(1.8)
        st.session_state["cg_view"] = "verified"
        st.rerun()

# ---------------------------------------------------------------------------
# State 2 — Verification Result
# ---------------------------------------------------------------------------

elif _view == "verified":
    _matched, _mismatched, _uncertain = _counts(MOCK_FIELDS)

    st.markdown("## Verification Result")
    st.caption(f"Shipment: **{MOCK_EMAIL['shipment_ref']}** · Customer: **{MOCK_EMAIL['customer']}**")

    _m1, _m2, _m3 = st.columns(3)
    _m1.metric("Matched", _matched)
    _m2.metric("Mismatches", _mismatched, delta=f"-{_mismatched} issues", delta_color="inverse")
    _m3.metric("Uncertain", _uncertain, delta=f"-{_uncertain} need review", delta_color="inverse")

    if _mismatched > 0 or _uncertain > 0:
        st.error(
            f"**{_mismatched} mismatch(es) and {_uncertain} uncertain field(s) found.** "
            "Review the flagged rows below, then send the draft amendment to the supplier."
        )
    else:
        st.success("All fields matched. You can auto-approve this shipment.")

    st.divider()

    # Table header
    _hcols = st.columns([3, 2, 3, 3, 2, 1])
    for _hcol, _hdr in zip(_hcols, ["Field", "Status", "Found", "Expected", "Confidence", "Detail"]):
        _hcol.markdown(f"**{_hdr}**")
    st.divider()

    # Field rows
    for _fd in MOCK_FIELDS:
        _st = _fd["status"]
        _row = st.columns([3, 2, 3, 3, 2, 1])
        _row[0].markdown(_fd["label"])

        _badge = _STATUS_BADGE[_st]
        if _st == "mismatch":
            _row[1].markdown(f":red[{_badge}]")
        elif _st == "uncertain":
            _row[1].markdown(f":orange[{_badge}]")
        else:
            _row[1].markdown(f":green[{_badge}]")

        _row[2].markdown(_fd["found"] or "—")
        _exp = _fd["expected"]
        _row[3].markdown(", ".join(_exp) if isinstance(_exp, list) else str(_exp))
        _row[4].progress(_fd["confidence"], text=f"{_fd['confidence']:.0%}")

        if _st in ("mismatch", "uncertain"):
            if _row[5].button("View →", key=f"detail_{_fd['field']}"):
                st.session_state["selected_field"] = _fd["field"]
                st.session_state["cg_view"] = "discrepancy"
                st.rerun()
        else:
            _row[5].markdown("—")

    st.divider()

    with st.expander("Cross-document consistency check"):
        st.caption("Fields checked across BOL, Invoice, and Packing List:")
        for _fn, _docs, _cs, _note in [
            ("consignee_name", "BOL vs Invoice",                "mismatch", "BOL: 'ACME Corp Ltd' ≠ Invoice: 'ACME Corporation Limited'"),
            ("hs_code",        "BOL vs Invoice vs Packing List","match",    "8471.30 consistent across all 3 documents"),
            ("gross_weight",   "Invoice vs Packing List",       "match",    "468.00 KG consistent across Invoice and Packing List"),
        ]:
            _xc1, _xc2, _xc3 = st.columns([2, 3, 4])
            _xc1.markdown(f"**{_fn}**")
            _xc2.caption(_docs)
            _xb = _STATUS_BADGE[_cs]
            if _cs == "mismatch":
                _xc3.markdown(f":red[{_xb}] — {_note}")
            else:
                _xc3.markdown(f":green[{_xb}] — {_note}")

    st.divider()

    if st.button("📝  Review Draft Reply to Supplier", type="primary", use_container_width=True):
        st.session_state["cg_view"] = "draft"
        st.rerun()

# ---------------------------------------------------------------------------
# State 3 — Discrepancy Detail
# ---------------------------------------------------------------------------

elif _view == "discrepancy":
    _fkey = st.session_state.get("selected_field")
    _fd = next((f for f in MOCK_FIELDS if f["field"] == _fkey), None)

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
        _badge = _STATUS_BADGE[_st]
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
                st.caption(f"Source: `{_fd['doc_source']}`")
                st.caption(f"Confidence: {_fd['confidence']:.0%} — {_confidence_label(_fd['confidence'])}")

        with _ce:
            with st.container(border=True):
                st.markdown("#### Expected by customer rule")
                _exp = _fd["expected"]
                if isinstance(_exp, list):
                    for _ei in _exp:
                        st.markdown(f"• **{_ei}**")
                else:
                    st.markdown(f"**{_exp}**")
                st.caption("Customer: ACME Corporation Limited")

        st.divider()

        if _st == "mismatch":
            st.error(f"**Validation reason:** {_fd['reason']}")
        else:
            st.warning(f"**Validation reason:** {_fd['reason']}")

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
            if st.button("📝  Go to Draft Reply", type="primary", use_container_width=True):
                st.session_state["cg_view"] = "draft"
                st.rerun()

# ---------------------------------------------------------------------------
# State 4 — Draft Reply
# ---------------------------------------------------------------------------

elif _view == "draft":
    _matched, _mismatched, _uncertain = _counts(MOCK_FIELDS)

    st.markdown("## Draft Reply to Supplier")
    st.caption(
        f"Shipment: **{MOCK_EMAIL['shipment_ref']}** · "
        f"To: **{MOCK_EMAIL['from_name']}** &lt;{MOCK_EMAIL['from_addr']}&gt;"
    )

    st.info(
        f"The agent drafted this reply based on **{_mismatched} mismatch(es)** "
        f"and **{_uncertain} uncertain field(s)**. "
        "Review and edit before sending — the agent never sends on its own."
    )

    _flagged = [f for f in MOCK_FIELDS if f["status"] in ("mismatch", "uncertain")]
    if _flagged:
        st.markdown("**Issues included in this reply:**")
        _chip_cols = st.columns(len(_flagged))
        for _col, _f in zip(_chip_cols, _flagged):
            _color = "red" if _f["status"] == "mismatch" else "orange"
            _col.markdown(f":{_color}[{_STATUS_BADGE[_f['status']]}  **{_f['label']}**]")

    st.divider()

    st.text_area(
        "Draft email to supplier (fully editable)",
        value=MOCK_AMENDMENT_EMAIL,
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
                f"Amendment email sent to {MOCK_EMAIL['from_name']} ({MOCK_EMAIL['from_addr']})",
                icon="✅",
            )
            st.session_state["sent_shipments"].add(MOCK_EMAIL["id"])
            st.session_state["cg_view"] = "incoming"
            st.session_state["selected_field"] = None
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
        st.session_state["cg_view"] = "incoming"
        st.rerun()

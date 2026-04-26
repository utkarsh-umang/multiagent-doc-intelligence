"""
Minimal Streamlit UI for running the Nova pipeline on one document.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st  # type: ignore[import-not-found]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.graph import run_pipeline as run_langgraph_pipeline  # noqa: E402
from storage.db import answer_question  # noqa: E402


def _db_path() -> str:
    return os.getenv("NOVA_DB_PATH", str(PROJECT_ROOT / "app.duckdb"))


def _load_local_env() -> None:
    """Load simple KEY=value pairs from .env for local Streamlit runs."""

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _save_upload(uploaded_file: Any) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def _run_pipeline(document_path: str, document_name: str, shipment_id: str | None) -> dict[str, Any]:
    with st.status("Running Nova pipeline...", expanded=True) as status:
        st.write("LangGraph is running extractor -> validator -> auditor -> storage")
        state = run_langgraph_pipeline(
            document_path,
            shipment_id=shipment_id,
            document_name=document_name,
            db_path=_db_path(),
        )
        if state["pipeline_status"] == "failed":
            status.update(label="Pipeline failed", state="error")
        else:
            status.update(label="Pipeline complete", state="complete")

    return state


def _extraction_rows(extracted: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for field_name, payload in extracted.items():
        rows.append(
            {
                "field": field_name,
                "value": payload.get("value"),
                "confidence": payload.get("confidence"),
                "source_snippet": payload.get("source_snippet"),
            }
        )
    return rows


def _validation_rows(validation_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for field_name, payload in validation_report.get("fields", {}).items():
        rows.append(
            {
                "field": field_name,
                "status": payload.get("status"),
                "found": payload.get("found"),
                "expected": payload.get("expected"),
                "confidence": payload.get("confidence"),
                "reason": payload.get("reason"),
            }
        )
    return rows


def _render_results(state: dict[str, Any]) -> None:
    if state["pipeline_status"] == "failed":
        st.error(state.get("error") or "Pipeline failed.")
        with st.expander("Failed Pipeline State"):
            st.json(state)
        return

    decision_report = state["decision_report"]
    validation_report = state["validation_report"]

    st.subheader("Decision")
    st.metric("Outcome", state["decision"])
    st.write(state["reasoning"])
    st.caption(f"Stored run id: {state['storage_id']}")

    if state.get("decision_audit_report"):
        st.subheader("Decision Audit Report")
        st.write(state["decision_audit_report"])

    if state.get("amendment_draft"):
        st.subheader("Draft Amendment Request")
        st.text_area(
            "Amendment email",
            value=state["amendment_draft"],
            height=260,
        )

    if decision_report.get("human_review_reasons"):
        st.subheader("Human Review Reasons")
        st.dataframe(decision_report["human_review_reasons"], width="stretch")

    st.subheader("Extracted Fields")
    st.dataframe(_extraction_rows(state["extracted_fields"]), width="stretch")

    st.subheader("Validation Results")
    cols = st.columns(3)
    cols[0].metric("Overall Status", validation_report["overall_status"])
    cols[1].metric("Has Mismatches", str(validation_report["has_mismatches"]))
    cols[2].metric("Has Uncertain", str(validation_report["has_uncertain"]))
    st.dataframe(_validation_rows(validation_report), width="stretch")

    with st.expander("Raw Pipeline State"):
        st.json(state)


def _render_query_box() -> None:
    st.subheader("Ask Stored Runs")
    question = st.text_input(
        "Natural-language question",
        value="how many shipments were flagged this week?",
    )
    if st.button("Ask DuckDB"):
        try:
            result = answer_question(question, db_path=_db_path())
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            return

        st.write(result["answer"])
        with st.expander("Grounding SQL and rows"):
            st.caption(f"Query planner: {result['query_source']}")
            st.code(result["sql"], language="sql")
            st.dataframe(result["rows"], width="stretch")


def main() -> None:
    _load_local_env()

    st.set_page_config(page_title="Nova Trade-Doc Pipeline", layout="wide")
    st.title("Nova Trade-Doc Pipeline")
    st.caption("Upload one PDF/image and run extractor -> validator -> auditor -> storage.")

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set. Add it to `.env` or your shell before running.")

    uploaded_file = st.file_uploader(
        "Trade document",
        type=["pdf", "jpg", "jpeg", "png", "webp"],
    )
    shipment_id = st.text_input("Shipment ID (optional)")

    if uploaded_file and st.button("Run Pipeline", type="primary"):
        document_path = _save_upload(uploaded_file)
        try:
            st.session_state["last_run"] = _run_pipeline(
                document_path,
                uploaded_file.name,
                shipment_id or None,
            )
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
        finally:
            Path(document_path).unlink(missing_ok=True)

    if "last_run" in st.session_state:
        _render_results(st.session_state["last_run"])

    st.divider()
    _render_query_box()


if __name__ == "__main__":
    main()


"""
DuckDB storage + simple natural-language query layer.

The query layer intentionally maps common business questions to fixed SQL
templates instead of allowing arbitrary generated SQL.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any


DEFAULT_DB_PATH = "app.duckdb"
REQUIRED_FIELDS = [
    "consignee_name",
    "hs_code",
    "port_of_loading",
    "port_of_discharge",
    "incoterms",
    "description_of_goods",
    "gross_weight",
    "invoice_number",
]


def connect(db_path: str = DEFAULT_DB_PATH):
    """Open a DuckDB connection and ensure required tables exist."""

    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "DuckDB is required for storage. Install dependencies with "
            "`pip install -r requirements.txt` inside a virtual environment."
        ) from exc

    conn = duckdb.connect(db_path)
    init_db(conn)
    return conn


def init_db(conn: Any) -> None:
    """Create the queryable shipment table if it does not exist."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shipment_runs (
            id VARCHAR PRIMARY KEY,
            shipment_id VARCHAR,
            document_name VARCHAR,
            customer_id VARCHAR,
            customer_name VARCHAR,
            outcome VARCHAR NOT NULL,
            overall_status VARCHAR,
            should_store BOOLEAN NOT NULL,
            consignee_name VARCHAR,
            hs_code VARCHAR,
            port_of_loading VARCHAR,
            port_of_discharge VARCHAR,
            incoterms VARCHAR,
            description_of_goods VARCHAR,
            gross_weight VARCHAR,
            invoice_number VARCHAR,
            mismatch_count INTEGER NOT NULL,
            uncertain_count INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL,
            extracted_json TEXT NOT NULL,
            validation_json TEXT NOT NULL,
            decision_json TEXT NOT NULL
        )
        """
    )


def store_pipeline_run(
    extracted: dict[str, Any],
    validation_report: dict[str, Any],
    decision: dict[str, Any],
    *,
    shipment_id: str | None = None,
    document_name: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
    conn: Any | None = None,
) -> str:
    """
    Persist one pipeline run in a queryable + auditable form.

    Returns the generated storage id.
    """

    owns_connection = conn is None
    conn = conn or connect(db_path)
    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    field_values = [_field_value(extracted, field_name) for field_name in REQUIRED_FIELDS]
    fields = validation_report.get("fields", {})
    mismatch_count = _count_field_status(fields, "mismatch")
    uncertain_count = _count_field_status(fields, "uncertain")

    conn.execute(
        """
        INSERT INTO shipment_runs (
            id,
            shipment_id,
            document_name,
            customer_id,
            customer_name,
            outcome,
            overall_status,
            should_store,
            consignee_name,
            hs_code,
            port_of_loading,
            port_of_discharge,
            incoterms,
            description_of_goods,
            gross_weight,
            invoice_number,
            mismatch_count,
            uncertain_count,
            created_at,
            extracted_json,
            validation_json,
            decision_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            shipment_id,
            document_name,
            validation_report.get("customer_id"),
            validation_report.get("customer_name"),
            decision["outcome"],
            validation_report.get("overall_status"),
            bool(decision.get("should_store", False)),
            *field_values,
            mismatch_count,
            uncertain_count,
            now,
            json.dumps(extracted),
            json.dumps(validation_report),
            json.dumps(decision),
        ],
    )

    if owns_connection:
        conn.close()

    return run_id


def answer_question(
    question: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
    conn: Any | None = None,
) -> dict[str, Any]:
    """
    Answer basic natural-language questions over stored shipment runs.

    Supported examples:
    - "how many shipments were flagged this week?"
    - "how many shipments were auto approved today?"
    - "show amendment requests this month"
    - "what is the straight-through processing rate this week?"
    """

    query_plan = _plan_query(question)
    owns_connection = conn is None
    conn = conn or connect(db_path)
    rows = conn.execute(query_plan["sql"]).fetchall()
    columns = [column[0] for column in conn.description]
    grounded_rows = [dict(zip(columns, row)) for row in rows]

    if owns_connection:
        conn.close()

    return {
        "question": question,
        "answer": _format_answer(query_plan, grounded_rows),
        "sql": query_plan["sql"],
        "rows": grounded_rows,
    }


def _field_value(extracted: dict[str, Any], field_name: str) -> str | None:
    field = extracted.get(field_name)
    if isinstance(field, dict):
        return field.get("value")
    return None


def _count_field_status(fields: dict[str, Any], status: str) -> int:
    return sum(1 for field in fields.values() if field.get("status") == status)


def _plan_query(question: str) -> dict[str, str]:
    normalized = question.strip().lower()
    time_filter = _time_filter(normalized)

    if _asks_for_rate(normalized):
        where = f"WHERE {time_filter}" if time_filter else ""
        return {
            "kind": "rate",
            "label": "straight-through processing rate",
            "sql": f"""
                SELECT
                    COUNT(*) AS total_shipments,
                    SUM(CASE WHEN outcome = 'auto_approve_and_store' THEN 1 ELSE 0 END)
                        AS auto_approved_shipments,
                    CASE
                        WHEN COUNT(*) = 0 THEN 0
                        ELSE ROUND(
                            100.0 * SUM(CASE WHEN outcome = 'auto_approve_and_store' THEN 1 ELSE 0 END)
                            / COUNT(*),
                            2
                        )
                    END AS stp_rate_percent
                FROM shipment_runs
                {where}
            """,
        }

    if _asks_for_list(normalized):
        return _list_query(normalized, time_filter)

    return _count_query(normalized, time_filter)


def _time_filter(question: str) -> str:
    if "today" in question:
        return "created_at >= date_trunc('day', current_timestamp)"
    if "this week" in question or "week" in question:
        return "created_at >= date_trunc('week', current_timestamp)"
    if "last 7 days" in question or "past 7 days" in question:
        return "created_at >= current_timestamp - INTERVAL 7 DAY"
    if "this month" in question or "month" in question:
        return "created_at >= date_trunc('month', current_timestamp)"
    return ""


def _asks_for_rate(question: str) -> bool:
    return "rate" in question or "percentage" in question or "straight-through" in question


def _asks_for_list(question: str) -> bool:
    return question.startswith(("show", "list", "which", "what shipments"))


def _count_query(question: str, time_filter: str) -> dict[str, str]:
    label, condition = _business_condition(question)
    where_parts = [condition] if condition else []
    if time_filter:
        where_parts.append(time_filter)
    where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    return {
        "kind": "count",
        "label": label,
        "sql": f"""
            SELECT COUNT(*) AS shipment_count
            FROM shipment_runs
            {where}
        """,
    }


def _list_query(question: str, time_filter: str) -> dict[str, str]:
    label, condition = _business_condition(question)
    where_parts = [condition] if condition else []
    if time_filter:
        where_parts.append(time_filter)
    where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    return {
        "kind": "list",
        "label": label,
        "sql": f"""
            SELECT
                id,
                shipment_id,
                document_name,
                outcome,
                overall_status,
                mismatch_count,
                uncertain_count,
                created_at
            FROM shipment_runs
            {where}
            ORDER BY created_at DESC
            LIMIT 20
        """,
    }


def _business_condition(question: str) -> tuple[str, str]:
    if re.search(r"\b(flagged|review|uncertain)\b", question):
        return "shipments flagged for human review", "outcome = 'flag_for_human_review'"
    if re.search(r"\b(amendment|mismatch|mismatched|failed)\b", question):
        return "shipments needing amendment", "outcome = 'draft_amendment_request'"
    if re.search(r"\b(approved|auto approved|auto-approved|straight through)\b", question):
        return "auto-approved shipments", "outcome = 'auto_approve_and_store'"
    return "shipments", ""


def _format_answer(query_plan: dict[str, str], rows: list[dict[str, Any]]) -> str:
    if query_plan["kind"] == "rate":
        row = rows[0] if rows else {}
        return (
            f"The {query_plan['label']} is {row.get('stp_rate_percent', 0)}% "
            f"({row.get('auto_approved_shipments', 0)} of "
            f"{row.get('total_shipments', 0)} shipments)."
        )

    if query_plan["kind"] == "list":
        count = len(rows)
        return f"Found {count} {query_plan['label']}."

    count = rows[0].get("shipment_count", 0) if rows else 0
    return f"Found {count} {query_plan['label']}."


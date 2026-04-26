"""
DuckDB storage + simple natural-language query layer.

The query layer maps common business questions to fixed SQL templates first,
then falls back to guarded LLM-generated SELECT queries for broader questions.
"""

from __future__ import annotations

import json
import os
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
QUERYABLE_COLUMNS = [
    "id",
    "shipment_id",
    "document_name",
    "customer_id",
    "customer_name",
    "outcome",
    "overall_status",
    "should_store",
    *REQUIRED_FIELDS,
    "mismatch_count",
    "uncertain_count",
    "created_at",
]
QUERY_ROW_LIMIT = 50
DISALLOWED_SQL_KEYWORDS = {
    "alter",
    "attach",
    "call",
    "copy",
    "create",
    "delete",
    "detach",
    "drop",
    "export",
    "import",
    "insert",
    "install",
    "load",
    "pragma",
    "replace",
    "set",
    "truncate",
    "update",
}


def connect(db_path: str = DEFAULT_DB_PATH):
    """Open a DuckDB connection and ensure required tables exist."""

    try:
        import duckdb  # type: ignore[import-not-found]
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
    allow_llm: bool = True,
    model: str | None = None,
    completion_fn: Any | None = None,
) -> dict[str, Any]:
    """
    Answer basic natural-language questions over stored shipment runs.

    Supported examples:
    - "how many shipments were flagged this week?"
    - "how many shipments were auto approved today?"
    - "show amendment requests this month"
    - "what is the straight-through processing rate this week?"
    """

    query_plan = _plan_rule_based_query(question)
    if query_plan is None:
        if not allow_llm:
            raise ValueError("This question is not supported by the rule-based query planner.")
        query_plan = _plan_llm_query(
            question,
            model=model,
            completion_fn=completion_fn,
        )

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
        "query_source": query_plan["source"],
        "rows": grounded_rows,
    }


def _field_value(extracted: dict[str, Any], field_name: str) -> str | None:
    field = extracted.get(field_name)
    if isinstance(field, dict):
        return field.get("value")
    return None


def _count_field_status(fields: dict[str, Any], status: str) -> int:
    return sum(1 for field in fields.values() if field.get("status") == status)


def _plan_rule_based_query(question: str) -> dict[str, str] | None:
    normalized = question.strip().lower()
    time_filter = _time_filter(normalized)

    if _asks_for_rate(normalized):
        where = f"WHERE {time_filter}" if time_filter else ""
        return {
            "source": "rules",
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

    if _asks_for_count(normalized):
        return _count_query(normalized, time_filter)

    return None


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


def _asks_for_count(question: str) -> bool:
    return (
        re.search(r"\b(how many|count|number of|total)\b", question) is not None
        and re.search(r"\b(shipment|shipments|runs|documents)\b", question) is not None
    )


def _count_query(question: str, time_filter: str) -> dict[str, str]:
    label, condition = _business_condition(question)
    where_parts = [condition] if condition else []
    if time_filter:
        where_parts.append(time_filter)
    where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    return {
        "source": "rules",
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
        "source": "rules",
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


def _plan_llm_query(
    question: str,
    *,
    model: str | None = None,
    completion_fn: Any | None = None,
) -> dict[str, str]:
    if completion_fn is None:
        try:
            import litellm  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "LLM query fallback requires LiteLLM. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc
        completion = litellm.completion
    else:
        completion = completion_fn

    selected_model = model or _default_query_model()
    prompt = {
        "task": "Translate a business question into a safe DuckDB SQL SELECT query.",
        "question": question,
        "table": "shipment_runs",
        "columns": QUERYABLE_COLUMNS,
        "business_values": {
            "outcome": [
                "auto_approve_and_store",
                "flag_for_human_review",
                "draft_amendment_request",
            ],
            "overall_status": ["passed", "failed", "needs_review"],
        },
        "rules": [
            "Return JSON only with keys: sql, label.",
            "Use only the shipment_runs table and listed columns.",
            "Generate DuckDB-compatible SQL.",
            "Generate exactly one read-only SELECT statement.",
            f"Use LIMIT {QUERY_ROW_LIMIT} for row-listing queries.",
            "Do not select extracted_json, validation_json, or decision_json.",
            "If the question cannot be answered from the schema, return an empty sql string.",
        ],
    }
    response = completion(
        model=selected_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": json.dumps(prompt)}],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Query model returned an empty response.")

    payload = json.loads(content)
    sql = str(payload.get("sql") or "").strip()
    if not sql:
        raise ValueError("This question cannot be answered from the stored shipment schema.")

    safe_sql = _validate_generated_sql(sql)
    return {
        "source": "llm",
        "kind": "llm",
        "label": str(payload.get("label") or "stored shipment query"),
        "sql": safe_sql,
        "model": selected_model,
    }


def _default_query_model() -> str:
    return (
        os.getenv("QUERY_SQL_MODEL")
        or os.getenv("VALIDATOR_MODEL")
        or os.getenv("AUDITOR_MODEL")
        or "gpt-4o"
    )


def _validate_generated_sql(sql: str) -> str:
    candidate = _strip_sql_markdown(sql).strip().rstrip(";").strip()
    lowered = candidate.lower()

    if not lowered.startswith("select "):
        raise ValueError("Generated SQL was rejected because only SELECT queries are allowed.")
    if ";" in candidate:
        raise ValueError("Generated SQL was rejected because multiple statements are not allowed.")
    if "--" in candidate or "/*" in candidate or "*/" in candidate:
        raise ValueError("Generated SQL was rejected because comments are not allowed.")
    if not re.search(r"\bfrom\s+shipment_runs\b", lowered):
        raise ValueError("Generated SQL must query the shipment_runs table.")

    tokens = set(re.findall(r"\b[a-z_]+\b", lowered))
    blocked = sorted(tokens & DISALLOWED_SQL_KEYWORDS)
    if blocked:
        raise ValueError(f"Generated SQL used disallowed keyword(s): {', '.join(blocked)}.")

    if not re.search(r"\blimit\s+\d+\b", lowered) and not _is_single_row_aggregate_query(lowered):
        candidate = f"{candidate}\nLIMIT {QUERY_ROW_LIMIT}"

    return candidate


def _strip_sql_markdown(sql: str) -> str:
    match = re.fullmatch(r"\s*```(?:sql)?\s*(.*?)\s*```\s*", sql, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else sql


def _is_single_row_aggregate_query(sql: str) -> bool:
    has_aggregate = re.search(r"\b(count|sum|avg|min|max)\s*\(", sql) is not None
    return bool(has_aggregate and not re.search(r"\bgroup\s+by\b", sql))


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

    if query_plan["kind"] == "llm":
        if not rows:
            return f"No matching rows found for {query_plan['label']}."
        if len(rows) == 1 and len(rows[0]) == 1:
            value = next(iter(rows[0].values()))
            return f"{query_plan['label']}: {value}."
        return f"Found {len(rows)} row(s) for {query_plan['label']}."

    count = rows[0].get("shipment_count", 0) if rows else 0
    return f"Found {count} {query_plan['label']}."


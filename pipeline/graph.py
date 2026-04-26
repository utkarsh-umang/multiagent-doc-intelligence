"""
LangGraph orchestration for the Nova document pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langfuse import observe  # type: ignore[import-not-found]
from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]

from agents.extractor import run as run_extractor
from agents.router import run as run_router
from agents.validator import run as run_validator
from pipeline.state import PipelineState
from rules.customer_rules import CUSTOMER_RULE_SET
from storage.db import DEFAULT_DB_PATH, store_pipeline_run


def _model_to_dict(model: PipelineState) -> dict[str, Any]:
    try:
        return model.model_dump()
    except AttributeError:
        return model.dict()


def _coerce_state(state: PipelineState | dict[str, Any]) -> PipelineState:
    if isinstance(state, PipelineState):
        return state
    try:
        return PipelineState.model_validate(state)
    except AttributeError:
        return PipelineState.parse_obj(state)


def _parse_pdf_text(document_url: str) -> str:
    path = Path(document_url)
    if path.suffix.lower() != ".pdf" or not path.exists():
        return ""

    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        return ""

    with fitz.open(path) as doc:
        return "\n".join(page.get_text("text") for page in doc)


def _failed_update(exc: Exception) -> dict[str, Any]:
    return {
        "pipeline_status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
    }


def _next_or_end(state: PipelineState | dict[str, Any]) -> str:
    current = _coerce_state(state)
    return "failed" if current.pipeline_status == "failed" else "continue"


def _decision_alias(outcome: str) -> str:
    mapping = {
        "auto_approve_and_store": "auto_approve",
        "flag_for_human_review": "flag_review",
        "draft_amendment_request": "amendment",
    }
    return mapping.get(outcome, outcome)


def _field_schema_from_rules(rule_set: dict[str, Any]) -> list[dict[str, Any]]:
    fields = rule_set.get("fields", {})
    return [
        {
            "name": field_name,
            "description": rule.get("description"),
            "required": bool(rule.get("required", True)),
            "aliases": rule.get("aliases", []),
        }
        for field_name, rule in fields.items()
    ]


@observe(name="nova.extractor_node")
def extractor_node(state: PipelineState | dict[str, Any]) -> dict[str, Any]:
    current = _coerce_state(state)
    try:
        rule_set = current.customer_rule_set or CUSTOMER_RULE_SET
        field_schema = current.field_schema or _field_schema_from_rules(rule_set)
        extraction_result = run_extractor(current.document_url, field_schema=field_schema)
        extraction_metadata = extraction_result["metadata"]
        return {
            "pipeline_status": "extracted",
            "raw_text": current.raw_text or _parse_pdf_text(current.document_url),
            "field_schema": field_schema,
            "extracted_fields": extraction_result["fields"],
            "extraction_metadata": extraction_metadata,
            "low_confidence_fields": extraction_metadata.get(
                "persistent_low_confidence_fields",
                [],
            ),
            "error": None,
        }
    except Exception as exc:
        return _failed_update(exc)


@observe(name="nova.validator_node")
def validator_node(state: PipelineState | dict[str, Any]) -> dict[str, Any]:
    current = _coerce_state(state)
    try:
        rule_set = current.customer_rule_set or CUSTOMER_RULE_SET
        validation_report = run_validator(current.extracted_fields, rule_set=rule_set)
        return {
            "pipeline_status": "validated",
            "validation_report": validation_report,
            "error": None,
        }
    except Exception as exc:
        return _failed_update(exc)


@observe(name="nova.router_node")
def router_node(state: PipelineState | dict[str, Any]) -> dict[str, Any]:
    current = _coerce_state(state)
    try:
        decision_report = run_router(current.validation_report)
        return {
            "pipeline_status": "routed",
            "decision": _decision_alias(decision_report["outcome"]),
            "reasoning": decision_report["explanation"],
            "amendment_draft": decision_report.get("amendment_request") or "",
            "decision_report": decision_report,
            "error": None,
        }
    except Exception as exc:
        return _failed_update(exc)


@observe(name="nova.storage_node")
def storage_node(state: PipelineState | dict[str, Any]) -> dict[str, Any]:
    current = _coerce_state(state)
    try:
        storage_id = store_pipeline_run(
            current.extracted_fields,
            current.validation_report,
            current.decision_report,
            shipment_id=current.shipment_id,
            document_name=current.document_name,
            db_path=current.storage_db_path,
        )
        return {
            "pipeline_status": "complete",
            "storage_id": storage_id,
            "error": None,
        }
    except Exception as exc:
        return _failed_update(exc)


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("extractor", extractor_node)
    graph.add_node("validator", validator_node)
    graph.add_node("router", router_node)
    graph.add_node("storage", storage_node)

    graph.add_edge(START, "extractor")
    graph.add_conditional_edges(
        "extractor",
        _next_or_end,
        {"continue": "validator", "failed": END},
    )
    graph.add_conditional_edges(
        "validator",
        _next_or_end,
        {"continue": "router", "failed": END},
    )
    graph.add_conditional_edges(
        "router",
        _next_or_end,
        {"continue": "storage", "failed": END},
    )
    graph.add_edge("storage", END)
    return graph.compile()


def run_pipeline(
    document_url: str,
    *,
    shipment_id: str | None = None,
    document_name: str | None = None,
    customer_rule_set: dict[str, Any] | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    app = build_graph()
    initial_state = PipelineState(
        document_url=document_url,
        document_name=document_name,
        shipment_id=shipment_id,
        storage_db_path=db_path,
        customer_rule_set=customer_rule_set or CUSTOMER_RULE_SET,
        field_schema=_field_schema_from_rules(customer_rule_set or CUSTOMER_RULE_SET),
    )
    result = app.invoke(_model_to_dict(initial_state))
    return _model_to_dict(_coerce_state(result))


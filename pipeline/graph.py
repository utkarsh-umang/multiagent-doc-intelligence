"""
LangGraph orchestration for the Nova document pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import uuid

from langfuse import observe  # type: ignore[import-not-found]
from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]

from agents.auditor import run as run_auditor
from agents.extractor import run as run_extractor
from agents.validator import run as run_validator
from pipeline.state import PipelineState
from rules.customer_rules import CUSTOMER_RULE_SET
from rules.master_data import MASTER_DATA
from storage.checkpoints import DEFAULT_CHECKPOINT_DIR, save_pipeline_checkpoint
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


def _auditor_outcome_or_end(state: PipelineState | dict[str, Any]) -> str:
    current = _coerce_state(state)
    if current.pipeline_status == "failed":
        return "failed"
    return current.decision or "failed"


def _decision_alias(outcome: str) -> str:
    mapping = {
        "auto_approve_and_store": "auto_approve",
        "flag_for_human_review": "flag_review",
        "draft_amendment_request": "amendment",
    }
    return mapping.get(outcome, outcome)


def _checkpoint(
    node_name: str,
    current: PipelineState,
    update: dict[str, Any],
) -> dict[str, Any]:
    save_pipeline_checkpoint(
        checkpoint_path=current.checkpoint_path,
        node_name=node_name,
        current_state=_model_to_dict(current),
        update=update,
    )
    return update


def _default_checkpoint_path(
    document_url: str,
    shipment_id: str | None,
    document_name: str | None,
) -> str:
    label = shipment_id or document_name or Path(document_url).stem or "pipeline-run"
    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "-", label).strip("-") or "pipeline-run"
    return str(DEFAULT_CHECKPOINT_DIR / f"{safe_label}-{uuid.uuid4().hex[:8]}.json")


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
        update = {
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
        return _checkpoint("extractor", current, update)
    except Exception as exc:
        return _checkpoint("extractor", current, _failed_update(exc))


@observe(name="nova.validator_node")
def validator_node(state: PipelineState | dict[str, Any]) -> dict[str, Any]:
    current = _coerce_state(state)
    try:
        rule_set = current.customer_rule_set or CUSTOMER_RULE_SET
        validation_report = run_validator(
            current.extracted_fields,
            rule_set=rule_set,
            master_data=current.master_data or MASTER_DATA,
        )
        update = {
            "pipeline_status": "validated",
            "validation_report": validation_report,
            "validation_metadata": validation_report.get("metadata", {}),
            "error": None,
        }
        return _checkpoint("validator", current, update)
    except Exception as exc:
        return _checkpoint("validator", current, _failed_update(exc))


@observe(name="nova.auditor_node")
def auditor_node(state: PipelineState | dict[str, Any]) -> dict[str, Any]:
    current = _coerce_state(state)
    try:
        decision_report = run_auditor(current.validation_report)
        update = {
            "pipeline_status": "audited",
            "decision": _decision_alias(decision_report["outcome"]),
            "reasoning": decision_report["explanation"],
            "decision_audit_report": decision_report["decision_audit_report"],
            "amendment_draft": decision_report.get("amendment_request") or "",
            "decision_report": decision_report,
            "auditor_metadata": decision_report.get("metadata", {}),
            "error": None,
        }
        return _checkpoint("auditor", current, update)
    except Exception as exc:
        return _checkpoint("auditor", current, _failed_update(exc))


@observe(name="nova.auto_approve_branch")
def auto_approve_branch_node(state: PipelineState | dict[str, Any]) -> dict[str, Any]:
    current = _coerce_state(state)
    update = {
        "pipeline_status": "auto_approved",
        "auditor_branch": "auto_approve",
        "error": None,
    }
    return _checkpoint("auto_approve_branch", current, update)


@observe(name="nova.human_review_branch")
def human_review_branch_node(state: PipelineState | dict[str, Any]) -> dict[str, Any]:
    current = _coerce_state(state)
    update = {
        "pipeline_status": "flagged_for_review",
        "auditor_branch": "flag_review",
        "error": None,
    }
    return _checkpoint("human_review_branch", current, update)


@observe(name="nova.amendment_branch")
def amendment_branch_node(state: PipelineState | dict[str, Any]) -> dict[str, Any]:
    current = _coerce_state(state)
    update = {
        "pipeline_status": "amendment_drafted",
        "auditor_branch": "amendment",
        "error": None,
    }
    return _checkpoint("amendment_branch", current, update)


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
        update = {
            "pipeline_status": "complete",
            "storage_id": storage_id,
            "error": None,
        }
        return _checkpoint("storage", current, update)
    except Exception as exc:
        return _checkpoint("storage", current, _failed_update(exc))


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("extractor", extractor_node)
    graph.add_node("validator", validator_node)
    graph.add_node("auditor", auditor_node)
    graph.add_node("auto_approve_branch", auto_approve_branch_node)
    graph.add_node("human_review_branch", human_review_branch_node)
    graph.add_node("amendment_branch", amendment_branch_node)
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
        {"continue": "auditor", "failed": END},
    )
    graph.add_conditional_edges(
        "auditor",
        _auditor_outcome_or_end,
        {
            "auto_approve": "auto_approve_branch",
            "flag_review": "human_review_branch",
            "amendment": "amendment_branch",
            "failed": END,
        },
    )
    graph.add_conditional_edges(
        "auto_approve_branch",
        _next_or_end,
        {"continue": "storage", "failed": END},
    )
    graph.add_conditional_edges(
        "human_review_branch",
        _next_or_end,
        {"continue": "storage", "failed": END},
    )
    graph.add_conditional_edges(
        "amendment_branch",
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
        checkpoint_path=_default_checkpoint_path(document_url, shipment_id, document_name),
        customer_rule_set=customer_rule_set or CUSTOMER_RULE_SET,
        master_data=MASTER_DATA,
        field_schema=_field_schema_from_rules(customer_rule_set or CUSTOMER_RULE_SET),
    )
    result = app.invoke(_model_to_dict(initial_state))
    return _model_to_dict(_coerce_state(result))


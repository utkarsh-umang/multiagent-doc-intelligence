"""
Shared LangGraph pipeline state.

Every agent reads from and writes to this object through graph node updates.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PipelineStatus = Literal[
    "pending",
    "extracting",
    "extracted",
    "validating",
    "validated",
    "auditing",
    "audited",
    "auto_approved",
    "flagged_for_review",
    "amendment_drafted",
    "storing",
    "complete",
    "failed",
]


class PipelineState(BaseModel):
    # Input
    document_url: str
    raw_text: str = ""
    field_schema: list[dict[str, Any]] = Field(default_factory=list)
    customer_rule_set: dict[str, Any] = Field(default_factory=dict)
    master_data: dict[str, Any] = Field(default_factory=dict)
    shipment_id: str | None = None
    document_name: str | None = None
    storage_db_path: str = "app.duckdb"
    checkpoint_path: str = "data/checkpoints/latest.json"

    # Extractor output
    extracted_fields: dict[str, Any] = Field(default_factory=dict)
    extraction_metadata: dict[str, Any] = Field(default_factory=dict)
    low_confidence_fields: list[str] = Field(default_factory=list)

    # Validator output
    validation_report: dict[str, Any] = Field(default_factory=dict)
    validation_metadata: dict[str, Any] = Field(default_factory=dict)

    # Auditor output
    decision: str = ""
    reasoning: str = ""
    decision_audit_report: str = ""
    amendment_draft: str = ""
    decision_report: dict[str, Any] = Field(default_factory=dict)
    auditor_metadata: dict[str, Any] = Field(default_factory=dict)
    auditor_branch: str = ""

    # Storage output
    storage_id: str | None = None

    # Pipeline health
    pipeline_status: PipelineStatus = "pending"
    error: str | None = None


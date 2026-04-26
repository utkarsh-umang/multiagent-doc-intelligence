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
    "routing",
    "routed",
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
    shipment_id: str | None = None
    document_name: str | None = None
    storage_db_path: str = "app.duckdb"

    # Extractor output
    extracted_fields: dict[str, Any] = Field(default_factory=dict)
    extraction_metadata: dict[str, Any] = Field(default_factory=dict)
    low_confidence_fields: list[str] = Field(default_factory=list)

    # Validator output
    validation_report: dict[str, Any] = Field(default_factory=dict)

    # Router output
    decision: str = ""
    reasoning: str = ""
    amendment_draft: str = ""
    decision_report: dict[str, Any] = Field(default_factory=dict)

    # Storage output
    storage_id: str | None = None

    # Pipeline health
    pipeline_status: PipelineStatus = "pending"
    error: str | None = None


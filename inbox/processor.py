"""
Multi-document shipment processor.

Takes one email bundle directory (email.json + attachment files), runs the full
Phase 2 pipeline, and returns a structured result dict.

    bundle_dir/
        email.json          ← sender, subject, customer_id, received_at, body
        BOL-2026-001.pdf    ← Bill of Lading
        INVOICE-2026-001.pdf ← Commercial Invoice
        PACKLIST-2026-001.pdf ← Packing List   (any number of attachments)

Pipeline executed here (all agents reused from Phase 1):
    1. ExtractorAgent  — per attachment
    2. cross_validate  — across all attachments
    3. ValidatorAgent  — on highest-confidence merged fields
    4. _inject_cross_doc — inject cross-doc discrepancies into ValidationReport
    5. AuditorAgent    — produce decision + amendment draft
    6. store_pipeline_run — persist to DuckDB

Public API:
    process_bundle(bundle_path, db_path) -> dict[str, Any]
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when run directly
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.auditor import AuditorAgent
from agents.extractor import ExtractorAgent, ExtractionResult
from agents.validator import FieldValidationResult, ValidatorAgent, ValidationReport, ValidationStatus
from pipeline.cross_validate import CrossDocReport, cross_validate
from rules.customer_rules import CUSTOMER_RULE_SET
from storage.db import DEFAULT_DB_PATH, store_pipeline_run


# ---------------------------------------------------------------------------
# Supported attachment extensions
# ---------------------------------------------------------------------------

ATTACHMENT_EXTENSIONS: set[str] = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Customer rule-set registry
# Extend this dict to support multiple customers without changing other code.
# ---------------------------------------------------------------------------

_RULE_REGISTRY: dict[str, dict[str, Any]] = {
    CUSTOMER_RULE_SET["customer_id"]: CUSTOMER_RULE_SET,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_plain_dict(model: Any) -> dict[str, Any]:
    """Serialise a Pydantic model or dataclass to a plain dict."""
    if hasattr(model, "model_dump"):
        try:
            return model.model_dump(mode="json")
        except TypeError:
            return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return asdict(model)  # dataclass fallback


def _read_email_json(bundle_path: Path) -> dict[str, Any]:
    """Parse email.json from the bundle directory."""
    email_file = bundle_path / "email.json"
    if not email_file.exists():
        return {}
    return json.loads(email_file.read_text(encoding="utf-8"))


def _find_attachments(bundle_path: Path) -> list[Path]:
    """Return all document files in the bundle, sorted by name, ignoring email.json."""
    return sorted(
        p
        for p in bundle_path.iterdir()
        if p.is_file() and p.suffix.lower() in ATTACHMENT_EXTENSIONS
    )


def _resolve_rule_set(customer_id: str | None) -> dict[str, Any]:
    """Look up a customer rule set by ID; fall back to the demo rule set."""
    if customer_id and customer_id in _RULE_REGISTRY:
        return _RULE_REGISTRY[customer_id]
    return CUSTOMER_RULE_SET


def _merge_extractions(
    extractions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Merge extracted fields from multiple documents into one dict.

    For each field, keep the value from whichever document extracted it with
    the highest confidence. This gives the Validator the best available
    evidence without duplicating fields.

    Each returned field dict has an additional ``doc_source`` key recording
    which document the winning value came from.
    """
    merged: dict[str, Any] = {}

    all_field_names: set[str] = set()
    for fields in extractions.values():
        all_field_names.update(fields.keys())

    for field_name in all_field_names:
        best_doc: str | None = None
        best_confidence: float = -1.0
        best_payload: dict[str, Any] = {}

        for doc_name, fields in extractions.items():
            payload = fields.get(field_name)
            if not isinstance(payload, dict):
                continue
            confidence = float(payload.get("confidence", 0.0))
            if confidence > best_confidence:
                best_confidence = confidence
                best_doc = doc_name
                best_payload = dict(payload)

        if best_doc is not None:
            best_payload["doc_source"] = best_doc
            merged[field_name] = best_payload

    return merged


def _inject_cross_doc(
    validation_report: ValidationReport,
    cross_report: CrossDocReport,
) -> None:
    """
    Inject cross-document discrepancies into a ValidationReport in-place.

    Rules:
    - If the field is already MISMATCH → augment the reason (don't double-penalise).
    - If the field was MATCH or UNCERTAIN → override to the cross-doc status and
      prepend "Cross-document inconsistency: " to the reason.
    - If the field is not in the report at all → add it as a new MISMATCH entry
      so the Auditor sees it.
    """
    for discrepancy in cross_report.discrepancies:
        field_name = discrepancy.field
        cross_reason_prefix = (
            f"Cross-document inconsistency ({discrepancy.status}): {discrepancy.reason}"
        )
        cross_status = (
            ValidationStatus.MISMATCH
            if discrepancy.status == "mismatch"
            else ValidationStatus.UNCERTAIN
        )

        existing = validation_report.fields.get(field_name)

        if existing is None:
            # Field not validated against customer rules — add it as a new entry
            validation_report.fields[field_name] = FieldValidationResult(
                field=field_name,
                status=cross_status,
                found=None,
                expected=None,
                confidence=0.0,
                reason=cross_reason_prefix,
                source_snippet=None,
            )
        elif existing.status == ValidationStatus.MATCH:
            # Was clean against rules but docs disagree — override
            validation_report.fields[field_name] = FieldValidationResult(
                field=field_name,
                status=cross_status,
                found=existing.found,
                expected=existing.expected,
                confidence=existing.confidence,
                reason=f"{cross_reason_prefix} (previously matched customer rule).",
                source_snippet=existing.source_snippet,
            )
        else:
            # Already mismatch/uncertain — just append cross-doc context to reason
            validation_report.fields[field_name] = FieldValidationResult(
                field=field_name,
                status=existing.status,
                found=existing.found,
                expected=existing.expected,
                confidence=existing.confidence,
                reason=f"{existing.reason} Additionally: {cross_reason_prefix}",
                source_snippet=existing.source_snippet,
            )

    # Recompute overall status and discrepancy summary after injection
    statuses = [r.status for r in validation_report.fields.values()]
    has_mismatches = ValidationStatus.MISMATCH in statuses
    has_uncertain = ValidationStatus.UNCERTAIN in statuses

    validation_report.has_mismatches = has_mismatches
    validation_report.has_uncertain = has_uncertain
    if has_uncertain:
        validation_report.overall_status = "needs_review"
    elif has_mismatches:
        validation_report.overall_status = "failed"
    else:
        validation_report.overall_status = "passed"

    # Refresh discrepancy summary
    from agents.validator import DiscrepancySummary
    validation_report.discrepancy_summary = DiscrepancySummary(
        matched_fields=[
            fn for fn, r in validation_report.fields.items()
            if r.status == ValidationStatus.MATCH
        ],
        mismatched_fields=[
            fn for fn, r in validation_report.fields.items()
            if r.status == ValidationStatus.MISMATCH
        ],
        uncertain_fields=[
            fn for fn, r in validation_report.fields.items()
            if r.status == ValidationStatus.UNCERTAIN
        ],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_bundle(
    bundle_path: Path,
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """
    Process one email bundle directory end-to-end.

    Parameters
    ----------
    bundle_path:
        Path to the bundle directory (e.g. ``inbox/pending/SU-2026-001``).
        Must contain ``email.json`` and at least one document file.
    db_path:
        Path to the DuckDB database file for storing the pipeline run.

    Returns
    -------
    dict with keys:
        shipment_id, email, attachments, cross_doc_report,
        validation_report, decision, storage_id, pipeline_status
    """
    bundle_path = Path(bundle_path)
    shipment_id = bundle_path.name

    email_meta = _read_email_json(bundle_path)
    attachments = _find_attachments(bundle_path)

    if not attachments:
        raise ValueError(
            f"Bundle '{bundle_path}' contains no document attachments. "
            "Add at least one PDF or image file alongside email.json."
        )

    # ------------------------------------------------------------------
    # Step 1: Extract each attachment independently
    # ------------------------------------------------------------------
    extractor = ExtractorAgent()
    extractions: dict[str, dict[str, Any]] = {}

    for att_path in attachments:
        result: ExtractionResult = extractor.extract(str(att_path))
        # Convert ExtractedField objects to plain dicts for downstream use
        extractions[att_path.name] = {
            field_name: _to_plain_dict(field_obj)
            for field_name, field_obj in result.fields.items()
        }

    # ------------------------------------------------------------------
    # Step 2: Cross-document consistency check
    # ------------------------------------------------------------------
    cross_report: CrossDocReport = cross_validate(extractions)

    # ------------------------------------------------------------------
    # Step 3: Merge extractions (highest-confidence per field)
    # ------------------------------------------------------------------
    merged_fields = _merge_extractions(extractions)

    # ------------------------------------------------------------------
    # Step 4: Validate merged fields against customer rules
    # ------------------------------------------------------------------
    rule_set = _resolve_rule_set(email_meta.get("customer_id"))
    validation_report: ValidationReport = ValidatorAgent(rule_set=rule_set).validate(
        merged_fields
    )

    # ------------------------------------------------------------------
    # Step 5: Inject cross-doc discrepancies into the validation report
    # ------------------------------------------------------------------
    _inject_cross_doc(validation_report, cross_report)

    # ------------------------------------------------------------------
    # Step 6: Route and draft amendment (Auditor Agent)
    # ------------------------------------------------------------------
    validation_dict = _to_plain_dict(validation_report)
    decision_report = AuditorAgent().decide(validation_dict)
    decision_dict = _to_plain_dict(decision_report)

    # ------------------------------------------------------------------
    # Step 7: Persist to DuckDB
    # ------------------------------------------------------------------
    storage_id = store_pipeline_run(
        merged_fields,
        validation_dict,
        decision_dict,
        shipment_id=shipment_id,
        document_name=f"{shipment_id} ({len(attachments)} docs)",
        db_path=db_path,
    )

    return {
        "shipment_id": shipment_id,
        "pipeline_status": "complete",
        "email": email_meta,
        "attachments": [a.name for a in attachments],
        "per_doc_extractions": extractions,
        "merged_fields": merged_fields,
        "cross_doc_report": cross_report.to_dict(),
        "validation_report": validation_dict,
        "decision": decision_dict,
        "storage_id": storage_id,
    }

"""
Cross-document field consistency checker.

When a shipment arrives with multiple attachments (BOL, Invoice, Packing List),
key fields like consignee_name and hs_code must be consistent across all documents.
This module compares those fields across all extracted documents and surfaces any
inconsistencies as discrepancies before the Validator / Auditor run.

Usage:
    from pipeline.cross_validate import cross_validate

    cross_report = cross_validate(extractions)  # extractions: {doc_name: extracted_fields}
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Fields compared across documents
# ---------------------------------------------------------------------------

CROSS_CHECK_FIELDS: list[str] = [
    "consignee_name",
    "hs_code",
    "invoice_number",
]

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.6
DEFAULT_SIMILARITY_THRESHOLD: float = 0.86


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class CrossDocDiscrepancy:
    """A field whose value differs (or is missing) across two or more documents."""

    field: str
    values_by_doc: dict[str, str]
    status: Literal["mismatch", "uncertain"]
    reason: str


@dataclass
class CrossDocReport:
    """Full report of all cross-document field checks for one shipment."""

    checked_fields: list[str]
    discrepancies: list[CrossDocDiscrepancy] = field(default_factory=list)
    consistent_fields: list[str] = field(default_factory=list)
    skipped_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def has_discrepancies(self) -> bool:
        return bool(self.discrepancies)

    @property
    def discrepancy_fields(self) -> list[str]:
        return [d.field for d in self.discrepancies]


# ---------------------------------------------------------------------------
# Normalisation helpers (mirrors validator.py logic — no shared import to
# keep this module dependency-free)
# ---------------------------------------------------------------------------


def _normalize(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value.strip().casefold())


def _normalize_code(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", value).casefold()


def _similar_enough(
    a: str,
    b: str,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> bool:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio() >= threshold


def _codes_similar(a: str, b: str) -> bool:
    """HS codes and invoice numbers: compare without punctuation / case."""
    return _normalize_code(a) == _normalize_code(b)


def _values_agree(field_name: str, a: str, b: str) -> bool:
    """Field-aware comparison: codes use exact normalised match; names use fuzzy."""
    if field_name in ("hs_code", "invoice_number"):
        return _codes_similar(a, b)
    return _similar_enough(a, b)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def cross_validate(
    extractions: dict[str, dict[str, Any]],
    fields_to_check: list[str] | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> CrossDocReport:
    """
    Compare key fields across all extracted documents in one shipment.

    Parameters
    ----------
    extractions:
        Mapping of ``doc_name -> extracted_fields`` where ``extracted_fields``
        is the ``fields`` dict from ``ExtractionResult`` (each value is a dict
        with ``value``, ``confidence``, and ``source_snippet`` keys).
    fields_to_check:
        Fields to compare across documents. Defaults to
        ``CROSS_CHECK_FIELDS`` (consignee_name, hs_code, invoice_number).
    confidence_threshold:
        Minimum extractor confidence for a field value to be included in
        the cross-document comparison. Low-confidence extractions are not
        compared — they are already marked uncertain by the Validator.

    Returns
    -------
    CrossDocReport
    """
    checked = fields_to_check or CROSS_CHECK_FIELDS
    report = CrossDocReport(checked_fields=checked)

    for field_name in checked:
        discrepancy = _check_field(
            field_name=field_name,
            extractions=extractions,
            confidence_threshold=confidence_threshold,
        )
        if discrepancy is None:
            # Not enough evidence to compare (< 2 docs with confident values)
            report.skipped_fields.append(field_name)
        elif discrepancy.status in ("mismatch", "uncertain"):
            report.discrepancies.append(discrepancy)
        else:
            report.consistent_fields.append(field_name)

    return report


def _check_field(
    *,
    field_name: str,
    extractions: dict[str, dict[str, Any]],
    confidence_threshold: float,
) -> CrossDocDiscrepancy | None:
    """
    Check one field across all documents.

    Returns
    -------
    None
        If fewer than 2 documents have a high-confidence value for this field
        (not enough data to cross-check).
    CrossDocDiscrepancy with status "consistent"
        All confident values agree.
    CrossDocDiscrepancy with status "mismatch"
        Two or more confident values disagree.
    CrossDocDiscrepancy with status "uncertain"
        Exactly one doc has a confident value; others have None or low
        confidence — cannot confirm consistency.
    """
    confident_values: dict[str, str] = {}
    all_values: dict[str, str] = {}

    for doc_name, fields in extractions.items():
        field_payload = fields.get(field_name, {})
        if not isinstance(field_payload, dict):
            continue
        value = field_payload.get("value")
        confidence = float(field_payload.get("confidence", 0.0))
        if value is not None:
            all_values[doc_name] = str(value)
            if confidence >= confidence_threshold:
                confident_values[doc_name] = str(value)

    if len(confident_values) < 2:
        if len(all_values) >= 2 and len(confident_values) == 1:
            # One doc has a confident value; others extracted something but below
            # threshold — surface as uncertain
            present_doc = next(iter(confident_values))
            present_val = confident_values[present_doc]
            low_conf_docs = {k: v for k, v in all_values.items() if k != present_doc}
            return CrossDocDiscrepancy(
                field=field_name,
                values_by_doc=all_values,
                status="uncertain",
                reason=(
                    f"Only '{present_doc}' has a high-confidence value "
                    f"('{present_val}'). "
                    + ", ".join(
                        f"'{d}' extracted '{v}' with low confidence"
                        for d, v in low_conf_docs.items()
                    )
                    + ". Manual confirmation recommended."
                ),
            )
        return None

    # Two or more docs have confident values — compare all pairs
    doc_names = list(confident_values.keys())
    reference_doc = doc_names[0]
    reference_val = confident_values[reference_doc]
    mismatches: list[tuple[str, str]] = []

    for other_doc in doc_names[1:]:
        other_val = confident_values[other_doc]
        if not _values_agree(field_name, reference_val, other_val):
            mismatches.append((other_doc, other_val))

    if mismatches:
        mismatch_lines = "; ".join(
            f"'{doc}' has '{val}'" for doc, val in mismatches
        )
        return CrossDocDiscrepancy(
            field=field_name,
            values_by_doc=confident_values,
            status="mismatch",
            reason=(
                f"'{reference_doc}' has '{reference_val}' but {mismatch_lines}. "
                "All documents in a shipment must agree on this field."
            ),
        )

    # All agree
    return CrossDocDiscrepancy(
        field=field_name,
        values_by_doc=confident_values,
        status="consistent",  # type: ignore[arg-type]
        reason="All documents agree.",
    )

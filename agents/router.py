"""
Router / decision agent.

Reads the validator report and chooses one of three outcomes:
1. auto-approve and store
2. flag for human review with reasoning
3. draft an amendment request listing discrepancies
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DecisionOutcome(str, Enum):
    AUTO_APPROVE_AND_STORE = "auto_approve_and_store"
    FLAG_FOR_HUMAN_REVIEW = "flag_for_human_review"
    DRAFT_AMENDMENT_REQUEST = "draft_amendment_request"


class ValidationField(BaseModel):
    field: str
    status: str
    found: str | None = None
    expected: str | list[str] | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    source_snippet: str | None = None


class DecisionIssue(BaseModel):
    field: str
    status: str
    found: str | None = None
    expected: str | list[str] | None = None
    reason: str


class DecisionReport(BaseModel):
    outcome: DecisionOutcome
    explanation: str
    should_store: bool
    human_review_reasons: list[DecisionIssue] = Field(default_factory=list)
    amendment_request: str | None = None


def _validate_model(model_class: type[BaseModel], data: dict[str, Any]) -> BaseModel:
    try:
        return model_class.model_validate(data)
    except AttributeError:
        return model_class.parse_obj(data)


def _to_plain_dict(model: BaseModel) -> dict[str, Any]:
    try:
        return model.model_dump(mode="json")
    except TypeError:
        return model.model_dump()
    except AttributeError:
        return json.loads(model.json())


def _format_expected(expected: str | list[str] | None) -> str:
    if expected is None:
        return "not specified"
    if isinstance(expected, list):
        return ", ".join(str(item) for item in expected)
    return str(expected)


def _humanize_field_name(field_name: str) -> str:
    return field_name.replace("_", " ").title()


class RouterAgent:
    """Turns validation results into an operational decision."""

    def decide(self, validation_report: dict[str, Any]) -> DecisionReport:
        fields = self._parse_fields(validation_report)
        uncertain = [field for field in fields if field.status == "uncertain"]
        mismatches = [field for field in fields if field.status == "mismatch"]

        if uncertain:
            return self._flag_for_review(uncertain, mismatches)

        if mismatches:
            return self._draft_amendment(mismatches)

        return DecisionReport(
            outcome=DecisionOutcome.AUTO_APPROVE_AND_STORE,
            explanation=(
                "All required fields matched the customer's rules with sufficient "
                "extractor confidence. The shipment can be auto-approved and stored."
            ),
            should_store=True,
        )

    def _parse_fields(self, validation_report: dict[str, Any]) -> list[ValidationField]:
        raw_fields = validation_report.get("fields", {})
        parsed_fields: list[ValidationField] = []

        for field_name, payload in raw_fields.items():
            field_payload = dict(payload)
            field_payload.setdefault("field", field_name)
            field = _validate_model(ValidationField, field_payload)
            if not isinstance(field, ValidationField):
                raise TypeError(f"Invalid validator field payload for {field_name}")
            parsed_fields.append(field)

        return parsed_fields

    def _flag_for_review(
        self,
        uncertain: list[ValidationField],
        mismatches: list[ValidationField],
    ) -> DecisionReport:
        review_issues = [
            self._issue(field)
            for field in [*uncertain, *mismatches]
        ]
        uncertain_names = ", ".join(_humanize_field_name(field.field) for field in uncertain)
        mismatch_note = ""
        if mismatches:
            mismatch_names = ", ".join(_humanize_field_name(field.field) for field in mismatches)
            mismatch_note = f" Mismatches were also found in: {mismatch_names}."

        return DecisionReport(
            outcome=DecisionOutcome.FLAG_FOR_HUMAN_REVIEW,
            explanation=(
                "The shipment cannot be auto-approved because one or more fields are "
                f"uncertain: {uncertain_names}.{mismatch_note} A human should review "
                "these fields before approval or amendment."
            ),
            should_store=False,
            human_review_reasons=review_issues,
        )

    def _draft_amendment(self, mismatches: list[ValidationField]) -> DecisionReport:
        amendment_request = self._build_amendment_request(mismatches)
        mismatch_names = ", ".join(_humanize_field_name(field.field) for field in mismatches)

        return DecisionReport(
            outcome=DecisionOutcome.DRAFT_AMENDMENT_REQUEST,
            explanation=(
                "The shipment has confident validation mismatches and should not be "
                f"approved as submitted. An amendment request was drafted for: {mismatch_names}."
            ),
            should_store=False,
            human_review_reasons=[self._issue(field) for field in mismatches],
            amendment_request=amendment_request,
        )

    def _issue(self, field: ValidationField) -> DecisionIssue:
        return DecisionIssue(
            field=field.field,
            status=field.status,
            found=field.found,
            expected=field.expected,
            reason=field.reason,
        )

    def _build_amendment_request(self, mismatches: list[ValidationField]) -> str:
        lines = [
            "Subject: Amendment required for submitted shipment documents",
            "",
            "Hello,",
            "",
            "We reviewed the submitted shipment documents and found the following discrepancies:",
        ]

        for field in mismatches:
            lines.extend(
                [
                    "",
                    f"- {_humanize_field_name(field.field)}",
                    f"  Found: {field.found or 'missing'}",
                    f"  Expected: {_format_expected(field.expected)}",
                    f"  Reason: {field.reason}",
                ]
            )

        lines.extend(
            [
                "",
                "Please amend the document(s) and resubmit the corrected version for review.",
                "",
                "Regards,",
                "GoComet Nova",
            ]
        )
        return "\n".join(lines)


def run(validation_report: dict[str, Any]) -> dict[str, Any]:
    """Create an auditable decision from the validator output."""

    decision = RouterAgent().decide(validation_report)
    return _to_plain_dict(decision)


def route(state: dict[str, Any]) -> str:
    """
    LangGraph-friendly conditional route.

    If `state` already contains a decision, use it. Otherwise, compute one from
    `state["validation_report"]`.
    """

    decision = state.get("decision")
    if not decision:
        validation_report = state.get("validation_report")
        if validation_report is None:
            raise ValueError("Router state must include `validation_report`.")
        decision = run(validation_report)
        state["decision"] = decision

    return str(decision["outcome"])


"""
Auditor / decision agent.

Reads the Validator discrepancy report and chooses one operational outcome:
1. auto-approve and store
2. flag for human review with reasoning
3. draft an amendment request listing discrepancies
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any, Literal

import litellm  # type: ignore[import-not-found]
from pydantic import BaseModel, Field


DEFAULT_AUDITOR_MODEL = os.getenv("AUDITOR_MODEL", os.getenv("OPENAI_AUDITOR_MODEL", "gpt-4o"))


class DecisionOutcome(str, Enum):
    AUTO_APPROVE_AND_STORE = "auto_approve_and_store"
    FLAG_FOR_HUMAN_REVIEW = "flag_for_human_review"
    DRAFT_AMENDMENT_REQUEST = "draft_amendment_request"


PolicyPath = Literal[
    "deterministic_auto_approve",
    "deterministic_human_review",
    "deterministic_amendment",
]


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


class AuditorMetadata(BaseModel):
    model: str
    policy_path: PolicyPath
    issues_considered: list[DecisionIssue] = Field(default_factory=list)
    used_model_report: bool = False
    used_model_amendment: bool = False
    fallback_error: str | None = None


class DecisionReport(BaseModel):
    outcome: DecisionOutcome
    explanation: str
    should_store: bool
    decision_audit_report: str
    human_review_reasons: list[DecisionIssue] = Field(default_factory=list)
    amendment_request: str | None = None
    metadata: AuditorMetadata


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


class AuditorAgent:
    """Turns Validator discrepancy reports into operational decisions."""

    def __init__(
        self,
        model: str = DEFAULT_AUDITOR_MODEL,
        completion_fn: Any | None = None,
    ) -> None:
        self.model = model
        self.completion_fn = completion_fn or litellm.completion

    def decide(self, validation_report: dict[str, Any]) -> DecisionReport:
        fields = self._parse_fields(validation_report)
        uncertain = [field for field in fields if field.status == "uncertain"]
        mismatches = [field for field in fields if field.status == "mismatch"]

        if uncertain:
            outcome = DecisionOutcome.FLAG_FOR_HUMAN_REVIEW
            policy_path: PolicyPath = "deterministic_human_review"
            issues = [self._issue(field) for field in [*uncertain, *mismatches]]
            should_store = False
        elif mismatches:
            outcome = DecisionOutcome.DRAFT_AMENDMENT_REQUEST
            policy_path = "deterministic_amendment"
            issues = [self._issue(field) for field in mismatches]
            should_store = False
        else:
            outcome = DecisionOutcome.AUTO_APPROVE_AND_STORE
            policy_path = "deterministic_auto_approve"
            issues = []
            should_store = True

        metadata = AuditorMetadata(
            model=self.model,
            policy_path=policy_path,
            issues_considered=issues,
        )

        audit_payload = self._generate_audit_payload(
            validation_report=validation_report,
            outcome=outcome,
            issues=issues,
            policy_path=policy_path,
            metadata=metadata,
        )
        amendment_request = None
        if outcome == DecisionOutcome.DRAFT_AMENDMENT_REQUEST:
            amendment_request = self._generate_amendment_request(
                validation_report,
                mismatches,
                metadata,
            )

        return DecisionReport(
            outcome=outcome,
            explanation=audit_payload["explanation"],
            should_store=should_store,
            decision_audit_report=audit_payload["decision_audit_report"],
            human_review_reasons=issues,
            amendment_request=amendment_request,
            metadata=metadata,
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

    def _generate_audit_payload(
        self,
        *,
        validation_report: dict[str, Any],
        outcome: DecisionOutcome,
        issues: list[DecisionIssue],
        policy_path: PolicyPath,
        metadata: AuditorMetadata,
    ) -> dict[str, str]:
        try:
            payload = self._call_auditor_model(
                validation_report=validation_report,
                outcome=outcome,
                issues=issues,
                policy_path=policy_path,
            )
            metadata.used_model_report = True
            return {
                "explanation": str(payload["explanation"]),
                "decision_audit_report": str(payload["decision_audit_report"]),
            }
        except Exception as exc:
            metadata.fallback_error = f"{type(exc).__name__}: {exc}"
            return self._fallback_audit_payload(validation_report, outcome, issues)

    def _call_auditor_model(
        self,
        *,
        validation_report: dict[str, Any],
        outcome: DecisionOutcome,
        issues: list[DecisionIssue],
        policy_path: PolicyPath,
    ) -> dict[str, Any]:
        prompt = {
            "task": "Write an audit report for a trade-document validation decision.",
            "deterministic_policy": {
                "selected_outcome": outcome.value,
                "policy_path": policy_path,
                "policy_must_not_be_overridden": True,
            },
            "validation_report": validation_report,
            "issues_considered": [_to_plain_dict(issue) for issue in issues],
            "instructions": [
                "Explain why the selected outcome follows from the Validator discrepancy report.",
                "Summarize what happened during validation using only the validation report.",
                "For human review, identify the exact uncertain fields the CG operator should inspect.",
                "For auto-approval, explain why approval is safe based on matched fields.",
                "For amendment, summarize the mismatches without drafting the email here.",
                "Return JSON only with keys: explanation, decision_audit_report.",
            ],
        }
        response = self.completion_fn(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(prompt),
                }
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Auditor model returned an empty audit report")
        return json.loads(content)

    def _generate_amendment_request(
        self,
        validation_report: dict[str, Any],
        mismatches: list[ValidationField],
        metadata: AuditorMetadata,
    ) -> str:
        try:
            amendment = self._call_amendment_model(validation_report, mismatches)
            metadata.used_model_amendment = True
            return amendment
        except Exception as exc:
            fallback = f"{type(exc).__name__}: {exc}"
            metadata.fallback_error = (
                f"{metadata.fallback_error}; {fallback}"
                if metadata.fallback_error
                else fallback
            )
            return self._fallback_amendment_request(mismatches)

    def _call_amendment_model(
        self,
        validation_report: dict[str, Any],
        mismatches: list[ValidationField],
    ) -> str:
        prompt = {
            "task": "Draft a supplier-facing amendment request for trade-document discrepancies.",
            "validation_report": validation_report,
            "mismatched_fields": [_to_plain_dict(field) for field in mismatches],
            "instructions": [
                "Write natural language email text, not JSON.",
                "List every mismatch clearly with found value, expected value, and reason.",
                "Do not mention internal model names or implementation details.",
                "Ask the supplier to amend and resubmit the corrected document.",
            ],
        }
        response = self.completion_fn(
            model=self.model,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(prompt),
                }
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Auditor model returned an empty amendment request")
        return str(content)

    def _fallback_audit_payload(
        self,
        validation_report: dict[str, Any],
        outcome: DecisionOutcome,
        issues: list[DecisionIssue],
    ) -> dict[str, str]:
        if outcome == DecisionOutcome.AUTO_APPROVE_AND_STORE:
            explanation = (
                "All fields in the Validator discrepancy report matched customer and "
                "compliance requirements, so the shipment can be auto-approved."
            )
        elif outcome == DecisionOutcome.FLAG_FOR_HUMAN_REVIEW:
            fields = ", ".join(_humanize_field_name(issue.field) for issue in issues)
            explanation = (
                "The shipment cannot be auto-approved because the Validator reported "
                f"uncertain field(s): {fields}. A CG operator should review them."
            )
        else:
            fields = ", ".join(_humanize_field_name(issue.field) for issue in issues)
            explanation = (
                "The shipment has confident validation mismatches and requires an "
                f"amendment request for: {fields}."
            )

        evidence = validation_report.get("evidence_summary", [])
        report_lines = [
            f"Decision: {outcome.value}",
            "",
            explanation,
            "",
            "Validation evidence:",
            *[f"- {item}" for item in evidence],
        ]
        return {
            "explanation": explanation,
            "decision_audit_report": "\n".join(report_lines),
        }

    def _issue(self, field: ValidationField) -> DecisionIssue:
        return DecisionIssue(
            field=field.field,
            status=field.status,
            found=field.found,
            expected=field.expected,
            reason=field.reason,
        )

    def _fallback_amendment_request(self, mismatches: list[ValidationField]) -> str:
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
    """Create an auditable decision from the Validator discrepancy report."""

    decision = AuditorAgent().decide(validation_report)
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
            raise ValueError("Auditor state must include `validation_report`.")
        decision = run(validation_report)
        state["decision"] = decision

    return str(decision["outcome"])

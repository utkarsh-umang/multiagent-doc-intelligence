"""
Validator agent for extracted trade-document fields.

Takes extractor JSON plus a customer rule set and produces field-by-field
validation results: match, mismatch, or uncertain.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from rules.customer_rules import CUSTOMER_RULE_SET


class ValidationStatus(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNCERTAIN = "uncertain"


class ExtractedFieldInput(BaseModel):
    value: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source_snippet: str | None = None


class FieldValidationResult(BaseModel):
    field: str
    status: ValidationStatus
    found: str | None = None
    expected: str | list[str] | None = None
    confidence: float
    reason: str
    source_snippet: str | None = None


class ValidationReport(BaseModel):
    customer_id: str
    customer_name: str
    overall_status: Literal["passed", "failed", "needs_review"]
    has_mismatches: bool
    has_uncertain: bool
    fields: dict[str, FieldValidationResult]


def _validate_model(model_class: type[BaseModel], data: dict[str, Any]) -> BaseModel:
    try:
        return model_class.model_validate(data)
    except AttributeError:
        return model_class.parse_obj(data)


def _normalize(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value.strip().casefold())


def _normalize_code(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", value).casefold()


def _similar_enough(found: str, expected: str, threshold: float = 0.86) -> bool:
    return SequenceMatcher(None, _normalize(found), _normalize(expected)).ratio() >= threshold


def _extract_number(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", value)
    if not match:
        return None
    return float(match.group(0).replace(",", ""))


def _expected_label(rule: dict[str, Any]) -> str | list[str] | None:
    if rule["type"] == "numeric_range":
        unit = rule.get("unit")
        suffix = f" {unit}" if unit else ""
        return f"{rule.get('min')} to {rule.get('max')}{suffix}"
    return rule.get("description") or rule.get("expected")


class ValidatorAgent:
    """Rule-based validator for one customer's logistics requirements."""

    def __init__(self, rule_set: dict[str, Any] | None = None) -> None:
        self.rule_set = rule_set or CUSTOMER_RULE_SET
        self.confidence_threshold = float(self.rule_set.get("confidence_threshold", 0.6))

    def validate(self, extracted: dict[str, Any]) -> ValidationReport:
        results: dict[str, FieldValidationResult] = {}

        for field_name, rule in self.rule_set["fields"].items():
            field = _validate_model(ExtractedFieldInput, extracted.get(field_name, {}))
            if not isinstance(field, ExtractedFieldInput):
                raise TypeError(f"Invalid extracted field payload for {field_name}")
            results[field_name] = self._validate_field(field_name, field, rule)

        statuses = [result.status for result in results.values()]
        has_mismatches = ValidationStatus.MISMATCH in statuses
        has_uncertain = ValidationStatus.UNCERTAIN in statuses
        if has_uncertain:
            overall_status: Literal["passed", "failed", "needs_review"] = "needs_review"
        elif has_mismatches:
            overall_status = "failed"
        else:
            overall_status = "passed"

        return ValidationReport(
            customer_id=self.rule_set["customer_id"],
            customer_name=self.rule_set["customer_name"],
            overall_status=overall_status,
            has_mismatches=has_mismatches,
            has_uncertain=has_uncertain,
            fields=results,
        )

    def _validate_field(
        self,
        field_name: str,
        field: ExtractedFieldInput,
        rule: dict[str, Any],
    ) -> FieldValidationResult:
        expected = _expected_label(rule)
        found = field.value

        if rule.get("required", False) and not found:
            return self._result(
                field_name,
                ValidationStatus.UNCERTAIN,
                field,
                expected,
                "Required field was not extracted.",
            )

        if field.confidence < self.confidence_threshold:
            return self._result(
                field_name,
                ValidationStatus.UNCERTAIN,
                field,
                expected,
                (
                    f"Extractor confidence {field.confidence:.2f} is below "
                    f"threshold {self.confidence_threshold:.2f}."
                ),
            )

        if not found:
            return self._result(
                field_name,
                ValidationStatus.UNCERTAIN,
                field,
                expected,
                "No value found to validate.",
            )

        rule_type = rule["type"]
        if rule_type == "equals":
            return self._validate_equals(field_name, field, rule)
        if rule_type == "allowed_values":
            return self._validate_allowed_values(field_name, field, rule)
        if rule_type == "prefix":
            return self._validate_prefix(field_name, field, rule)
        if rule_type == "contains_any":
            return self._validate_contains_any(field_name, field, rule)
        if rule_type == "numeric_range":
            return self._validate_numeric_range(field_name, field, rule)
        if rule_type == "regex":
            return self._validate_regex(field_name, field, rule)

        return self._result(
            field_name,
            ValidationStatus.UNCERTAIN,
            field,
            expected,
            f"Unsupported validation rule type: {rule_type}.",
        )

    def _validate_equals(
        self,
        field_name: str,
        field: ExtractedFieldInput,
        rule: dict[str, Any],
    ) -> FieldValidationResult:
        candidates = [rule["expected"], *rule.get("aliases", [])]
        matched = any(_similar_enough(field.value or "", candidate) for candidate in candidates)
        if matched:
            return self._result(
                field_name,
                ValidationStatus.MATCH,
                field,
                rule["expected"],
                "Extracted value matches the expected customer value.",
            )
        return self._result(
            field_name,
            ValidationStatus.MISMATCH,
            field,
            rule["expected"],
            "Extracted value does not match the expected customer value.",
        )

    def _validate_allowed_values(
        self,
        field_name: str,
        field: ExtractedFieldInput,
        rule: dict[str, Any],
    ) -> FieldValidationResult:
        allowed_values = rule["expected"]
        matched = any(_similar_enough(field.value or "", allowed) for allowed in allowed_values)
        if matched:
            return self._result(
                field_name,
                ValidationStatus.MATCH,
                field,
                allowed_values,
                "Extracted value is in the allowed customer rule set.",
            )
        return self._result(
            field_name,
            ValidationStatus.MISMATCH,
            field,
            allowed_values,
            "Extracted value is not in the allowed customer rule set.",
        )

    def _validate_prefix(
        self,
        field_name: str,
        field: ExtractedFieldInput,
        rule: dict[str, Any],
    ) -> FieldValidationResult:
        found_code = _normalize_code(field.value)
        expected_prefixes = [_normalize_code(prefix) for prefix in rule["expected"]]
        matched = any(found_code.startswith(prefix) for prefix in expected_prefixes)
        if matched:
            return self._result(
                field_name,
                ValidationStatus.MATCH,
                field,
                rule["expected"],
                "Extracted code starts with an allowed prefix.",
            )
        return self._result(
            field_name,
            ValidationStatus.MISMATCH,
            field,
            rule["expected"],
            "Extracted code does not start with an allowed prefix.",
        )

    def _validate_contains_any(
        self,
        field_name: str,
        field: ExtractedFieldInput,
        rule: dict[str, Any],
    ) -> FieldValidationResult:
        found = _normalize(field.value)
        expected_terms = rule["expected"]
        matched = any(_normalize(term) in found for term in expected_terms)
        if matched:
            return self._result(
                field_name,
                ValidationStatus.MATCH,
                field,
                expected_terms,
                "Extracted text contains an expected goods keyword.",
            )
        return self._result(
            field_name,
            ValidationStatus.MISMATCH,
            field,
            expected_terms,
            "Extracted text does not contain an expected goods keyword.",
        )

    def _validate_numeric_range(
        self,
        field_name: str,
        field: ExtractedFieldInput,
        rule: dict[str, Any],
    ) -> FieldValidationResult:
        number = _extract_number(field.value)
        expected = _expected_label(rule)
        if number is None:
            return self._result(
                field_name,
                ValidationStatus.UNCERTAIN,
                field,
                expected,
                "Could not parse a numeric value from the extracted field.",
            )

        minimum = float(rule["min"])
        maximum = float(rule["max"])
        if minimum <= number <= maximum:
            return self._result(
                field_name,
                ValidationStatus.MATCH,
                field,
                expected,
                "Extracted numeric value is within the allowed range.",
            )
        return self._result(
            field_name,
            ValidationStatus.MISMATCH,
            field,
            expected,
            "Extracted numeric value is outside the allowed range.",
        )

    def _validate_regex(
        self,
        field_name: str,
        field: ExtractedFieldInput,
        rule: dict[str, Any],
    ) -> FieldValidationResult:
        matched = re.fullmatch(rule["expected"], field.value or "") is not None
        expected = _expected_label(rule)
        if matched:
            return self._result(
                field_name,
                ValidationStatus.MATCH,
                field,
                expected,
                "Extracted value matches the required format.",
            )
        return self._result(
            field_name,
            ValidationStatus.MISMATCH,
            field,
            expected,
            "Extracted value does not match the required format.",
        )

    def _result(
        self,
        field_name: str,
        status: ValidationStatus,
        field: ExtractedFieldInput,
        expected: str | list[str] | None,
        reason: str,
    ) -> FieldValidationResult:
        return FieldValidationResult(
            field=field_name,
            status=status,
            found=field.value,
            expected=expected,
            confidence=field.confidence,
            reason=reason,
            source_snippet=field.source_snippet,
        )


def _to_plain_dict(model: BaseModel) -> dict[str, Any]:
    try:
        return model.model_dump()
    except AttributeError:
        return model.dict()


def run(
    extracted: dict[str, Any],
    rule_set: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate extracted JSON against customer rules."""

    report = ValidatorAgent(rule_set=rule_set).validate(extracted)
    return _to_plain_dict(report)


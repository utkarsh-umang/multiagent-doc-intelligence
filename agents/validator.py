"""
Validator agent for extracted trade-document fields.

Takes extractor JSON plus a customer rule set and produces field-by-field
validation results: match, mismatch, or uncertain.
"""

from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Literal

import litellm  # type: ignore[import-not-found]
from pydantic import BaseModel, Field

from rules.customer_rules import CUSTOMER_RULE_SET
from rules.master_data import MASTER_DATA
from rules.retrieval import resolve_port, retrieve_validation_context


DEFAULT_VALIDATOR_MODEL = os.getenv(
    "VALIDATOR_MODEL",
    os.getenv("ANTHROPIC_VALIDATOR_MODEL", "anthropic/claude-3-5-sonnet-20241022"),
)


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


class ValidationMetadata(BaseModel):
    model: str
    deterministic_checks: list[str] = Field(default_factory=list)
    semantic_checks: list[str] = Field(default_factory=list)
    retrieved_context: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    retrieval_tool: str = "rules.retrieval.retrieve_validation_context"
    semantic_errors: dict[str, str] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    customer_id: str
    customer_name: str
    overall_status: Literal["passed", "failed", "needs_review"]
    has_mismatches: bool
    has_uncertain: bool
    fields: dict[str, FieldValidationResult]
    metadata: ValidationMetadata


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


def _field_value(extracted: dict[str, Any], field_name: str) -> str | None:
    payload = extracted.get(field_name, {})
    if not isinstance(payload, dict):
        return None
    value = payload.get("value")
    return str(value) if value is not None else None


def _expected_label(rule: dict[str, Any]) -> str | list[str] | None:
    if rule["type"] == "numeric_range":
        unit = rule.get("unit")
        suffix = f" {unit}" if unit else ""
        return f"{rule.get('min')} to {rule.get('max')}{suffix}"
    return rule.get("description") or rule.get("expected")


class ValidatorAgent:
    """Rule and semantic validator for one customer's logistics requirements."""

    def __init__(
        self,
        rule_set: dict[str, Any] | None = None,
        master_data: dict[str, Any] | None = None,
        model: str = DEFAULT_VALIDATOR_MODEL,
        completion_fn: Any | None = None,
        retrieval_fn: Any | None = None,
    ) -> None:
        self.rule_set = rule_set or CUSTOMER_RULE_SET
        self.master_data = master_data or MASTER_DATA
        self.model = model
        self.completion_fn = completion_fn or litellm.completion
        self.retrieval_fn = retrieval_fn or retrieve_validation_context
        self.confidence_threshold = float(self.rule_set.get("confidence_threshold", 0.6))

    def validate(self, extracted: dict[str, Any]) -> ValidationReport:
        results: dict[str, FieldValidationResult] = {}
        metadata = ValidationMetadata(model=self.model)

        for field_name, rule in self.rule_set["fields"].items():
            field = _validate_model(ExtractedFieldInput, extracted.get(field_name, {}))
            if not isinstance(field, ExtractedFieldInput):
                raise TypeError(f"Invalid extracted field payload for {field_name}")
            context = self.retrieval_fn(
                field_name=field_name,
                extracted_fields=extracted,
                rule_set=self.rule_set,
                master_data=self.master_data,
            )
            metadata.retrieved_context[field_name] = context
            results[field_name] = self._validate_field(
                field_name,
                field,
                rule,
                context,
                metadata,
            )

        self._apply_contextual_checks(extracted, results, metadata)

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
            metadata=metadata,
        )

    def _validate_field(
        self,
        field_name: str,
        field: ExtractedFieldInput,
        rule: dict[str, Any],
        context: list[dict[str, Any]],
        metadata: ValidationMetadata,
    ) -> FieldValidationResult:
        expected = _expected_label(rule)
        found = field.value

        if rule.get("required", False) and not found:
            metadata.deterministic_checks.append(field_name)
            return self._result(
                field_name,
                ValidationStatus.UNCERTAIN,
                field,
                expected,
                "Required field was not extracted.",
            )

        if field.confidence < self.confidence_threshold:
            metadata.deterministic_checks.append(field_name)
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
            metadata.deterministic_checks.append(field_name)
            return self._result(
                field_name,
                ValidationStatus.UNCERTAIN,
                field,
                expected,
                "No value found to validate.",
            )

        rule_type = rule["type"]
        if rule_type == "equals":
            metadata.deterministic_checks.append(field_name)
            return self._validate_equals(field_name, field, rule)
        if rule_type == "allowed_values":
            metadata.deterministic_checks.append(field_name)
            return self._validate_allowed_values(field_name, field, rule)
        if rule_type == "prefix":
            metadata.deterministic_checks.append(field_name)
            return self._validate_prefix(field_name, field, rule)
        if rule_type == "contains_any":
            return self._validate_semantic_contains_any(
                field_name,
                field,
                rule,
                context,
                metadata,
            )
        if rule_type == "numeric_range":
            metadata.deterministic_checks.append(field_name)
            return self._validate_numeric_range(field_name, field, rule)
        if rule_type == "regex":
            metadata.deterministic_checks.append(field_name)
            return self._validate_regex(field_name, field, rule)

        metadata.deterministic_checks.append(field_name)
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

    def _validate_semantic_contains_any(
        self,
        field_name: str,
        field: ExtractedFieldInput,
        rule: dict[str, Any],
        context: list[dict[str, Any]],
        metadata: ValidationMetadata,
    ) -> FieldValidationResult:
        metadata.semantic_checks.append(field_name)
        try:
            return self._call_semantic_validator(field_name, field, rule, context)
        except Exception as exc:
            metadata.semantic_errors[field_name] = f"{type(exc).__name__}: {exc}"
            metadata.deterministic_checks.append(field_name)
            fallback = self._validate_contains_any(field_name, field, rule)
            return FieldValidationResult(
                field=fallback.field,
                status=fallback.status,
                found=fallback.found,
                expected=fallback.expected,
                confidence=fallback.confidence,
                reason=(
                    "Semantic validator was unavailable, so deterministic keyword "
                    f"validation was used. {fallback.reason}"
                ),
                source_snippet=fallback.source_snippet,
            )

    def _call_semantic_validator(
        self,
        field_name: str,
        field: ExtractedFieldInput,
        rule: dict[str, Any],
        context: list[dict[str, Any]],
    ) -> FieldValidationResult:
        expected = _expected_label(rule)
        prompt = {
            "task": "Validate an extracted logistics document field against customer and domain rules.",
            "field": field_name,
            "found": field.value,
            "expected": expected,
            "source_snippet": field.source_snippet,
            "retrieved_context": context,
            "allowed_statuses": ["match", "mismatch", "uncertain"],
            "instructions": [
                "Use retrieved context for semantic equivalence and logistics terminology.",
                "Return JSON only with keys: status, reason.",
                "Use uncertain if the evidence is insufficient.",
                "Do not make an approval/routing decision.",
            ],
        }
        response = self.completion_fn(
            model=self.model,
            temperature=0,
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
            raise ValueError("Validator model returned an empty response")
        payload = json.loads(content)
        status = ValidationStatus(payload["status"])
        return self._result(
            field_name,
            status,
            field,
            expected,
            str(payload["reason"]),
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

    def _apply_contextual_checks(
        self,
        extracted: dict[str, Any],
        results: dict[str, FieldValidationResult],
        metadata: ValidationMetadata,
    ) -> None:
        metadata.deterministic_checks.extend(
            [
                "master_port_resolution",
                "country_compliance_rules",
                "hs_description_consistency",
            ]
        )
        self._validate_port_master(results)
        self._validate_country_compliance(extracted, results)
        self._validate_hs_description_consistency(extracted, results)

    def _validate_port_master(self, results: dict[str, FieldValidationResult]) -> None:
        for field_name in ("port_of_loading", "port_of_discharge"):
            result = results.get(field_name)
            if not result or result.status == ValidationStatus.UNCERTAIN or not result.found:
                continue
            if resolve_port(result.found, self.master_data):
                continue
            results[field_name] = FieldValidationResult(
                field=result.field,
                status=ValidationStatus.UNCERTAIN,
                found=result.found,
                expected=result.expected,
                confidence=result.confidence,
                reason=(
                    f"{result.reason} However, the port could not be resolved in "
                    "master data, so it requires review."
                ),
                source_snippet=result.source_snippet,
            )

    def _validate_country_compliance(
        self,
        extracted: dict[str, Any],
        results: dict[str, FieldValidationResult],
    ) -> None:
        incoterm = _field_value(extracted, "incoterms")
        discharge_port = resolve_port(_field_value(extracted, "port_of_discharge"), self.master_data)
        if not incoterm or not discharge_port:
            return

        destination_country = discharge_port.get("country")
        compliance_context = retrieve_validation_context(
            field_name="incoterms",
            extracted_fields=extracted,
            rule_set=self.rule_set,
            master_data=self.master_data,
        )
        import_rules = [
            item.get("data", {})
            for item in compliance_context
            if item.get("source") == "country_compliance"
            and item.get("country") == destination_country
            and item.get("direction") == "imports"
        ]
        if not import_rules:
            return

        allowed_incoterms = import_rules[0].get("allowed_incoterms", [])
        if allowed_incoterms and _normalize(incoterm).upper() not in allowed_incoterms:
            current = results["incoterms"]
            results["incoterms"] = FieldValidationResult(
                field=current.field,
                status=ValidationStatus.MISMATCH,
                found=current.found,
                expected=allowed_incoterms,
                confidence=current.confidence,
                reason=(
                    f"Incoterm is not allowed for imports into {destination_country} "
                    "under retrieved country compliance rules."
                ),
                source_snippet=current.source_snippet,
            )

    def _validate_hs_description_consistency(
        self,
        extracted: dict[str, Any],
        results: dict[str, FieldValidationResult],
    ) -> None:
        hs_code = _field_value(extracted, "hs_code")
        description = _field_value(extracted, "description_of_goods")
        if not hs_code or not description:
            return

        hs_rules = self.master_data.get("hs_code_rules", {})
        matching_rule = None
        for prefix, rule in hs_rules.items():
            if _normalize_code(hs_code).startswith(_normalize_code(prefix)):
                matching_rule = rule
                break
        if not matching_rule:
            return

        allowed_descriptions = matching_rule.get("allowed_descriptions", [])
        matched = any(
            _normalize(term) in _normalize(description)
            or _similar_enough(description, term, threshold=0.72)
            for term in allowed_descriptions
        )
        if matched:
            return

        current = results["description_of_goods"]
        results["description_of_goods"] = FieldValidationResult(
            field=current.field,
            status=ValidationStatus.MISMATCH,
            found=current.found,
            expected=allowed_descriptions,
            confidence=current.confidence,
            reason=(
                "Goods description does not semantically align with the retrieved "
                f"HS category `{matching_rule.get('category')}`."
            ),
            source_snippet=current.source_snippet,
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
    master_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate extracted JSON against customer rules."""

    report = ValidatorAgent(rule_set=rule_set, master_data=master_data).validate(extracted)
    return _to_plain_dict(report)


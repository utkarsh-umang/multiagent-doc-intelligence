"""
Local retrieval facade for Validator rule lookups.

The PRD calls for the Validator to fetch customer, country, and domain rules via
tools. This module keeps that boundary explicit while using local dummy data.
"""

from __future__ import annotations

import re
from typing import Any

from rules.compliance_rules import COUNTRY_COMPLIANCE_RULES
from rules.master_data import MASTER_DATA


def _normalize(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value.strip().casefold())


def _field_value(extracted_fields: dict[str, Any], field_name: str) -> str | None:
    payload = extracted_fields.get(field_name, {})
    if not isinstance(payload, dict):
        return None
    value = payload.get("value")
    return str(value) if value is not None else None


_UNLOCODE_TOKEN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{3}\b")


def _word_bounded_contains(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return re.search(rf"(?<![\w]){re.escape(needle)}(?![\w])", haystack) is not None


def resolve_port(value: str | None, master_data: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Resolve a port from canonical names, aliases, or UN/LOCODE.

    Tolerates real-world Bill of Lading formats such as
    ``"SHANGHAI, CHINA (CNSHA)"`` or ``"NHAVA SHEVA (JNPT), INDIA"`` by:

    1. Exact normalized equality against canonical name / unlocode / aliases.
    2. Any UN/LOCODE-shaped token in the value matching a port's unlocode
       (or an alias, since aliases sometimes carry the code).
    3. Whole-word (token-bounded) substring match of any candidate against
       the normalized value.
    """

    if not value:
        return None

    data = master_data or MASTER_DATA
    ports = data.get("ports", {})
    needle = _normalize(value)

    for canonical_name, payload in ports.items():
        candidates = [
            canonical_name,
            payload.get("unlocode"),
            *payload.get("aliases", []),
        ]
        if any(_normalize(candidate) == needle for candidate in candidates if candidate):
            return {"name": canonical_name, **payload}

    unlocode_tokens = {tok.upper() for tok in _UNLOCODE_TOKEN_RE.findall(value.upper())}
    if unlocode_tokens:
        for canonical_name, payload in ports.items():
            unlocode = (payload.get("unlocode") or "").upper()
            aliases_upper = {str(a).upper() for a in payload.get("aliases", [])}
            if unlocode and unlocode in unlocode_tokens:
                return {"name": canonical_name, **payload}
            if unlocode_tokens & aliases_upper:
                return {"name": canonical_name, **payload}

    for canonical_name, payload in ports.items():
        candidates = [canonical_name, *payload.get("aliases", [])]
        for candidate in candidates:
            if not candidate:
                continue
            normalized_candidate = _normalize(candidate)
            if len(normalized_candidate) < 3:
                continue
            if _word_bounded_contains(needle, normalized_candidate):
                return {"name": canonical_name, **payload}

    return None


def infer_trade_countries(
    extracted_fields: dict[str, Any],
    master_data: dict[str, Any] | None = None,
) -> dict[str, str | None]:
    loading_port = resolve_port(_field_value(extracted_fields, "port_of_loading"), master_data)
    discharge_port = resolve_port(_field_value(extracted_fields, "port_of_discharge"), master_data)
    return {
        "origin_country": loading_port.get("country") if loading_port else None,
        "destination_country": discharge_port.get("country") if discharge_port else None,
    }


def retrieve_validation_context(
    *,
    field_name: str,
    extracted_fields: dict[str, Any],
    rule_set: dict[str, Any],
    master_data: dict[str, Any] | None = None,
    compliance_rules: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Return field-relevant facts for validation.

    This is intentionally deterministic for the POC, but the call shape matches a
    future RAG/tool-backed implementation.
    """

    data = master_data or MASTER_DATA
    country_rules = compliance_rules or COUNTRY_COMPLIANCE_RULES
    context: list[dict[str, Any]] = []
    trade_countries = infer_trade_countries(extracted_fields, data)
    customer_id = rule_set.get("customer_id")

    if customer_id in data.get("customers", {}):
        context.append(
            {
                "source": "customer_master",
                "customer_id": customer_id,
                "data": data["customers"][customer_id],
            }
        )

    if field_name in {"port_of_loading", "port_of_discharge"}:
        context.append({"source": "port_master", "data": data.get("ports", {})})

    if field_name == "incoterms":
        context.append({"source": "incoterms_master", "data": data.get("incoterms", {})})

    if field_name in {"hs_code", "description_of_goods"}:
        context.append({"source": "hs_code_master", "data": data.get("hs_code_rules", {})})

    destination_country = trade_countries.get("destination_country")
    if destination_country and destination_country in country_rules:
        context.append(
            {
                "source": "country_compliance",
                "country": destination_country,
                "direction": "imports",
                "data": country_rules[destination_country].get("imports", {}),
            }
        )

    origin_country = trade_countries.get("origin_country")
    if origin_country and origin_country in country_rules:
        context.append(
            {
                "source": "country_compliance",
                "country": origin_country,
                "direction": "exports",
                "data": country_rules[origin_country].get("exports", {}),
            }
        )

    return context

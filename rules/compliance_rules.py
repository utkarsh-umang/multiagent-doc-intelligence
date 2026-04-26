"""
Dummy country-specific compliance rules for the Validator POC.

These simulate the dynamic customs and regulatory facts the PRD expects the
Validator to retrieve through tools instead of baking into the prompt.
"""

from __future__ import annotations


COUNTRY_COMPLIANCE_RULES = {
    "IN": {
        "imports": {
            "required_fields": [
                "hs_code",
                "consignee_name",
                "gross_weight",
                "invoice_number",
            ],
            "hs_code_format": r"^\d{4,8}$",
            "allowed_incoterms": ["FOB", "CIF"],
            "notes": [
                "Indian import filings require HS code, consignee, invoice number, and gross weight.",
                "For this POC, CIF and FOB are treated as acceptable import terms.",
            ],
        }
    },
    "CN": {
        "exports": {
            "required_fields": [
                "port_of_loading",
                "description_of_goods",
                "gross_weight",
            ],
            "requires_port_unlocode": False,
            "notes": [
                "Chinese export documents should include loading port, goods description, and gross weight.",
            ],
        }
    },
}

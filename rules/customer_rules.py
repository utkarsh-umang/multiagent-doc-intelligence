"""
Hardcoded customer rule set for the validator POC.

In production this should move to a retrieval-backed rules store. For the
assignment, one explicit customer profile is enough to exercise validation.
"""

from __future__ import annotations

CUSTOMER_ID = "gocomet_demo_customer"
CUSTOMER_NAME = "GoComet Demo Importer"

CUSTOMER_RULE_SET = {
    "customer_id": CUSTOMER_ID,
    "customer_name": CUSTOMER_NAME,
    "confidence_threshold": 0.6,
    "fields": {
        "consignee_name": {
            "type": "equals",
            "expected": "GoComet Demo Importer Pvt Ltd",
            "aliases": [
                "GoComet Demo Importer",
                "GoComet Demo Importer Private Limited",
            ],
            "required": True,
        },
        "hs_code": {
            "type": "prefix",
            "expected": ["8708", "8409"],
            "required": True,
        },
        "port_of_loading": {
            "type": "allowed_values",
            "expected": ["Shanghai", "Shenzhen", "Ningbo"],
            "required": True,
        },
        "port_of_discharge": {
            "type": "allowed_values",
            "expected": ["Nhava Sheva", "JNPT", "Mumbai"],
            "required": True,
        },
        "incoterms": {
            "type": "allowed_values",
            "expected": ["FOB", "CIF"],
            "required": True,
        },
        "description_of_goods": {
            "type": "contains_any",
            "expected": ["auto parts", "automotive parts", "spare parts"],
            "required": True,
        },
        "gross_weight": {
            "type": "numeric_range",
            "min": 1,
            "max": 25000,
            "unit": "kg",
            "required": True,
        },
        "invoice_number": {
            "type": "regex",
            "expected": r"^[A-Z]{2,5}[-/]?\d{4,}$",
            "description": "2-5 uppercase letters followed by at least 4 digits",
            "required": True,
        },
    },
}

# Backwards-friendly name for modules that import RULES directly.
RULES = CUSTOMER_RULE_SET


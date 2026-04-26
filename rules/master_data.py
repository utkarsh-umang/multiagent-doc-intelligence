"""
Dummy logistics master data for the Validator POC.

In production this data would live in customer systems, master-data tables, or a
retrieval-backed knowledge base. The local shape mirrors that future interface.
"""

from __future__ import annotations


MASTER_DATA = {
    "ports": {
        "Shanghai": {
            "country": "CN",
            "aliases": ["Port of Shanghai", "CNSHA"],
            "unlocode": "CNSHA",
        },
        "Shenzhen": {
            "country": "CN",
            "aliases": ["Port of Shenzhen", "CNSZX"],
            "unlocode": "CNSZX",
        },
        "Ningbo": {
            "country": "CN",
            "aliases": ["Ningbo-Zhoushan", "CNNGB"],
            "unlocode": "CNNGB",
        },
        "Nhava Sheva": {
            "country": "IN",
            "aliases": ["JNPT", "Jawaharlal Nehru Port", "INNSA"],
            "unlocode": "INNSA",
        },
        "Mumbai": {
            "country": "IN",
            "aliases": ["Bombay Port", "INBOM"],
            "unlocode": "INBOM",
        },
    },
    "incoterms": {
        "FOB": {
            "description": "Free on Board",
            "requires_port_of_loading": True,
            "seller_responsible_until": "loaded_on_vessel",
        },
        "CIF": {
            "description": "Cost, Insurance, and Freight",
            "requires_port_of_discharge": True,
            "requires_insurance": True,
        },
    },
    "hs_code_rules": {
        "8708": {
            "category": "automotive_parts",
            "allowed_descriptions": [
                "auto parts",
                "automotive parts",
                "spare parts",
                "vehicle components",
            ],
        },
        "8409": {
            "category": "engine_parts",
            "allowed_descriptions": [
                "engine parts",
                "automotive parts",
                "spare parts",
                "piston components",
            ],
        },
    },
    "customers": {
        "gocomet_demo_customer": {
            "approved_consignee_names": [
                "GoComet Demo Importer Pvt Ltd",
                "GoComet Demo Importer",
                "GoComet Demo Importer Private Limited",
            ],
            "approved_lanes": [
                {
                    "origin_country": "CN",
                    "destination_country": "IN",
                    "allowed_incoterms": ["FOB", "CIF"],
                }
            ],
        }
    },
}

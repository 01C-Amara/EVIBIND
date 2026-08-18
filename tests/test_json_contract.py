from __future__ import annotations

import pytest

from evibind.core import EvidenceTypeError, EvidenceTypeRegistry
from tapbench.json_contract import json_contract_accepts


def test_numeric_string_and_array_constraints_are_enforced() -> None:
    schema = {
        "type": "object",
        "properties": {
            "amount": {
                "type": "number",
                "minimum": 10,
                "exclusiveMaximum": 100,
                "multipleOf": 5,
            },
            "reference": {
                "type": "string",
                "minLength": 4,
                "pattern": r"^INV-[0-9]+$",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "uniqueItems": True,
            },
        },
        "required": ["amount", "reference", "tags"],
        "additionalProperties": False,
    }
    valid = {
        "amount": 20,
        "reference": "INV-42",
        "tags": ["urgent", "finance"],
    }

    assert json_contract_accepts(valid, schema)
    assert not json_contract_accepts({**valid, "amount": 101}, schema)
    assert not json_contract_accepts({**valid, "amount": 21}, schema)
    assert not json_contract_accepts(
        {**valid, "reference": "invoice-42"},
        schema,
    )
    assert not json_contract_accepts(
        {**valid, "tags": ["urgent", "urgent"]},
        schema,
    )


def test_composition_and_dependent_requirements_are_enforced() -> None:
    schema = {
        "type": "object",
        "properties": {
            "mode": {"enum": ["email", "sms"]},
            "address": {
                "oneOf": [
                    {"type": "string", "pattern": "@"},
                    {"type": "string", "pattern": r"^\+[0-9]+$"},
                ]
            },
            "confirmation": {"type": "string"},
        },
        "required": ["mode", "address"],
        "dependentRequired": {"confirmation": ["mode"]},
        "additionalProperties": False,
    }

    assert json_contract_accepts(
        {"mode": "email", "address": "a@example.com"},
        schema,
    )
    assert not json_contract_accepts(
        {"mode": "email", "address": "not-an-address"},
        schema,
    )


@pytest.mark.parametrize("evidence_type", ["number", "schema_enum"])
def test_numeric_evidence_types_reject_non_finite_values(
    evidence_type: str,
) -> None:
    with pytest.raises(EvidenceTypeError):
        EvidenceTypeRegistry.standard().get(evidence_type).validate(
            float("inf"),
            {"span"},
        )
    assert not json_contract_accepts(float("inf"), {"type": "number"})

from __future__ import annotations

from tapbench.schema_lint import lint_tool_schemas


def _request() -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "pay_invoice",
                    "description": "Pay one invoice.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "invoice_id": {
                                "type": "string",
                                "x-evibind-evidence-type": "opaque_registry_id",
                                "x-evibind-sources": ["state.invoices"],
                                "x-evibind-slot-role": "identifier",
                                "x-evibind-resolution-type": "referential",
                            },
                            "amount": {
                                "type": "number",
                                "x-evibind-slot-role": "control",
                                "x-evibind-evidence-type": "number",
                                "x-evibind-sources": ["user.current_turn"],
                                "x-evibind-resolution-type": "normalizable",
                                "x-evibind-extraction-cue": "amount",
                            },
                        },
                        "required": ["invoice_id", "amount"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    }


def test_linter_accepts_complete_private_contract() -> None:
    report = lint_tool_schemas(_request())
    assert report["valid"]
    assert report["error_count"] == 0
    assert report["warning_count"] == 0
    assert report["private_contract_sha256"] != report["provider_schema_sha256"]


def test_linter_fails_invalid_annotation_and_required_slot() -> None:
    request = _request()
    schema = request["tools"][0]["function"]["parameters"]
    schema["required"].append("missing")
    schema["properties"]["invoice_id"]["x-evibind-slot-role"] = "mystery"
    schema["properties"]["amount"]["x-evibind-evidence-type"] = "magic_number"
    report = lint_tool_schemas(request)
    assert not report["valid"]
    codes = {issue["code"] for issue in report["issues"]}
    assert "required.unknown_property" in codes
    assert "annotation.slot_role" in codes
    assert "annotation.evidence_type" in codes


def test_linter_warns_for_open_untyped_critical_schema() -> None:
    request = _request()
    schema = request["tools"][0]["function"]["parameters"]
    schema.pop("additionalProperties")
    schema["properties"]["amount"] = {"type": "number"}
    report = lint_tool_schemas(request)
    assert report["valid"]
    codes = {issue["code"] for issue in report["issues"]}
    assert "parameters.open_object" in codes
    assert "annotation.critical_slot_untyped" in codes

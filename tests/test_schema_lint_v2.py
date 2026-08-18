from __future__ import annotations

from tapbench.schema_lint import lint_tool_schemas


def _nested_request() -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "send",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "payload": {
                                "type": "object",
                                "properties": {
                                    "recipient": {
                                        "type": "string",
                                        "format": "email",
                                        "x-evibind-evidence-type": ("email_address"),
                                        "x-evibind-sources": ["user.current_turn"],
                                    }
                                },
                                "required": ["recipient"],
                            }
                        },
                        "required": ["payload"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    }


def test_linter_recurses_into_nested_parameter_objects() -> None:
    request = _nested_request()
    request["tools"][0]["function"]["parameters"]["properties"]["payload"][
        "properties"
    ]["recipient"]["x-evibind-evidence-type"] = "magic"

    report = lint_tool_schemas(request)
    codes = {issue["code"] for issue in report["issues"]}

    assert not report["valid"]
    assert "annotation.evidence_type" in codes
    assert "parameters.nested_open_object" in codes


def test_linter_rejects_value_class_that_disagrees_with_type() -> None:
    request = _nested_request()
    recipient = request["tools"][0]["function"]["parameters"]["properties"]["payload"][
        "properties"
    ]["recipient"]
    recipient["x-evibind-value-class"] = "opaque_content"
    request["tools"][0]["function"]["parameters"]["properties"]["payload"][
        "additionalProperties"
    ] = False

    report = lint_tool_schemas(request)
    codes = {issue["code"] for issue in report["issues"]}

    assert not report["valid"]
    assert "annotation.value_class_mismatch" in codes


def test_nested_required_declaration_is_validated() -> None:
    request = _nested_request()
    nested = request["tools"][0]["function"]["parameters"]["properties"]["payload"]
    nested["required"] = ["recipient", "missing"]

    report = lint_tool_schemas(request)
    codes = {issue["code"] for issue in report["issues"]}

    assert not report["valid"]
    assert "required.unknown_property" in codes


def test_nested_required_must_be_a_string_list() -> None:
    request = _nested_request()
    nested = request["tools"][0]["function"]["parameters"]["properties"]["payload"]
    nested["required"] = "recipient"

    report = lint_tool_schemas(request)
    codes = {issue["code"] for issue in report["issues"]}

    assert not report["valid"]
    assert "required.not_string_list" in codes

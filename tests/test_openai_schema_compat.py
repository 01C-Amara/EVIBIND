"""Keep the action tool's schema inside OpenAI's function-schema rules.

OpenAI rejects a function whose ``parameters`` carry ``oneOf`` / ``anyOf`` /
``allOf`` / ``enum`` / ``const`` / ``not`` at the top level:

    Invalid schema for function 'evibind_action': schema must have type
    'object' and not have 'oneOf'/'anyOf'/'allOf'/'enum'/'const'/'not' at the
    top level.  (HTTP 400, code invalid_function_parameters)

The action schema used to return ``{"type": "object", "oneOf": [...]}``, so
``evibind serve`` could not reach api.openai.com at all — every request 400'd
before the model was consulted. The union now sits under a single required
``action`` property, which keeps every branch constraint while satisfying the
rule. These tests stop that from regressing.
"""

from __future__ import annotations

from tapbench.one_call_gateway import (
    ACTION_ENVELOPE_KEY,
    _indexed_action_schema,
    action_branches,
)

FORBIDDEN_AT_TOP_LEVEL = ("oneOf", "anyOf", "allOf", "enum", "const", "not")


def test_action_schema_is_accepted_by_openai_function_rules() -> None:
    schema = _indexed_action_schema()
    assert schema.get("type") == "object"
    offending = [key for key in FORBIDDEN_AT_TOP_LEVEL if key in schema]
    assert not offending, (
        f"OpenAI rejects these top-level keywords in function parameters: {offending}")


def test_the_union_is_reachable_and_required() -> None:
    schema = _indexed_action_schema()
    assert schema["required"] == [ACTION_ENVELOPE_KEY]
    assert schema["additionalProperties"] is False
    assert "oneOf" in schema["properties"][ACTION_ENVELOPE_KEY]


def test_action_schema_branches_are_still_well_formed() -> None:
    """Whatever the envelope, the three branches must stay intact."""
    branches = action_branches(_indexed_action_schema())
    modes = {branch["properties"]["mode"]["const"] for branch in branches}
    assert modes == {"call", "need_input", "no_tool"}
    for branch in branches:
        assert branch["additionalProperties"] is False
        assert "mode" in branch["required"]


def test_action_branches_reads_the_flat_form_too() -> None:
    """Certificates recorded before the envelope must still be readable."""
    flat = {"type": "object", "oneOf": [{"properties": {"mode": {"const": "no_tool"}}}]}
    assert len(action_branches(flat)) == 1

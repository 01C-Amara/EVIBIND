"""Pin the OpenAI function-schema constraint the action tool currently violates.

OpenAI rejects a function whose ``parameters`` carry ``oneOf`` / ``anyOf`` /
``allOf`` / ``enum`` / ``const`` / ``not`` at the top level:

    Invalid schema for function 'evibind_action': schema must have type
    'object' and not have 'oneOf'/'anyOf'/'allOf'/'enum'/'const'/'not' at the
    top level.  (HTTP 400, code invalid_function_parameters)

``_indexed_action_schema`` returns ``{"type": "object", "oneOf": [...]}``, so
the serving path in ``tapbench.one_call_gateway`` cannot reach api.openai.com at
all — every request 400s before the model is consulted. Reproduced live on
2026-08-18 against ``gpt-5.4-nano`` with ``EVIBIND_UPSTREAM_BASE_URL`` set to
``https://api.openai.com/v1``.

Note this constrains the *gateway* only. InjectBench scores
``protect_chat_completion`` against a model response that the harness fetches
itself, so the benchmark results are unaffected.

The fix is to move the branch union off the top level -- e.g. a single required
``action`` property holding the ``oneOf`` -- and to unwrap it in
``_parse_action_proposal``. That changes the wire contract with the model and
touches the schemas asserted across the suite, so it is deliberately not done
here. This test is the tripwire: it is ``strict``, so once the schema is fixed
it XPASSes and fails until the marker is removed.
"""

from __future__ import annotations

import pytest

from tapbench.one_call_gateway import _indexed_action_schema

FORBIDDEN_AT_TOP_LEVEL = ("oneOf", "anyOf", "allOf", "enum", "const", "not")


@pytest.mark.xfail(strict=True,
                   reason="action schema uses a top-level oneOf; OpenAI rejects it")
def test_action_schema_is_accepted_by_openai_function_rules() -> None:
    schema = _indexed_action_schema()
    assert schema.get("type") == "object"
    offending = [key for key in FORBIDDEN_AT_TOP_LEVEL if key in schema]
    assert not offending, (
        f"OpenAI rejects these top-level keywords in function parameters: {offending}")


def test_action_schema_branches_are_still_well_formed() -> None:
    """Whatever the envelope, the three branches must stay intact."""
    schema = _indexed_action_schema()
    branches = schema.get("oneOf") or schema.get("properties", {}).get(
        "action", {}).get("oneOf", [])
    modes = {b["properties"]["mode"]["const"] for b in branches}
    assert modes == {"call", "need_input", "no_tool"}

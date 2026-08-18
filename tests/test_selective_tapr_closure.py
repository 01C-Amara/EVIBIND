from tapbench.selective_tapr_closure import (
    apply_online_semantic_closure,
    decode_json_stream,
)


def _tool() -> dict:
    return {
        "name": "register",
        "canonical_name": "register",
        "parameters": {
            "type": "object",
            "properties": {
                "attendee": {
                    "type": "string",
                    "x-tap-semantic-envelope": "head_number",
                }
            },
            "required": ["attendee"],
            "additionalProperties": False,
        },
    }


def _metadata(raw: str) -> dict:
    return {
        "selected_tools": ["register", "register"],
        "proposal_attempts": [
            {"proposal_method": "full_tap_b2"},
        ],
        "raw_text": raw,
        "generation_calls": 8,
    }


def test_decode_json_stream_accepts_concatenated_objects() -> None:
    assert decode_json_stream('{"a":1}\n{"b":2}') == [
        {"a": 1},
        {"b": 2},
    ]


def test_online_closure_recovers_unique_source_backed_fragment() -> None:
    proposal = (
        '{"mode":"call","tool":"register",'
        '"arguments":{"attendee":"3193"},"payload":{}}'
    )
    action, metadata = apply_online_semantic_closure(
        {"mode": "refuse", "tool": None, "arguments": {}, "payload": {}},
        _metadata(proposal),
        messages=[{"role": "user", "content": "Register Attendee 3193."}],
        tools=[_tool()],
    )
    assert action["arguments"]["attendee"] == "Attendee 3193"
    assert metadata["semantic_closure"]["status"] == "recovered"
    assert metadata["generation_calls"] == 8
    assert metadata["certificate_count"] == 1


def test_online_closure_preserves_refusal_when_fragment_is_ambiguous() -> None:
    proposal = (
        '{"mode":"call","tool":"register",'
        '"arguments":{"attendee":"Attendee"},"payload":{}}'
    )
    original = {
        "mode": "refuse",
        "tool": None,
        "arguments": {},
        "payload": {},
    }
    action, metadata = apply_online_semantic_closure(
        original,
        _metadata(proposal),
        messages=[
            {
                "role": "user",
                "content": "Compare Attendee 3193 and Attendee 9914.",
            }
        ],
        tools=[_tool()],
    )
    assert action == original
    assert metadata["semantic_closure"]["status"] == "not_recovered"


def test_online_closure_does_not_change_existing_call() -> None:
    original = {
        "mode": "call",
        "tool": "register",
        "arguments": {"attendee": "Attendee 3193"},
        "payload": {},
    }
    action, metadata = apply_online_semantic_closure(
        original,
        {},
        messages=[],
        tools=[_tool()],
    )
    assert action == original
    assert (
        metadata["semantic_closure"]["status"]
        == "not_needed_existing_certified_call"
    )

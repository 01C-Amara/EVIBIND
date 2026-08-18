from __future__ import annotations

import json
from typing import Any

from tapbench.capc_projected import (
    projected_action_schema,
    replay_source_certificate,
    run_projected_capc_resolution,
    source_certificate,
)


def _tool(name: str, slot: str = "person") -> dict[str, Any]:
    return {
        "name": name,
        "canonical_name": name,
        "description": f"Perform {name}.",
        "parameters": {
            "type": "object",
            "properties": {slot: {"type": "string"}},
            "required": [],
            "additionalProperties": False,
        },
    }


def _metadata() -> dict[str, Any]:
    return {
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "prompt_ms": 1.0,
        "generation_ms": 1.0,
        "context_truncated": False,
        "rendered_input_tokens": 10,
        "context_headroom_tokens": 32000,
        "preflight_prompt_token_delta": 0,
    }


def _catalog(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    content = messages[-1]["content"]
    payload = content.split("Effect catalog:\n", 1)[1]
    payload = payload.rsplit("\nReturn", 1)[0]
    return json.loads(payload)


def _request(
    proposals: list[dict[str, Any]],
    *,
    desired_tool: str = "email.send",
    dissent_last: bool = False,
    strict_disagreement: bool = False,
):
    election_index = 0
    proposal_index = 0

    def request(endpoint, messages, *, response_schema, **kwargs):
        nonlocal election_index, proposal_index
        properties = response_schema.get("properties", {})
        if "english_effect" in properties:
            return {"english_effect": "send an email"}, _metadata()
        if "selection_id" in properties:
            catalog = _catalog(messages)
            selected = desired_tool
            if strict_disagreement and election_index == 1:
                selected = "NO_CALL"
            if dissent_last and election_index == 2:
                selected = "calendar.remove"
            selection_id = next(
                row["selection_id"]
                for row in catalog
                if row["effect"] == selected
            )
            election_index += 1
            return {"selection_id": selection_id}, _metadata()
        arguments_schema = properties["arguments"]
        assert properties["tool"]["enum"] == [desired_tool]
        assert arguments_schema["additionalProperties"] is False
        proposal = proposals[min(proposal_index, len(proposals) - 1)]
        proposal_index += 1
        return proposal, _metadata()

    return request


def _call(value: str) -> dict[str, Any]:
    return {
        "mode": "call",
        "tool": "email.send",
        "arguments": {"person": value},
        "payload": {},
    }


def test_unicode_source_certificate_replays_persian_and_japanese() -> None:
    for text, value in (
        ("به ماهان ایمیل بزن", "ماهان"),
        ("渋谷まで案内して", "渋谷"),
    ):
        certificate = source_certificate(text, value, {"type": "string"})
        assert certificate is not None
        assert certificate["source_text"] == value
        assert replay_source_certificate(text, certificate)


def test_unicode_casefold_certificate_preserves_source_span() -> None:
    text = "Email Alice now"
    certificate = source_certificate(text, "alice", {"type": "string"})
    assert certificate is not None
    assert certificate["transform"] == "unicode_casefold_match"
    assert certificate["source_text"] == "Alice"
    assert replay_source_certificate(text, certificate)


def test_projected_schema_closes_arguments_to_selected_tool() -> None:
    schema = projected_action_schema(_tool("email.send"))
    assert schema["properties"]["tool"]["enum"] == ["email.send"]
    arguments = schema["properties"]["arguments"]
    assert set(arguments["properties"]) == {"person"}
    assert arguments["additionalProperties"] is False


def test_majority_election_tolerates_one_order_dissent() -> None:
    tools = [_tool("calendar.remove"), _tool("email.send")]
    action, metadata = run_projected_capc_resolution(
        messages=[{"role": "user", "content": "به ماهان ایمیل بزن"}],
        tools=tools,
        endpoint="http://unused",
        condition="tap_r_capc_projected_majority",
        max_tokens=64,
        seed=1,
        request_fn=_request([_call("ماهان")], dissent_last=True),
    )
    assert action == _call("ماهان")
    assert metadata["election_winner_votes"] == 2
    assert metadata["proposal_admitted"]
    assert metadata["generation_calls"] == 4
    certificate = metadata["evidence_certificates"]["person"]
    assert replay_source_certificate("به ماهان ایمیل بزن", certificate)


def test_strict_election_fails_closed_on_order_disagreement() -> None:
    action, metadata = run_projected_capc_resolution(
        messages=[{"role": "user", "content": "Email Alice."}],
        tools=[_tool("email.send")],
        endpoint="http://unused",
        condition="tap_r_capc_projected_strict",
        max_tokens=64,
        seed=1,
        request_fn=_request([_call("Alice")], strict_disagreement=True),
    )
    assert action["mode"] == "refuse"
    assert not metadata["tool_agreement"]
    assert metadata["generation_calls"] == 2


def test_unsupported_primary_literal_uses_source_grounded_repair() -> None:
    action, metadata = run_projected_capc_resolution(
        messages=[{"role": "user", "content": "Email Alice."}],
        tools=[_tool("email.send")],
        endpoint="http://unused",
        condition="tap_r_capc_projected_majority",
        max_tokens=64,
        seed=1,
        request_fn=_request([_call("Bob"), _call("Alice")]),
    )
    assert action == _call("Alice")
    assert metadata["accepted_proposal_index"] == 1
    assert [row["status"] for row in metadata["proposal_attempts"]] == [
        "unsupported_argument",
        "certified",
    ]
    assert metadata["generation_calls"] == 5


def test_pivot_is_advisory_and_recorded_without_entering_action() -> None:
    action, metadata = run_projected_capc_resolution(
        messages=[{"role": "user", "content": "به ماهان ایمیل بزن"}],
        tools=[_tool("email.send")],
        endpoint="http://unused",
        condition="tap_r_capc_projected_pivot",
        max_tokens=64,
        seed=1,
        request_fn=_request([_call("ماهان")]),
    )
    assert action == _call("ماهان")
    assert metadata["effect_pivot"]["used"]
    assert "english_effect" not in json.dumps(action, ensure_ascii=False)
    assert metadata["generation_calls"] == 5


def test_explicit_firewall_makes_no_projected_model_calls() -> None:
    def should_not_run(*args, **kwargs):
        raise AssertionError("explicit firewall should terminate")

    action, metadata = run_projected_capc_resolution(
        messages=[
            {
                "role": "user",
                "content": "Explain only; do not call a tool.",
            }
        ],
        tools=[_tool("email.send")],
        endpoint="http://unused",
        condition="tap_r_capc_projected_majority",
        max_tokens=64,
        seed=1,
        request_fn=should_not_run,
    )
    assert action["mode"] == "direct_answer"
    assert metadata["generation_calls"] == 0

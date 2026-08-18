from __future__ import annotations

from typing import Any

from tapbench.capc import run_capc_resolution


def _tool() -> dict[str, Any]:
    return {
        "name": "submit_count",
        "canonical_name": "submit_count",
        "description": "Submit the requested count.",
        "parameters": {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
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


def _call(count: int) -> dict[str, Any]:
    return {
        "mode": "call",
        "tool": "submit_count",
        "arguments": {"count": count},
        "payload": {},
    }


def _request(proposals: list[dict[str, Any]], *, disagree: bool = False):
    proposal_index = 0
    election_index = 0

    def request(endpoint, messages, *, response_schema, **kwargs):
        nonlocal proposal_index, election_index
        if "selection_id" in response_schema.get("properties", {}):
            selected = 0
            if disagree and election_index == 1:
                selected = -1
            election_index += 1
            return {"selection_id": selected}, _metadata()
        proposal = proposals[min(proposal_index, len(proposals) - 1)]
        proposal_index += 1
        return proposal, _metadata()

    return request


def test_primary_certified_proposal_is_admitted_without_fallback() -> None:
    action, metadata = run_capc_resolution(
        messages=[{"role": "user", "content": "Submit count 3."}],
        tools=[_tool()],
        endpoint="http://unused",
        condition="tap_r_capc_dual",
        max_tokens=64,
        seed=1,
        request_fn=_request([_call(3)]),
    )
    assert action == _call(3)
    assert metadata["accepted_proposal_index"] == 0
    assert metadata["generation_calls"] == 3
    certificate = metadata["evidence_certificates"]["count"]
    assert certificate["value"] == 3
    assert certificate["source_span"]
    assert metadata["preflight_prompt_token_delta_max_abs"] == 0


def test_dual_cascade_uses_certified_fallback_after_unsupported_literal() -> None:
    action, metadata = run_capc_resolution(
        messages=[{"role": "user", "content": "Submit count 3."}],
        tools=[_tool()],
        endpoint="http://unused",
        condition="tap_r_capc_dual",
        max_tokens=64,
        seed=1,
        request_fn=_request([_call(999), _call(3)]),
    )
    assert action == _call(3)
    assert metadata["accepted_proposal_index"] == 1
    assert metadata["action_risk_score"] == 0.05
    assert metadata["risk_gate_passed"]
    assert [row["status"] for row in metadata["proposal_attempts"]] == [
        "unsupported_argument",
        "certified",
    ]
    assert metadata["generation_calls"] == 4


def test_all_unsupported_proposals_fail_closed() -> None:
    action, metadata = run_capc_resolution(
        messages=[{"role": "user", "content": "Submit count 3."}],
        tools=[_tool()],
        endpoint="http://unused",
        condition="tap_r_capc_dual",
        max_tokens=64,
        seed=1,
        request_fn=_request([_call(999), _call(998)]),
    )
    assert action["mode"] == "refuse"
    assert not metadata["proposal_admitted"]
    assert metadata["action_risk_score"] == 1.0


def test_guardian_disagreement_stops_before_literal_generation() -> None:
    action, metadata = run_capc_resolution(
        messages=[{"role": "user", "content": "Submit count 3."}],
        tools=[_tool()],
        endpoint="http://unused",
        condition="tap_r_capc_dual",
        max_tokens=64,
        seed=1,
        request_fn=_request([_call(3)], disagree=True),
    )
    assert action["mode"] == "refuse"
    assert not metadata["tool_agreement"]
    assert metadata["generation_calls"] == 2
    assert metadata["proposal_attempts"] == []


def test_explicit_firewall_makes_no_model_calls() -> None:
    def should_not_run(*args, **kwargs):
        raise AssertionError("explicit firewall should terminate")

    action, metadata = run_capc_resolution(
        messages=[
            {
                "role": "user",
                "content": "Explain only; do not call a tool.",
            }
        ],
        tools=[_tool()],
        endpoint="http://unused",
        condition="tap_r_capc_dual",
        max_tokens=64,
        seed=1,
        request_fn=should_not_run,
    )
    assert action["mode"] == "direct_answer"
    assert metadata["generation_calls"] == 0

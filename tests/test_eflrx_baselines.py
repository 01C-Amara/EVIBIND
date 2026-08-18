from __future__ import annotations

from typing import Any

from tapbench.eflrx_baselines import run_raw_baseline


def _case() -> dict[str, Any]:
    return {
        "case_id": "bfcl_case",
        "family": "bfcl",
        "messages": [{"role": "user", "content": "Submit count 3."}],
        "tools": [
            {
                "name": "submit_count",
                "canonical_name": "submit_count",
                "description": "Submit a count.",
                "parameters": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                    "additionalProperties": False,
                },
            }
        ],
    }


def _metadata() -> dict[str, Any]:
    return {
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "prompt_ms": 1.0,
        "generation_ms": 1.0,
        "rendered_input_tokens": 10,
        "context_headroom_tokens": 32000,
        "preflight_http_calls": 2,
    }


def _valid() -> dict[str, Any]:
    return {
        "mode": "call",
        "tool": "submit_count",
        "arguments": {"count": 3},
        "payload": {},
    }


def test_prompt_few_shot_is_one_raw_schema_generation() -> None:
    calls = []

    def request(endpoint, messages, **kwargs):
        calls.append(messages)
        return _valid(), _metadata()

    action, metadata = run_raw_baseline(
        _case(),
        endpoint="http://unused",
        condition="prompt_few_shot",
        max_tokens=64,
        seed=1,
        request_fn=request,
    )
    assert action == _valid()
    assert len(calls) == 1
    assert metadata["repair_budget"] == 0
    assert metadata["generation_calls"] == 1


def test_full_tap_b2_repairs_only_contract_violations() -> None:
    outputs = iter(
        [
            {
                "mode": "call",
                "tool": "submit_count",
                "arguments": {},
                "payload": {},
            },
            _valid(),
        ]
    )
    messages_seen = []

    def request(endpoint, messages, **kwargs):
        messages_seen.append(messages)
        return next(outputs), _metadata()

    action, metadata = run_raw_baseline(
        _case(),
        endpoint="http://unused",
        condition="full_tap_b2",
        max_tokens=64,
        seed=1,
        request_fn=request,
    )
    assert action == _valid()
    assert len(messages_seen) == 2
    assert metadata["repair_budget"] == 2
    assert metadata["repair_attempts_used"] == 1
    assert metadata["final_contract_valid"]
    feedback = messages_seen[1][-1]["content"]
    assert "call_contract_invalid" in feedback
    assert "gold" not in feedback.casefold()


def test_full_tap_b2_stops_after_two_repairs() -> None:
    invalid = {
        "mode": "call",
        "tool": "submit_count",
        "arguments": {},
        "payload": {},
    }
    count = 0

    def request(endpoint, messages, **kwargs):
        nonlocal count
        count += 1
        return invalid, _metadata()

    action, metadata = run_raw_baseline(
        _case(),
        endpoint="http://unused",
        condition="full_tap_b2",
        max_tokens=64,
        seed=1,
        request_fn=request,
    )
    assert action == invalid
    assert count == 3
    assert metadata["repair_attempts_used"] == 2
    assert not metadata["final_contract_valid"]

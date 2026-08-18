from __future__ import annotations

from typing import Any

from tapbench.hierarchy_runner import _selective_action


def test_extent_conditions_differ_only_in_declared_gate() -> None:
    captured: list[dict[str, Any]] = []

    def request(*args, **kwargs):
        raise AssertionError("the fake selective controller owns requests")

    def selective(**kwargs):
        captured.append(dict(kwargs))
        return (
            {
                "mode": "no_tool",
                "tool": None,
                "arguments": {},
                "payload": {},
            },
            {
                "generation_calls": 2,
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "total_tokens": 24,
                "raw_text": ["shared", "trace"],
            },
        )

    runtime = {
        "messages": [{"role": "user", "content": "Submit item 7."}],
        "tools": [
            {
                "name": "submit_item",
                "parameters": {
                    "type": "object",
                    "properties": {"item_id": {"type": "integer"}},
                    "required": ["item_id"],
                    "additionalProperties": False,
                },
            }
        ],
    }
    outputs = [
        _selective_action(
            runtime,
            "http://unused",
            max_tokens=64,
            seed=1,
            semantic_extent_enabled=enabled,
            request_fn=request,
            selective_fn=selective,
        )
        for enabled in (False, True)
    ]

    assert len(captured) == 2
    source, full = captured
    assert source["semantic_extent_enabled"] is False
    assert full["semantic_extent_enabled"] is True
    assert source["exhaust_proposal_budget"] is True
    assert full["exhaust_proposal_budget"] is True
    assert {
        key: value
        for key, value in source.items()
        if key != "semantic_extent_enabled"
    } == {
        key: value
        for key, value in full.items()
        if key != "semantic_extent_enabled"
    }
    assert outputs[0][1]["model_trace_sha256"] == outputs[1][1][
        "model_trace_sha256"
    ]

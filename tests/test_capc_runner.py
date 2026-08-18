from __future__ import annotations

import json
from typing import Any

from tapbench.capc_runner import run_capc_cases


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


def _call(count: int) -> dict[str, Any]:
    return {
        "mode": "call",
        "tool": "submit_count",
        "arguments": {"count": count},
        "payload": {},
    }


def _request(proposal: dict[str, Any]):
    def request(endpoint, messages, *, response_schema, **kwargs):
        metadata = {
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
        if "selection_id" in response_schema.get("properties", {}):
            return {"selection_id": 0}, metadata
        return proposal, metadata

    return request


def test_capc_runner_filters_cases_and_records_identity(tmp_path) -> None:
    case = {
        "case_id": "bfcl_dev_0",
        "task_kind": "call",
        "messages": [{"role": "user", "content": "Submit count 3."}],
        "tools": [_tool()],
        "metadata": {"bfcl_category": "simple_python"},
        "factors": {"bfcl_category": "simple_python"},
    }
    heldout = {
        **case,
        "case_id": "bfcl_holdout_0",
        "metadata": {"bfcl_category": "simple_java"},
        "factors": {"bfcl_category": "simple_java"},
    }
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps(case) + "\n" + json.dumps(heldout) + "\n")
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"test-artifact")
    output = tmp_path / "predictions.jsonl"
    timings = tmp_path / "timings.jsonl"
    manifest = tmp_path / "manifest.yaml"

    report = run_capc_cases(
        cases,
        output,
        timings,
        manifest,
        endpoint="http://unused",
        model_id="test/model",
        model_key="test",
        model_artifact=str(artifact),
        chat_template="test",
        conditions=("tap_r_capc_dual",),
        seeds=(1,),
        categories=("simple_python",),
        request_fn=_request(_call(3)),
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["prediction"] == _call(3)
    assert rows[0]["capc_version"] == "tapbench.capc.v1"
    assert rows[0]["capc_runner_version"] == "tapbench.capc_runner.v1"
    assert rows[0]["thinking_marker_detected"] is False
    assert rows[0]["response_metadata"][
        "preflight_prompt_token_delta_max_abs"
    ] == 0
    assert report["case_count"] == 1
    assert report["actual_model_calls"] == 3

from __future__ import annotations

import json
from typing import Any

from tapbench.eflrx import _merge_metadata, run_eflrx_resolution


def _tool() -> dict[str, Any]:
    return {
        "name": "submit_count",
        "canonical_name": "submit_count",
        "description": "Submit an explicitly requested count.",
        "parameters": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "count to submit",
                }
            },
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
    }


def _candidate_view(messages: list[dict[str, str]]) -> dict[str, Any]:
    content = messages[-1]["content"]
    encoded = content.split("Certified candidates:\n", 1)[1].split(
        "\nReturn ",
        1,
    )[0]
    return json.loads(encoded)


def _selector(targets: list[int]):
    pointer_index = 0

    def request(endpoint, messages, *, response_schema, **kwargs):
        nonlocal pointer_index
        if "selection_id" in response_schema.get("properties", {}):
            return {"selection_id": 0}, _metadata()
        view = _candidate_view(messages)
        target = targets[min(pointer_index, len(targets) - 1)]
        pointer_index += 1
        row = next(
            candidate
            for candidate in view["count"]
            if candidate["value"] == target
        )
        return {
            "arguments": {"count": row["candidate_id"]}
        }, _metadata()

    return request


def test_consensus_materializes_only_selected_certificate_values() -> None:
    action, metadata = run_eflrx_resolution(
        messages=[{"role": "user", "content": "Submit count 3."}],
        tools=[_tool()],
        endpoint="http://unused",
        condition="tap_r_eflrx_consensus",
        max_tokens=64,
        seed=1,
        request_fn=_selector([3, 3]),
    )
    assert action == {
        "mode": "call",
        "tool": "submit_count",
        "arguments": {"count": 3},
        "payload": {},
    }
    assert metadata["tool_agreement"]
    assert metadata["pointer_agreement"]
    assert metadata["risk_gate_passed"]
    assert metadata["action_risk_score"] == 0.013
    certificate = metadata["evidence_certificates"]["count"]
    assert certificate["source_text"] == "3."
    assert certificate["source_span"]


def test_consensus_refuses_when_canonical_pointer_values_disagree() -> None:
    action, metadata = run_eflrx_resolution(
        messages=[{"role": "user", "content": "Submit count 3 or 4."}],
        tools=[_tool()],
        endpoint="http://unused",
        condition="tap_r_eflrx_consensus",
        max_tokens=64,
        seed=1,
        request_fn=_selector([3, 4]),
    )
    assert action["mode"] == "refuse"
    assert not metadata["pointer_agreement"]
    assert metadata["action_risk_score"] == 1.0


def test_no_call_sentinel_terminates_before_pointer_selection() -> None:
    def no_call(endpoint, messages, *, response_schema, **kwargs):
        return {"selection_id": -1}, _metadata()

    action, metadata = run_eflrx_resolution(
        messages=[{"role": "user", "content": "Explain photosynthesis."}],
        tools=[_tool()],
        endpoint="http://unused",
        condition="tap_r_eflrx_consensus",
        max_tokens=64,
        seed=1,
        request_fn=no_call,
    )
    assert action["mode"] == "no_tool"
    assert metadata["generation_calls"] == 2
    assert metadata["pointer_elections"] == []


def test_out_of_domain_pointer_cannot_inject_a_literal_value() -> None:
    def invalid_pointer(endpoint, messages, *, response_schema, **kwargs):
        if "selection_id" in response_schema.get("properties", {}):
            return {"selection_id": 0}, _metadata()
        return {"arguments": {"count": 999}}, _metadata()

    action, metadata = run_eflrx_resolution(
        messages=[{"role": "user", "content": "Submit count 3."}],
        tools=[_tool()],
        endpoint="http://unused",
        condition="tap_r_eflrx_single",
        max_tokens=64,
        seed=1,
        request_fn=invalid_pointer,
    )
    assert action["mode"] == "refuse"
    assert metadata["pointer_failure"]["status"] == "out_of_domain_pointer"
    assert metadata["action_risk_score"] == 1.0


def test_explicit_firewall_bypasses_all_model_calls() -> None:
    def should_not_run(*args, **kwargs):
        raise AssertionError("firewall should terminate before generation")

    action, metadata = run_eflrx_resolution(
        messages=[
            {
                "role": "user",
                "content": "Explain only; do not call a tool.",
            }
        ],
        tools=[_tool()],
        endpoint="http://unused",
        condition="tap_r_eflrx_consensus",
        max_tokens=64,
        seed=1,
        request_fn=should_not_run,
    )
    assert action["mode"] == "direct_answer"
    assert metadata["generation_calls"] == 0


def test_runner_filters_categories_and_writes_provenance(tmp_path) -> None:
    from tapbench.eflrx_runner import run_eflrx_cases

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
    cases.write_text(
        json.dumps(case) + "\n" + json.dumps(heldout) + "\n"
    )
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"test-artifact")
    output = tmp_path / "predictions.jsonl"
    timings = tmp_path / "timings.jsonl"
    manifest = tmp_path / "manifest.yaml"

    report = run_eflrx_cases(
        cases,
        output,
        timings,
        manifest,
        endpoint="http://unused",
        model_id="test/model",
        model_key="test",
        model_artifact=str(artifact),
        chat_template="test",
        conditions=("tap_r_eflrx_single",),
        seeds=(1,),
        categories=("simple_python",),
        request_fn=_selector([3]),
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["case_id"] == "bfcl_dev_0"
    assert rows[0]["prediction"]["arguments"] == {"count": 3}
    assert rows[0]["action_ir_normalized"] is True
    assert rows[0]["thinking_marker_detected"] is False
    assert report["case_count"] == 1
    assert report["actual_model_calls"] == 2
    assert report["source_sha256"]["model_artifact"]


def test_context_preflight_records_headroom_and_blocks_overflow(monkeypatch) -> None:
    import pytest
    import tapbench.eflrx as eflrx

    responses = iter(
        [
            {"prompt": "templated prompt"},
            {"tokens": [1, 2, 3]},
        ]
    )
    monkeypatch.setattr(
        eflrx,
        "_post_json",
        lambda endpoint, route, payload: next(responses),
    )
    monkeypatch.setattr(
        eflrx,
        "_request_schema_json",
        lambda *args, **kwargs: ({"ok": True}, _metadata()),
    )
    raw, metadata = eflrx.preflight_schema_request(
        "http://unused",
        [{"role": "user", "content": "hello"}],
        response_schema={"type": "object"},
        max_tokens=4,
        temperature=0.0,
        seed=1,
        context_tokens=10,
    )
    assert raw == {"ok": True}
    assert metadata["rendered_input_tokens"] == 3
    assert metadata["context_headroom_tokens"] == 3
    assert metadata["preflight_prompt_token_delta"] == 7

    overflow_responses = iter(
        [
            {"prompt": "templated prompt"},
            {"tokens": list(range(8))},
        ]
    )
    monkeypatch.setattr(
        eflrx,
        "_post_json",
        lambda endpoint, route, payload: next(overflow_responses),
    )
    with pytest.raises(eflrx.ContextOverflowError, match="context_overflow"):
        eflrx.preflight_schema_request(
            "http://unused",
            [{"role": "user", "content": "hello"}],
            response_schema={"type": "object"},
            max_tokens=4,
            temperature=0.0,
            seed=1,
            context_tokens=10,
        )


def test_multi_call_metadata_preserves_worst_preflight_delta() -> None:
    first = {
        **_metadata(),
        "rendered_input_tokens": 20,
        "context_headroom_tokens": 32000,
        "preflight_prompt_token_delta": -1,
    }
    second = {
        **_metadata(),
        "rendered_input_tokens": 30,
        "context_headroom_tokens": 31990,
        "preflight_prompt_token_delta": 1,
    }
    merged = _merge_metadata([first, second])
    assert merged["rendered_input_tokens_max"] == 30
    assert merged["context_headroom_tokens_min"] == 31990
    assert merged["preflight_prompt_token_delta_max_abs"] == 1


def test_multi_call_metadata_sums_nested_repair_generations() -> None:
    nested = {
        **_metadata(),
        "generation_calls": 3,
    }
    merged = _merge_metadata([_metadata(), nested])
    assert merged["generation_calls"] == 4


def _tool_with_optional_unit() -> dict[str, Any]:
    tool = _tool()
    tool["parameters"]["properties"]["unit"] = {
        "type": "string",
        "enum": ["meters", "feet"],
    }
    return tool


def _optional_selector(include_unit: bool):
    def request(endpoint, messages, *, response_schema, **kwargs):
        if "selection_id" in response_schema.get("properties", {}):
            return {"selection_id": 0}, _metadata()
        view = _candidate_view(messages)
        arguments = {}
        for slot, rows in view.items():
            if slot == "count":
                row = next(item for item in rows if item["value"] == 3)
            elif include_unit:
                row = next(item for item in rows if item["value"] == "meters")
            else:
                row = next(item for item in rows if item["candidate_id"] == -1)
            arguments[slot] = row["candidate_id"]
        return {"arguments": arguments}, _metadata()

    return request


def test_optional_slot_is_bound_only_by_explicit_candidate_id() -> None:
    action, metadata = run_eflrx_resolution(
        messages=[
            {
                "role": "user",
                "content": "Submit count 3 measured in meters.",
            }
        ],
        tools=[_tool_with_optional_unit()],
        endpoint="http://unused",
        condition="tap_r_eflrx_consensus",
        max_tokens=64,
        seed=1,
        request_fn=_optional_selector(include_unit=True),
    )
    assert action["arguments"] == {"count": 3, "unit": "meters"}
    assert metadata["evidence_certificates"]["unit"]["source_text"] == "meters"


def test_optional_slot_without_explicit_enum_evidence_is_not_exposed() -> None:
    action, metadata = run_eflrx_resolution(
        messages=[{"role": "user", "content": "Submit count 3."}],
        tools=[_tool_with_optional_unit()],
        endpoint="http://unused",
        condition="tap_r_eflrx_consensus",
        max_tokens=64,
        seed=1,
        request_fn=_optional_selector(include_unit=False),
    )
    assert action["arguments"] == {"count": 3}
    assert "unit" not in metadata["evidence_certificates"]


def test_enum_slot_rejects_generic_non_enum_spans() -> None:
    from tapbench.extractive_candidates import build_extractive_candidate_table

    tool = {
        "name": "choose",
        "parameters": {
            "type": "object",
            "properties": {
                "choice": {
                    "type": "string",
                    "enum": ["alpha", "beta"],
                }
            },
            "required": ["choice"],
        },
    }
    table = build_extractive_candidate_table(
        [{"role": "user", "content": "Choose mystery."}],
        tool,
    )
    assert table["slots"]["choice"] == []

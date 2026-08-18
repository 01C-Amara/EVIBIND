import json

from tapbench import large_model_closure_runner as closure_runner
from tapbench import native_bfcl_runner


def _case() -> dict:
    return {
        "case_id": "case-1",
        "hypothesis_grid_id": "test-grid",
        "family": "registrations",
        "task_kind": "call",
        "factors": {"catalog_mutation": "none"},
        "messages": [
            {"role": "user", "content": "Register Attendee 3193."}
        ],
        "tools": [
            {
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
                },
            }
        ],
        "tool_aliases": {"register": "register"},
        "argument_aliases": {},
        "dialogue_state": {},
        "reference_context": {},
        "gold_action": {
            "mode": "call",
            "tool": "register",
            "arguments": {"attendee": "Attendee 3193"},
            "payload": {},
        },
    }


def _write_inputs(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps(_case()) + "\n", encoding="utf-8")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("study_id: test\n", encoding="utf-8")
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"model")
    return cases, protocol, artifact


def test_large_model_closure_runner_shares_model_calls(
    tmp_path, monkeypatch
) -> None:
    cases, protocol, artifact = _write_inputs(tmp_path)
    monkeypatch.setattr(
        closure_runner,
        "_literal_action",
        lambda *args, **kwargs: (
            {"mode": "no_tool", "tool": None, "arguments": {}, "payload": {}},
            {"finish_reason": "stop", "generation_calls": 1},
        ),
    )
    monkeypatch.setattr(
        closure_runner,
        "run_selective_tapr_resolution",
        lambda **kwargs: (
            {"mode": "refuse", "tool": None, "arguments": {}, "payload": {}},
            {"finish_reason": "stop", "generation_calls": 7},
        ),
    )
    monkeypatch.setattr(
        closure_runner,
        "apply_online_semantic_closure",
        lambda action, metadata, **kwargs: (
            {
                "mode": "call",
                "tool": "register",
                "arguments": {"attendee": "Attendee 3193"},
                "payload": {},
            },
            {
                **metadata,
                "semantic_closure": {"status": "recovered"},
            },
        ),
    )
    output = tmp_path / "predictions.jsonl"
    timings = tmp_path / "timings.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    manifest = closure_runner.run_large_model_closure(
        cases,
        output,
        timings,
        manifest_path,
        endpoint="http://unused",
        model_id="large-model",
        model_key="large",
        model_artifact=str(artifact),
        quantization="Q4",
        chat_template="native",
        protocol_path=protocol,
        study_id="test-grid",
        context_tokens=8192,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 3
    assert manifest["actual_model_calls"] == 8
    original = next(row for row in rows if row["method"] == closure_runner.ORIGINAL)
    closure = next(row for row in rows if row["method"] == closure_runner.CLOSURE)
    assert closure["shared_generation_trace_id"] == original["shared_generation_trace_id"]
    assert closure["model_calls_charged"] == 0


def test_native_bfcl_runner_records_normalized_action(
    tmp_path, monkeypatch
) -> None:
    cases, protocol, artifact = _write_inputs(tmp_path)
    monkeypatch.setattr(
        native_bfcl_runner,
        "request_native_tool",
        lambda *args, **kwargs: (
            {
                "mode": "call",
                "tool": "register",
                "arguments": {"attendee": "Attendee 3193"},
                "payload": {},
            },
            {
                "finish_reason": "tool_calls",
                "generation_calls": 1,
                "reasoning_content": "matched function",
            },
        ),
    )
    output = tmp_path / "native_predictions.jsonl"
    manifest = native_bfcl_runner.run_native_bfcl(
        cases,
        output,
        tmp_path / "native_timings.jsonl",
        tmp_path / "native_manifest.yaml",
        endpoint="http://unused",
        model_id="large-model",
        model_key="large",
        model_profile="qwen",
        model_artifact=str(artifact),
        quantization="Q4",
        chat_template="native",
        protocol_path=protocol,
        study_id="native-grid",
        context_tokens=8192,
        max_tokens=512,
        reasoning_budget=128,
        seed=1,
    )
    row = json.loads(output.read_text().splitlines()[0])
    assert row["prediction"]["tool"] == "register"
    assert row["reasoning_content_detected"] is True
    assert manifest["actual_model_calls"] == 1


def test_native_bfcl_runner_resumes_contiguous_prefix(tmp_path, monkeypatch) -> None:
    cases, protocol, artifact = _write_inputs(tmp_path)
    second = {**_case(), "case_id": "case-2"}
    cases.write_text(
        "\n".join(json.dumps(row) for row in (_case(), second)) + "\n",
        encoding="utf-8",
    )
    calls = []

    def request(*args, **kwargs):
        calls.append(kwargs["seed"])
        return (
            {"mode": "no_tool", "tool": None, "arguments": {}, "payload": {}},
            {"finish_reason": "stop", "generation_calls": 1},
        )

    monkeypatch.setattr(native_bfcl_runner, "request_native_tool", request)
    output = tmp_path / "native_predictions.jsonl"
    timings = tmp_path / "native_timings.jsonl"
    manifest_path = tmp_path / "native_manifest.yaml"
    common = {
        "endpoint": "http://unused",
        "model_id": "large-model",
        "model_key": "large",
        "model_profile": "qwen",
        "model_artifact": str(artifact),
        "quantization": "Q4",
        "chat_template": "native",
        "protocol_path": protocol,
        "study_id": "native-grid",
        "context_tokens": 8192,
        "max_tokens": 512,
        "reasoning_budget": 128,
        "seed": 1,
    }
    native_bfcl_runner.run_native_bfcl(
        cases,
        output,
        timings,
        manifest_path,
        max_cases=1,
        **common,
    )
    checkpoint = json.loads(output.read_text().splitlines()[0])
    calls.clear()

    manifest = native_bfcl_runner.run_native_bfcl(
        cases,
        output,
        timings,
        manifest_path,
        resume=True,
        **common,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert calls == [1]
    assert rows[0] == checkpoint
    assert len(rows) == 2
    assert manifest["resumed_prediction_count"] == 1
    assert manifest["resume_checkpoint_sha256"]

from __future__ import annotations

from pathlib import Path
import json

from tapbench.config import load_experiment_config
from tapbench.generator import generate_cases_from_config
from tapbench.io import read_jsonl, write_jsonl
from tapbench.runner import (
    _generation_jobs,
    _grid_by_id,
    _llama_response_metadata,
    _request_llama_json,
    action_ir_json_schema,
    method_instruction,
    parse_xlam_native_text,
    render_chat_messages,
    render_xlam_tools,
    run_cases,
    tools_for_method,
)
from tapbench.r2_model_runner import _request_schema_json
from tapbench.scoring import score_predictions
import tapbench.runner as runner_module


def test_oracle_runner_expands_methods_models_and_writes_manifest(tmp_path: Path) -> None:
    cfg = load_experiment_config()
    cases = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["H1_prompt_verbosity"])[:2]
    cases_path = tmp_path / "cases.jsonl"
    write_jsonl(cases_path, cases)

    manifest = run_cases(
        cases_path=cases_path,
        output_path=tmp_path / "predictions.jsonl",
        timings_path=tmp_path / "timings.jsonl",
        manifest_path=tmp_path / "manifest.yaml",
        backend="oracle",
        methods="prompt_strict_json",
        models="qwen3_1_7b",
        model_artifact="local-test.gguf",
        quantization="test_quant",
    )

    predictions = read_jsonl(tmp_path / "predictions.jsonl")
    timings = read_jsonl(tmp_path / "timings.jsonl")
    assert manifest["dry_run"] is True
    assert manifest["generation_count"] == 2
    assert manifest["identity_overrides"]["model_artifact"] == "local-test.gguf"
    assert len(predictions) == 2
    assert predictions[0]["model_artifact"] == "local-test.gguf"
    assert predictions[0]["quantization"] == "test_quant"
    assert len(timings) == 2
    scores = score_predictions(cases, predictions)
    assert all(row["execution_success"] for row in scores)


def test_oracle_runner_exposes_h6_call_only_fabrication(tmp_path: Path) -> None:
    cfg = load_experiment_config()
    cases = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["H6_abstention_suppression"])[:1]
    cases_path = tmp_path / "cases.jsonl"
    write_jsonl(cases_path, cases)

    run_cases(
        cases_path=cases_path,
        output_path=tmp_path / "predictions.jsonl",
        timings_path=tmp_path / "timings.jsonl",
        manifest_path=tmp_path / "manifest.yaml",
        backend="oracle",
        methods="constrained_call_only_b2",
        models="qwen3_1_7b",
    )

    scores = score_predictions(cases, read_jsonl(tmp_path / "predictions.jsonl"))
    assert scores[0]["fabrication"] is True
    assert scores[0]["execution_success"] is False



def test_llama_response_metadata_extracts_usage_and_timing() -> None:
    metadata = _llama_response_metadata(
        {
            "id": "chatcmpl-test",
            "choices": [{"finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "timings": {"prompt_per_second": 1000.0, "predicted_per_second": 150.0, "prompt_ms": 10.0, "predicted_ms": 33.3},
        }
    )
    assert metadata["response_id"] == "chatcmpl-test"
    assert metadata["finish_reason"] == "stop"
    assert metadata["prompt_tokens"] == 10
    assert metadata["completion_tokens"] == 5
    assert metadata["generated_tokens_per_second"] == 150.0


def test_raw_schema_request_forwards_and_records_stop_sequences(monkeypatch) -> None:
    requests = []

    def fake_request(request, endpoint):
        payload = json.loads(request.data)
        requests.append(payload)
        if request.full_url.endswith("/apply-template"):
            return {"prompt": "rendered"}, 0
        return {
            "content": '{"selection_id":0}',
            "stop_type": "word",
            "timings": {"prompt_n": 3, "predicted_n": 4},
        }, 0

    monkeypatch.setattr(
        "tapbench.r2_model_runner._request_llama_json",
        fake_request,
    )
    parsed, metadata = _request_schema_json(
        "http://test",
        [{"role": "user", "content": "choose"}],
        response_schema={"type": "object"},
        max_tokens=32,
        temperature=0.0,
        seed=1,
        stop_sequences=("<tool_call|>",),
    )
    assert parsed == {"selection_id": 0}
    assert requests[1]["stop"] == ["<tool_call|>"]
    assert metadata["finish_reason"] == "stop"
    assert metadata["stop_sequences"] == ["<tool_call|>"]



def test_render_chat_messages_include_method_policy() -> None:
    cfg = load_experiment_config()
    case = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["H6_abstention_suppression"])[0]
    messages = render_chat_messages(case, "constrained_abstain_b2")
    assert "Method policy:" in messages[1]["content"]
    assert "abstention-aware" in messages[1]["content"]
    assert "call-only baseline" in method_instruction("constrained_call_only_b2")



def test_runner_records_llama_server_errors_without_aborting(tmp_path: Path, monkeypatch) -> None:
    cfg = load_experiment_config()
    cases = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["H1_prompt_verbosity"])[:1]
    cases_path = tmp_path / "cases.jsonl"
    write_jsonl(cases_path, cases)

    def fail_once(*args, **kwargs):
        raise RuntimeError("synthetic server failure")

    monkeypatch.setattr(runner_module, "llama_server_action", fail_once)
    manifest = run_cases(
        cases_path=cases_path,
        output_path=tmp_path / "predictions.jsonl",
        timings_path=tmp_path / "timings.jsonl",
        manifest_path=tmp_path / "manifest.yaml",
        backend="llama-server",
        methods="prompt_strict_json",
        models="qwen3_1_7b",
        max_generations=1,
    )
    predictions = read_jsonl(tmp_path / "predictions.jsonl")
    timings = read_jsonl(tmp_path / "timings.jsonl")
    assert manifest["generation_count"] == 1
    assert predictions[0]["runner_error"] == "synthetic server failure"
    assert predictions[0]["response_metadata"]["finish_reason"] == "runner_error"
    assert timings[0]["error_type"] == "RuntimeError"

def test_retrieval_methods_limit_prompt_tools_to_top_k() -> None:
    cfg = load_experiment_config()
    case = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["R1_retrieval_ablation_N64"])[0]
    assert len(tools_for_method(case, "tap_no_retrieval")) == 64
    retrieved = tools_for_method(case, "tap_tfidf_char")
    assert len(retrieved) == 8
    assert any(tool.get("canonical_name") == case["gold_action"]["tool"] for tool in retrieved)
    messages = render_chat_messages(case, "tap_tfidf_char")
    encoded_tools = messages[1]["content"].split("Available tools: ", 1)[1].split("\nReturn the Action IR JSON now.", 1)[0]
    prompt_tools = json.loads(encoded_tools)
    assert len(prompt_tools) == 8

def test_runner_allows_explicit_local_model_outside_grid_group(tmp_path: Path) -> None:
    cfg = load_experiment_config()
    cases = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["H1_prompt_verbosity"])[:1]
    cases_path = tmp_path / "cases.jsonl"
    write_jsonl(cases_path, cases)

    run_cases(
        cases_path=cases_path,
        output_path=tmp_path / "predictions.jsonl",
        timings_path=tmp_path / "timings.jsonl",
        manifest_path=tmp_path / "manifest.yaml",
        backend="oracle",
        methods="prompt_strict_json",
        models="gemma4_12b_it_qat_local",
        max_generations=1,
    )

    predictions = read_jsonl(tmp_path / "predictions.jsonl")
    assert predictions[0]["model_id"] == "google/gemma-4-12B-it"
    assert predictions[0]["quantization"] == "none"



def test_xlam_native_json_array_is_normalized() -> None:
    action = parse_xlam_native_text('[{"name":"send_email","arguments":{"recipient":"a@example.com"}}]')
    assert action["mode"] == "call"
    assert action["tool"] == "send_email"
    assert action["arguments"] == {"recipient": "a@example.com"}
    assert action["payload"]["native_tool_call_count"] == 1


def test_xlam_tool_render_uses_native_function_schema() -> None:
    cfg = load_experiment_config()
    case = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["H5_specialist_gap"])[0]
    tools = render_xlam_tools(case, "native_specialist")
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == case["tools"][0]["name"]
    assert "parameters" in tools[0]["function"]


def test_action_ir_json_schema_limits_top_level_surface() -> None:
    cfg = load_experiment_config()
    case = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["H4_legal_token_mass"])[0]
    schema = action_ir_json_schema(case)
    assert schema["required"] == ["mode", "tool", "arguments", "payload"]
    assert schema["additionalProperties"] is False
    tool_enum = schema["properties"]["tool"]["anyOf"][0]["enum"]
    assert case["tools"][0]["name"] in tool_enum


def test_h5_hf_jobs_do_not_cross_specialist_formats() -> None:
    cfg = load_experiment_config()
    grids = _grid_by_id(cfg.subgrids)
    cases = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["H5_specialist_gap"])[:1]
    seeds = [1]
    xgrammar_jobs = _generation_jobs(
        cases,
        grids,
        cfg.models,
        backend="hf-xgrammar",
        method_filter=None,
        model_filter=None,
        seed_values=seeds,
        max_generations=None,
    )
    native_jobs = _generation_jobs(
        cases,
        grids,
        cfg.models,
        backend="hf-native",
        method_filter=None,
        model_filter=None,
        seed_values=seeds,
        max_generations=None,
    )
    assert {job[1] for job in xgrammar_jobs} == {"full_tap_b2", "best_of_n_budget_matched"}
    assert {job[2] for job in xgrammar_jobs} == {"qwen3_1_7b", "gemma4_e2b_it", "liquid_lfm25_8b_a1b"}
    assert {job[1] for job in native_jobs} == {"native_specialist"}
    assert {job[2] for job in native_jobs} == {"xlam2_1b_fc_r", "xlam2_3b_fc_r"}


def test_llama_request_retries_transient_http_500(monkeypatch) -> None:
    import io
    import urllib.error
    import urllib.request

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"choices": []}'

    attempts = iter([
        urllib.error.HTTPError("http://localhost", 500, "temporary", {}, io.BytesIO()),
        Response(),
    ])

    def urlopen(*args, **kwargs):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(runner_module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(runner_module.time, "sleep", lambda _: None)
    data, retry_count = _request_llama_json(
        urllib.request.Request("http://localhost"),
        "http://localhost",
    )
    assert data == {"choices": []}
    assert retry_count == 1

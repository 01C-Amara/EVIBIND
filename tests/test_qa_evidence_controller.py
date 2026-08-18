from __future__ import annotations

import hashlib
from pathlib import Path

from tapbench.discipline import coefficient_discipline_failures
from tapbench.extractive_qa_verifier import (
    EXTRACTIVE_QA_ARTIFACT_SCHEMA,
    EXTRACTIVE_QA_MARGIN_THRESHOLD,
    EXTRACTIVE_QA_MAX_ANSWER_TOKENS,
    EXTRACTIVE_QA_MAX_INPUT_TOKENS,
    EXTRACTIVE_QA_MODEL_ID,
    EXTRACTIVE_QA_MODEL_REVISION,
    EXTRACTIVE_QA_QUESTION_VERSION,
    EXTRACTIVE_QA_VERIFIER_VERSION,
    _json_sha256,
    verifier_question,
)
from tapbench.io import read_jsonl, write_jsonl
from tapbench.massive_agents import audit_qa_evidence_certificates
from tapbench.massive_runner import run_massive_cases
from tapbench.multilingual_retriever import (
    MULTILINGUAL_RETRIEVER_VERSION,
    RETRIEVAL_RANKING_SCHEMA_VERSION,
    RETRIEVER_MODEL_ID,
    RETRIEVER_REVISION,
    catalog_sha256,
    ranking_sha256,
)
from tapbench.qa_evidence_controller import (
    index_verifier_rows,
    run_qa_evidence_resolution,
    validate_verifier_record,
)
from tapbench.semantic_surface_projection import _public_slots
from tapbench.source_span_projection import source_span_catalog


def _tool() -> dict:
    return {
        "name": "email.send",
        "canonical_name": "email.send",
        "description": "Send an email to a person.",
        "parameters": {
            "type": "object",
            "properties": {
                "person": {"type": "string", "description": "recipient"},
                "date": {"type": "string", "description": "explicit date"},
            },
            "required": [],
            "additionalProperties": False,
        },
    }


def _ranking(case_id: str, request: str, tools: list[dict]) -> dict:
    ranked = [{"rank": 1, "tool": "email.send", "cosine_score": 0.9}]
    return {
        "schema_version": RETRIEVAL_RANKING_SCHEMA_VERSION,
        "case_id": case_id,
        "language": "en-US",
        "retriever_version": MULTILINGUAL_RETRIEVER_VERSION,
        "retriever_model_id": RETRIEVER_MODEL_ID,
        "retriever_revision": RETRIEVER_REVISION,
        "serialization_arm": "effect_only_v1",
        "k": 8,
        "request_sha256": hashlib.sha256(request.encode()).hexdigest(),
        "catalog_sha256": catalog_sha256(tools, "effect_only_v1"),
        "ranking": ranked,
        "ranking_sha256": ranking_sha256(ranked),
    }


def _records(request: str, ranking: dict) -> list[dict]:
    tool = _tool()
    slots, _ = _public_slots(tool, "forward")
    lattice = source_span_catalog(request, "en-US")
    alice = next(row for row in lattice["spans"] if row["source_text"] == "Alice")
    rows = []
    for slot in slots:
        admitted = slot["name"] == "person"
        row = {
                "schema_version": EXTRACTIVE_QA_ARTIFACT_SCHEMA,
                "verifier_version": EXTRACTIVE_QA_VERIFIER_VERSION,
                "question_version": EXTRACTIVE_QA_QUESTION_VERSION,
                "model_id": EXTRACTIVE_QA_MODEL_ID,
                "model_revision": EXTRACTIVE_QA_MODEL_REVISION,
                "backend": "huggingface_transformers_cpu",
                "dtype": "float32",
                "margin_threshold": EXTRACTIVE_QA_MARGIN_THRESHOLD,
                "max_input_tokens": EXTRACTIVE_QA_MAX_INPUT_TOKENS,
                "max_answer_tokens": EXTRACTIVE_QA_MAX_ANSWER_TOKENS,
                "case_id": "c1",
                "language": "en-US",
                "request_sha256": lattice["request_sha256"],
                "span_catalog_sha256": lattice["catalog_sha256"],
                "ranking_sha256": ranking["ranking_sha256"],
                "tool": "email.send",
                "slot_id": slot["slot_id"],
                "surface_name": slot["name"],
                "required": False,
                "question": verifier_question(slot["name"]),
                "input_truncated": False,
                "gold_loaded": False,
                "status": "admitted" if admitted else "null_margin",
                "admitted": admitted,
                "answer": "Alice" if admitted else None,
                "answer_span": alice["source_span"] if admitted else None,
                "span_id": alice["span_id"] if admitted else None,
                "non_null_margin": 5.0 if admitted else -5.0,
            }
        row["row_sha256"] = _json_sha256(row)
        rows.append(row)
    return rows


def _metadata() -> dict:
    return {
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "prompt_ms": 1.0,
        "generation_ms": 1.0,
        "generated_tokens_per_second": 100.0,
        "context_truncated": False,
        "rendered_input_tokens": 10,
        "context_headroom_tokens": 32000,
        "preflight_prompt_token_delta": 0,
    }


def _request(endpoint, messages, *, response_schema, **kwargs):
    del endpoint, messages, kwargs
    properties = response_schema["properties"]
    if "tool" in properties:
        return {"tool": "email.send"}, _metadata()
    if "active_slots" in properties:
        return {"active_slots": ["date", "person"]}, _metadata()
    raise AssertionError("unexpected schema")


def _run(condition: str):
    request = "Email Alice"
    tools = [_tool()]
    ranking = _ranking("c1", request, tools)
    return run_qa_evidence_resolution(
        case_id="c1",
        messages=[{"role": "user", "content": request}],
        tools=tools,
        language="en-US",
        ranking_row=ranking,
        ranking_artifact_sha256="c" * 64,
        verifier_index=index_verifier_rows(_records(request, ranking)),
        verifier_artifact_sha256="d" * 64,
        endpoint="http://unused",
        condition=condition,
        max_tokens=128,
        seed=1,
        request_fn=_request,
    )


def test_verifier_record_matches_runtime_case_and_lattice() -> None:
    request = "Email Alice"
    ranking = _ranking("c1", request, [_tool()])
    records = _records(request, ranking)
    slots, _ = _public_slots(_tool(), "forward")
    by_name = {slot["name"]: slot for slot in slots}
    for record in records:
        assert validate_verifier_record(
            record,
            case_id="c1",
            request_text=request,
            language="en-US",
            ranking_sha256=ranking["ranking_sha256"],
            tool="email.send",
            slot=by_name[record["surface_name"]],
        ) == []


def test_all_slots_arm_uses_one_small_model_call_and_qa_null() -> None:
    action, metadata = _run("tap_r_qa_all_slots_single")
    assert action["arguments"] == {"person": "Alice"}
    assert metadata["generation_calls"] == 1
    assert metadata["qa_verifier_rows_consulted"] == 2
    assert metadata["qa_verifier_null_count"] == 1
    assert metadata["small_model_supplies_argument_values"] is False


def test_active_arm_uses_tool_and_active_calls_then_qa_filter() -> None:
    action, metadata = _run("tap_r_qa_active_slots_single")
    assert action["arguments"] == {"person": "Alice"}
    assert metadata["generation_calls"] == 2
    assert metadata["active_slots_pre_verifier"] == ["date", "person"]
    assert metadata["active_slots_post_verifier"] == ["person"]


def test_consensus_arm_requires_four_small_model_calls() -> None:
    action, metadata = _run("tap_r_qa_active_slots_consensus")
    assert action["arguments"] == {"person": "Alice"}
    assert metadata["generation_calls"] == 4
    assert metadata["tool_agreement"] is True
    assert metadata["active_slot_agreement"] is True


def test_tampered_verifier_answer_fails_closed() -> None:
    request = "Email Alice"
    tools = [_tool()]
    ranking = _ranking("c1", request, tools)
    records = _records(request, ranking)
    person = next(row for row in records if row["surface_name"] == "person")
    person["answer_span"] = [0, 5]
    action, metadata = run_qa_evidence_resolution(
        case_id="c1",
        messages=[{"role": "user", "content": request}],
        tools=tools,
        language="en-US",
        ranking_row=ranking,
        ranking_artifact_sha256="c" * 64,
        verifier_index=index_verifier_rows(records),
        verifier_artifact_sha256="d" * 64,
        endpoint="http://unused",
        condition="tap_r_qa_all_slots_single",
        max_tokens=128,
        seed=1,
        request_fn=_request,
    )
    assert action["mode"] == "refuse"
    assert metadata["controller_stage_failure"] == "qa_verifier_record_invalid"


def test_qa_runner_and_independent_auditor_replay_provenance(
    tmp_path: Path,
) -> None:
    request = "Email Alice"
    tools = [_tool()]
    ranking = _ranking("c1", request, tools)
    case = {
        "case_id": "c1",
        "hypothesis_grid_id": "massive_qa_hybrid_smoke_v7",
        "messages": [{"role": "user", "content": request}],
        "tools": tools,
        "metadata": {"language": "en-US"},
    }
    cases_path = tmp_path / "cases.jsonl"
    rankings_path = tmp_path / "rankings.jsonl"
    verifier_path = tmp_path / "verifier.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"model")
    write_jsonl(cases_path, [case])
    write_jsonl(rankings_path, [ranking])
    write_jsonl(verifier_path, _records(request, ranking))

    manifest = run_massive_cases(
        cases_path,
        predictions_path,
        tmp_path / "timings.jsonl",
        tmp_path / "manifest.yaml",
        endpoint="http://unused",
        model_id="model",
        model_key="model",
        model_artifact=str(artifact),
        chat_template="qwen3",
        conditions=[
            "tap_r_qa_all_slots_single",
            "tap_r_qa_active_slots_single",
            "tap_r_qa_active_slots_consensus",
        ],
        rankings_path=rankings_path,
        qa_verifier_path=verifier_path,
        request_fn=_request,
    )
    predictions = read_jsonl(predictions_path)
    assert manifest["generation_count"] == 3
    assert manifest["actual_model_calls"] == 7
    assert manifest["runner_errors"] == 0
    assert coefficient_discipline_failures(predictions) == []

    report = audit_qa_evidence_certificates(
        cases_path,
        predictions_path,
        verifier_path,
        tmp_path / "audit.jsonl",
        tmp_path / "audit_summary.json",
    )
    assert report["rows"] == 3
    assert report["accepted_calls"] == 3
    assert report["failed"] == 0
    assert report["verifier_artifact_failures"] == []


def test_auditor_accepts_verified_fail_closed_materialization(
    tmp_path: Path,
) -> None:
    request = "Email Alice"
    tools = [_tool()]
    ranking = _ranking("c1", request, tools)
    case = {
        "case_id": "c1",
        "messages": [{"role": "user", "content": request}],
        "tools": tools,
        "metadata": {"language": "en-US"},
    }
    cases_path = tmp_path / "cases.jsonl"
    verifier_path = tmp_path / "verifier.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    write_jsonl(cases_path, [case])
    write_jsonl(verifier_path, _records(request, ranking))
    action, metadata = _run("tap_r_qa_all_slots_single")
    assert action["mode"] == "call"
    metadata.update(
        {
            "controller_stage_failure": "qa_materialize_required_slot_omitted",
            "active_slots_post_verifier": None,
            "risk_gate_passed": False,
        }
    )
    for field in (
        "qa_verifier_row_sha256",
        "qa_verifier_rows_consulted",
        "qa_verifier_null_count",
        "evidence_certificates",
        "selected_span_ids",
        "materialized_action_sha256",
        "no_unconstrained_action_critical_tokens",
    ):
        metadata.pop(field, None)
    verifier_sha256 = hashlib.sha256(verifier_path.read_bytes()).hexdigest()
    write_jsonl(
        predictions_path,
        [
            {
                "case_id": "c1",
                "model_id": "model",
                "method": "tap_r_qa_all_slots_single",
                "seed": 1,
                "prediction": {
                    "mode": "refuse",
                    "tool": None,
                    "arguments": {},
                    "payload": {"reason": "required evidence absent"},
                },
                "response_metadata": metadata,
                "qa_evidence_controller_version": (
                    metadata["qa_evidence_controller_version"]
                ),
                "qa_evidence_system_label": metadata[
                    "qa_evidence_system_label"
                ],
                "qa_verifier_version": metadata["qa_verifier_version"],
                "qa_verifier_question_version": metadata[
                    "qa_verifier_question_version"
                ],
                "qa_verifier_model_id": metadata["qa_verifier_model_id"],
                "qa_verifier_model_revision": metadata[
                    "qa_verifier_model_revision"
                ],
                "qa_verifier_backend": metadata["qa_verifier_backend"],
                "qa_verifier_dtype": metadata["qa_verifier_dtype"],
                "qa_verifier_artifact_sha256": verifier_sha256,
            }
        ],
    )
    metadata["qa_verifier_artifact_sha256"] = verifier_sha256
    predictions = read_jsonl(predictions_path)
    predictions[0]["response_metadata"] = metadata
    write_jsonl(predictions_path, predictions)

    report = audit_qa_evidence_certificates(
        cases_path,
        predictions_path,
        verifier_path,
        tmp_path / "audit.jsonl",
        tmp_path / "audit_summary.json",
    )
    assert report["rows"] == 1
    assert report["accepted_calls"] == 0
    assert report["failed"] == 0

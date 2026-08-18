from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tapbench.io import read_jsonl, write_jsonl
from tapbench.massive_agents import audit_retrieve_pointer_certificates
from tapbench.massive_runner import run_massive_cases
from tapbench.multilingual_retriever import (
    MULTILINGUAL_RETRIEVER_VERSION,
    RETRIEVAL_RANKING_SCHEMA_VERSION,
    RETRIEVER_MODEL_ID,
    RETRIEVER_REVISION,
    catalog_sha256,
    file_sha256,
    ranking_sha256,
)
from tapbench.retrieve_pointer import (
    run_retrieve_pointer_resolution,
    validate_external_ranking_row,
)


def _tool() -> dict:
    return {
        "name": "email.send",
        "canonical_name": "email.send",
        "description": "Send an email to a person.",
        "parameters": {
            "type": "object",
            "properties": {
                "person": {"type": "string", "description": "recipient"}
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
    properties = response_schema.get("properties", {})
    if "selection_id" in properties:
        content = messages[-1]["content"]
        catalog = json.loads(
            content.split("Candidate effects:\n", 1)[1].rsplit("\nReturn", 1)[0]
        )
        selected = next(
            row["selection_id"]
            for row in catalog
            if row["effect"] == "email.send"
        )
        return {"selection_id": selected}, _metadata()
    content = messages[-1]["content"]
    spans = json.loads(
        content.split("Finite source spans:\n", 1)[1].rsplit("\nReturn", 1)[0]
    )
    alice = next(row["span_id"] for row in spans if row["text"] == "Alice")
    slot_ids = response_schema["properties"]["bindings"]["required"]
    return {"bindings": {slot_id: alice for slot_id in slot_ids}}, _metadata()


def test_consensus_controller_is_call_only_and_replayable() -> None:
    request = "Email Alice"
    tools = [_tool()]
    action, metadata = run_retrieve_pointer_resolution(
        case_id="c1",
        messages=[{"role": "user", "content": request}],
        tools=tools,
        language="en-US",
        ranking_row=_ranking("c1", request, tools),
        ranking_artifact_sha256="a" * 64,
        endpoint="http://unused",
        condition="tap_r_retrieve_pointer_consensus",
        max_tokens=128,
        seed=1,
        request_fn=_request,
    )
    assert action == {
        "mode": "call",
        "tool": "email.send",
        "arguments": {"person": "Alice"},
        "payload": {},
    }
    assert metadata["generation_calls"] == 4
    assert metadata["no_call_election_option"] is False
    assert metadata["pointer_agreement"] is True
    assert metadata["no_generated_action_critical_literals"] is True


def test_external_ranking_contract_is_explicit_and_gold_free() -> None:
    request = "Email Alice"
    tools = [_tool()]
    ranked = [{"rank": 1, "tool": "email.send", "cosine_score": 0.8}]
    row = {
        "schema_version": "external.ranking.v1",
        "case_id": "c1",
        "retriever_version": "external.router.v1",
        "retriever_model_id": "centroid",
        "retriever_revision": "train_only",
        "k": 8,
        "request_sha256": hashlib.sha256(request.encode()).hexdigest(),
        "ranking": ranked,
        "ranking_sha256": ranking_sha256(ranked),
    }
    assert validate_external_ranking_row(
        row,
        case_id="c1",
        request_text=request,
        tools=tools,
        schema_version="external.ranking.v1",
        retriever_version="external.router.v1",
        retriever_model_id="centroid",
        retriever_revision="train_only",
    ) == ranked

    row["gold_tool"] = "email.send"
    try:
        validate_external_ranking_row(
            row,
            case_id="c1",
            request_text=request,
            tools=tools,
            schema_version="external.ranking.v1",
            retriever_version="external.router.v1",
            retriever_model_id="centroid",
            retriever_revision="train_only",
        )
    except ValueError as error:
        assert "scorer-only" in str(error)
    else:
        raise AssertionError("external ranking validator accepted a gold field")


def test_runner_records_v2_provenance_and_certificate_replays(
    tmp_path: Path,
) -> None:
    request = "Email Alice"
    tools = [_tool()]
    cases_path = tmp_path / "cases.jsonl"
    rankings_path = tmp_path / "rankings.jsonl"
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"model")
    write_jsonl(
        cases_path,
        [
            {
                "case_id": "c1",
                "hypothesis_grid_id": "MASSIVE_Agents_CAPC_language_disjoint_v1",
                "messages": [{"role": "user", "content": request}],
                "tools": tools,
                "metadata": {"language": "en-US"},
            }
        ],
    )
    write_jsonl(rankings_path, [_ranking("c1", request, tools)])
    output = tmp_path / "predictions.jsonl"
    manifest = run_massive_cases(
        cases_path,
        output,
        tmp_path / "timings.jsonl",
        tmp_path / "manifest.yaml",
        endpoint="http://unused",
        model_id="model",
        model_key="model",
        model_artifact=str(artifact),
        chat_template="qwen3",
        conditions=["tap_r_retrieve_pointer_consensus"],
        request_fn=_request,
        rankings_path=rankings_path,
    )
    row = read_jsonl(output)[0]
    assert manifest["actual_model_calls"] == 4
    assert manifest["ranking_artifact_sha256"] == file_sha256(rankings_path)
    assert row["retrieve_pointer_version"] == "tapbench.retrieve_pointer.v2"
    assert row["source_span_certificate_version"] == (
        "tapbench.source_span_certificate.v2"
    )
    assert row["ranking_artifact_sha256"] == file_sha256(rankings_path)
    report = audit_retrieve_pointer_certificates(
        cases_path,
        output,
        tmp_path / "audit.jsonl",
        tmp_path / "audit_summary.json",
    )
    assert report["rows"] == 1
    assert report["accepted_calls"] == 1
    assert report["failed"] == 0

from __future__ import annotations

import hashlib
from pathlib import Path

from tapbench.discipline import coefficient_discipline_failures
from tapbench.io import read_jsonl, write_jsonl
from tapbench.massive_agents import audit_slotwise_surface_certificates
from tapbench.massive_runner import run_massive_cases
from tapbench.multilingual_retriever import (
    MULTILINGUAL_RETRIEVER_VERSION,
    RETRIEVAL_RANKING_SCHEMA_VERSION,
    RETRIEVER_MODEL_ID,
    RETRIEVER_REVISION,
    catalog_sha256,
    ranking_sha256,
)
from tapbench.slotwise_surface_projection import (
    _slot_messages,
    minimal_surface_catalog,
    run_slotwise_surface_resolution,
    slotwise_value_schema,
    validate_slotwise_value,
)
from tapbench.source_span_projection import replay_span_certificate


def _tool(*, required: bool = False, with_slots: bool = True) -> dict:
    properties = (
        {
            "person": {"type": "string", "description": "recipient"},
            "date": {"type": "string", "description": "explicit date"},
        }
        if with_slots
        else {}
    )
    return {
        "name": "email.send",
        "canonical_name": "email.send",
        "description": "Send an email to a person.",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": ["person"] if required else [],
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
    del endpoint, kwargs
    if "tool" in response_schema["properties"]:
        return {"tool": "email.send"}, _metadata()
    content = messages[-1]["content"]
    if '"name":"person"' in content:
        return {"value": "Alice"}, _metadata()
    assert '"name":"date"' in content
    return {"value": None}, _metadata()


def test_minimal_catalog_preserves_support_and_reverses_only_order() -> None:
    forward, _ = minimal_surface_catalog("Email Alice now", "en-US", "forward")
    reverse, _ = minimal_surface_catalog("Email Alice now", "en-US", "reverse")
    forward_values = [row["source_text"] for row in forward]
    reverse_values = [row["source_text"] for row in reverse]
    assert forward_values == list(reversed(reverse_values))
    assert forward_values[:3] == ["Email", "Alice", "now"]
    assert "Email Alice now" == forward_values[-1]


def test_optional_schema_admits_null_but_required_schema_does_not() -> None:
    surfaces, _ = minimal_surface_catalog("Email Alice", "en-US", "forward")
    optional = {
        "name": "person",
        "description": "recipient",
        "required": False,
    }
    required = {**optional, "required": True}
    assert slotwise_value_schema(optional, surfaces)["properties"]["value"][
        "enum"
    ][0] is None
    assert None not in slotwise_value_schema(required, surfaces)["properties"][
        "value"
    ]["enum"]
    assert validate_slotwise_value(
        {"value": None}, slot=optional, surfaces=surfaces
    )[1]["status"] == "validated_null"
    assert validate_slotwise_value(
        {"value": None}, slot=required, surfaces=surfaces
    )[1]["status"] == "required_slot_null"


def test_prompt_exposes_candidates_and_fixed_multilingual_examples() -> None:
    surfaces, _ = minimal_surface_catalog("Email Alice", "en-US", "forward")
    slot = {"name": "person", "description": "recipient", "required": False}
    messages = _slot_messages("Email Alice", _tool(), slot, surfaces)
    assert "Amira" in messages[0]["content"]
    assert "شیراز" in messages[0]["content"]
    assert "京都" in messages[0]["content"]
    assert '"Alice"' in messages[1]["content"]
    assert "shortest exact" in messages[0]["content"]


def test_slotwise_single_omits_null_and_replays_selected_surface() -> None:
    request = "Email Alice"
    tools = [_tool()]
    action, metadata = run_slotwise_surface_resolution(
        case_id="c1",
        messages=[{"role": "user", "content": request}],
        tools=tools,
        language="en-US",
        ranking_row=_ranking("c1", request, tools),
        ranking_artifact_sha256="a" * 64,
        endpoint="http://unused",
        condition="tap_r_slotwise_surface_single",
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
    assert metadata["generation_calls"] == 3
    assert metadata["slotwise_null_count"] == 1
    assert metadata["slotwise_independent_generation"] is True
    assert metadata["no_unconstrained_action_critical_tokens"] is True
    assert replay_span_certificate(
        request, "en-US", metadata["evidence_certificates"]["person"]
    )


def test_counterbalanced_views_agree_on_materialized_action() -> None:
    request = "Email Alice"
    tools = [_tool()]
    action, metadata = run_slotwise_surface_resolution(
        case_id="c1",
        messages=[{"role": "user", "content": request}],
        tools=tools,
        language="en-US",
        ranking_row=_ranking("c1", request, tools),
        ranking_artifact_sha256="b" * 64,
        endpoint="http://unused",
        condition="tap_r_slotwise_surface_consensus",
        max_tokens=128,
        seed=1,
        request_fn=_request,
    )
    assert action["arguments"] == {"person": "Alice"}
    assert metadata["generation_calls"] == 6
    assert metadata["slotwise_action_agreement"] is True


def test_zero_slot_tool_requires_only_tool_election() -> None:
    request = "Open email"
    tools = [_tool(with_slots=False)]

    def tool_only(endpoint, messages, *, response_schema, **kwargs):
        del endpoint, messages, kwargs
        assert "tool" in response_schema["properties"]
        return {"tool": "email.send"}, _metadata()

    action, metadata = run_slotwise_surface_resolution(
        case_id="c0",
        messages=[{"role": "user", "content": request}],
        tools=tools,
        language="en-US",
        ranking_row=_ranking("c0", request, tools),
        ranking_artifact_sha256="c" * 64,
        endpoint="http://unused",
        condition="tap_r_slotwise_surface_single",
        max_tokens=128,
        seed=1,
        request_fn=tool_only,
    )
    assert action["arguments"] == {}
    assert metadata["generation_calls"] == 1


def test_runner_provenance_discipline_and_independent_audit(
    tmp_path: Path,
) -> None:
    request = "Email Alice"
    tools = [_tool()]
    cases_path = tmp_path / "cases.jsonl"
    rankings_path = tmp_path / "rankings.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"model")
    write_jsonl(
        cases_path,
        [
            {
                "case_id": "c1",
                "hypothesis_grid_id": "slotwise_surface_smoke_v5",
                "messages": [{"role": "user", "content": request}],
                "tools": tools,
                "metadata": {"language": "en-US"},
            }
        ],
    )
    write_jsonl(rankings_path, [_ranking("c1", request, tools)])
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
        conditions=["tap_r_slotwise_surface_single"],
        rankings_path=rankings_path,
        request_fn=_request,
    )
    predictions = read_jsonl(predictions_path)
    row = predictions[0]
    assert manifest["runner_version"] == "tapbench.massive_runner.v5"
    assert manifest["slotwise_surface_version"] == (
        "tapbench.slotwise_surface_projection.v1"
    )
    assert row["slotwise_surface_version"] == (
        "tapbench.slotwise_surface_projection.v1"
    )
    assert row["semantic_surface_materializer_version"] == (
        "tapbench.semantic_surface_projection.v1"
    )
    assert coefficient_discipline_failures(predictions) == []
    report = audit_slotwise_surface_certificates(
        cases_path,
        predictions_path,
        tmp_path / "certificate_details.jsonl",
        tmp_path / "certificate_summary.json",
    )
    assert report["rows"] == report["accepted_calls"] == report["passed"] == 1
    assert report["failed"] == 0

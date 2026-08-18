from __future__ import annotations

import hashlib
from pathlib import Path

from tapbench.discipline import coefficient_discipline_failures
from tapbench.io import read_jsonl, write_jsonl
from tapbench.massive_agents import audit_semantic_surface_certificates
from tapbench.massive_runner import run_massive_cases
from tapbench.multilingual_retriever import (
    MULTILINGUAL_RETRIEVER_VERSION,
    RETRIEVAL_RANKING_SCHEMA_VERSION,
    RETRIEVER_MODEL_ID,
    RETRIEVER_REVISION,
    catalog_sha256,
    ranking_sha256,
)
from tapbench.semantic_surface_projection import (
    _active_slot_schema,
    _binding_schema,
    _public_slots,
    _tool_catalog,
    _tool_schema,
    materialize_surface_bindings,
    run_semantic_surface_resolution,
    validate_active_slots,
)
from tapbench.source_span_projection import replay_span_certificate


def _tool(*, required: bool = False, with_slot: bool = True) -> dict:
    properties = (
        {"person": {"type": "string", "description": "recipient"}}
        if with_slot
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
    del endpoint, messages, kwargs
    properties = response_schema["properties"]
    if "tool" in properties:
        return {"tool": "email.send"}, _metadata()
    if "active_slots" in properties:
        return {"active_slots": ["person"]}, _metadata()
    assert "Alice" in properties["bindings"]["properties"]["person"]["enum"]
    return {"bindings": {"person": "Alice"}}, _metadata()


def test_schemas_expose_semantic_labels_not_opaque_ids() -> None:
    tool = _tool()
    catalog = _tool_catalog([tool], "forward")
    slots, _ = _public_slots(tool, "forward")
    tool_values = _tool_schema(catalog)["properties"]["tool"]["enum"]
    slot_values = _active_slot_schema(slots)["properties"]["active_slots"][
        "items"
    ]["enum"]
    binding = _binding_schema(
        ["person"], [{"source_text": "Alice", "span_id": "p0001"}]
    )
    surface_values = binding["properties"]["bindings"]["properties"][
        "person"
    ]["enum"]
    assert tool_values == ["email.send"]
    assert slot_values == ["person"]
    assert surface_values == ["Alice"]
    assert "slot_id" not in str(binding)
    assert "selection_id" not in str(_tool_schema(catalog))


def test_required_slots_are_restored_but_optional_slots_can_be_omitted() -> None:
    optional, optional_meta = validate_active_slots(
        {"active_slots": []}, tool=_tool(required=False)
    )
    required, required_meta = validate_active_slots(
        {"active_slots": []}, tool=_tool(required=True)
    )
    assert optional == []
    assert optional_meta["required_slots_added"] == []
    assert required == ["person"]
    assert required_meta["required_slots_added"] == ["person"]


def test_surface_binding_materializes_and_replays_source_certificate() -> None:
    request = "Email Alice"
    tool = _tool()
    action, metadata = materialize_surface_bindings(
        {"bindings": {"person": "Alice"}},
        active_slots=["person"],
        selected_tool="email.send",
        tool=tool,
        tools=[tool],
        request_text=request,
        language="en-US",
    )
    assert action == {
        "mode": "call",
        "tool": "email.send",
        "arguments": {"person": "Alice"},
        "payload": {},
    }
    assert replay_span_certificate(
        request, "en-US", metadata["certificates"]["person"]
    )
    assert metadata["no_unconstrained_action_critical_tokens"] is True


def test_consensus_controller_uses_semantic_surfaces_and_agrees() -> None:
    request = "Email Alice"
    tools = [_tool()]
    action, metadata = run_semantic_surface_resolution(
        case_id="c1",
        messages=[{"role": "user", "content": request}],
        tools=tools,
        language="en-US",
        ranking_row=_ranking("c1", request, tools),
        ranking_artifact_sha256="a" * 64,
        endpoint="http://unused",
        condition="tap_r_surface_active_consensus",
        max_tokens=128,
        seed=1,
        request_fn=_request,
    )
    assert action["arguments"] == {"person": "Alice"}
    assert metadata["generation_calls"] == 6
    assert metadata["tool_agreement"] is True
    assert metadata["active_slot_agreement"] is True
    assert metadata["surface_action_agreement"] is True
    assert metadata["no_unconstrained_action_critical_tokens"] is True


def test_zero_argument_tool_skips_empty_slot_enum() -> None:
    request = "Open the email application"
    tools = [_tool(with_slot=False)]

    def tool_only_request(endpoint, messages, *, response_schema, **kwargs):
        del endpoint, messages, kwargs
        assert "tool" in response_schema["properties"]
        return {"tool": "email.send"}, _metadata()

    action, metadata = run_semantic_surface_resolution(
        case_id="c0",
        messages=[{"role": "user", "content": request}],
        tools=tools,
        language="en-US",
        ranking_row=_ranking("c0", request, tools),
        ranking_artifact_sha256="b" * 64,
        endpoint="http://unused",
        condition="tap_r_surface_active_consensus",
        max_tokens=128,
        seed=1,
        request_fn=tool_only_request,
    )
    assert action["arguments"] == {}
    assert metadata["generation_calls"] == 2
    assert metadata["active_slot_views_skipped"] is True
    assert metadata["surface_binding_views_skipped"] is True


def test_runner_provenance_and_independent_audit(tmp_path: Path) -> None:
    request = "Email Alice"
    tools = [_tool()]
    case = {
        "case_id": "c1",
        "hypothesis_grid_id": "semantic_surface_smoke_v4",
        "messages": [{"role": "user", "content": request}],
        "tools": tools,
        "metadata": {"language": "en-US"},
    }
    cases_path = tmp_path / "cases.jsonl"
    rankings_path = tmp_path / "rankings.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"model")
    write_jsonl(cases_path, [case])
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
        conditions=["tap_r_surface_active_single"],
        rankings_path=rankings_path,
        request_fn=_request,
    )
    predictions = read_jsonl(predictions_path)
    row = predictions[0]
    assert manifest["semantic_surface_version"] == (
        "tapbench.semantic_surface_projection.v1"
    )
    assert row["semantic_surface_version"] == (
        "tapbench.semantic_surface_projection.v1"
    )
    assert row["source_span_certificate_version"] == (
        "tapbench.source_span_certificate.v2"
    )
    assert coefficient_discipline_failures(predictions) == []
    report = audit_semantic_surface_certificates(
        cases_path,
        predictions_path,
        tmp_path / "certificate_details.jsonl",
        tmp_path / "certificate_summary.json",
    )
    assert report["rows"] == report["accepted_calls"] == report["passed"] == 1
    assert report["failed"] == 0

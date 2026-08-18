from __future__ import annotations

import hashlib

from tapbench.extractive_qa_verifier import (
    EXTRACTIVE_QA_ARTIFACT_SCHEMA,
    EXTRACTIVE_QA_MAX_ANSWER_TOKENS,
    EXTRACTIVE_QA_MAX_INPUT_TOKENS,
    EXTRACTIVE_QA_MODEL_ID,
    EXTRACTIVE_QA_MODEL_REVISION,
    EXTRACTIVE_QA_QUESTION_VERSION,
    EXTRACTIVE_QA_VERIFIER_VERSION,
    _json_sha256,
    verifier_question,
)
from tapbench.multilingual_retriever import ranking_sha256
from tapbench.qa_evidence_controller import index_verifier_rows
from tapbench.semantic_surface_projection import _public_slots
from tapbench.source_span_projection import source_span_catalog
from tapbench.supervised_router_qa import materialize_router_qa_action
from tapbench.supervised_router_small_model_qa import (
    run_small_model_router_qa_action,
)


def _fixture() -> tuple[dict, dict, dict]:
    request = "Email Alice"
    tool = {
        "name": "email.send",
        "canonical_name": "email.send",
        "description": "Send email",
        "parameters": {
            "type": "object",
            "properties": {"person": {"type": "string", "description": "recipient"}},
            "required": [],
            "additionalProperties": False,
        },
    }
    case = {
        "case_id": "c1",
        "messages": [{"role": "user", "content": request}],
        "tools": [tool],
        "metadata": {"language": "en-US"},
    }
    ranked = [{"rank": 1, "tool": "email.send", "cosine_score": 0.9}]
    ranking = {
        "schema_version": "tapbench.supervised_tool_ranking.v1",
        "case_id": "c1",
        "retriever_version": "tapbench.massive_supervised_intent_router.v1",
        "retriever_model_id": "hashed_tfidf_nearest_centroid",
        "retriever_revision": "massive_v1.1_train_only",
        "k": 8,
        "request_sha256": hashlib.sha256(request.encode()).hexdigest(),
        "ranking": ranked,
        "ranking_sha256": ranking_sha256(ranked),
    }
    lattice = source_span_catalog(request, "en-US")
    span = next(row for row in lattice["spans"] if row["source_text"] == "Alice")
    slots, _ = _public_slots(tool, "forward")
    record = {
        "schema_version": EXTRACTIVE_QA_ARTIFACT_SCHEMA,
        "verifier_version": EXTRACTIVE_QA_VERIFIER_VERSION,
        "question_version": EXTRACTIVE_QA_QUESTION_VERSION,
        "model_id": EXTRACTIVE_QA_MODEL_ID,
        "model_revision": EXTRACTIVE_QA_MODEL_REVISION,
        "backend": "huggingface_transformers_cpu",
        "dtype": "float32",
        "margin_threshold": 0.0,
        "max_input_tokens": EXTRACTIVE_QA_MAX_INPUT_TOKENS,
        "max_answer_tokens": EXTRACTIVE_QA_MAX_ANSWER_TOKENS,
        "input_truncated": False,
        "gold_loaded": False,
        "case_id": "c1",
        "language": "en-US",
        "request_sha256": hashlib.sha256(request.encode()).hexdigest(),
        "span_catalog_sha256": lattice["catalog_sha256"],
        "ranking_sha256": ranking["ranking_sha256"],
        "tool": "email.send",
        "tool_rank": 1,
        "slot_id": slots[0]["slot_id"],
        "surface_name": "person",
        "required": False,
        "question": verifier_question("person"),
        "input_tokens": 8,
        "status": "admitted",
        "admitted": True,
        "answer": "Alice",
        "answer_span": span["source_span"],
        "span_id": span["span_id"],
        "non_null_margin": 10.0,
    }
    record["row_sha256"] = _json_sha256(record)
    return case, ranking, record


def test_router_qa_materializes_only_verified_source_value() -> None:
    case, ranking, record = _fixture()
    action, metadata = materialize_router_qa_action(
        case=case,
        ranking_row=ranking,
        verifier_index=index_verifier_rows([record]),
        method="tap_r_supervised_router_qa_all",
        dev_threshold=0.2,
    )
    assert action["tool"] == "email.send"
    assert action["arguments"] == {"person": "Alice"}
    assert metadata["small_model_in_path"] is False
    assert metadata["no_unconstrained_action_critical_tokens"] is True


def test_dev95_condition_abstains_below_locked_threshold() -> None:
    case, ranking, record = _fixture()
    action, metadata = materialize_router_qa_action(
        case=case,
        ranking_row=ranking,
        verifier_index=index_verifier_rows([record]),
        method="tap_r_supervised_router_qa_dev95",
        dev_threshold=0.2,
    )
    assert action["mode"] == "refuse"
    assert metadata["router_selected"] is False
    assert metadata["qa_verifier_rows_consulted"] == 0


def test_slot_knn_condition_uses_only_declared_active_slots() -> None:
    case, ranking, record = _fixture()
    action, metadata = materialize_router_qa_action(
        case=case,
        ranking_row=ranking,
        verifier_index=index_verifier_rows([record]),
        method="tap_r_supervised_router_slot_knn_qa_all",
        dev_threshold=0.2,
        slot_prediction_row={
            "schema_version": "tapbench.massive_supervised_slot_knn.v1",
            "case_id": "c1",
            "predicted_tool": "email.send",
            "active_slots": ["person"],
            "k": 3,
            "vote_threshold": 0.5,
            "neighbor_ids": ["train-1"],
        },
    )
    assert action["arguments"] == {"person": "Alice"}
    assert metadata["active_slot_policy"] == "massive_train_dev_supervised_knn"


def test_small_model_bridge_uses_model_only_for_active_slots() -> None:
    case, ranking, record = _fixture()
    calls = []

    def request_fn(endpoint, messages, **kwargs):
        calls.append((endpoint, messages, kwargs))
        return {"active_slots": ["person"]}, {
            "raw_text": '{"active_slots":["person"]}',
            "completion_tokens": 8,
            "generation_ms": 100.0,
        }

    action, metadata = run_small_model_router_qa_action(
        case=case,
        ranking_row=ranking,
        verifier_index=index_verifier_rows([record]),
        method="tap_r_supervised_router_small_model_slots_qa_all",
        dev_threshold=0.2,
        endpoint="http://localhost:1234",
        max_tokens=384,
        seed=1,
        request_fn=request_fn,
    )
    assert len(calls) == 1
    assert action["arguments"] == {"person": "Alice"}
    assert metadata["small_model_in_path"] is True
    assert metadata["small_model_role"] == "active_slot_selection_only"
    assert metadata["small_model_supplies_argument_values"] is False
    assert metadata["no_unconstrained_action_critical_tokens"] is True


def test_small_model_bridge_dev95_abstains_without_model_call() -> None:
    case, ranking, record = _fixture()

    def request_fn(*args, **kwargs):
        raise AssertionError("model must not be called below the router threshold")

    action, metadata = run_small_model_router_qa_action(
        case=case,
        ranking_row=ranking,
        verifier_index=index_verifier_rows([record]),
        method="tap_r_supervised_router_small_model_slots_qa_dev95",
        dev_threshold=0.2,
        endpoint="http://localhost:1234",
        max_tokens=384,
        seed=1,
        request_fn=request_fn,
    )
    assert action["mode"] == "refuse"
    assert metadata["generation_calls"] == 0

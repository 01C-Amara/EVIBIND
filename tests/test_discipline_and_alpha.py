from __future__ import annotations

import pytest

from tapbench.alpha import fragmentation_stats
from tapbench.config import load_experiment_config
from tapbench.discipline import assert_coefficient_discipline, coefficient_discipline_failures
from tapbench.generator import generate_cases_from_config


def test_backend_discipline_flags_mixed_grid_backend() -> None:
    rows = [
        {"hypothesis_grid_id": "H1_prompt_verbosity", "model_id": "m", "backend": "llama.cpp", "quantization": "Q4_K_M", "chat_template": "t", "grammar_engine": "gbnf", "model_artifact": "a"},
        {"hypothesis_grid_id": "H1_prompt_verbosity", "model_id": "m", "backend": "hf", "quantization": "Q4_K_M", "chat_template": "t", "grammar_engine": "gbnf", "model_artifact": "a"},
    ]
    failures = coefficient_discipline_failures(rows)
    assert failures
    with pytest.raises(AssertionError):
        assert_coefficient_discipline(rows)


def test_discipline_rejects_mixed_chat_parser_modes() -> None:
    common = {
        "hypothesis_grid_id": "R2A_component_evaluation",
        "model_id": "m",
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "chat_template": "t",
        "grammar_engine": "llama.cpp_json_schema_gbnf",
        "model_artifact": "a",
        "thinking_mode": "off",
        "reasoning_budget": 0,
    }
    failures = coefficient_discipline_failures([
        {**common, "chat_parser": "pure_content"},
        {**common, "chat_parser": "peg_native"},
    ])
    assert any(failure["field"] == "chat_parser" for failure in failures)




def test_discipline_rejects_thinking_markers_and_budget_modes() -> None:
    rows = [
        {
            "hypothesis_grid_id": "H6_abstention_suppression",
            "model_id": "m",
            "backend": "llama.cpp",
            "quantization": "Q4_K_M",
            "chat_template": "t",
            "grammar_engine": "gbnf",
            "model_artifact": "a",
            "thinking_mode": "budget_128",
            "reasoning_budget": 128,
            "thinking_marker_detected": True,
        }
    ]
    failures = coefficient_discipline_failures(rows)
    assert {failure["field"] for failure in failures} >= {"thinking_mode", "thinking_marker_detected"}

def test_alpha_proxy_separates_fragmented_from_aligned_names() -> None:
    cfg = load_experiment_config()
    cases = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["H3_schema_alpha"])
    report = fragmentation_stats(cases)
    means = report["mean_by_alpha"]
    assert means["fragmented"] > means["aligned"]


def test_discipline_rejects_mixed_tap_r_component_versions() -> None:
    common = {
        "hypothesis_grid_id": "BFCL_v4_external_anchor",
        "method": "tap_r_hybrid_span_tier_b",
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "chat_template": "qwen3",
        "grammar_engine": "gbnf",
        "thinking_mode": "off",
        "reasoning_budget": 0,
        "deployable_resolution_version": "tapbench.deployable_resolution.v2",
        "evidence_contract_version": "tapbench.evidence_contract.v2",
    }
    rows = [
        {**common, "model_id": "m1", "model_artifact": "m1.gguf", "contract_solver_version": "tapbench.contract_solver.v1"},
        {**common, "model_id": "m2", "model_artifact": "m2.gguf", "contract_solver_version": "tapbench.contract_solver.v2"},
    ]
    failures = coefficient_discipline_failures(rows)
    assert any(failure["field"] == "contract_solver_version" for failure in failures)


def test_discipline_requires_one_tep_version_for_tep_coefficients() -> None:
    common = {
        "hypothesis_grid_id": "R2A_tep_evidence",
        "method": "tap_r_tep_tier_a",
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "chat_template": "qwen3",
        "grammar_engine": "gbnf",
        "thinking_mode": "off",
        "reasoning_budget": 0,
        "deployable_resolution_version": "tapbench.deployable_resolution.v4",
        "evidence_contract_version": "tapbench.evidence_contract.v2",
        "contract_solver_version": "tapbench.contract_solver.v2",
        "model_id": "m1",
        "model_artifact": "m1.gguf",
    }
    failures = coefficient_discipline_failures([common])
    assert any(failure["field"] == "typed_evidence_program_version" for failure in failures)
    assert coefficient_discipline_failures([{**common, "typed_evidence_program_version": "tapbench.typed_evidence_program.v2"}]) == []



def test_discipline_requires_effect_first_version_and_risk_threshold() -> None:
    row = {
        "hypothesis_grid_id": "R2C_effect_first_confirmation",
        "method": "tap_r_effect_first_consensus_locked",
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "chat_template": "qwen3",
        "grammar_engine": "gbnf",
        "chat_parser": "raw",
        "inference_path": "completion",
        "model_artifact": "model.gguf",
        "model_id": "model",
        "thinking_mode": "off",
        "reasoning_budget": 0,
        "deployable_resolution_version": "deployable",
        "evidence_contract_version": "evidence",
        "contract_solver_version": "contract",
    }
    fields = {row["field"] for row in coefficient_discipline_failures([row])}
    assert {"effect_first_version", "action_risk_threshold"} <= fields
    complete = {
        **row,
        "effect_first_version": "effect-first-v1",
        "action_risk_threshold": 0.05,
    }
    assert coefficient_discipline_failures([complete]) == []


def test_discipline_uses_eflrx_specific_component_contract() -> None:
    row = {
        "hypothesis_grid_id": "BFCL_v4_EFLRX_development",
        "method": "tap_r_eflrx_consensus",
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "chat_template": "qwen3",
        "grammar_engine": "gbnf",
        "chat_parser": "raw",
        "inference_path": "completion",
        "model_artifact": "model.gguf",
        "model_id": "model",
        "thinking_mode": "off",
        "reasoning_budget": 0,
    }
    fields = {item["field"] for item in coefficient_discipline_failures([row])}
    expected_fields = {
        "eflrx_version",
        "extractive_candidate_version",
        "eflrx_runner_version",
        "action_risk_threshold",
    }
    assert expected_fields <= fields
    complete = {
        **row,
        "eflrx_version": "tapbench.eflrx.v1",
        "extractive_candidate_version": "tapbench.extractive_candidates.v4",
        "eflrx_runner_version": "tapbench.eflrx_runner.v2",
        "action_risk_threshold": 0.05,
    }
    assert coefficient_discipline_failures([complete]) == []


def test_discipline_uses_capc_specific_component_contract() -> None:
    row = {
        "hypothesis_grid_id": "BFCL_v4_CAPC_development",
        "method": "tap_r_capc_dual",
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "chat_template": "qwen3",
        "grammar_engine": "gbnf",
        "chat_parser": "raw",
        "inference_path": "completion",
        "model_artifact": "model.gguf",
        "model_id": "model",
        "thinking_mode": "off",
        "reasoning_budget": 0,
    }
    fields = {item["field"] for item in coefficient_discipline_failures([row])}
    assert {
        "capc_version",
        "extractive_candidate_version",
        "capc_runner_version",
        "action_risk_threshold",
    } <= fields
    complete = {
        **row,
        "capc_version": "tapbench.capc.v1",
        "extractive_candidate_version": "tapbench.extractive_candidates.v4",
        "capc_runner_version": "tapbench.capc_runner.v1",
        "action_risk_threshold": 0.05,
    }
    assert coefficient_discipline_failures([complete]) == []


def test_discipline_uses_selective_tapr_component_contract() -> None:
    row = {
        "hypothesis_grid_id": "R2D_selective_composite_confirmation_v7",
        "method": "tap_r_selective_full",
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "chat_template": "qwen3",
        "grammar_engine": "gbnf",
        "chat_parser": "raw",
        "inference_path": "completion",
        "model_artifact": "model.gguf",
        "model_id": "model",
        "thinking_mode": "off",
        "reasoning_budget": 0,
    }
    fields = {item["field"] for item in coefficient_discipline_failures([row])}
    assert {
        "selective_tapr_version",
        "admission_version",
        "effect_support_version",
        "scope_guard_version",
        "certificate_span_policy_version",
        "extractive_candidate_version",
        "r2d_model_runner_version",
        "action_risk_threshold",
        "stop_sequence_policy_version",
        "stop_sequences",
    } <= fields
    complete = {
        **row,
        "selective_tapr_version": "tapbench.selective_tapr.v4",
        "admission_version": "tapbench.speech_act_admission.v3",
        "effect_support_version": "tapbench.effect_support_verifier.v1",
        "scope_guard_version": "tapbench.deterministic_scope_guard.v1",
        "certificate_span_policy_version": "tapbench.certificate_span_policy.v2",
        "extractive_candidate_version": "tapbench.extractive_candidates.v4",
        "r2d_model_runner_version": "tapbench.r2d_model_runner.v7",
        "action_risk_threshold": 0.05,
        "stop_sequence_policy_version": "tapbench.r2d_stop_sequences.v1",
        "stop_sequences": [],
    }
    assert coefficient_discipline_failures([complete]) == []


def test_discipline_uses_projected_capc_component_contract() -> None:
    row = {
        "hypothesis_grid_id": "MASSIVE_Agents_CAPC_language_disjoint_v1",
        "method": "tap_r_capc_projected_majority",
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "chat_template": "qwen3",
        "grammar_engine": "gbnf",
        "chat_parser": "raw",
        "inference_path": "completion",
        "model_artifact": "model.gguf",
        "model_id": "model",
        "thinking_mode": "off",
        "reasoning_budget": 0,
    }
    fields = {item["field"] for item in coefficient_discipline_failures([row])}
    assert {
        "projected_capc_version",
        "source_certificate_version",
        "massive_runner_version",
        "action_risk_threshold",
    } <= fields
    complete = {
        **row,
        "projected_capc_version": "tapbench.capc_projected.v1",
        "source_certificate_version": "tapbench.source_certificate.unicode.v1",
        "massive_runner_version": "tapbench.massive_runner.v1",
        "action_risk_threshold": 0.05,
    }
    assert coefficient_discipline_failures([complete]) == []


def test_discipline_uses_qa_contract_for_supervised_router_bridge() -> None:
    row = {
        "hypothesis_grid_id": "MASSIVE_Agents_CAPC_language_disjoint_v1",
        "method": "tap_r_supervised_router_small_model_slots_qa_all",
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "chat_template": "qwen3",
        "grammar_engine": "llama.cpp_raw_completion_json_schema_gbnf",
        "chat_parser": "bypassed_after_native_template",
        "inference_path": "supervised_router_then_small_model_slots_then_qa",
        "model_artifact": "model.gguf",
        "model_id": "model",
        "thinking_mode": "off",
        "reasoning_budget": 0,
        "qa_evidence_controller_version": "bridge-v1",
        "qa_evidence_system_label": "small-model-router-qa",
        "qa_verifier_version": "qa-v1",
        "qa_verifier_question_version": "questions-v1",
        "qa_verifier_model_id": "qa-model",
        "qa_verifier_model_revision": "qa-revision",
        "qa_verifier_backend": "huggingface_transformers_cpu",
        "qa_verifier_dtype": "float32",
        "qa_verifier_artifact_sha256": "verifier-sha",
        "retriever_version": "router-v1",
        "retriever_model_id": "router-model",
        "retriever_revision": "train-only",
        "retriever_serialization_arm": "supervised_intent_labels",
        "retriever_k": 8,
        "ranking_artifact_sha256": "ranking-sha",
        "source_span_projection_version": "projection-v1",
        "source_span_certificate_version": "certificate-v1",
        "massive_runner_version": "bridge-v1",
        "action_risk_threshold": 0.05,
    }
    assert coefficient_discipline_failures([row]) == []

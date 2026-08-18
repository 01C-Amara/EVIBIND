from __future__ import annotations

from pathlib import Path
from typing import Any

from . import NORMALIZER_VERSION, SCORER_VERSION, VALIDATOR_VERSION
from .io import read_jsonl, write_jsonl
from .ir import parse_and_normalize_prediction
from .slot_errors import slot_errors_for_predictions
from .thinking import prediction_has_thinking_marker
from .validation import validate_action


def _prediction_identity(prediction: dict[str, Any]) -> dict[str, Any]:
    metadata = prediction.get("response_metadata", {}) if isinstance(prediction.get("response_metadata"), dict) else {}
    return {
        "method": prediction.get("method", "unknown"),
        "model_id": prediction.get("model_id", "unknown"),
        "seed": prediction.get("seed", 0),
        "backend": prediction.get("backend", "unknown"),
        "quantization": prediction.get("quantization", "unknown"),
        "chat_template": prediction.get("chat_template", "unknown"),
        "grammar_engine": prediction.get("grammar_engine", "unknown"),
        "model_artifact": prediction.get("model_artifact", "unknown"),
        "thinking_mode": prediction.get("thinking_mode", "not_applicable"),
        "reasoning_budget": prediction.get("reasoning_budget"),
        "thinking_marker_detected": prediction_has_thinking_marker(prediction),
        "finish_reason": metadata.get("finish_reason"),
        **{
            key: prediction[key]
            for key in (
                "chat_parser",
                "inference_path",
                "deployable_resolution_version",
                "evidence_contract_version",
                "contract_solver_version",
                "typed_evidence_program_version",
                "tier_b_verifier_version",
                "tier_b_verifier_artifact_sha256",
                "r2a_model_runner_version",
                "r2b_model_runner_version",
                "r2b_action_schema_version",
                "r2c_model_runner_version",
                "r2d_model_runner_version",
                "effect_first_version",
                "capc_version",
                "capc_runner_version",
                "selective_tapr_version",
                "admission_version",
                "effect_support_version",
                "scope_guard_version",
                "certificate_span_policy_version",
                "extractive_candidate_version",
                "action_risk_threshold",
                "action_risk_score",
                "action_ir_normalized",
                "max_output_tokens",
                "stop_sequence_policy_version",
                "stop_sequences",
            )
            if key in prediction
        },
    }


def score_prediction(case: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    action, format_valid = parse_and_normalize_prediction(prediction, case)
    metrics = validate_action(case, action)
    identity = _prediction_identity(prediction)
    row = {
        "case_id": case["case_id"],
        "hypothesis_grid_id": case["hypothesis_grid_id"],
        "hypothesis": case.get("hypothesis", ""),
        "family": case["family"],
        "task_kind": case["task_kind"],
        "format_valid": bool(format_valid),
        **metrics,
        **identity,
        "scorer_version": SCORER_VERSION,
        "normalizer_version": NORMALIZER_VERSION,
        "validator_version": VALIDATOR_VERSION,
    }
    factors = case.get("factors", {})
    for key in ("N", "q", "d", "e", "sigma", "alpha", "task_kind", "retriever", "prompt_condition", "operator_stratum", "catalog_mutation", "variant"):
        if key in factors:
            row[key] = factors[key]
    return row


def score_predictions(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case_by_id = {case["case_id"]: case for case in cases}
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        case_id = prediction.get("case_id")
        if case_id not in case_by_id:
            raise KeyError(f"prediction references unknown case_id: {case_id}")
        rows.append(score_prediction(case_by_id[case_id], prediction))
    return rows


def score_files(
    cases_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
    *,
    slot_errors_path: str | Path | None = None,
) -> int:
    cases = read_jsonl(cases_path)
    predictions = read_jsonl(predictions_path)
    rows = score_predictions(cases, predictions)
    if slot_errors_path is not None:
        write_jsonl(slot_errors_path, slot_errors_for_predictions(cases, predictions))
    return write_jsonl(output_path, rows)

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Iterable

from . import NORMALIZER_VERSION, SCORER_VERSION, VALIDATOR_VERSION
from .capc import CAPC_VERSION, run_capc_resolution
from .deployable_resolution import DEPLOYABLE_RESOLUTION_VERSION
from .extractive_candidates import EXTRACTIVE_CANDIDATE_VERSION
from .families import FamilySpec
from .io import read_jsonl, write_jsonl, write_yaml
from .r2_model_runner import (
    R2A_CHAT_PARSER,
    R2A_GRAMMAR_ENGINE,
    _request_schema_json,
)
from .r2b import (
    R2B_ACTION_SCHEMA_VERSION,
    _catalog,
    _gold,
    _literal_action,
    _pointer_action,
    _runtime,
    score_r2b_files,
)
from .r2c import _all_values
from .r2c_families import R2C_PILOT_FAMILIES
from .r2d_families import (
    R2D_CONFIRM_FAMILIES,
    R2D_MISSING_REQUESTS,
    R2D_UNSUPPORTED_REQUESTS,
)
from .selective_tapr import (
    ADMISSION_VERSION,
    CERTIFICATE_SPAN_POLICY_VERSION,
    EFFECT_SUPPORT_VERSION,
    SCOPE_GUARD_VERSION,
    SELECTIVE_TAPR_CONDITION,
    SELECTIVE_TAPR_VERSION,
    run_selective_tapr_resolution,
)
from .thinking import prediction_has_thinking_marker
from .tier_b_verifier import FrozenTierBVerifier


R2D_CASE_VERSION = "tapbench.r2d_case.v7"
R2D_RUNNER_VERSION = "tapbench.r2d_model_runner.v7"
R2D_REPORT_VERSION = "tapbench.r2d_report.v7"
R2D_GRID_ID = "R2D_selective_composite_confirmation_v7"
R2D_STOP_SEQUENCE_POLICY_VERSION = "tapbench.r2d_stop_sequences.v1"
R2D_CHAT_TEMPLATE_STOP_SEQUENCES = {
    "gemma4": ("<tool_call|>",),
}
R2D_TASK_KINDS = ("call", "missing_info", "no_tool", "direct_answer")
R2D_MUTATIONS = (
    "unseen_tool_names",
    "unseen_argument_names",
    "enum_alias_shift",
    "near_duplicate_distractor_tools",
    "reordered_schema",
)
R2D_CONDITIONS = (
    "constrained_abstention",
    "tap_r_literal_evidence",
    "tap_r_capc_dual",
    SELECTIVE_TAPR_CONDITION,
)


def _family_values(family: FamilySpec, index: int) -> dict[str, Any]:
    values = _all_values(index)
    return {slot: values[slot] for slot in family.required_slots}


def _implicit_request(
    family: FamilySpec,
    values: dict[str, Any],
    task_kind: str,
) -> str:
    if task_kind == "call":
        return family.request_template.format(**values)
    if task_kind == "missing_info":
        return R2D_MISSING_REQUESTS[family.name].format(**values)
    if task_kind == "no_tool":
        return R2D_UNSUPPORTED_REQUESTS[family.name].format(**values)
    if task_kind == "direct_answer":
        return family.no_tool_request
    raise ValueError(task_kind)


def generate_r2d_cases(*, scope: str = "pilot") -> list[dict[str, Any]]:
    if scope not in {"smoke", "pilot", "confirmation"}:
        raise ValueError("R2-D scope must be smoke, pilot, or confirmation")
    if scope == "confirmation":
        families = R2D_CONFIRM_FAMILIES
        variants = 32
        split = "confirmation"
        offset = 3000
    else:
        families = R2C_PILOT_FAMILIES
        variants = 1 if scope == "smoke" else 16
        split = "engineering_pilot"
        offset = 2000

    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        for variant in range(variants):
            global_index = offset + family_index * 64 + variant
            task_kind = R2D_TASK_KINDS[(variant + family_index) % 4]
            mutation = R2D_MUTATIONS[(variant + 2 * family_index) % len(R2D_MUTATIONS)]
            values = _family_values(family, global_index)
            tools, tool_aliases, argument_aliases = _catalog(
                family,
                200 + family_index,
                values,
                mutation,
            )
            request = _implicit_request(family, values, task_kind)
            derivable = dict(values) if task_kind in {"call", "missing_info"} else {}
            if task_kind == "missing_info":
                derivable.pop(family.missing_slot, None)
            rows.append(
                {
                    "schema_version": R2D_CASE_VERSION,
                    "case_id": f"r2d_{scope}_{family.name}_{variant:02d}",
                    "hypothesis_grid_id": R2D_GRID_ID,
                    "hypothesis": "R2D",
                    "split": split,
                    "family": family.name,
                    "task_kind": task_kind,
                    "factors": {
                        "task_kind": task_kind,
                        "catalog_mutation": mutation,
                        "variant": variant,
                        "cue_regime": "implicit_unlabeled",
                    },
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Return one Action IR outcome. Do not invent "
                                "authorization or unsupported argument values."
                            ),
                        },
                        {"role": "user", "content": request},
                    ],
                    "tools": tools,
                    "tool_aliases": tool_aliases,
                    "argument_aliases": argument_aliases,
                    "dialogue_state": {},
                    "reference_context": {
                        "reference_date": "2026-07-17",
                        "timezone": "Europe/London",
                        "action_risk_budget": 0.05,
                    },
                    "gold_action": _gold(family, values, task_kind),
                    "derivable_values": derivable,
                    "r2d_oracle": {
                        "family": family.name,
                        "mutation": mutation,
                        "variant": variant,
                    },
                    "metadata": {
                        "backend_namespace": "llama_cpp_q4km_r2d",
                        "coefficient_backend": "llama.cpp",
                        "quantization": "Q4_K_M",
                        "thinking_mode": "off",
                        "reasoning_budget": 0,
                        "runtime_allowed_fields": [
                            "messages",
                            "tools",
                            "tool_aliases",
                            "argument_aliases",
                            "dialogue_state",
                            "reference_context",
                        ],
                        "offline_only_fields": [
                            "gold_action",
                            "derivable_values",
                            "r2d_oracle",
                            "task_kind",
                        ],
                    },
                }
            )
    return rows


def write_r2d_cases(output: str | Path, *, scope: str) -> int:
    return write_jsonl(output, generate_r2d_cases(scope=scope))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_r2d_model_conditions(
    cases_path: str | Path,
    output_path: str | Path,
    timings_path: str | Path,
    manifest_path: str | Path,
    *,
    endpoint: str,
    model_id: str,
    model_key: str,
    model_artifact: str,
    chat_template: str,
    tier_b_verifier_path: str | Path,
    conditions: Iterable[str] = R2D_CONDITIONS,
    seeds: Iterable[int] = (1,),
    max_tokens: int = 768,
    max_generations: int | None = None,
    quantization: str = "Q4_K_M",
    protocol_path: str | Path = "configs/r2d_selective_preregistration_v7.yaml",
    study_id: str = R2D_GRID_ID,
    runner_version: str = R2D_RUNNER_VERSION,
    manifest_schema_version: str = "tapbench.r2d_run_manifest.v7",
    stop_sequences: Iterable[str] | None = None,
    stop_sequence_policy_version: str = R2D_STOP_SEQUENCE_POLICY_VERSION,
) -> dict[str, Any]:
    verifier = FrozenTierBVerifier.load(tier_b_verifier_path)
    selected_conditions = tuple(str(value) for value in conditions)
    unknown = set(selected_conditions) - set(R2D_CONDITIONS)
    if unknown:
        raise ValueError(f"unknown R2-D conditions: {sorted(unknown)}")
    selected_seeds = tuple(int(value) for value in seeds)
    selected_stop_sequences = (
        tuple(str(value) for value in stop_sequences)
        if stop_sequences is not None
        else R2D_CHAT_TEMPLATE_STOP_SEQUENCES.get(chat_template, ())
    )
    request_fn = partial(
        _request_schema_json,
        stop_sequences=selected_stop_sequences,
    )
    jobs = [
        (case, condition, seed)
        for case in read_jsonl(cases_path)
        for condition in selected_conditions
        for seed in selected_seeds
    ]
    if max_generations is not None:
        jobs = jobs[:max_generations]

    predictions: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    for case, condition, seed in jobs:
        started = time.perf_counter()
        error = None
        try:
            runtime = _runtime(case)
            if condition == SELECTIVE_TAPR_CONDITION:
                action, metadata = run_selective_tapr_resolution(
                    messages=runtime["messages"],
                    tools=runtime["tools"],
                    endpoint=endpoint,
                    max_tokens=max_tokens,
                    seed=seed,
                    request_fn=request_fn,
                )
            elif condition == "tap_r_capc_dual":
                action, metadata = run_capc_resolution(
                    messages=runtime["messages"],
                    tools=runtime["tools"],
                    endpoint=endpoint,
                    condition=condition,
                    max_tokens=max_tokens,
                    seed=seed,
                    request_fn=request_fn,
                )
            elif condition == "tap_r_literal_evidence":
                action, metadata = _pointer_action(
                    runtime,
                    condition,
                    endpoint,
                    verifier,
                    max_tokens=max_tokens,
                    seed=seed,
                    request_fn=request_fn,
                )
            else:
                action, metadata = _literal_action(
                    runtime,
                    condition,
                    endpoint,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    seed=seed,
                    request_fn=request_fn,
                )
                metadata["generation_calls"] = 1
        except Exception as exc:
            error = str(exc)
            action = {"runner_error": error}
            metadata = {
                "finish_reason": "runner_error",
                "error_type": exc.__class__.__name__,
                "error_message": error,
                "generation_calls": 0,
                "action_risk_score": 1.0,
            }

        elapsed = time.perf_counter() - started
        terminal = action.get("mode") if isinstance(action, dict) else "runner_error"
        resolution = (
            metadata.get("resolution")
            if isinstance(metadata.get("resolution"), dict)
            else {
                "terminal_state": terminal,
                "elapsed_seconds": elapsed,
                "generation_calls": metadata.get("generation_calls", 0),
            }
        )
        row = {
            "case_id": case["case_id"],
            "method": condition,
            "model_id": model_id,
            "seed": seed,
            "prediction": action,
            "action_ir_normalized": True,
            "response_metadata": metadata,
            "resolution": resolution,
            "runner_error": error,
            "backend": "llama.cpp",
            "quantization": quantization,
            "chat_template": chat_template,
            "grammar_engine": R2A_GRAMMAR_ENGINE,
            "chat_parser": R2A_CHAT_PARSER,
            "inference_path": "apply_template_then_raw_completion",
            "model_artifact": model_artifact,
            "thinking_mode": "off",
            "reasoning_budget": 0,
            "r2b_action_schema_version": R2B_ACTION_SCHEMA_VERSION,
            "r2d_model_runner_version": runner_version,
            "stop_sequence_policy_version": stop_sequence_policy_version,
            "stop_sequences": list(selected_stop_sequences),
            "selective_tapr_version": (
                SELECTIVE_TAPR_VERSION
                if condition == SELECTIVE_TAPR_CONDITION
                else None
            ),
            "admission_version": (
                ADMISSION_VERSION
                if condition == SELECTIVE_TAPR_CONDITION
                else None
            ),
            "effect_support_version": (
                EFFECT_SUPPORT_VERSION
                if condition == SELECTIVE_TAPR_CONDITION
                else None
            ),
            "scope_guard_version": (
                SCOPE_GUARD_VERSION
                if condition == SELECTIVE_TAPR_CONDITION
                else None
            ),
            "certificate_span_policy_version": (
                CERTIFICATE_SPAN_POLICY_VERSION
                if condition == SELECTIVE_TAPR_CONDITION
                else None
            ),
            "capc_version": CAPC_VERSION if condition == "tap_r_capc_dual" else None,
            "extractive_candidate_version": (
                EXTRACTIVE_CANDIDATE_VERSION
                if condition in {"tap_r_capc_dual", SELECTIVE_TAPR_CONDITION}
                else None
            ),
            "capc_runner_version": (
                R2D_RUNNER_VERSION if condition == "tap_r_capc_dual" else None
            ),
            "deployable_resolution_version": (
                DEPLOYABLE_RESOLUTION_VERSION
                if condition == "tap_r_literal_evidence"
                else None
            ),
            "action_risk_threshold": metadata.get("action_risk_threshold"),
            "action_risk_score": metadata.get("action_risk_score"),
            "max_output_tokens": max_tokens,
            "contract_solver_version": resolution.get("schema_version"),
            "evidence_contract_version": resolution.get("evidence_contract_version"),
        }
        row["thinking_marker_detected"] = prediction_has_thinking_marker(row)
        predictions.append(row)
        timings.append(
            {
                "case_id": case["case_id"],
                "hypothesis_grid_id": study_id,
                "catalog_mutation": case["factors"]["catalog_mutation"],
                "task_kind": case["task_kind"],
                "method": condition,
                "model_key": model_key,
                "model_id": model_id,
                "seed": seed,
                "backend": "llama.cpp",
                "quantization": quantization,
                "elapsed_seconds": elapsed,
                "generation_calls": metadata.get("generation_calls", 0),
                "runner_error": error,
                "thinking_mode": "off",
                "thinking_marker_detected": row["thinking_marker_detected"],
                "stop_sequence_policy_version": stop_sequence_policy_version,
                "stop_sequences": list(selected_stop_sequences),
                "admission_agreement": metadata.get("admission_agreement"),
                "tool_agreement": metadata.get("tool_agreement"),
                "effect_support_agreement": metadata.get(
                    "effect_support_agreement"
                ),
                "scope_guard_blocked": metadata.get(
                    "scope_guard", {}
                ).get("blocked"),
                "cross_slot_span_conflict_detected": any(
                    certification.get("status")
                    == "cross_slot_source_span_conflict"
                    for attempt in metadata.get("proposal_attempts", [])
                    for certification in attempt.get("certification", [])
                ),
                "semantic_envelope_violation_detected": any(
                    certification.get("status")
                    == "semantic_envelope_violation"
                    for attempt in metadata.get("proposal_attempts", [])
                    for certification in attempt.get("certification", [])
                ),
                "accepted_evidence_tier": metadata.get("accepted_evidence_tier"),
                "accepted_proposal_index": metadata.get("accepted_proposal_index"),
                **{
                    key: metadata.get(key)
                    for key in (
                        "finish_reason",
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "generated_tokens_per_second",
                        "context_truncated",
                    )
                },
            }
        )

    write_jsonl(output_path, predictions)
    write_jsonl(timings_path, timings)
    manifest = {
        "schema_version": manifest_schema_version,
        "study_id": study_id,
        "runner_version": runner_version,
        "stop_sequence_policy_version": stop_sequence_policy_version,
        "stop_sequences": list(selected_stop_sequences),
        "selective_tapr_version": SELECTIVE_TAPR_VERSION,
        "admission_version": ADMISSION_VERSION,
        "effect_support_version": EFFECT_SUPPORT_VERSION,
        "scope_guard_version": SCOPE_GUARD_VERSION,
        "certificate_span_policy_version": CERTIFICATE_SPAN_POLICY_VERSION,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path),
        "predictions_path": str(output_path),
        "timings_path": str(timings_path),
        "protocol_path": str(protocol_path),
        "model_key": model_key,
        "model_id": model_id,
        "model_artifact": model_artifact,
        "backend": "llama.cpp",
        "quantization": quantization,
        "chat_template": chat_template,
        "grammar_engine": R2A_GRAMMAR_ENGINE,
        "thinking_mode": "off",
        "reasoning_budget": 0,
        "context_tokens": 8192,
        "max_output_tokens": max_tokens,
        "conditions": list(selected_conditions),
        "seeds": list(selected_seeds),
        "case_count": len(read_jsonl(cases_path)),
        "generation_count": len(predictions),
        "actual_model_calls": sum(
            int(row["response_metadata"].get("generation_calls", 0))
            for row in predictions
        ),
        "runner_errors": sum(row["runner_error"] is not None for row in predictions),
        "thinking_markers": sum(
            bool(row.get("thinking_marker_detected")) for row in predictions
        ),
        "length_stops": sum(
            row.get("response_metadata", {}).get("finish_reason") == "length"
            for row in predictions
        ),
        "source_sha256": {
            "cases": _sha256(cases_path),
            "protocol": _sha256(protocol_path),
            "model_artifact": _sha256(model_artifact),
        },
        "tier_b_verifier_version": verifier.version,
        "tier_b_verifier_artifact_sha256": verifier.artifact_sha256,
    }
    write_yaml(manifest_path, manifest)
    return manifest


def score_r2d_files(
    cases_path: str | Path,
    predictions_path: str | Path,
    scores_path: str | Path,
    slot_errors_path: str | Path,
    report_path: str | Path,
    *,
    expected_model_count: int | None = None,
    expected_condition_count: int | None = None,
    expected_seed_count: int | None = None,
    study_id: str = R2D_GRID_ID,
    report_version: str = R2D_REPORT_VERSION,
) -> dict[str, Any]:
    report = score_r2b_files(
        cases_path,
        predictions_path,
        scores_path,
        slot_errors_path,
        report_path,
        expected_model_count=expected_model_count,
        expected_condition_count=expected_condition_count,
        expected_seed_count=expected_seed_count,
    )
    predictions = {
        (
            row["case_id"],
            row.get("method"),
            row.get("model_id"),
            row.get("seed"),
        ): row
        for row in read_jsonl(predictions_path)
    }
    scores = read_jsonl(scores_path)
    for row in scores:
        prediction = predictions[
            (
                row["case_id"],
                row.get("method"),
                row.get("model_id"),
                row.get("seed"),
            )
        ]
        metadata = prediction.get("response_metadata", {})
        row["r2d_report_version"] = R2D_REPORT_VERSION
        row["admission_agreement"] = metadata.get("admission_agreement")
        row["tool_agreement"] = metadata.get("tool_agreement")
        row["effect_support_agreement"] = metadata.get(
            "effect_support_agreement"
        )
        row["scope_guard_blocked"] = metadata.get(
            "scope_guard", {}
        ).get("blocked")
        row["cross_slot_span_conflict_detected"] = any(
            certification.get("status")
            == "cross_slot_source_span_conflict"
            for attempt in metadata.get("proposal_attempts", [])
            for certification in attempt.get("certification", [])
        )
        row["semantic_envelope_violation_detected"] = any(
            certification.get("status") == "semantic_envelope_violation"
            for attempt in metadata.get("proposal_attempts", [])
            for certification in attempt.get("certification", [])
        )
        row["accepted_evidence_tier"] = metadata.get("accepted_evidence_tier")
        row["accepted_proposal_index"] = metadata.get("accepted_proposal_index")
        row["clarification_source"] = metadata.get("clarification_source")
        row["model_literal_entered_action"] = metadata.get(
            "model_literal_entered_action"
        )
        if row.get("method") == SELECTIVE_TAPR_CONDITION:
            row["selective_tapr_version"] = (
                prediction.get("selective_tapr_version")
                or metadata.get("selective_tapr_version")
                or SELECTIVE_TAPR_VERSION
            )
            row["admission_version"] = (
                prediction.get("admission_version")
                or metadata.get("admission_version")
                or ADMISSION_VERSION
            )
            row["effect_support_version"] = (
                prediction.get("effect_support_version")
                or metadata.get("effect_support_version")
                or EFFECT_SUPPORT_VERSION
            )
            row["scope_guard_version"] = (
                prediction.get("scope_guard_version")
                or metadata.get("scope_guard_version")
                or SCOPE_GUARD_VERSION
            )
            row["certificate_span_policy_version"] = (
                prediction.get("certificate_span_policy_version")
                or metadata.get("certificate_span_policy_version")
                or CERTIFICATE_SPAN_POLICY_VERSION
            )
            row["extractive_candidate_version"] = (
                prediction.get("extractive_candidate_version")
                or metadata.get("extractive_candidate_version")
                or EXTRACTIVE_CANDIDATE_VERSION
            )
            row["r2d_model_runner_version"] = (
                prediction.get("r2d_model_runner_version")
                or prediction.get("runner_version")
                or R2D_RUNNER_VERSION
            )
            row["action_risk_threshold"] = (
                prediction.get("action_risk_threshold")
                if prediction.get("action_risk_threshold") is not None
                else metadata.get("action_risk_threshold")
            )
    write_jsonl(scores_path, scores)
    report["schema_version"] = report_version
    report["study_id"] = study_id
    report["scorer_version"] = SCORER_VERSION
    report["normalizer_version"] = NORMALIZER_VERSION
    report["validator_version"] = VALIDATOR_VERSION
    Path(report_path).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report

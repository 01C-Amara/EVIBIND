from __future__ import annotations

import hashlib
import time
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Iterable

from .extractive_candidates import EXTRACTIVE_CANDIDATE_VERSION
from .io import read_jsonl, write_jsonl, write_yaml
from .r2_model_runner import (
    R2A_CHAT_PARSER,
    R2A_GRAMMAR_ENGINE,
    _request_schema_json,
)
from .r2b import R2B_ACTION_SCHEMA_VERSION, _literal_action, _runtime
from .r2d import (
    R2D_CHAT_TEMPLATE_STOP_SEQUENCES,
    R2D_STOP_SEQUENCE_POLICY_VERSION,
)
from .selective_tapr import (
    ADMISSION_VERSION,
    CERTIFICATE_SPAN_POLICY_VERSION,
    EFFECT_SUPPORT_VERSION,
    SCOPE_GUARD_VERSION,
    SELECTIVE_TAPR_VERSION,
    run_selective_tapr_resolution,
)
from .selective_tapr_closure import (
    ONLINE_SEMANTIC_CLOSURE_VERSION,
    apply_online_semantic_closure,
)
from .semantic_closure import SEMANTIC_CLOSURE_VERSION
from .thinking import prediction_has_thinking_marker


LARGE_MODEL_CLOSURE_RUNNER_VERSION = "tapbench.large_model_closure_runner.v1"
BASELINE = "constrained_abstention"
ORIGINAL = "tap_r_selective_full"
CLOSURE = "tap_r_selective_semantic_closure"
CONDITIONS = (BASELINE, ORIGINAL, CLOSURE)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolution(
    action: dict[str, Any],
    metadata: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    existing = metadata.get("resolution")
    if isinstance(existing, dict):
        return deepcopy(existing)
    return {
        "terminal_state": action.get("mode", "runner_error"),
        "elapsed_seconds": elapsed,
        "generation_calls": metadata.get("generation_calls", 0),
    }


def _prediction_row(
    *,
    case: dict[str, Any],
    method: str,
    model_id: str,
    model_artifact: str,
    quantization: str,
    chat_template: str,
    seed: int,
    max_tokens: int,
    action: dict[str, Any],
    metadata: dict[str, Any],
    elapsed: float,
    error: str | None,
    trace_id: str,
    model_calls_charged: int,
    stop_sequences: tuple[str, ...],
) -> dict[str, Any]:
    selective = method in {ORIGINAL, CLOSURE}
    resolution = _resolution(action, metadata, elapsed)
    row = {
        "case_id": case["case_id"],
        "method": method,
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
        "r2d_model_runner_version": LARGE_MODEL_CLOSURE_RUNNER_VERSION,
        "stop_sequence_policy_version": R2D_STOP_SEQUENCE_POLICY_VERSION,
        "stop_sequences": list(stop_sequences),
        "selective_tapr_version": SELECTIVE_TAPR_VERSION if selective else None,
        "admission_version": ADMISSION_VERSION if selective else None,
        "effect_support_version": EFFECT_SUPPORT_VERSION if selective else None,
        "scope_guard_version": SCOPE_GUARD_VERSION if selective else None,
        "certificate_span_policy_version": (
            CERTIFICATE_SPAN_POLICY_VERSION if selective else None
        ),
        "extractive_candidate_version": (
            EXTRACTIVE_CANDIDATE_VERSION if selective else None
        ),
        "online_semantic_closure_version": (
            ONLINE_SEMANTIC_CLOSURE_VERSION if method == CLOSURE else None
        ),
        "semantic_closure_version": (
            SEMANTIC_CLOSURE_VERSION if method == CLOSURE else None
        ),
        "shared_generation_trace_id": trace_id,
        "model_calls_charged": model_calls_charged,
        "action_risk_threshold": metadata.get("action_risk_threshold"),
        "action_risk_score": metadata.get("action_risk_score"),
        "max_output_tokens": max_tokens,
        "contract_solver_version": resolution.get("schema_version"),
        "evidence_contract_version": resolution.get(
            "evidence_contract_version"
        ),
    }
    row["thinking_marker_detected"] = prediction_has_thinking_marker(row)
    return row


def _timing_row(
    row: dict[str, Any],
    case: dict[str, Any],
    *,
    model_key: str,
    study_id: str,
    elapsed: float,
) -> dict[str, Any]:
    metadata = row["response_metadata"]
    return {
        "case_id": case["case_id"],
        "hypothesis_grid_id": study_id,
        "family": case.get("family"),
        "catalog_mutation": case.get("factors", {}).get(
            "catalog_mutation"
        ),
        "task_kind": case.get("task_kind"),
        "method": row["method"],
        "model_key": model_key,
        "model_id": row["model_id"],
        "seed": row["seed"],
        "backend": row["backend"],
        "quantization": row["quantization"],
        "elapsed_seconds": elapsed,
        "generation_calls": metadata.get("generation_calls", 0),
        "model_calls_charged": row["model_calls_charged"],
        "shared_generation_trace_id": row["shared_generation_trace_id"],
        "runner_error": row["runner_error"],
        "thinking_mode": row["thinking_mode"],
        "thinking_marker_detected": row["thinking_marker_detected"],
        "semantic_closure_status": metadata.get(
            "semantic_closure", {}
        ).get("status"),
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


def run_large_model_closure(
    cases_path: str | Path,
    output_path: str | Path,
    timings_path: str | Path,
    manifest_path: str | Path,
    *,
    endpoint: str,
    model_id: str,
    model_key: str,
    model_artifact: str,
    quantization: str,
    chat_template: str,
    protocol_path: str | Path,
    study_id: str,
    context_tokens: int,
    max_tokens: int = 768,
    seeds: Iterable[int] = (1,),
    max_cases: int | None = None,
) -> dict[str, Any]:
    cases = read_jsonl(cases_path)
    if max_cases is not None:
        cases = cases[:max_cases]
    selected_seeds = tuple(int(value) for value in seeds)
    stop_sequences = R2D_CHAT_TEMPLATE_STOP_SEQUENCES.get(
        chat_template, ()
    )
    request_fn = partial(
        _request_schema_json,
        stop_sequences=stop_sequences,
    )
    predictions: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    charged_calls = 0

    for case in cases:
        runtime = _runtime(case)
        for seed in selected_seeds:
            baseline_started = time.perf_counter()
            baseline_error = None
            try:
                baseline_action, baseline_metadata = _literal_action(
                    runtime,
                    BASELINE,
                    endpoint,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    seed=seed,
                    request_fn=request_fn,
                )
                baseline_metadata["generation_calls"] = 1
            except Exception as exc:
                baseline_error = str(exc)
                baseline_action = {"runner_error": baseline_error}
                baseline_metadata = {
                    "finish_reason": "runner_error",
                    "error_type": exc.__class__.__name__,
                    "error_message": baseline_error,
                    "generation_calls": 0,
                    "action_risk_score": 1.0,
                }
            baseline_elapsed = time.perf_counter() - baseline_started
            baseline_calls = int(
                baseline_metadata.get("generation_calls", 0)
            )
            charged_calls += baseline_calls
            baseline_trace = f"{case['case_id']}:{seed}:baseline"
            baseline_row = _prediction_row(
                case=case,
                method=BASELINE,
                model_id=model_id,
                model_artifact=model_artifact,
                quantization=quantization,
                chat_template=chat_template,
                seed=seed,
                max_tokens=max_tokens,
                action=baseline_action,
                metadata=baseline_metadata,
                elapsed=baseline_elapsed,
                error=baseline_error,
                trace_id=baseline_trace,
                model_calls_charged=baseline_calls,
                stop_sequences=stop_sequences,
            )
            predictions.append(baseline_row)
            timings.append(
                _timing_row(
                    baseline_row,
                    case,
                    model_key=model_key,
                    study_id=study_id,
                    elapsed=baseline_elapsed,
                )
            )

            selective_started = time.perf_counter()
            selective_error = None
            try:
                selective_action, selective_metadata = (
                    run_selective_tapr_resolution(
                        messages=runtime["messages"],
                        tools=runtime["tools"],
                        endpoint=endpoint,
                        max_tokens=max_tokens,
                        seed=seed,
                        request_fn=request_fn,
                    )
                )
            except Exception as exc:
                selective_error = str(exc)
                selective_action = {"runner_error": selective_error}
                selective_metadata = {
                    "finish_reason": "runner_error",
                    "error_type": exc.__class__.__name__,
                    "error_message": selective_error,
                    "generation_calls": 0,
                    "action_risk_score": 1.0,
                }
            selective_elapsed = time.perf_counter() - selective_started
            selective_calls = int(
                selective_metadata.get("generation_calls", 0)
            )
            charged_calls += selective_calls
            selective_trace = f"{case['case_id']}:{seed}:selective"
            original_row = _prediction_row(
                case=case,
                method=ORIGINAL,
                model_id=model_id,
                model_artifact=model_artifact,
                quantization=quantization,
                chat_template=chat_template,
                seed=seed,
                max_tokens=max_tokens,
                action=selective_action,
                metadata=selective_metadata,
                elapsed=selective_elapsed,
                error=selective_error,
                trace_id=selective_trace,
                model_calls_charged=selective_calls,
                stop_sequences=stop_sequences,
            )
            predictions.append(original_row)
            timings.append(
                _timing_row(
                    original_row,
                    case,
                    model_key=model_key,
                    study_id=study_id,
                    elapsed=selective_elapsed,
                )
            )

            closure_started = time.perf_counter()
            if selective_error is None:
                closure_action, closure_metadata = (
                    apply_online_semantic_closure(
                        selective_action,
                        selective_metadata,
                        messages=runtime["messages"],
                        tools=runtime["tools"],
                    )
                )
                closure_error = None
            else:
                closure_action = deepcopy(selective_action)
                closure_metadata = deepcopy(selective_metadata)
                closure_error = selective_error
            closure_overhead = time.perf_counter() - closure_started
            closure_elapsed = selective_elapsed + closure_overhead
            closure_metadata["semantic_closure_overhead_seconds"] = (
                closure_overhead
            )
            closure_row = _prediction_row(
                case=case,
                method=CLOSURE,
                model_id=model_id,
                model_artifact=model_artifact,
                quantization=quantization,
                chat_template=chat_template,
                seed=seed,
                max_tokens=max_tokens,
                action=closure_action,
                metadata=closure_metadata,
                elapsed=closure_elapsed,
                error=closure_error,
                trace_id=selective_trace,
                model_calls_charged=0,
                stop_sequences=stop_sequences,
            )
            predictions.append(closure_row)
            timings.append(
                _timing_row(
                    closure_row,
                    case,
                    model_key=model_key,
                    study_id=study_id,
                    elapsed=closure_elapsed,
                )
            )

    write_jsonl(output_path, predictions)
    write_jsonl(timings_path, timings)
    manifest = {
        "schema_version": "tapbench.large_model_closure_manifest.v1",
        "study_id": study_id,
        "runner_version": LARGE_MODEL_CLOSURE_RUNNER_VERSION,
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
        "inference_path": "apply_template_then_raw_completion",
        "thinking_mode": "off",
        "reasoning_budget": 0,
        "context_tokens": context_tokens,
        "max_output_tokens": max_tokens,
        "conditions": list(CONDITIONS),
        "seeds": list(selected_seeds),
        "case_count": len(cases),
        "prediction_count": len(predictions),
        "actual_model_calls": charged_calls,
        "shared_trace_policy": (
            "original TAP-R and semantic closure share one model trace; "
            "closure adds deterministic postprocessing only"
        ),
        "runner_errors": sum(
            row["runner_error"] is not None for row in predictions
        ),
        "thinking_markers": sum(
            bool(row["thinking_marker_detected"]) for row in predictions
        ),
        "context_truncations": sum(
            bool(row["response_metadata"].get("context_truncated"))
            for row in predictions
        ),
        "length_stops": sum(
            row["response_metadata"].get("finish_reason") == "length"
            for row in predictions
        ),
        "versions": {
            "selective_tapr": SELECTIVE_TAPR_VERSION,
            "online_semantic_closure": ONLINE_SEMANTIC_CLOSURE_VERSION,
            "semantic_closure": SEMANTIC_CLOSURE_VERSION,
        },
        "source_sha256": {
            "cases": _sha256(cases_path),
            "protocol": _sha256(protocol_path),
            "model_artifact": _sha256(model_artifact),
        },
    }
    write_yaml(manifest_path, manifest)
    return manifest

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .eflrx import (
    CONTEXT_TOKENS,
    EFLRX_CONDITIONS,
    EFLRX_VERSION,
    ContextOverflowError,
    RequestFn,
    preflight_schema_request,
    run_eflrx_resolution,
)
from .eflrx_baselines import (
    RAW_BASELINE_CONDITIONS,
    RAW_BASELINE_VERSION,
    run_raw_baseline,
)
from .extractive_candidates import EXTRACTIVE_CANDIDATE_VERSION
from .io import read_jsonl, write_jsonl, write_yaml
from .r2_model_runner import R2A_CHAT_PARSER, R2A_GRAMMAR_ENGINE
from .thinking import prediction_has_thinking_marker


EFLRX_RUNNER_VERSION = "tapbench.eflrx_runner.v2"
EFLRX_STUDY_CONDITIONS = (*RAW_BASELINE_CONDITIONS, *EFLRX_CONDITIONS)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _category(case: dict[str, Any]) -> str:
    metadata = case.get("metadata", {})
    factors = case.get("factors", {})
    return str(
        metadata.get("bfcl_category")
        or factors.get("bfcl_category")
        or ""
    )


def _select_cases(
    cases: list[dict[str, Any]],
    *,
    categories: Iterable[str] | None,
    limit_per_category: int | None,
) -> list[dict[str, Any]]:
    allowed = (
        {str(value).strip() for value in categories if str(value).strip()}
        if categories is not None
        else None
    )
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for case in cases:
        category = _category(case)
        if allowed is not None and category not in allowed:
            continue
        if (
            limit_per_category is not None
            and counts.get(category, 0) >= limit_per_category
        ):
            continue
        selected.append(case)
        counts[category] = counts.get(category, 0) + 1
    return selected


def run_eflrx_cases(
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
    conditions: Iterable[str] = EFLRX_STUDY_CONDITIONS,
    seeds: Iterable[int] = (1,),
    max_tokens: int = 384,
    categories: Iterable[str] | None = None,
    limit_per_category: int | None = None,
    request_fn: RequestFn = preflight_schema_request,
    preregistration_path: str | Path | None = None,
    amendment_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    selected_conditions = tuple(str(value) for value in conditions)
    unknown = set(selected_conditions) - set(EFLRX_STUDY_CONDITIONS)
    if unknown:
        raise ValueError(f"unknown EFLR-X conditions: {sorted(unknown)}")
    selected_seeds = tuple(int(value) for value in seeds)
    cases = _select_cases(
        read_jsonl(cases_path),
        categories=categories,
        limit_per_category=limit_per_category,
    )
    jobs = [
        (case, condition, seed)
        for case in cases
        for condition in selected_conditions
        for seed in selected_seeds
    ]

    predictions: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    for case, condition, seed in jobs:
        started = time.perf_counter()
        error = None
        try:
            if condition in EFLRX_CONDITIONS:
                action, metadata = run_eflrx_resolution(
                    messages=list(case.get("messages", [])),
                    tools=list(case.get("tools", [])),
                    endpoint=endpoint,
                    condition=condition,
                    max_tokens=max_tokens,
                    seed=seed,
                    request_fn=request_fn,
                )
            else:
                action, metadata = run_raw_baseline(
                    case,
                    endpoint=endpoint,
                    condition=condition,
                    max_tokens=max_tokens,
                    seed=seed,
                    request_fn=request_fn,
                )
        except Exception as exc:
            error = str(exc)
            overflow = isinstance(exc, ContextOverflowError)
            action = {"runner_error": error}
            metadata = {
                "finish_reason": (
                    "context_overflow" if overflow else "runner_error"
                ),
                "error_type": exc.__class__.__name__,
                "error_message": error,
                "generation_calls": 0,
                "action_risk_score": 1.0,
                "context_overflow": overflow,
            }
        elapsed = time.perf_counter() - started
        row = {
            "case_id": case["case_id"],
            "method": condition,
            "model_id": model_id,
            "seed": seed,
            "prediction": action,
            "action_ir_normalized": True,
            "response_metadata": metadata,
            "resolution": {
                "terminal_state": (
                    action.get("mode")
                    if isinstance(action, dict)
                    else "runner_error"
                ),
                "materialized_action": action,
                "elapsed_seconds": elapsed,
            },
            "runner_error": error,
            "backend": "llama.cpp",
            "quantization": "Q4_K_M",
            "chat_template": chat_template,
            "grammar_engine": R2A_GRAMMAR_ENGINE,
            "chat_parser": R2A_CHAT_PARSER,
            "inference_path": "apply_template_then_raw_completion",
            "model_artifact": model_artifact,
            "thinking_mode": "off",
            "reasoning_budget": 0,
            "eflrx_version": (
                EFLRX_VERSION if condition in EFLRX_CONDITIONS else None
            ),
            "extractive_candidate_version": (
                EXTRACTIVE_CANDIDATE_VERSION
                if condition in EFLRX_CONDITIONS
                else None
            ),
            "raw_baseline_version": (
                RAW_BASELINE_VERSION
                if condition in RAW_BASELINE_CONDITIONS
                else None
            ),
            "eflrx_runner_version": EFLRX_RUNNER_VERSION,
            "action_risk_threshold": metadata.get("action_risk_threshold"),
            "action_risk_score": metadata.get("action_risk_score"),
            "max_output_tokens": max_tokens,
        }
        row["thinking_marker_detected"] = prediction_has_thinking_marker(row)
        predictions.append(row)
        timings.append(
            {
                "case_id": case["case_id"],
                "bfcl_category": _category(case),
                "task_kind": case.get("task_kind"),
                "method": condition,
                "model_key": model_key,
                "model_id": model_id,
                "seed": seed,
                "backend": "llama.cpp",
                "quantization": "Q4_K_M",
                "elapsed_seconds": elapsed,
                "generation_calls": metadata.get("generation_calls", 0),
                "runner_error": error,
                "thinking_mode": "off",
                "thinking_marker_detected": row[
                    "thinking_marker_detected"
                ],
                "action_risk_score": metadata.get("action_risk_score"),
                "tool_agreement": metadata.get("tool_agreement"),
                "pointer_agreement": metadata.get("pointer_agreement"),
                "risk_gate_passed": metadata.get("risk_gate_passed"),
                **{
                    key: metadata.get(key)
                    for key in (
                        "finish_reason",
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "generated_tokens_per_second",
                        "context_truncated",
                        "rendered_input_tokens_max",
                        "preflight_prompt_token_delta_max_abs",
                        "context_headroom_tokens_min",
                        "context_overflow",
                    )
                },
            }
        )

    write_jsonl(output_path, predictions)
    write_jsonl(timings_path, timings)
    source_hashes = {
        "cases": _sha256(cases_path),
        "model_artifact": _sha256(model_artifact),
    }
    if preregistration_path is not None:
        source_hashes["preregistration"] = _sha256(preregistration_path)
    amendment_hashes = {
        str(path): _sha256(path)
        for path in amendment_paths
    }
    manifest = {
        "schema_version": "tapbench.eflrx_run_manifest.v1",
        "runner_version": EFLRX_RUNNER_VERSION,
        "eflrx_version": EFLRX_VERSION,
        "extractive_candidate_version": EXTRACTIVE_CANDIDATE_VERSION,
        "raw_baseline_version": RAW_BASELINE_VERSION,
        "action_ir_normalized": True,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path),
        "predictions_path": str(output_path),
        "timings_path": str(timings_path),
        "model_key": model_key,
        "model_id": model_id,
        "model_artifact": model_artifact,
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "chat_template": chat_template,
        "grammar_engine": R2A_GRAMMAR_ENGINE,
        "chat_parser": R2A_CHAT_PARSER,
        "thinking_mode": "off",
        "reasoning_budget": 0,
        "context_tokens": CONTEXT_TOKENS,
        "max_output_tokens": max_tokens,
        "categories": sorted(
            {_category(case) for case in cases}
        ),
        "limit_per_category": limit_per_category,
        "conditions": list(selected_conditions),
        "seeds": list(selected_seeds),
        "case_count": len(cases),
        "generation_count": len(predictions),
        "actual_model_calls": sum(
            int(row["response_metadata"].get("generation_calls", 0))
            for row in predictions
        ),
        "runner_errors": sum(
            row["runner_error"] is not None for row in predictions
        ),
        "source_sha256": source_hashes,
        "amendment_sha256": amendment_hashes,
    }
    write_yaml(manifest_path, manifest)
    return manifest

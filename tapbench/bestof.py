from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl, write_yaml
from .runner import llama_server_action
from .tapr import contract_validator_error
from .thinking import prediction_has_thinking_marker

BEST_OF_VERSION = "tapbench.best_of_n.v1"


def candidate_rank(case: dict[str, Any], action: Any) -> tuple[int, int, int, int]:
    report = contract_validator_error(case, action)
    errors = report["errors"]
    unsafe_required = sum(
        error["error_class"] in {"unsupported_required_value", "missing_required_slot_no_evidence"}
        for error in errors
    )
    unsupported = sum(error.get("evidence_status") in {"unsupported", "contradicted"} for error in errors)
    return (
        int(report["contract_valid"]),
        -unsafe_required,
        -len(errors),
        -unsupported,
    )


def select_candidate(case: dict[str, Any], actions: list[Any]) -> tuple[int, list[dict[str, Any]]]:
    if not actions:
        raise ValueError("best-of-n requires at least one candidate")
    diagnostics = []
    for index, action in enumerate(actions):
        report = contract_validator_error(case, action)
        diagnostics.append(
            {
                "candidate_index": index,
                "rank": list(candidate_rank(case, action)),
                "contract_valid": report["contract_valid"],
                "error_classes": [error["error_class"] for error in report["errors"]],
            }
        )
    selected = max(range(len(actions)), key=lambda index: candidate_rank(case, actions[index]))
    return selected, diagnostics


def run_best_of_n(
    cases_path: str | Path,
    output_path: str | Path,
    timings_path: str | Path,
    manifest_path: str | Path,
    *,
    endpoint: str,
    n: int,
    model_id: str,
    model_artifact: str,
    quantization: str,
    chat_template: str,
    grammar_engine: str,
    thinking_mode: str = "off",
    reasoning_budget: int = 0,
    max_tokens: int = 256,
    temperature: float = 0.2,
    seed: int = 1,
) -> dict[str, Any]:
    if n < 1:
        raise ValueError("n must be positive")
    cases = read_jsonl(cases_path)
    predictions = []
    timings = []
    started_at = datetime.now(timezone.utc).isoformat()
    for case_index, case in enumerate(cases):
        actions = []
        candidate_metadata = []
        started = time.perf_counter()
        for sample_index in range(n):
            action, metadata = llama_server_action(
                case,
                "best_of_n_budget_matched",
                endpoint,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_mode=thinking_mode,
                seed=seed + case_index * n + sample_index,
            )
            actions.append(action)
            candidate_metadata.append(metadata)
        selected_index, diagnostics = select_candidate(case, actions)
        elapsed = time.perf_counter() - started
        selected_metadata = candidate_metadata[selected_index]
        row = {
            "case_id": case["case_id"],
            "method": "best_of_n_budget_matched",
            "model_id": model_id,
            "seed": seed,
            "prediction": actions[selected_index],
            "response_metadata": selected_metadata,
            "runner_error": None,
            "backend": "llama.cpp",
            "quantization": quantization,
            "chat_template": chat_template,
            "grammar_engine": grammar_engine,
            "model_artifact": model_artifact,
            "thinking_mode": thinking_mode,
            "reasoning_budget": reasoning_budget,
            "best_of_n": {
                "schema_version": BEST_OF_VERSION,
                "n": n,
                "selected_index": selected_index,
                "candidate_diagnostics": diagnostics,
                "generation_calls": n,
                "elapsed_seconds": elapsed,
            },
        }
        row["thinking_marker_detected"] = prediction_has_thinking_marker(row)
        predictions.append(row)
        token_rates = [float(meta["generated_tokens_per_second"]) for meta in candidate_metadata if meta.get("generated_tokens_per_second")]
        timings.append(
            {
                "case_id": case["case_id"],
                "hypothesis_grid_id": case["hypothesis_grid_id"],
                "method": "best_of_n_budget_matched",
                "model_id": model_id,
                "backend": "llama.cpp",
                "elapsed_seconds": elapsed,
                "generation_calls": n,
                "best_of_n": n,
                "selected_index": selected_index,
                "generated_tokens_per_second_mean": statistics.mean(token_rates) if token_rates else None,
                "thinking_mode": thinking_mode,
                "reasoning_budget": reasoning_budget,
                "thinking_marker_detected": row["thinking_marker_detected"],
            }
        )
    write_jsonl(output_path, predictions)
    write_jsonl(timings_path, timings)
    manifest = {
        "schema_version": "tapbench.best_of_manifest.v1",
        "best_of_version": BEST_OF_VERSION,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path),
        "predictions_path": str(output_path),
        "timings_path": str(timings_path),
        "generation_count": len(cases) * n,
        "resolved_case_count": len(cases),
        "n": n,
        "selection_rule": "inference_safe_contract_validator_rank",
        "backend": "llama.cpp",
        "endpoint": endpoint,
        "model_id": model_id,
        "model_artifact": model_artifact,
        "quantization": quantization,
        "chat_template": chat_template,
        "grammar_engine": grammar_engine,
        "thinking_mode": thinking_mode,
        "reasoning_budget": reasoning_budget,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    write_yaml(manifest_path, manifest)
    return manifest

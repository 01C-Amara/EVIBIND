from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .io import read_jsonl, write_jsonl, write_yaml
from .r2_model_runner import R2A_CHAT_PARSER, R2A_GRAMMAR_ENGINE, _request_schema_json
from .runner import action_ir_json_schema, render_chat_messages
from .thinking import prediction_has_thinking_marker


EXTERNAL_RUNNER_VERSION = "tapbench.external_anchor_runner.v1"


def run_external_cases(
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
    methods: Iterable[str],
    seeds: Iterable[int] = (1,),
    max_tokens: int = 384,
    temperature: float = 0.0,
    max_generations: int | None = None,
) -> dict:
    methods = tuple(str(method) for method in methods)
    seeds = tuple(int(seed) for seed in seeds)
    cases = read_jsonl(cases_path)
    jobs = [
        (case, method, seed)
        for case in cases
        for method in methods
        for seed in seeds
    ]
    if max_generations is not None:
        jobs = jobs[:max_generations]
    predictions = []
    timings = []
    started_at = datetime.now(timezone.utc).isoformat()
    for case, method, seed in jobs:
        started = time.perf_counter()
        error = None
        try:
            action, metadata = _request_schema_json(
                endpoint,
                render_chat_messages(case, method, thinking_mode="off"),
                response_schema=action_ir_json_schema(case),
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
        except Exception as exc:
            error = str(exc)
            action = {"runner_error": error}
            metadata = {
                "finish_reason": "runner_error",
                "error_type": exc.__class__.__name__,
                "error_message": error,
            }
        elapsed = time.perf_counter() - started
        row = {
            "case_id": case["case_id"],
            "method": method,
            "model_id": model_id,
            "seed": seed,
            "prediction": action,
            "response_metadata": metadata,
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
            "external_runner_version": EXTERNAL_RUNNER_VERSION,
        }
        row["thinking_marker_detected"] = prediction_has_thinking_marker(row)
        predictions.append(row)
        timings.append({
            "case_id": case["case_id"],
            "hypothesis_grid_id": case.get("hypothesis_grid_id"),
            "method": method,
            "model_key": model_key,
            "model_id": model_id,
            "seed": seed,
            "backend": "llama.cpp",
            "quantization": "Q4_K_M",
            "elapsed_seconds": elapsed,
            "runner_error": error,
            "thinking_mode": "off",
            "thinking_marker_detected": row["thinking_marker_detected"],
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
        })
    write_jsonl(output_path, predictions)
    write_jsonl(timings_path, timings)
    manifest = {
        "schema_version": "tapbench.external_anchor_run_manifest.v1",
        "runner_version": EXTERNAL_RUNNER_VERSION,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path),
        "predictions_path": str(output_path),
        "timings_path": str(timings_path),
        "hypothesis_grid_ids": sorted({
            str(case.get("hypothesis_grid_id")) for case in cases
        }),
        "model_key": model_key,
        "model_id": model_id,
        "model_artifact": model_artifact,
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "chat_template": chat_template,
        "grammar_engine": R2A_GRAMMAR_ENGINE,
        "thinking_mode": "off",
        "reasoning_budget": 0,
        "context_tokens": 8192,
        "methods": list(methods),
        "seeds": list(seeds),
        "generation_count": len(predictions),
        "runner_errors": sum(row["runner_error"] is not None for row in predictions),
        "thinking_markers": sum(row["thinking_marker_detected"] for row in predictions),
        "context_truncations": sum(
            bool(row["response_metadata"].get("context_truncated"))
            for row in predictions
        ),
    }
    write_yaml(manifest_path, manifest)
    return manifest

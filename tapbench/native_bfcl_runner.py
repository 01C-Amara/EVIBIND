from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl, write_yaml
from .native_tool_runtime import (
    NATIVE_TOOL_RUNTIME_VERSION,
    request_native_tool,
)
from .thinking import prediction_has_thinking_marker


NATIVE_BFCL_RUNNER_VERSION = "tapbench.native_bfcl_runner.v2"
NATIVE_BFCL_METHOD = "native_tool_reasoning_512"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def _load_resume_prefix(
    cases: list[dict[str, Any]],
    output_path: str | Path,
    timings_path: str | Path,
    *,
    model_id: str,
    model_key: str,
    model_artifact: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    output = Path(output_path)
    timing_file = Path(timings_path)
    if not output.exists() and not timing_file.exists():
        return [], [], {}
    if not output.is_file() or not timing_file.is_file():
        raise ValueError("native resume requires both prediction and timing files")
    predictions = read_jsonl(output)
    timings = read_jsonl(timing_file)
    if len(predictions) != len(timings):
        raise ValueError("native resume prediction and timing row counts differ")
    if len(predictions) > len(cases):
        raise ValueError("native resume prefix exceeds the frozen case count")
    for index, (prediction, timing) in enumerate(zip(predictions, timings, strict=True)):
        expected_case_id = str(cases[index]["case_id"])
        if str(prediction.get("case_id")) != expected_case_id:
            raise ValueError("native prediction rows are not a contiguous case prefix")
        if str(timing.get("case_id")) != expected_case_id:
            raise ValueError("native timing rows are not a contiguous case prefix")
        expected_prediction = {
            "method": NATIVE_BFCL_METHOD,
            "model_id": model_id,
            "model_artifact": model_artifact,
            "seed": seed,
        }
        expected_timing = {
            "method": NATIVE_BFCL_METHOD,
            "model_id": model_id,
            "model_key": model_key,
            "seed": seed,
        }
        if any(prediction.get(key) != value for key, value in expected_prediction.items()):
            raise ValueError(f"native prediction metadata mismatch at {expected_case_id}")
        if any(timing.get(key) != value for key, value in expected_timing.items()):
            raise ValueError(f"native timing metadata mismatch at {expected_case_id}")
    return (
        predictions,
        timings,
        {"predictions": _sha256(output), "timings": _sha256(timing_file)},
    )


def run_native_bfcl(
    cases_path: str | Path,
    output_path: str | Path,
    timings_path: str | Path,
    manifest_path: str | Path,
    *,
    endpoint: str,
    model_id: str,
    model_key: str,
    model_profile: str,
    model_artifact: str,
    quantization: str,
    chat_template: str,
    protocol_path: str | Path,
    study_id: str,
    context_tokens: int,
    max_tokens: int,
    reasoning_budget: int,
    seed: int,
    max_cases: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    cases = read_jsonl(cases_path)
    if max_cases is not None:
        cases = cases[:max_cases]
    resume_checkpoint_sha256: dict[str, str] = {}
    if resume:
        predictions, timings, resume_checkpoint_sha256 = _load_resume_prefix(
            cases,
            output_path,
            timings_path,
            model_id=model_id,
            model_key=model_key,
            model_artifact=model_artifact,
            seed=seed,
        )
    else:
        predictions = []
        timings = []
    resumed_prediction_count = len(predictions)
    started_at = datetime.now(timezone.utc).isoformat()
    for case in cases[resumed_prediction_count:]:
        started = time.perf_counter()
        error = None
        try:
            action, metadata = request_native_tool(
                endpoint,
                case,
                model_profile=model_profile,
                max_tokens=max_tokens,
                seed=seed,
                reasoning_budget=reasoning_budget,
            )
        except Exception as exc:
            error = str(exc)
            action = {"runner_error": error}
            metadata = {
                "finish_reason": "runner_error",
                "error_type": exc.__class__.__name__,
                "error_message": error,
                "generation_calls": 0,
            }
        elapsed = time.perf_counter() - started
        row = {
            "case_id": case["case_id"],
            "method": NATIVE_BFCL_METHOD,
            "model_id": model_id,
            "seed": seed,
            "prediction": action,
            "action_ir_normalized": True,
            "response_metadata": metadata,
            "resolution": {
                "terminal_state": action.get("mode", "runner_error"),
                "elapsed_seconds": elapsed,
                "generation_calls": metadata.get("generation_calls", 0),
            },
            "runner_error": error,
            "backend": "llama.cpp",
            "quantization": quantization,
            "chat_template": chat_template,
            "grammar_engine": "llama.cpp_native_tool_grammar",
            "chat_parser": "llama.cpp_native_tool_parser",
            "inference_path": "openai_chat_completions_native_tools",
            "model_artifact": model_artifact,
            "thinking_mode": "native_recommended",
            "reasoning_budget": reasoning_budget,
            "reasoning_effort": (
                "medium" if model_profile == "gpt_oss" else "enabled"
            ),
            "native_tool_runtime_version": NATIVE_TOOL_RUNTIME_VERSION,
            "native_bfcl_runner_version": NATIVE_BFCL_RUNNER_VERSION,
            "max_output_tokens": max_tokens,
        }
        row["thinking_marker_detected"] = prediction_has_thinking_marker(row)
        row["reasoning_content_detected"] = bool(
            metadata.get("reasoning_content")
        )
        predictions.append(row)
        timings.append(
            {
                "case_id": case["case_id"],
                "hypothesis_grid_id": study_id,
                "bfcl_category": case.get("metadata", {}).get(
                    "bfcl_category"
                ),
                "task_kind": case.get("task_kind"),
                "method": NATIVE_BFCL_METHOD,
                "model_key": model_key,
                "model_id": model_id,
                "seed": seed,
                "backend": "llama.cpp",
                "quantization": quantization,
                "elapsed_seconds": elapsed,
                "generation_calls": metadata.get("generation_calls", 0),
                "runner_error": error,
                "thinking_mode": "native_recommended",
                "reasoning_budget": reasoning_budget,
                "reasoning_content_detected": row[
                    "reasoning_content_detected"
                ],
                "thinking_marker_detected": row[
                    "thinking_marker_detected"
                ],
                **{
                    key: metadata.get(key)
                    for key in (
                        "finish_reason",
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "reasoning_tokens",
                        "generated_tokens_per_second",
                        "context_truncated",
                        "native_tool_call_count",
                    )
                },
            }
        )

        write_jsonl(output_path, predictions)
        write_jsonl(timings_path, timings)
    write_jsonl(output_path, predictions)
    write_jsonl(timings_path, timings)
    manifest = {
        "schema_version": "tapbench.native_bfcl_manifest.v1",
        "study_id": study_id,
        "runner_version": NATIVE_BFCL_RUNNER_VERSION,
        "runtime_version": NATIVE_TOOL_RUNTIME_VERSION,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path),
        "predictions_path": str(output_path),
        "timings_path": str(timings_path),
        "protocol_path": str(protocol_path),
        "model_key": model_key,
        "model_id": model_id,
        "model_profile": model_profile,
        "model_artifact": model_artifact,
        "backend": "llama.cpp",
        "quantization": quantization,
        "chat_template": chat_template,
        "grammar_engine": "llama.cpp_native_tool_grammar",
        "chat_parser": "llama.cpp_native_tool_parser",
        "inference_path": "openai_chat_completions_native_tools",
        "thinking_mode": "native_recommended",
        "reasoning_budget": reasoning_budget,
        "reasoning_effort": (
            "medium" if model_profile == "gpt_oss" else "enabled"
        ),
        "context_tokens": context_tokens,
        "max_output_tokens": max_tokens,
        "method": NATIVE_BFCL_METHOD,
        "seed": seed,
        "case_count": len(cases),
        "prediction_count": len(predictions),
        "resumed_prediction_count": resumed_prediction_count,
        "resume_checkpoint_sha256": resume_checkpoint_sha256,
        "actual_model_calls": sum(
            int(row["response_metadata"].get("generation_calls", 0))
            for row in predictions
        ),
        "runner_errors": sum(
            row["runner_error"] is not None for row in predictions
        ),
        "length_stops": sum(
            row["response_metadata"].get("finish_reason") == "length"
            for row in predictions
        ),
        "visible_thinking_markers": sum(
            bool(row["thinking_marker_detected"]) for row in predictions
        ),
        "reasoning_content_rows": sum(
            bool(row["reasoning_content_detected"])
            for row in predictions
        ),
        "source_sha256": {
            "cases": _sha256(cases_path),
            "protocol": _sha256(protocol_path),
            "model_artifact": _sha256(model_artifact),
        },
    }
    write_yaml(manifest_path, manifest)
    return manifest

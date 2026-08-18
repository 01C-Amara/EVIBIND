from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bestof import select_candidate
from .extractive_candidates import build_extractive_candidate_table
from .io import read_jsonl, write_jsonl, write_yaml
from .r2_model_runner import R2A_CHAT_PARSER, R2A_GRAMMAR_ENGINE, _request_schema_json
from .r2b import _action_schema, _literal_action, _runtime
from .selective_tapr import (
    SELECTIVE_TAPR_VERSION,
    run_selective_tapr_resolution,
)
from .thinking import prediction_has_thinking_marker


HIERARCHY_RUNNER_VERSION = "evibind.hierarchy_runner.v3"
HIERARCHY_CONDITIONS = (
    "constrained_abstention",
    "best_of_compute_matched",
    "deterministic_candidates_ordinary_validation",
    "source_role_contract",
    "tap_r_selective_full",
)

RequestFn = Callable[..., tuple[dict[str, Any], dict[str, Any]]]
LiteralFn = Callable[..., tuple[dict[str, Any], dict[str, Any]]]
SelectiveFn = Callable[..., tuple[dict[str, Any], dict[str, Any]]]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _candidate_catalog(case: dict[str, Any]) -> dict[str, Any]:
    tools = []
    candidate_pools: dict[str, list[Any]] = {}
    pool_id_by_values: dict[str, str] = {}
    for tool in case.get("tools", []):
        table = build_extractive_candidate_table(
            case.get("messages", []),
            tool,
            include_optional=True,
        )
        properties = (
            tool.get("parameters", {}).get("properties", {})
            if isinstance(tool.get("parameters"), dict)
            else {}
        )
        surface_by_canonical = {
            str(prop.get("x-ir-name") or surface): str(surface)
            for surface, prop in properties.items()
            if isinstance(prop, dict)
        }
        slot_candidate_pools = {}
        for canonical, candidates in table.get("slots", {}).items():
            surface = surface_by_canonical.get(str(canonical), str(canonical))
            values = []
            for candidate in candidates:
                value = candidate.get("value")
                if value not in values:
                    values.append(value)
            values_key = json.dumps(
                values,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            pool_id = pool_id_by_values.get(values_key)
            if pool_id is None:
                pool_id = f"pool_{len(candidate_pools)}"
                pool_id_by_values[values_key] = pool_id
                candidate_pools[pool_id] = values
            slot_candidate_pools[surface] = pool_id
        tools.append(
            {
                "tool": str(tool.get("name")),
                "description": str(tool.get("description", "")),
                "slot_candidate_pools": slot_candidate_pools,
            }
        )
    return {"tools": tools, "candidate_pools": candidate_pools}


def deterministic_candidates_ordinary_action(
    case: dict[str, Any],
    endpoint: str,
    *,
    max_tokens: int,
    seed: int,
    request_fn: RequestFn = _request_schema_json,
) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog = _candidate_catalog(case)
    messages = [
        {
            "role": "system",
            "content": (
                "Return one ordinary Action IR JSON object. Candidate values are "
                "hints, not certificates. Use call only when a listed tool and "
                "all required values satisfy the request; otherwise clarify, "
                "no_tool, or direct_answer. Never output reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                "Request:\n"
                + "\n".join(
                    str(message.get("content", ""))
                    for message in case.get("messages", [])
                    if message.get("role") == "user"
                )
                + "\nDeterministic candidate catalog:\n"
                + json.dumps(catalog, sort_keys=True, separators=(",", ":"))
            ),
        },
    ]
    action, metadata = request_fn(
        endpoint,
        messages,
        response_schema=_action_schema(case),
        max_tokens=max_tokens,
        temperature=0.0,
        seed=seed,
    )
    metadata.update(
        {
            "generation_calls": 1,
            "candidate_catalog_sha256": _stable_hash(catalog),
            "certificate_gate": "none",
            "selection_rule": "ordinary_schema_constrained_generation",
        }
    )
    return action, metadata


def allocate_compute_matched_best_of(
    cases: Sequence[dict[str, Any]],
    endpoint: str,
    *,
    full_aggregate_total_tokens: int,
    max_tokens: int,
    seed: int,
    literal_fn: LiteralFn = _literal_action,
    validation_cases: Sequence[dict[str, Any]] | None = None,
    max_samples: int = 32,
) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    if not cases:
        raise ValueError("best-of compute matching requires at least one case")
    if validation_cases is None:
        validation_cases = cases
    if len(validation_cases) != len(cases):
        raise ValueError("generation and validation case counts must match")
    if full_aggregate_total_tokens <= 0:
        raise ValueError("full controller aggregate token budget must be positive")
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    actions: list[list[dict[str, Any]]] = [[] for _ in cases]
    metadata_rows: list[list[dict[str, Any]]] = [[] for _ in cases]
    elapsed_by_case = [0.0 for _ in cases]
    prompt_reservations: list[int | None] = [None for _ in cases]
    used = 0
    for round_index in range(max_samples):
        additions = 0
        for case_index, case in enumerate(cases):
            remaining = full_aggregate_total_tokens - used
            prompt_reservation = prompt_reservations[case_index]
            if round_index == 0:
                output_budget = max_tokens
            else:
                if prompt_reservation is None:
                    raise RuntimeError("best-of first-sample accounting is missing")
                output_budget = min(max_tokens, remaining - prompt_reservation)
                if output_budget < 1:
                    continue
            sample_seed = seed * 1_000_000 + round_index * len(cases) + case_index
            started = time.perf_counter()
            action, metadata = literal_fn(
                case,
                "best_of_compute_matched",
                endpoint,
                max_tokens=output_budget,
                temperature=0.2,
                seed=sample_seed,
            )
            elapsed_by_case[case_index] += time.perf_counter() - started
            prompt_tokens = int(metadata.get("prompt_tokens") or 0)
            total_tokens = int(metadata.get("total_tokens") or 0)
            if prompt_tokens <= 0 or total_tokens <= 0:
                raise RuntimeError(
                    "best-of compute matching requires token accounting"
                )
            if round_index > 0 and total_tokens > remaining:
                raise RuntimeError(
                    "best-of sample exceeded the remaining aggregate budget"
                )
            if prompt_reservation is not None and prompt_tokens != prompt_reservation:
                raise RuntimeError("best-of prompt token count changed between rounds")
            prompt_reservations[case_index] = prompt_tokens
            actions[case_index].append(action)
            metadata_rows[case_index].append(metadata)
            used += total_tokens
            additions += 1
        if round_index == 0 and used > full_aggregate_total_tokens:
            raise RuntimeError(
                "one ordinary sample per case exceeds the full-controller "
                "aggregate token budget"
            )
        if additions < len(cases):
            break

    results = []
    for validation_case, case_actions, case_metadata, elapsed in zip(
        validation_cases,
        actions,
        metadata_rows,
        elapsed_by_case,
        strict=True,
    ):
        if not case_actions:
            raise RuntimeError("every case must receive one best-of sample")
        selected, diagnostics = select_candidate(validation_case, case_actions)
        prompt_total = sum(
            int(row.get("prompt_tokens") or 0) for row in case_metadata
        )
        completion_total = sum(
            int(row.get("completion_tokens") or 0) for row in case_metadata
        )
        case_total = sum(int(row.get("total_tokens") or 0) for row in case_metadata)
        results.append(
            (
                case_actions[selected],
                {
                    "generation_calls": len(case_actions),
                    "best_of_n": len(case_actions),
                    "selected_index": selected,
                    "candidate_diagnostics": diagnostics,
                    "selection_rule": "ordinary_contract_validator_rank",
                    "certificate_gate": "none",
                    "prompt_tokens": prompt_total,
                    "completion_tokens": completion_total,
                    "total_tokens": case_total,
                    "full_controller_aggregate_total_tokens": (
                        full_aggregate_total_tokens
                    ),
                    "best_of_aggregate_total_tokens": used,
                    "aggregate_budget_utilization": (
                        used / full_aggregate_total_tokens
                    ),
                    "allocation_rule": "deterministic_balanced_rounds",
                    "minimum_samples_per_case": 1,
                    "finish_reason": (
                        "length"
                        if any(
                            row.get("finish_reason") == "length"
                            for row in case_metadata
                        )
                        else "stop"
                    ),
                    "context_truncated": any(
                        bool(row.get("context_truncated"))
                        for row in case_metadata
                    ),
                    "model_trace_sha256": _stable_hash(
                        [row.get("raw_text", "") for row in case_metadata]
                    ),
                },
                elapsed,
            )
        )
    return results
def _selective_action(
    runtime: dict[str, Any],
    endpoint: str,
    *,
    max_tokens: int,
    seed: int,
    semantic_extent_enabled: bool,
    request_fn: RequestFn,
    selective_fn: SelectiveFn,
) -> tuple[dict[str, Any], dict[str, Any]]:
    action, metadata = selective_fn(
        messages=runtime["messages"],
        tools=runtime["tools"],
        endpoint=endpoint,
        max_tokens=max_tokens,
        seed=seed,
        request_fn=request_fn,
        semantic_extent_enabled=semantic_extent_enabled,
        exhaust_proposal_budget=True,
    )
    metadata["semantic_extent_enabled"] = semantic_extent_enabled
    metadata["exhaust_proposal_budget"] = True
    metadata["model_trace_sha256"] = _stable_hash(metadata.get("raw_text", ""))
    return action, metadata


def _prediction_row(
    case: dict[str, Any],
    *,
    condition: str,
    action: dict[str, Any],
    metadata: dict[str, Any],
    error: str | None,
    elapsed: float,
    model_id: str,
    model_artifact: str,
    quantization: str,
    chat_template: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolution = {
        "terminal_state": action.get("mode", "runner_error"),
        "elapsed_seconds": elapsed,
        "generation_calls": int(metadata.get("generation_calls", 0)),
    }
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
        "hierarchy_runner_version": HIERARCHY_RUNNER_VERSION,
        "selective_tapr_version": (
            SELECTIVE_TAPR_VERSION
            if condition in {"source_role_contract", "tap_r_selective_full"}
            else None
        ),
        "semantic_extent_enabled": metadata.get("semantic_extent_enabled"),
        "max_output_tokens": metadata.get("max_output_tokens"),
    }
    row["thinking_marker_detected"] = prediction_has_thinking_marker(row)
    timing = {
        "case_id": case["case_id"],
        "hypothesis_grid_id": case["hypothesis_grid_id"],
        "family": case["family"],
        "extent_stratum": case["factors"]["extent_stratum"],
        "catalog_mutation": case["factors"]["catalog_mutation"],
        "task_kind": case["task_kind"],
        "method": condition,
        "model_id": model_id,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "generation_calls": metadata.get("generation_calls", 0),
        "prompt_tokens": metadata.get("prompt_tokens"),
        "completion_tokens": metadata.get("completion_tokens"),
        "total_tokens": metadata.get("total_tokens"),
        "finish_reason": metadata.get("finish_reason"),
        "context_truncated": metadata.get("context_truncated"),
        "runner_error": error,
        "thinking_marker_detected": row["thinking_marker_detected"],
    }
    return row, timing



def _load_resume_prefix(
    cases: Sequence[dict[str, Any]],
    selected: Sequence[str],
    execution_order: Sequence[str],
    output_path: str | Path,
    timings_path: str | Path,
) -> tuple[
    list[dict[str, tuple[dict[str, Any], dict[str, Any]]]],
    int,
    dict[str, str],
]:
    output = Path(output_path)
    timings = Path(timings_path)
    if not output.is_file() or not timings.is_file():
        raise ValueError("resume requires existing prediction and timing files")
    predictions = read_jsonl(output)
    timing_rows = read_jsonl(timings)
    if len(predictions) != len(timing_rows):
        raise ValueError("resume prediction and timing row counts differ")

    expected_conditions = tuple(
        condition for condition in execution_order if condition in selected
    )
    allowed_case_ids = [str(case["case_id"]) for case in cases]
    allowed_keys = {
        (case_id, condition)
        for case_id in allowed_case_ids
        for condition in expected_conditions
    }
    prediction_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    timing_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for rows, destination, label in (
        (predictions, prediction_by_key, "prediction"),
        (timing_rows, timing_by_key, "timing"),
    ):
        for row in rows:
            key = (str(row.get("case_id")), str(row.get("method")))
            if key not in allowed_keys:
                raise ValueError(f"resume has unknown {label} key: {key}")
            if key in destination:
                raise ValueError(f"resume has duplicate {label} key: {key}")
            destination[key] = row
    if set(prediction_by_key) != set(timing_by_key):
        raise ValueError("resume prediction and timing keys differ")

    case_rows: list[
        dict[str, tuple[dict[str, Any], dict[str, Any]]]
    ] = [{} for _ in cases]
    completed_cases = 0
    incomplete_seen = False
    expected_set = set(expected_conditions)
    for case_index, case_id in enumerate(allowed_case_ids):
        present = {
            condition
            for condition in expected_conditions
            if (case_id, condition) in prediction_by_key
        }
        if present == expected_set:
            if incomplete_seen:
                raise ValueError("resume rows are not a contiguous case prefix")
            completed_cases += 1
            for condition in expected_conditions:
                key = (case_id, condition)
                case_rows[case_index][condition] = (
                    prediction_by_key[key],
                    timing_by_key[key],
                )
        elif present:
            raise ValueError(f"resume has a partial condition set for {case_id}")
        else:
            incomplete_seen = True

    return (
        case_rows,
        completed_cases,
        {
            "predictions": _sha256(output),
            "timings": _sha256(timings),
        },
    )


def run_hierarchy_conditions(
    cases_path: str | Path,
    output_path: str | Path,
    timings_path: str | Path,
    manifest_path: str | Path,
    *,
    endpoint: str,
    model_id: str,
    model_artifact: str,
    chat_template: str,
    protocol_path: str | Path,
    study_id: str,
    context_tokens: int,
    conditions: Iterable[str] = HIERARCHY_CONDITIONS,
    max_tokens: int = 768,
    max_cases: int | None = None,
    seed: int = 1,
    quantization: str = "Q4_K_M",
    request_fn: RequestFn = _request_schema_json,
    literal_fn: LiteralFn = _literal_action,
    selective_fn: SelectiveFn = run_selective_tapr_resolution,
    resume: bool = False,
) -> dict[str, Any]:
    selected = tuple(str(value) for value in conditions)
    if len(selected) != len(set(selected)):
        raise ValueError("hierarchy conditions must be unique")
    unknown = set(selected) - set(HIERARCHY_CONDITIONS)
    if unknown:
        raise ValueError(f"unknown hierarchy conditions: {sorted(unknown)}")
    required = {"tap_r_selective_full", "source_role_contract"}
    if not required <= set(selected):
        raise ValueError("hierarchy run requires both extent conditions")
    cases = read_jsonl(cases_path)
    if max_cases is not None:
        if max_cases < 1:
            raise ValueError("max_cases must be positive")
        cases = cases[:max_cases]
    if not cases:
        raise ValueError("hierarchy run requires at least one case")
    started_at = datetime.now(timezone.utc).isoformat()
    execution_order = (
        "tap_r_selective_full",
        "source_role_contract",
        "constrained_abstention",
        "deterministic_candidates_ordinary_validation",
    )
    resumed_case_count = 0
    resume_checkpoint_sha256: dict[str, str] = {}
    if resume:
        (
            case_rows,
            resumed_case_count,
            resume_checkpoint_sha256,
        ) = _load_resume_prefix(
            cases,
            selected,
            execution_order,
            output_path,
            timings_path,
        )
    else:
        case_rows = [{} for _ in cases]
    for case_index, case in enumerate(cases):
        runtime = _runtime(case)
        for condition in execution_order:
            if condition not in selected:
                continue
            if condition in case_rows[case_index]:
                continue
            started = time.perf_counter()
            error = None
            try:
                if condition == "tap_r_selective_full":
                    action, metadata = _selective_action(
                        runtime,
                        endpoint,
                        max_tokens=max_tokens,
                        seed=seed,
                        semantic_extent_enabled=True,
                        request_fn=request_fn,
                        selective_fn=selective_fn,
                    )
                elif condition == "source_role_contract":
                    action, metadata = _selective_action(
                        runtime,
                        endpoint,
                        max_tokens=max_tokens,
                        seed=seed,
                        semantic_extent_enabled=False,
                        request_fn=request_fn,
                        selective_fn=selective_fn,
                    )
                elif condition == "deterministic_candidates_ordinary_validation":
                    action, metadata = deterministic_candidates_ordinary_action(
                        runtime,
                        endpoint,
                        max_tokens=max_tokens,
                        seed=seed,
                        request_fn=request_fn,
                    )
                else:
                    action, metadata = literal_fn(
                        runtime,
                        condition,
                        endpoint,
                        max_tokens=max_tokens,
                        temperature=0.0,
                        seed=seed,
                    )
                    metadata["generation_calls"] = 1
                    metadata["certificate_gate"] = "none"
                    metadata["model_trace_sha256"] = _stable_hash(
                        metadata.get("raw_text", "")
                    )
            except Exception as exc:
                error = f"{exc.__class__.__name__}: {exc}"
                action = {"runner_error": error}
                metadata = {
                    "finish_reason": "runner_error",
                    "generation_calls": 0,
                    "total_tokens": 0,
                }
            elapsed = time.perf_counter() - started
            case_rows[case_index][condition] = _prediction_row(
                case,
                condition=condition,
                action=action,
                metadata=metadata,
                error=error,
                elapsed=elapsed,
                model_id=model_id,
                model_artifact=model_artifact,
                quantization=quantization,
                chat_template=chat_template,
                seed=seed,
            )
        partial_predictions = [
            rows[condition][0]
            for rows in case_rows
            for condition in selected
            if condition in rows
        ]
        partial_timings = [
            rows[condition][1]
            for rows in case_rows
            for condition in selected
            if condition in rows
        ]
        write_jsonl(output_path, partial_predictions)
        write_jsonl(timings_path, partial_timings)

    if "best_of_compute_matched" in selected:
        full_aggregate_total_tokens = sum(
            int(
                rows["tap_r_selective_full"][0]
                .get("response_metadata", {})
                .get("total_tokens", 0)
            )
            for rows in case_rows
        )
        best_results = allocate_compute_matched_best_of(
            [_runtime(case) for case in cases],
            endpoint,
            validation_cases=cases,
            full_aggregate_total_tokens=full_aggregate_total_tokens,
            max_tokens=max_tokens,
            seed=seed,
            literal_fn=literal_fn,
        )
        for case_index, (action, metadata, elapsed) in enumerate(best_results):
            case_rows[case_index]["best_of_compute_matched"] = _prediction_row(
                cases[case_index],
                condition="best_of_compute_matched",
                action=action,
                metadata=metadata,
                error=None,
                elapsed=elapsed,
                model_id=model_id,
                model_artifact=model_artifact,
                quantization=quantization,
                chat_template=chat_template,
                seed=seed,
            )

    predictions = [
        rows[condition][0]
        for rows in case_rows
        for condition in selected
    ]
    timings = [
        rows[condition][1]
        for rows in case_rows
        for condition in selected
    ]
    write_jsonl(output_path, predictions)
    write_jsonl(timings_path, timings)
    manifest = {
        "schema_version": "evibind.hierarchy_run_manifest.v2",
        "runner_version": HIERARCHY_RUNNER_VERSION,
        "study_id": study_id,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path),
        "protocol_path": str(protocol_path),
        "endpoint": endpoint,
        "model_id": model_id,
        "model_artifact": model_artifact,
        "backend": "llama.cpp",
        "quantization": quantization,
        "chat_template": chat_template,
        "grammar_engine": R2A_GRAMMAR_ENGINE,
        "chat_parser": R2A_CHAT_PARSER,
        "thinking_mode": "off",
        "reasoning_budget": 0,
        "context_tokens": context_tokens,
        "max_output_tokens": max_tokens,
        "seed": seed,
        "conditions": list(selected),
        "case_count": len(cases),
        "prediction_count": len(predictions),
        "resumed_case_count": resumed_case_count,
        "resume_checkpoint_sha256": resume_checkpoint_sha256,
        "actual_model_calls": sum(
            int(row["response_metadata"].get("generation_calls", 0))
            for row in predictions
        ),
        "runner_errors": sum(row["runner_error"] is not None for row in predictions),
        "thinking_markers": sum(
            bool(row["thinking_marker_detected"]) for row in predictions
        ),
        "length_stops": sum(
            row["response_metadata"].get("finish_reason") == "length"
            for row in predictions
        ),
        "context_truncations": sum(
            bool(row["response_metadata"].get("context_truncated"))
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

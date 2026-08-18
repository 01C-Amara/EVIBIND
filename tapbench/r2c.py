from __future__ import annotations

import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import NORMALIZER_VERSION, SCORER_VERSION, VALIDATOR_VERSION
from .deployable_resolution import DEPLOYABLE_RESOLUTION_VERSION
from .effect_first import (
    EFFECT_FIRST_CONDITIONS,
    EFFECT_FIRST_VERSION,
    run_effect_first_resolution,
)
from .io import read_jsonl, write_jsonl, write_yaml
from .r2_model_runner import R2A_CHAT_PARSER, R2A_GRAMMAR_ENGINE
from .r2b import (
    R2B_ACTION_SCHEMA_VERSION,
    _catalog,
    _gold,
    _pointer_action,
    score_r2b_files,
)
from .r2c_families import R2C_CONFIRM_FAMILIES, R2C_PILOT_FAMILIES
from .thinking import prediction_has_thinking_marker
from .tier_b_verifier import FrozenTierBVerifier


R2C_CASE_VERSION = "tapbench.r2c_case.v1"
R2C_RUNNER_VERSION = "tapbench.r2c_model_runner.v2"
R2C_REPORT_VERSION = "tapbench.r2c_report.v1"
R2C_GRID_ID = "R2C_effect_first_confirmation"
R2C_MUTATIONS = (
    "unseen_tool_names",
    "unseen_argument_names",
    "enum_alias_shift",
    "near_duplicate_distractor_tools",
    "reordered_schema",
    "state_version_change",
)
R2C_TASK_KINDS = ("call", "missing_info", "no_tool", "direct_answer")
R2C_CONDITIONS = (
    "tap_r_literal_evidence",
    "tap_r_full",
    "tap_r_effect_first_single_locked",
    "tap_r_effect_first_consensus_locked",
)


def _all_values(index: int) -> dict[str, Any]:
    day = index % 24 + 1
    return {
        "employee_id": f"EMP-{3000 + index}",
        "amount": float(40 + index * 2),
        "currency": ("GBP", "EUR", "USD", "CAD")[index % 4],
        "category": ("travel", "meals", "lodging", "supplies")[index % 4],
        "receipt_id": f"RCT-{8000 + index}",
        "workspace": f"workspace-{index % 7}",
        "user_id": f"user-{2000 + index}",
        "role": ("viewer", "editor", "admin", "auditor")[index % 4],
        "expires_at": f"2027-01-{day:02d}T18:00:00",
        "reason_code": f"ACCESS-{index % 9}",
        "event_id": f"EVT-{5000 + index}",
        "attendee": f"Attendee {index}",
        "quantity": index % 5 + 1,
        "ticket_class": ("standard", "premium", "vip", "balcony")[index % 4],
        "delivery_email": f"attendee{index}@example.org",
        "resource_id": f"resource-{6000 + index}",
        "region": ("eu_west", "us_east", "ap_south", "eu_central")[index % 4],
        "retention_days": (7, 14, 30, 90)[index % 4],
        "snapshot_name": f"snapshot-{index}",
        "encryption_key_id": f"KEY-{7000 + index}",
        "supplier_id": f"SUP-{4000 + index}",
        "item_code": f"ITEM-{9000 + index}",
        "delivery_date": f"2026-12-{day:02d}",
        "origin": f"Station {index % 9}",
        "destination": f"Station {(index + 3) % 9}",
        "travel_date": f"2026-11-{day:02d}",
        "service_class": ("standard", "first", "sleeper", "flexible")[index % 4],
        "passenger": f"Passenger {index}",
        "restaurant": f"Restaurant {index % 11}",
        "date": f"2026-11-{day:02d}",
        "time": f"{17 + index % 5:02d}:30",
        "party_size": index % 7 + 2,
        "seating": ("indoor", "outdoor", "bar", "private_room")[index % 4],
        "document_id": f"DOC-{10000 + index}",
        "signer_email": f"signer{index}@example.org",
        "due_date": f"2026-12-{day:02d}",
        "signature_order": ("parallel", "sequential", "witnessed", "notarized")[index % 4],
        "message_code": f"MSG-{index % 13}",
        "service": f"service-{index % 8}",
        "version": f"{2 + index % 4}.{index % 10}.{index % 6}",
        "environment": ("development", "staging", "production", "canary")[index % 4],
        "change_ticket": f"CHG-{11000 + index}",
        "equipment_type": ("excavator", "generator", "lift", "compressor")[index % 4],
        "start_date": f"2026-11-{day:02d}",
        "end_date": f"2026-12-{day:02d}",
        "location": f"Depot {index % 6}",
        "rental_plan": ("daily", "weekly", "weekend", "long_term")[index % 4],
    }


def _family_values(family: Any, index: int) -> dict[str, Any]:
    values = _all_values(index)
    return {slot: values[slot] for slot in family.required_slots}


def _mode_request(
    family: Any,
    values: dict[str, Any],
    task_kind: str,
    variant: int,
    mutation: str,
    argument_aliases: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    dialogue_state: dict[str, Any] = {}
    excluded: set[str] = set()

    if task_kind == "call":
        request = family.request_template.format(**values)
        if mutation == "state_version_change":
            slot = family.required_slots[0]
            dialogue_state[slot] = {"value": values[slot], "version": 5}
            request = request.replace(
                str(values[slot]),
                f"the currently verified {slot}",
            )
            excluded.add(slot)
    elif task_kind == "missing_info":
        excluded.add(family.missing_slot)
        effect = family.call_tool.replace("_", " ")
        request = (
            f"Please {effect}. One required detail is unavailable; "
            "use only supplied evidence and ask for the unresolved field."
        )
    elif task_kind == "no_tool":
        prefixes = (
            "This is background context, not authorization to perform an external action. ",
            "I am not asking you to execute or submit anything. ",
            "For discussion only, no domain action is authorized. ",
            "Treat this as hypothetical context only. ",
        )
        request = prefixes[variant % len(prefixes)] + family.no_tool_request
    elif task_kind == "direct_answer":
        prefixes = (
            "Answer directly without calling a tool: ",
            "Respond only in chat with a brief explanation: ",
            "Give me a brief answer in chat: ",
            "Explain only; do not execute a tool: ",
        )
        request = prefixes[variant % len(prefixes)] + family.no_tool_request
    else:
        raise ValueError(task_kind)

    if task_kind in {"call", "missing_info"}:
        fields = [
            f"{surface}={values[canonical]}"
            for surface, canonical in argument_aliases.items()
            if canonical not in excluded
        ]
        request += "\nEvidence fields: " + "; ".join(fields) + ";"
    return request, dialogue_state


def generate_r2c_cases(*, scope: str = "pilot") -> list[dict[str, Any]]:
    if scope not in {"smoke", "pilot", "confirmation"}:
        raise ValueError("R2-C scope must be smoke, pilot, or confirmation")
    if scope == "confirmation":
        families = R2C_CONFIRM_FAMILIES
        variants = 16
        split = "confirmation"
    else:
        families = R2C_PILOT_FAMILIES
        variants = 1 if scope == "smoke" else 8
        split = "engineering_pilot"

    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        for variant in range(variants):
            global_index = family_index * 32 + variant + (
                1000 if split == "confirmation" else 0
            )
            task_kind = R2C_TASK_KINDS[
                (variant + family_index) % len(R2C_TASK_KINDS)
            ]
            mutation = R2C_MUTATIONS[
                (variant + 2 * family_index) % len(R2C_MUTATIONS)
            ]
            values = _family_values(family, global_index)
            tools, tool_aliases, argument_aliases = _catalog(
                family,
                family_index + (100 if split == "confirmation" else 80),
                values,
                mutation,
            )
            request, dialogue_state = _mode_request(
                family,
                values,
                task_kind,
                variant,
                mutation,
                argument_aliases,
            )
            derivable = dict(values)
            if task_kind == "missing_info":
                derivable.pop(family.missing_slot, None)
            rows.append(
                {
                    "schema_version": R2C_CASE_VERSION,
                    "case_id": f"r2c_{family.name}_{variant:02d}",
                    "hypothesis_grid_id": R2C_GRID_ID,
                    "hypothesis": "R2C",
                    "split": split,
                    "family": family.name,
                    "task_kind": task_kind,
                    "factors": {
                        "task_kind": task_kind,
                        "catalog_mutation": mutation,
                        "variant": variant,
                    },
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Return exactly one Action IR object. Do not infer "
                                "authorization from values that merely fit a schema."
                            ),
                        },
                        {"role": "user", "content": request},
                    ],
                    "tools": tools,
                    "tool_aliases": tool_aliases,
                    "argument_aliases": argument_aliases,
                    "dialogue_state": dialogue_state,
                    "reference_context": {
                        "reference_date": "2026-07-15",
                        "timezone": "Europe/London",
                        "action_risk_budget": 0.05,
                    },
                    "gold_action": _gold(family, values, task_kind),
                    "derivable_values": derivable,
                    "r2c_oracle": {
                        "family": family.name,
                        "mutation": mutation,
                        "variant": variant,
                    },
                    "metadata": {
                        "backend_namespace": "llama_cpp_q4km_r2c",
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
                            "r2c_oracle",
                            "task_kind",
                        ],
                    },
                }
            )
    return rows


def write_r2c_cases(output: str | Path, *, scope: str) -> int:
    return write_jsonl(output, generate_r2c_cases(scope=scope))


def _runtime(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": deepcopy(case.get("messages", [])),
        "tools": deepcopy(case.get("tools", [])),
        "tool_aliases": deepcopy(case.get("tool_aliases", {})),
        "argument_aliases": deepcopy(case.get("argument_aliases", {})),
        "dialogue_state": deepcopy(case.get("dialogue_state", {})),
        "reference_context": deepcopy(case.get("reference_context", {})),
    }


def run_r2c_model_conditions(
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
    conditions: Iterable[str] = R2C_CONDITIONS,
    seeds: Iterable[int] = (1,),
    max_tokens: int = 128,
    max_generations: int | None = None,
) -> dict[str, Any]:
    verifier = FrozenTierBVerifier.load(tier_b_verifier_path)
    selected_conditions = tuple(str(item) for item in conditions)
    unknown = set(selected_conditions) - set(R2C_CONDITIONS)
    if unknown:
        raise ValueError(f"unknown R2-C conditions: {sorted(unknown)}")
    jobs = [
        (case, condition, int(seed))
        for case in read_jsonl(cases_path)
        for condition in selected_conditions
        for seed in seeds
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
            if condition in EFFECT_FIRST_CONDITIONS:
                runtime = _runtime(case)
                action, metadata = run_effect_first_resolution(
                    messages=runtime["messages"],
                    tools=runtime["tools"],
                    dialogue_state=runtime["dialogue_state"],
                    reference_context=runtime["reference_context"],
                    endpoint=endpoint,
                    condition=condition,
                    max_tokens=max_tokens,
                    seed=seed,
                )
            else:
                action, metadata = _pointer_action(
                    case,
                    condition,
                    endpoint,
                    verifier,
                    max_tokens=max_tokens,
                    seed=seed,
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
        resolution = (
            metadata.get("resolution")
            if isinstance(metadata.get("resolution"), dict)
            else {
                "terminal_state": (
                    action.get("mode") if isinstance(action, dict) else "refuse"
                ),
                "elapsed_seconds": elapsed,
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
            "quantization": "Q4_K_M",
            "chat_template": chat_template,
            "grammar_engine": R2A_GRAMMAR_ENGINE,
            "chat_parser": R2A_CHAT_PARSER,
            "inference_path": "apply_template_then_raw_completion",
            "model_artifact": model_artifact,
            "thinking_mode": "off",
            "reasoning_budget": 0,
            "r2b_action_schema_version": R2B_ACTION_SCHEMA_VERSION,
            "r2c_model_runner_version": R2C_RUNNER_VERSION,
            "deployable_resolution_version": DEPLOYABLE_RESOLUTION_VERSION,
            "effect_first_version": (
                EFFECT_FIRST_VERSION
                if condition in EFFECT_FIRST_CONDITIONS
                else None
            ),
            "action_risk_threshold": metadata.get("action_risk_threshold"),
            "action_risk_score": metadata.get("action_risk_score"),
            "max_output_tokens": max_tokens,
            "contract_solver_version": resolution.get("schema_version"),
            "evidence_contract_version": metadata.get(
                "evidence_contract_version"
            )
            or resolution.get("evidence_contract_version"),
        }
        row["thinking_marker_detected"] = prediction_has_thinking_marker(row)
        predictions.append(row)
        timings.append(
            {
                "case_id": case["case_id"],
                "hypothesis_grid_id": R2C_GRID_ID,
                "catalog_mutation": case["factors"]["catalog_mutation"],
                "task_kind": case["task_kind"],
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
                "counterbalanced_agreement": metadata.get(
                    "counterbalanced_agreement"
                ),
                "evidence_lock_status": (
                    metadata.get("evidence_lock", {}).get("status")
                    if isinstance(metadata.get("evidence_lock"), dict)
                    else None
                ),
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
        "schema_version": "tapbench.r2c_model_run_manifest.v1",
        "runner_version": R2C_RUNNER_VERSION,
        "effect_first_version": EFFECT_FIRST_VERSION,
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
        "thinking_mode": "off",
        "reasoning_budget": 0,
        "context_tokens": 8192,
        "max_output_tokens": max_tokens,
        "conditions": list(selected_conditions),
        "seeds": list(seeds),
        "generation_count": len(predictions),
        "actual_model_calls": sum(
            int(row["response_metadata"].get("generation_calls", 0))
            for row in predictions
        ),
        "runner_errors": sum(
            row["runner_error"] is not None for row in predictions
        ),
        "tier_b_verifier_version": verifier.version,
        "tier_b_verifier_artifact_sha256": verifier.artifact_sha256,
    }
    write_yaml(manifest_path, manifest)
    return manifest


def score_r2c_files(
    cases_path: str | Path,
    predictions_path: str | Path,
    scores_path: str | Path,
    slot_errors_path: str | Path,
    report_path: str | Path,
    *,
    expected_model_count: int | None = None,
    expected_condition_count: int | None = None,
    expected_seed_count: int | None = None,
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
    predictions = read_jsonl(predictions_path)
    by_key = {
        (
            row["case_id"],
            row.get("method"),
            row.get("model_id"),
            row.get("seed"),
        ): row
        for row in predictions
    }
    scores = read_jsonl(scores_path)
    for row in scores:
        prediction = by_key[
            (
                row["case_id"],
                row.get("method"),
                row.get("model_id"),
                row.get("seed"),
            )
        ]
        metadata = prediction.get("response_metadata", {})
        row["r2c_report_version"] = R2C_REPORT_VERSION
        row["action_risk_score"] = metadata.get("action_risk_score")
        row["counterbalanced_agreement"] = metadata.get(
            "counterbalanced_agreement"
        )
        row["evidence_lock_status"] = (
            metadata.get("evidence_lock", {}).get("status")
            if isinstance(metadata.get("evidence_lock"), dict)
            else None
        )
        row["effect_admission_basis"] = (
            metadata.get("effect_admission", {}).get("basis")
            if isinstance(metadata.get("effect_admission"), dict)
            else None
        )
    write_jsonl(scores_path, scores)

    report["schema_version"] = R2C_REPORT_VERSION
    report["study_id"] = R2C_GRID_ID
    report["scorer_version"] = SCORER_VERSION
    report["normalizer_version"] = NORMALIZER_VERSION
    report["validator_version"] = VALIDATOR_VERSION
    Path(report_path).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report

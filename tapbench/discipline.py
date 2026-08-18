from __future__ import annotations

from collections import defaultdict
from typing import Any


GRID_SINGLETON_FIELDS = ("backend", "quantization", "thinking_mode", "reasoning_budget")
MODEL_SINGLETON_FIELDS = ("backend", "quantization", "chat_template", "grammar_engine", "chat_parser", "inference_path", "model_artifact")
RESOLVER_SINGLETON_FIELDS = ("deployable_resolution_version", "evidence_contract_version", "contract_solver_version")
EFLRX_SINGLETON_FIELDS = (
    "eflrx_version",
    "extractive_candidate_version",
    "eflrx_runner_version",
    "action_risk_threshold",
)
CAPC_SINGLETON_FIELDS = (
    "capc_version",
    "extractive_candidate_version",
    "capc_runner_version",
    "action_risk_threshold",
)
SELECTIVE_TAPR_SINGLETON_FIELDS = (
    "selective_tapr_version",
    "admission_version",
    "effect_support_version",
    "scope_guard_version",
    "certificate_span_policy_version",
    "extractive_candidate_version",
    "r2d_model_runner_version",
    "action_risk_threshold",
)
PROJECTED_CAPC_SINGLETON_FIELDS = (
    "projected_capc_version",
    "source_certificate_version",
    "massive_runner_version",
    "action_risk_threshold",
)
RETRIEVE_POINTER_SINGLETON_FIELDS = (
    "retrieve_pointer_version",
    "retriever_version",
    "retriever_model_id",
    "retriever_revision",
    "retriever_serialization_arm",
    "retriever_k",
    "ranking_artifact_sha256",
    "source_span_projection_version",
    "source_span_certificate_version",
    "massive_runner_version",
    "action_risk_threshold",
)
SEMANTIC_SURFACE_SINGLETON_FIELDS = (
    "semantic_surface_version",
    "retriever_version",
    "retriever_model_id",
    "retriever_revision",
    "retriever_serialization_arm",
    "retriever_k",
    "ranking_artifact_sha256",
    "source_span_projection_version",
    "source_span_certificate_version",
    "massive_runner_version",
    "action_risk_threshold",
)
SLOTWISE_SURFACE_SINGLETON_FIELDS = (
    "slotwise_surface_version",
    "semantic_surface_materializer_version",
    "retriever_version",
    "retriever_model_id",
    "retriever_revision",
    "retriever_serialization_arm",
    "retriever_k",
    "ranking_artifact_sha256",
    "source_span_projection_version",
    "source_span_certificate_version",
    "massive_runner_version",
    "action_risk_threshold",
)
QA_EVIDENCE_SINGLETON_FIELDS = (
    "qa_evidence_controller_version",
    "qa_evidence_system_label",
    "qa_verifier_version",
    "qa_verifier_question_version",
    "qa_verifier_model_id",
    "qa_verifier_model_revision",
    "qa_verifier_backend",
    "qa_verifier_dtype",
    "qa_verifier_artifact_sha256",
    "retriever_version",
    "retriever_model_id",
    "retriever_revision",
    "retriever_serialization_arm",
    "retriever_k",
    "ranking_artifact_sha256",
    "source_span_projection_version",
    "source_span_certificate_version",
    "massive_runner_version",
    "action_risk_threshold",
)
COEFFICIENT_THINKING_MODES = {"off", "not_applicable", None}


def coefficient_discipline_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    by_grid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_grid_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_grid_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grid = str(row.get("hypothesis_grid_id", "unknown"))
        model = str(row.get("model_id", "unknown"))
        by_grid[grid].append(row)
        by_grid_model[(grid, model)].append(row)
        by_grid_method[(grid, str(row.get("method", "unknown")))].append(row)

    for grid, grid_rows in by_grid.items():
        for index, row in enumerate(grid_rows):
            thinking_mode = row.get("thinking_mode", "not_applicable")
            if thinking_mode not in COEFFICIENT_THINKING_MODES:
                failures.append({"scope": "row", "grid": grid, "row_index": index, "field": "thinking_mode", "value": str(thinking_mode)})
            if bool(row.get("thinking_marker_detected", False)):
                failures.append({"scope": "row", "grid": grid, "row_index": index, "field": "thinking_marker_detected", "value": "true"})
            if row.get("finish_reason") == "length" and thinking_mode not in {"off", "not_applicable", None}:
                failures.append({"scope": "row", "grid": grid, "row_index": index, "field": "finish_reason", "value": "length_with_reasoning"})
            if grid == "R2D_selective_composite_confirmation_v7":
                expected_stops = (
                    ["<tool_call|>"]
                    if row.get("chat_template") == "gemma4"
                    else []
                )
                for field, expected in (
                    (
                        "stop_sequence_policy_version",
                        "tapbench.r2d_stop_sequences.v1",
                    ),
                    ("stop_sequences", expected_stops),
                ):
                    if row.get(field) != expected:
                        failures.append(
                            {
                                "scope": "row",
                                "grid": grid,
                                "row_index": index,
                                "field": field,
                                "value": str(row.get(field)),
                            }
                        )
        for field in GRID_SINGLETON_FIELDS:
            values = {row.get(field) for row in grid_rows}
            if len(values) > 1:
                failures.append({"scope": "grid", "grid": grid, "field": field, "values": sorted(map(str, values))})

    for (grid, method), method_rows in by_grid_method.items():
        if not method.startswith("tap_r_"):
            continue
        if method.startswith("tap_r_eflrx"):
            for field in EFLRX_SINGLETON_FIELDS:
                values = {row.get(field) for row in method_rows}
                if None in values or len(values) > 1:
                    failures.append({
                        "scope": "grid_method",
                        "grid": grid,
                        "method": method,
                        "field": field,
                        "values": sorted(map(str, values)),
                    })
            continue
        if method.startswith("tap_r_selective"):
            for field in SELECTIVE_TAPR_SINGLETON_FIELDS:
                values = {row.get(field) for row in method_rows}
                if None in values or len(values) > 1:
                    failures.append({
                        "scope": "grid_method",
                        "grid": grid,
                        "method": method,
                        "field": field,
                        "values": sorted(map(str, values)),
                    })
            continue
        if method.startswith("tap_r_retrieve_pointer"):
            for field in RETRIEVE_POINTER_SINGLETON_FIELDS:
                values = {row.get(field) for row in method_rows}
                if None in values or len(values) > 1:
                    failures.append({
                        "scope": "grid_method",
                        "grid": grid,
                        "method": method,
                        "field": field,
                        "values": sorted(map(str, values)),
                    })
            continue
        if method.startswith("tap_r_surface_active"):
            for field in SEMANTIC_SURFACE_SINGLETON_FIELDS:
                values = {row.get(field) for row in method_rows}
                if None in values or len(values) > 1:
                    failures.append({
                        "scope": "grid_method",
                        "grid": grid,
                        "method": method,
                        "field": field,
                        "values": sorted(map(str, values)),
                    })
            continue
        if method.startswith("tap_r_slotwise_surface"):
            for field in SLOTWISE_SURFACE_SINGLETON_FIELDS:
                values = {row.get(field) for row in method_rows}
                if None in values or len(values) > 1:
                    failures.append({
                        "scope": "grid_method",
                        "grid": grid,
                        "method": method,
                        "field": field,
                        "values": sorted(map(str, values)),
                    })
            continue
        if method.startswith("tap_r_qa_") or method.startswith(
            "tap_r_supervised_router_small_model_slots_qa_"
        ):
            for field in QA_EVIDENCE_SINGLETON_FIELDS:
                values = {row.get(field) for row in method_rows}
                if None in values or len(values) > 1:
                    failures.append({
                        "scope": "grid_method",
                        "grid": grid,
                        "method": method,
                        "field": field,
                        "values": sorted(map(str, values)),
                    })
            continue
        if method.startswith("tap_r_capc_projected"):
            for field in PROJECTED_CAPC_SINGLETON_FIELDS:
                values = {row.get(field) for row in method_rows}
                if None in values or len(values) > 1:
                    failures.append({
                        "scope": "grid_method",
                        "grid": grid,
                        "method": method,
                        "field": field,
                        "values": sorted(map(str, values)),
                    })
            continue
        if method.startswith("tap_r_capc"):
            for field in CAPC_SINGLETON_FIELDS:
                values = {row.get(field) for row in method_rows}
                if None in values or len(values) > 1:
                    failures.append({
                        "scope": "grid_method",
                        "grid": grid,
                        "method": method,
                        "field": field,
                        "values": sorted(map(str, values)),
                    })
            continue
        for field in RESOLVER_SINGLETON_FIELDS:
            values = {row.get(field) for row in method_rows}
            if None in values or len(values) > 1:
                failures.append({
                    "scope": "grid_method",
                    "grid": grid,
                    "method": method,
                    "field": field,
                    "values": sorted(map(str, values)),
                })
        if method.startswith("tap_r_effect_first"):
            for field in ("effect_first_version", "action_risk_threshold"):
                values = {row.get(field) for row in method_rows}
                if None in values or len(values) > 1:
                    failures.append({
                        "scope": "grid_method",
                        "grid": grid,
                        "method": method,
                        "field": field,
                        "values": sorted(map(str, values)),
                    })
        tep_values = {row.get("typed_evidence_program_version") for row in method_rows}
        tep_required = "tep" in method
        non_null_tep_values = {value for value in tep_values if value is not None}
        if (tep_required and (None in tep_values or len(non_null_tep_values) != 1)) or len(non_null_tep_values) > 1:
            failures.append({
                "scope": "grid_method",
                "grid": grid,
                "method": method,
                "field": "typed_evidence_program_version",
                "values": sorted(map(str, tep_values)),
            })

    for (grid, model), model_rows in by_grid_model.items():
        for field in MODEL_SINGLETON_FIELDS:
            values = {row.get(field) for row in model_rows}
            if len(values) > 1:
                failures.append(
                    {
                        "scope": "grid_model",
                        "grid": grid,
                        "model_id": model,
                        "field": field,
                        "values": sorted(map(str, values)),
                    }
                )
    return failures


def assert_coefficient_discipline(rows: list[dict[str, Any]]) -> None:
    failures = coefficient_discipline_failures(rows)
    if failures:
        raise AssertionError(f"hypothesis coefficient discipline failures: {failures}")

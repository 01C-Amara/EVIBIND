from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .extractive_qa_verifier import (
    EXTRACTIVE_QA_VERIFIER_VERSION,
    file_sha256,
    validate_extractive_qa_rows,
)
from .io import read_jsonl
from .massive_v2_analysis import (
    EXPECTED_MODELS,
    _group_metrics,
    _percentile,
    _raw_metrics,
)
from .qa_evidence_controller import (
    QA_EVIDENCE_CONDITIONS,
    QA_EVIDENCE_SYSTEM_LABEL,
    QA_EVIDENCE_CONTROLLER_VERSION,
)


QA_HYBRID_DESIGN_ANALYSIS_VERSION = "tapbench.massive_qa_hybrid_design.v1"
QA_CANDIDATES = QA_EVIDENCE_CONDITIONS
PAIRED_CONTROLS = (
    "prompt_few_shot",
    "full_tap_b2",
    "tap_r_surface_active_single",
)
EXPECTED_METHODS = (*PAIRED_CONTROLS, *QA_CANDIDATES)


def analyze_qa_hybrid_design(
    cases_path: str | Path,
    predictions_path: str | Path,
    timings_path: str | Path,
    official_details_path: str | Path,
    qa_certificate_summary_path: str | Path,
    surface_certificate_summary_path: str | Path,
    discipline_failures_path: str | Path,
    verifier_path: str | Path,
) -> dict[str, Any]:
    cases = read_jsonl(cases_path)
    predictions = read_jsonl(predictions_path)
    timings = read_jsonl(timings_path)
    details = read_jsonl(official_details_path)
    qa_certificate = json.loads(
        Path(qa_certificate_summary_path).read_text(encoding="utf-8")
    )
    surface_certificate = json.loads(
        Path(surface_certificate_summary_path).read_text(encoding="utf-8")
    )
    discipline = read_jsonl(discipline_failures_path)
    verifier_rows = read_jsonl(verifier_path)
    verifier_failures = validate_extractive_qa_rows(verifier_rows)
    verifier_sha256 = file_sha256(verifier_path)

    case_ids = {str(row["case_id"]) for row in cases}
    models = sorted({str(row.get("model_id")) for row in predictions})
    methods = sorted({str(row.get("method")) for row in predictions})
    expected_count = len(cases) * len(EXPECTED_MODELS) * len(EXPECTED_METHODS)
    expected_qa_count = len(cases) * len(EXPECTED_MODELS) * len(QA_CANDIDATES)
    expected_surface_count = len(cases) * len(EXPECTED_MODELS)
    keys = [
        (
            str(row.get("case_id")),
            str(row.get("model_id")),
            str(row.get("method")),
            int(row.get("seed", 0)),
        )
        for row in predictions
    ]
    timing_keys = [
        (
            str(row.get("case_id")),
            str(row.get("model_id") or row.get("model_key")),
            str(row.get("method")),
            int(row.get("seed", 0)),
        )
        for row in timings
    ]
    qa_predictions = [
        row for row in predictions if row.get("method") in QA_CANDIDATES
    ]
    surface_predictions = [
        row
        for row in predictions
        if row.get("method") == "tap_r_surface_active_single"
    ]
    accepted_qa = [
        row
        for row in qa_predictions
        if isinstance(row.get("prediction"), dict)
        and row["prediction"].get("mode") == "call"
    ]
    accepted_surface = [
        row
        for row in surface_predictions
        if isinstance(row.get("prediction"), dict)
        and row["prediction"].get("mode") == "call"
    ]

    required_qa_provenance = (
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
        "ranking_sha256",
        "ranking_artifact_sha256",
        "source_span_projection_version",
        "source_span_certificate_version",
        "massive_runner_version",
    )
    ranked_predictions = [*qa_predictions, *surface_predictions]
    ranking_by_case: dict[str, set[str]] = defaultdict(set)
    for row in ranked_predictions:
        ranking_by_case[str(row.get("case_id"))].add(
            str(row.get("ranking_sha256"))
        )

    engineering = {
        "prediction_row_count": len(predictions) == expected_count,
        "official_row_count": len(details) == expected_count,
        "timing_row_count": len(timings) == expected_count,
        "qa_prediction_row_count": len(qa_predictions) == expected_qa_count,
        "surface_prediction_row_count": (
            len(surface_predictions) == expected_surface_count
        ),
        "no_duplicate_prediction_cells": len(keys) == len(set(keys)),
        "no_duplicate_timing_cells": len(timing_keys) == len(set(timing_keys)),
        "all_case_ids_known": all(
            str(row.get("case_id")) in case_ids for row in predictions
        ),
        "exact_models": models == sorted(EXPECTED_MODELS),
        "exact_methods": methods == sorted(EXPECTED_METHODS),
        "all_llama_cpp_q4_k_m": all(
            row.get("backend") == "llama.cpp"
            and row.get("quantization") == "Q4_K_M"
            for row in predictions
        ),
        "no_runner_errors": not any(
            row.get("runner_error") for row in predictions
        ),
        "no_visible_thinking": not any(
            bool(row.get("thinking_marker_detected")) for row in predictions
        ),
        "thinking_off": all(
            row.get("thinking_mode") == "off" for row in predictions
        ),
        "no_context_overflow": not any(
            bool(row.get("response_metadata", {}).get("context_overflow"))
            for row in predictions
        ),
        "no_context_truncation": not any(
            bool(row.get("response_metadata", {}).get("context_truncated"))
            for row in predictions
        ),
        "no_output_length_stops": not any(
            row.get("response_metadata", {}).get("finish_reason") == "length"
            for row in predictions
        ),
        "exact_context_preflight": all(
            float(row.get("preflight_prompt_token_delta_max_abs") or 0) <= 1
            for row in timings
        ),
        "qa_certificate_all_rows": (
            int(qa_certificate.get("rows", -1)) == expected_qa_count
        ),
        "qa_certificate_replay": int(qa_certificate.get("failed", -1)) == 0,
        "qa_accepted_certificate_count_matches": int(
            qa_certificate.get("accepted_calls", -1)
        )
        == len(accepted_qa),
        "surface_certificate_all_rows": (
            int(surface_certificate.get("rows", -1))
            == expected_surface_count
        ),
        "surface_certificate_replay": (
            int(surface_certificate.get("failed", -1)) == 0
        ),
        "surface_accepted_certificate_count_matches": int(
            surface_certificate.get("accepted_calls", -1)
        )
        == len(accepted_surface),
        "coefficient_discipline": not discipline,
        "complete_qa_provenance": all(
            all(row.get(field) is not None for field in required_qa_provenance)
            for row in qa_predictions
        ),
        "hybrid_label_exact": all(
            row.get("qa_evidence_system_label") == QA_EVIDENCE_SYSTEM_LABEL
            and row.get("qa_evidence_controller_version")
            == QA_EVIDENCE_CONTROLLER_VERSION
            and row.get("qa_verifier_version")
            == EXTRACTIVE_QA_VERIFIER_VERSION
            for row in qa_predictions
        ),
        "one_verifier_artifact": {
            row.get("qa_verifier_artifact_sha256") for row in qa_predictions
        }
        == {verifier_sha256},
        "verifier_artifact_valid": not verifier_failures,
        "verifier_no_truncation": not any(
            bool(row.get("input_truncated")) for row in verifier_rows
        ),
        "verifier_gold_firewall": not any(
            bool(row.get("gold_loaded")) for row in verifier_rows
        ),
        "verifier_audit_artifact_matches": (
            qa_certificate.get("qa_verifier_artifact_sha256")
            == verifier_sha256
        ),
        "one_ranking_per_case": all(
            len(values) == 1 and "None" not in values
            for values in ranking_by_case.values()
        )
        and set(ranking_by_case) == case_ids,
        "call_only_hybrid_election": all(
            row.get("response_metadata", {}).get("call_only_tool_election")
            is True
            and row.get("response_metadata", {}).get("no_call_election_option")
            is False
            for row in qa_predictions
        ),
        "small_model_never_supplies_argument_values": all(
            row.get("response_metadata", {}).get(
                "small_model_supplies_argument_values"
            )
            is False
            for row in qa_predictions
        ),
        "finite_hybrid_values_for_calls": all(
            row.get("response_metadata", {}).get(
                "no_unconstrained_action_critical_tokens"
            )
            is True
            for row in accepted_qa
        ),
    }

    by_method = _group_metrics(details, ("method",))
    by_model_method = _group_metrics(details, ("model_id", "method"))
    by_language_method = _group_metrics(details, ("language", "method"))
    by_cell = _group_metrics(details, ("model_id", "language", "method"))
    pooled_lookup = {str(row["method"]): row for row in by_method}
    model_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    language_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cell_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in by_model_method:
        model_rows[str(row["method"])].append(row)
    for row in by_language_method:
        language_rows[str(row["method"])].append(row)
    for row in by_cell:
        cell_rows[str(row["method"])].append(row)

    timing_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in timings:
        timing_groups[str(row.get("method"))].append(row)
    stage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in predictions:
        method = str(row.get("method"))
        failure = row.get("response_metadata", {}).get(
            "controller_stage_failure"
        )
        stage_counts[method][str(failure or "accepted_call")] += 1

    candidates: list[dict[str, Any]] = []
    for method in QA_CANDIDATES:
        pooled = pooled_lookup.get(method, _raw_metrics([]))
        models_for_method = model_rows.get(method, [])
        languages_for_method = language_rows.get(method, [])
        cells_for_method = cell_rows.get(method, [])
        timing = timing_groups.get(method, [])
        precision = pooled["accepted_call_exact_precision"]
        coverage = pooled["call_coverage"]
        gates = {
            "engineering": all(engineering.values()),
            "raw_pooled_precision": precision is not None
            and precision >= 0.95,
            "raw_pooled_coverage": coverage is not None and coverage >= 0.50,
            "minimum_model_language_coverage": bool(cells_for_method)
            and all(
                float(row["call_coverage"] or 0.0) >= 0.25
                for row in cells_for_method
            ),
            "minimum_emitted_calls_per_model_language": bool(cells_for_method)
            and all(
                int(row["emitted_calls"]) >= 12 for row in cells_for_method
            ),
            "minimum_model_pooled_precision": bool(models_for_method)
            and all(
                row["accepted_call_exact_precision"] is not None
                and float(row["accepted_call_exact_precision"]) >= 0.90
                for row in models_for_method
            ),
            "minimum_language_pooled_precision": bool(languages_for_method)
            and all(
                row["accepted_call_exact_precision"] is not None
                and float(row["accepted_call_exact_precision"]) >= 0.90
                for row in languages_for_method
            ),
        }
        latencies = [
            float(row.get("elapsed_seconds") or 0.0) for row in timing
        ]
        calls = [
            float(row.get("generation_calls") or 0.0) for row in timing
        ]
        candidates.append(
            {
                "method": method,
                **pooled,
                "minimum_model_language_coverage": min(
                    (
                        float(row["call_coverage"] or 0.0)
                        for row in cells_for_method
                    ),
                    default=0.0,
                ),
                "minimum_model_language_emitted_calls": min(
                    (int(row["emitted_calls"]) for row in cells_for_method),
                    default=0,
                ),
                "minimum_model_pooled_precision": min(
                    (
                        float(row["accepted_call_exact_precision"])
                        for row in models_for_method
                        if row["accepted_call_exact_precision"] is not None
                    ),
                    default=None,
                ),
                "minimum_language_pooled_precision": min(
                    (
                        float(row["accepted_call_exact_precision"])
                        for row in languages_for_method
                        if row["accepted_call_exact_precision"] is not None
                    ),
                    default=None,
                ),
                "mean_generation_calls": (
                    sum(calls) / len(calls) if calls else None
                ),
                "p50_latency_seconds": _percentile(latencies, 0.50),
                "p95_latency_seconds": _percentile(latencies, 0.95),
                "stage_failures": dict(sorted(stage_counts[method].items())),
                "gates": gates,
                "hard_gates_passed": all(gates.values()),
            }
        )

    preference = {
        "tap_r_qa_active_slots_consensus": 0,
        "tap_r_qa_active_slots_single": 1,
        "tap_r_qa_all_slots_single": 2,
    }
    passing = [row for row in candidates if row["hard_gates_passed"]]
    passing.sort(
        key=lambda row: (
            -float(row["safe_utility_lambda_4"]),
            -float(row["official_ast_accuracy"]),
            -float(row["minimum_model_language_coverage"]),
            float(row["mean_generation_calls"]),
            float(row["p95_latency_seconds"]),
            preference[str(row["method"])],
            str(row["method"]),
        )
    )
    chosen = str(passing[0]["method"]) if passing else None

    comparisons: list[dict[str, Any]] = []
    for method in QA_CANDIDATES:
        candidate = pooled_lookup.get(method, _raw_metrics([]))
        for control in ("full_tap_b2", "tap_r_surface_active_single"):
            baseline = pooled_lookup.get(control, _raw_metrics([]))
            comparisons.append(
                {
                    "candidate": method,
                    "control": control,
                    "official_ast_accuracy_delta": (
                        float(candidate["official_ast_accuracy"] or 0.0)
                        - float(baseline["official_ast_accuracy"] or 0.0)
                    ),
                    "call_coverage_delta": (
                        float(candidate["call_coverage"] or 0.0)
                        - float(baseline["call_coverage"] or 0.0)
                    ),
                    "accepted_call_exact_precision_delta": (
                        None
                        if candidate["accepted_call_exact_precision"] is None
                        or baseline["accepted_call_exact_precision"] is None
                        else float(candidate["accepted_call_exact_precision"])
                        - float(baseline["accepted_call_exact_precision"])
                    ),
                    "safe_utility_lambda_4_delta": (
                        float(candidate["safe_utility_lambda_4"] or 0.0)
                        - float(baseline["safe_utility_lambda_4"] or 0.0)
                    ),
                }
            )

    return {
        "schema_version": QA_HYBRID_DESIGN_ANALYSIS_VERSION,
        "scope": "development_design_sha256_ranks_3_through_48",
        "coefficient_eligible": False,
        "reporting_label": QA_EVIDENCE_SYSTEM_LABEL,
        "case_count": len(cases),
        "prediction_count": len(predictions),
        "expected_prediction_count": expected_count,
        "models": models,
        "methods": methods,
        "languages": sorted(
            {
                str(row.get("metadata", {}).get("language"))
                for row in cases
            }
        ),
        "engineering_gates": engineering,
        "engineering_passed": all(engineering.values()),
        "verifier": {
            "artifact_sha256": verifier_sha256,
            "rows": len(verifier_rows),
            "admitted": sum(
                bool(row.get("admitted")) for row in verifier_rows
            ),
            "validation_failures": verifier_failures,
            "input_truncations": sum(
                bool(row.get("input_truncated")) for row in verifier_rows
            ),
        },
        "raw_method_summary": by_method,
        "model_method_summary": by_model_method,
        "language_method_summary": by_language_method,
        "model_language_method_summary": by_cell,
        "selection_candidates": candidates,
        "paired_control_comparisons": comparisons,
        "chosen_method": chosen,
        "confirmation_permitted": chosen is not None,
        "holdout_access_permitted": False,
        "official_error_types": dict(
            sorted(
                Counter(
                    row.get("error_type")
                    for row in details
                    if row.get("error_type")
                ).items()
            )
        ),
    }


def write_qa_hybrid_design_outputs(
    report: dict[str, Any],
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = report["selection_candidates"]
    csv_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"gates", "stage_failures"}
        }
        for row in rows
    ]
    with (output / "method_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fields = list(csv_rows[0]) if csv_rows else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        if csv_rows:
            writer.writeheader()
            writer.writerows(csv_rows)

    lines = [
        "# MASSIVE-Agents QA-Evidence V7 Design Selection",
        "",
        "- Scope: development SHA ranks 3-48; method selection only",
        f"- System: `{QA_EVIDENCE_SYSTEM_LABEL}`",
        f"- Engineering gates: {'PASS' if report['engineering_passed'] else 'FAIL'}",
        f"- Chosen method: {report['chosen_method'] or 'none'}",
        f"- Confirmation permitted: {report['confirmation_permitted']}",
        "- Holdout access permitted: False",
        "",
        "| Method | Correct/Emitted | AST | Coverage | Raw precision | Utility | Gate |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        precision = row["accepted_call_exact_precision"]
        precision_text = "undefined" if precision is None else f"{precision:.1%}"
        lines.append(
            "| {method} | {correct}/{emitted} | {ast:.1%} | {coverage:.1%} | "
            "{precision} | {utility:.3f} | {gate} |".format(
                method=row["method"],
                correct=row["correct_emitted_calls"],
                emitted=row["emitted_calls"],
                ast=float(row["official_ast_accuracy"] or 0.0),
                coverage=float(row["call_coverage"] or 0.0),
                precision=precision_text,
                utility=float(row["safe_utility_lambda_4"] or 0.0),
                gate="PASS" if row["hard_gates_passed"] else "FAIL",
            )
        )
    (output / "RESULTS.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

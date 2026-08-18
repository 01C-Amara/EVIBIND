from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl


V2_CANDIDATES = (
    "tap_r_retrieve_pointer_single",
    "tap_r_retrieve_pointer_consensus",
    "tap_r_retrieve_pointer_consensus_top1",
)
EXPECTED_MODELS = (
    "LiquidAI/LFM2.5-8B-A1B",
    "Qwen/Qwen3-1.7B",
)


def _raw_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    emitted = sum(bool(row.get("emitted_call")) for row in rows)
    correct = sum(bool(row.get("official_ast_correct")) for row in rows)
    incorrect = sum(bool(row.get("incorrect_emitted_call")) for row in rows)
    return {
        "all_cases": n,
        "emitted_calls": emitted,
        "correct_emitted_calls": correct,
        "incorrect_emitted_calls": incorrect,
        "official_ast_accuracy": correct / n if n else None,
        "accepted_call_exact_precision": (
            correct / emitted if emitted else None
        ),
        "call_coverage": emitted / n if n else None,
        "incorrect_emitted_calls_per_100": (
            100.0 * incorrect / n if n else None
        ),
        "safe_utility_lambda_4": (
            (correct - 4 * incorrect) / n if n else None
        ),
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _group_metrics(
    details: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[tuple(str(row.get(field)) for field in fields)].append(row)
    return [
        {
            **dict(zip(fields, key)),
            **_raw_metrics(rows),
        }
        for key, rows in sorted(grouped.items())
    ]


def analyze_v2_design(
    cases_path: str | Path,
    predictions_path: str | Path,
    timings_path: str | Path,
    official_details_path: str | Path,
    certificate_summary_path: str | Path,
    discipline_failures_path: str | Path,
) -> dict[str, Any]:
    cases = read_jsonl(cases_path)
    predictions = read_jsonl(predictions_path)
    timings = read_jsonl(timings_path)
    details = read_jsonl(official_details_path)
    certificate = json.loads(Path(certificate_summary_path).read_text())
    discipline = read_jsonl(discipline_failures_path)
    case_ids = {str(row["case_id"]) for row in cases}
    models = sorted({str(row.get("model_id")) for row in predictions})
    methods = sorted({str(row.get("method")) for row in predictions})
    expected_count = len(cases) * len(EXPECTED_MODELS) * len(V2_CANDIDATES)
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
    ranking_by_case: dict[str, set[str]] = defaultdict(set)
    for row in predictions:
        ranking_by_case[str(row.get("case_id"))].add(
            str(row.get("ranking_sha256"))
        )
    accepted_predictions = [
        row
        for row in predictions
        if isinstance(row.get("prediction"), dict)
        and row["prediction"].get("mode") == "call"
    ]
    required_provenance = (
        "retrieve_pointer_version",
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
    engineering = {
        "prediction_row_count": len(predictions) == expected_count,
        "official_row_count": len(details) == expected_count,
        "timing_row_count": len(timings) == expected_count,
        "no_duplicate_prediction_cells": len(keys) == len(set(keys)),
        "no_duplicate_timing_cells": len(timing_keys) == len(set(timing_keys)),
        "all_case_ids_known": all(
            str(row.get("case_id")) in case_ids for row in predictions
        ),
        "exact_models": models == sorted(EXPECTED_MODELS),
        "exact_methods": methods == sorted(V2_CANDIDATES),
        "no_runner_errors": not any(row.get("runner_error") for row in predictions),
        "no_visible_thinking": not any(
            bool(row.get("thinking_marker_detected")) for row in predictions
        ),
        "thinking_off": all(row.get("thinking_mode") == "off" for row in predictions),
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
        "certificate_all_rows": int(certificate.get("rows", -1)) == expected_count,
        "certificate_replay": int(certificate.get("failed", -1)) == 0,
        "accepted_certificate_count_matches": int(
            certificate.get("accepted_calls", -1)
        )
        == len(accepted_predictions),
        "coefficient_discipline": not discipline,
        "complete_provenance": all(
            all(row.get(field) is not None for field in required_provenance)
            for row in predictions
        ),
        "one_ranking_per_case": all(
            len(values) == 1 and "None" not in values
            for values in ranking_by_case.values()
        )
        and set(ranking_by_case) == case_ids,
        "one_ranking_artifact": len(
            {row.get("ranking_artifact_sha256") for row in predictions}
        )
        == 1,
        "call_only_election": all(
            row.get("response_metadata", {}).get("call_only_tool_election") is True
            and row.get("response_metadata", {}).get("no_call_election_option") is False
            for row in predictions
        ),
        "finite_literal_invariant_for_calls": all(
            row.get("response_metadata", {}).get(
                "no_generated_action_critical_literals"
            )
            is True
            for row in accepted_predictions
        ),
    }

    by_method = _group_metrics(details, ("method",))
    by_model_method = _group_metrics(details, ("model_id", "method"))
    by_language_method = _group_metrics(details, ("language", "method"))
    by_cell = _group_metrics(details, ("model_id", "language", "method"))
    timing_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in timings:
        timing_groups[str(row.get("method"))].append(row)
    stage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in predictions:
        method = str(row.get("method"))
        failure = row.get("response_metadata", {}).get("controller_stage_failure")
        stage_counts[method][str(failure or "accepted_call")] += 1

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

    candidates: list[dict[str, Any]] = []
    for method in V2_CANDIDATES:
        pooled = pooled_lookup.get(method, _raw_metrics([]))
        models_for_method = model_rows.get(method, [])
        languages_for_method = language_rows.get(method, [])
        cells_for_method = cell_rows.get(method, [])
        timing = timing_groups.get(method, [])
        precision = pooled["accepted_call_exact_precision"]
        coverage = pooled["call_coverage"]
        gates = {
            "engineering": all(engineering.values()),
            "raw_pooled_precision": precision is not None and precision >= 0.95,
            "raw_pooled_coverage": coverage is not None and coverage >= 0.50,
            "minimum_model_language_coverage": bool(cells_for_method)
            and all(float(row["call_coverage"] or 0.0) >= 0.25 for row in cells_for_method),
            "minimum_emitted_calls_per_model_language": bool(cells_for_method)
            and all(int(row["emitted_calls"]) >= 12 for row in cells_for_method),
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
        latencies = [float(row.get("elapsed_seconds") or 0.0) for row in timing]
        calls = [float(row.get("generation_calls") or 0.0) for row in timing]
        candidates.append(
            {
                "method": method,
                **pooled,
                "minimum_model_language_coverage": min(
                    (float(row["call_coverage"] or 0.0) for row in cells_for_method),
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
        "tap_r_retrieve_pointer_consensus": 0,
        "tap_r_retrieve_pointer_consensus_top1": 1,
        "tap_r_retrieve_pointer_single": 2,
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
    return {
        "schema_version": "tapbench.massive_agents_development_analysis.v2",
        "scope": "development_design_ranks_1_through_48",
        "coefficient_eligible": False,
        "case_count": len(cases),
        "prediction_count": len(predictions),
        "expected_prediction_count": expected_count,
        "models": models,
        "methods": methods,
        "languages": sorted(
            {str(row.get("metadata", {}).get("language")) for row in cases}
        ),
        "engineering_gates": engineering,
        "engineering_passed": all(engineering.values()),
        "raw_method_summary": by_method,
        "model_method_summary": by_model_method,
        "language_method_summary": by_language_method,
        "model_language_method_summary": by_cell,
        "selection_candidates": candidates,
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


def write_v2_outputs(report: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = report["selection_candidates"]
    csv_rows = [
        {key: value for key, value in row.items() if key not in {"gates", "stage_failures"}}
        for row in rows
    ]
    with (output / "method_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = list(csv_rows[0]) if csv_rows else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        if csv_rows:
            writer.writeheader()
            writer.writerows(csv_rows)

    lines = [
        "# MASSIVE-Agents Retrieve-Pointer Design Selection",
        "",
        "- Scope: development SHA ranks 1-48; method selection only",
        f"- Engineering gates: {'PASS' if report['engineering_passed'] else 'FAIL'}",
        f"- Chosen method: {report['chosen_method'] or 'none'}",
        f"- Confirmation permitted: {report['confirmation_permitted']}",
        "- Holdout access permitted: False",
        "",
        "| Method | Correct/Emitted | AST | Coverage | Raw precision | Utility | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|",
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
        "\n".join(lines) + "\n", encoding="utf-8"
    )

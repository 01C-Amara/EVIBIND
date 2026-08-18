from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl


REPORT_VERSION = "tapbench.r2a_model_release_report.v1"
TEP_METHOD = "r2_pointer_tep_tier_ab"
BASELINE_METHODS = ("r2_literal_generation", "r2_pointer_unrestricted")


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(bool(row.get(field)) for row in rows) / len(rows)


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_r2a_model_report(
    scores: list[dict[str, Any]],
    timings: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    discipline_failures: list[dict[str, Any]],
    *,
    case_count: int,
    expected_model_count: int,
    expected_condition_count: int,
    expected_seed_count: int,
    context_window: int,
) -> dict[str, Any]:
    expected_rows = case_count * expected_model_count * expected_condition_count * expected_seed_count
    keys = [
        (row.get("case_id"), row.get("model_id"), row.get("method"), row.get("seed"))
        for row in scores
    ]
    duplicate_rows = len(keys) - len(set(keys))
    runner_errors = sum(bool(row.get("runner_error")) for row in predictions)
    thinking_markers = sum(bool(row.get("thinking_marker_detected")) for row in predictions)
    length_stops = sum(row.get("finish_reason") == "length" for row in timings)
    context_truncations = sum(bool(row.get("context_truncated")) for row in timings)
    prompt_tokens = [int(row.get("prompt_tokens") or 0) for row in timings]
    generation_rates = [
        float(row["generated_tokens_per_second"])
        for row in timings
        if row.get("generated_tokens_per_second") is not None
    ]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        grouped[(str(row.get("model_id")), str(row.get("method")))].append(row)

    aggregates: list[dict[str, Any]] = []
    for (model_id, method), rows in sorted(grouped.items()):
        aggregates.append(
            {
                "model_id": model_id,
                "method": method,
                "n": len(rows),
                "execution_success": _rate(rows, "execution_success"),
                "fabrication": _rate(rows, "fabrication"),
                "format_valid": _rate(rows, "format_valid"),
                "mode_correct": _rate(rows, "mode_correct"),
                "tool_correct": _rate(rows, "tool_correct"),
            }
        )

    by_model_method = {(row["model_id"], row["method"]): row for row in aggregates}
    models = sorted({row["model_id"] for row in aggregates})
    directional_checks: list[dict[str, Any]] = []
    for model_id in models:
        tep = by_model_method.get((model_id, TEP_METHOD))
        baselines = [by_model_method.get((model_id, method)) for method in BASELINE_METHODS]
        complete = tep is not None and all(row is not None for row in baselines)
        execution_pass = bool(
            complete
            and tep["execution_success"] > max(row["execution_success"] for row in baselines if row is not None)
        )
        fabrication_pass = bool(
            complete
            and all(tep["fabrication"] <= row["fabrication"] for row in baselines if row is not None)
        )
        directional_checks.append(
            {
                "model_id": model_id,
                "complete": complete,
                "tep_execution_success": None if tep is None else tep["execution_success"],
                "best_baseline_execution_success": None
                if not complete
                else max(row["execution_success"] for row in baselines if row is not None),
                "tep_fabrication": None if tep is None else tep["fabrication"],
                "lowest_baseline_fabrication": None
                if not complete
                else min(row["fabrication"] for row in baselines if row is not None),
                "execution_direction_passed": execution_pass,
                "fabrication_noninferiority_passed": fabrication_pass,
                "passed": execution_pass and fabrication_pass,
            }
        )

    observed_conditions = {row.get("method") for row in scores}
    observed_seeds = {row.get("seed") for row in scores}
    gates = {
        "score_row_completeness": len(scores) == expected_rows,
        "prediction_row_completeness": len(predictions) == expected_rows,
        "timing_row_completeness": len(timings) == expected_rows,
        "unique_score_keys": duplicate_rows == 0,
        "expected_models_observed": len(models) == expected_model_count,
        "expected_conditions_observed": len(observed_conditions) == expected_condition_count,
        "expected_seeds_observed": len(observed_seeds) == expected_seed_count,
        "runner_errors_absent": runner_errors == 0,
        "discipline_failures_absent": len(discipline_failures) == 0,
        "thinking_markers_absent": thinking_markers == 0,
        "length_stops_absent": length_stops == 0,
        "context_truncations_absent": context_truncations == 0,
        "context_window_sufficient": max(prompt_tokens, default=0) < context_window,
        "format_validity_complete": all(row["format_valid"] == 1.0 for row in aggregates),
        "per_model_directional_release": len(directional_checks) == expected_model_count
        and all(row["passed"] for row in directional_checks),
    }
    return {
        "report_version": REPORT_VERSION,
        "design": {
            "case_count": case_count,
            "expected_model_count": expected_model_count,
            "expected_condition_count": expected_condition_count,
            "expected_seed_count": expected_seed_count,
            "expected_rows": expected_rows,
            "context_window": context_window,
        },
        "integrity": {
            "score_rows": len(scores),
            "prediction_rows": len(predictions),
            "timing_rows": len(timings),
            "duplicate_score_keys": duplicate_rows,
            "runner_errors": runner_errors,
            "discipline_failures": len(discipline_failures),
            "thinking_markers": thinking_markers,
            "length_stops": length_stops,
            "context_truncations": context_truncations,
            "max_prompt_tokens": max(prompt_tokens, default=0),
            "generated_tokens_per_second_p50": _percentile(generation_rates, 0.50),
            "generated_tokens_per_second_p95": _percentile(generation_rates, 0.95),
        },
        "aggregates": aggregates,
        "directional_checks": directional_checks,
        "release_decision": {"passed": all(gates.values()), "gates": gates},
    }


def write_r2a_model_report(
    scores_path: str | Path,
    timings_path: str | Path,
    predictions_path: str | Path,
    discipline_failures_path: str | Path,
    cases_path: str | Path,
    output_path: str | Path,
    *,
    expected_model_count: int,
    expected_condition_count: int,
    expected_seed_count: int,
    context_window: int,
) -> dict[str, Any]:
    report = build_r2a_model_report(
        read_jsonl(scores_path),
        read_jsonl(timings_path),
        read_jsonl(predictions_path),
        read_jsonl(discipline_failures_path),
        case_count=len(read_jsonl(cases_path)),
        expected_model_count=expected_model_count,
        expected_condition_count=expected_condition_count,
        expected_seed_count=expected_seed_count,
        context_window=context_window,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

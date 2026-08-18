from __future__ import annotations

import json

from tapbench.io import write_jsonl
from tapbench.r1_report import write_r1_report


def test_full_report_counts_unique_cases_models_and_dynamic_compute(tmp_path) -> None:
    initial = tmp_path / "initial.jsonl"
    best = tmp_path / "best.jsonl"
    tapr = tmp_path / "tapr.json"
    calibrated = tmp_path / "cal.json"
    initial_timings = tmp_path / "initial_timings.jsonl"
    best_timings = tmp_path / "best_timings.jsonl"
    iterations = tmp_path / "iterations.jsonl"
    output = tmp_path / "report.json"
    output_csv = tmp_path / "report.csv"

    rows = []
    for case_id in ("full_R1_typed_resolution_00000", "full_R1_typed_resolution_00001"):
        for model_id in ("m1", "m2"):
            rows.append({
                "case_id": case_id, "model_id": model_id, "method": "full_tap_b2",
                "task_kind": "call", "execution_success": True, "fabrication": False, "format_valid": True,
            })
    write_jsonl(initial, rows)
    write_jsonl(best, [{**row, "method": "best_of_n_budget_matched"} for row in rows])
    write_jsonl(initial_timings, [{"method": "full_tap_b2", "elapsed_seconds": 1.0} for _ in rows])
    write_jsonl(best_timings, [{"method": "best_of_n_budget_matched", "elapsed_seconds": 2.0, "generation_calls": 2} for _ in rows])
    write_jsonl(iterations, [])
    group = {
        "n": 4, "safe_resolution_rate": 1.0, "unsafe_fabrication_rate": 0.0,
        "clarify_accuracy": 1.0, "accepted_call_precision": 1.0, "escalation_rate": 0.0,
        "non_escalated_coverage": 1.0, "mean_generation_calls": 1.0, "mean_validation_rounds": 1.0,
    }
    tapr.write_text(json.dumps({"groups": [{"method": "tap_r_no_calibrator", **group}]}))
    calibrated.write_text(json.dumps({"groups": [{"method": "tap_r_three_way", **group}]}))

    report = write_r1_report(initial, best, tapr, calibrated, initial_timings, best_timings, iterations, output, output_csv)
    assert report["design"]["scope"] == "full"
    assert report["design"]["case_count"] == 2
    assert report["design"]["model_count"] == 2
    assert report["design"]["scored_rows_per_condition"] == 4
    assert report["compute_ratio_bestof_to_onepass"] == 2.0

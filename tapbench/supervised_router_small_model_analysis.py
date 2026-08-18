from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_jsonl, write_jsonl
from .supervised_router_qa_analysis import (
    _accepted_metrics,
    paired_comparison,
    slot_error_rows,
)


BRIDGE_ANALYSIS_VERSION = (
    "tapbench.supervised_router_small_model_analysis.v1"
)
BASELINE_METHOD = "tap_r_qa_active_slots_single"
BRIDGE_ALL_METHOD = "tap_r_supervised_router_small_model_slots_qa_all"
BRIDGE_DEV95_METHOD = "tap_r_supervised_router_small_model_slots_qa_dev95"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _timing_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = np.asarray(
        [float(row["elapsed_seconds"]) for row in rows], dtype=np.float64
    )
    rates = np.asarray(
        [
            float(row["generated_tokens_per_second"])
            for row in rows
            if row.get("generated_tokens_per_second") is not None
        ],
        dtype=np.float64,
    )
    return {
        "n": len(rows),
        "actual_model_calls": sum(
            int(row.get("generation_calls") or 0) for row in rows
        ),
        "elapsed_seconds_total": float(np.sum(elapsed)) if len(elapsed) else 0.0,
        "elapsed_seconds_p50": float(np.quantile(elapsed, 0.5)) if len(elapsed) else None,
        "elapsed_seconds_p95": float(np.quantile(elapsed, 0.95)) if len(elapsed) else None,
        "generated_tokens_per_second_p50": float(np.quantile(rates, 0.5)) if len(rates) else None,
        "generated_tokens_per_second_p95": float(np.quantile(rates, 0.95)) if len(rates) else None,
    }


def analyze(
    *,
    cases_path: str | Path,
    gold_path: str | Path,
    baseline_predictions_path: str | Path,
    baseline_details_path: str | Path,
    bridge_predictions_path: str | Path,
    bridge_details_path: str | Path,
    bridge_timings_path: str | Path,
    certificate_summary_path: str | Path,
    deterministic_details_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(cases_path)
    gold = read_jsonl(gold_path)
    baseline_predictions = [
        row
        for row in read_jsonl(baseline_predictions_path)
        if row.get("method") == BASELINE_METHOD
    ]
    bridge_predictions = read_jsonl(bridge_predictions_path)
    baseline_details = [
        row
        for row in read_jsonl(baseline_details_path)
        if row.get("method") == BASELINE_METHOD
    ]
    bridge_details = read_jsonl(bridge_details_path)
    deterministic_details = read_jsonl(deterministic_details_path)
    timings = read_jsonl(bridge_timings_path)
    certificate_summary = json.loads(
        Path(certificate_summary_path).read_text(encoding="utf-8")
    )
    if certificate_summary.get("failed") or not certificate_summary.get("rows"):
        raise ValueError("bridge certificate audit did not pass on nonzero rows")
    if any(row.get("runner_error") for row in bridge_predictions):
        raise ValueError("bridge predictions contain runner errors")
    if any(bool(row.get("thinking_marker_detected")) for row in bridge_predictions):
        raise ValueError("bridge predictions contain visible thinking markers")

    languages = {
        str(case["case_id"]): str(case.get("metadata", {}).get("language"))
        for case in cases
    }
    model_ids = sorted({str(row["model_id"]) for row in bridge_details})
    comparisons: list[dict[str, Any]] = []
    for model_id in model_ids:
        left = [
            row for row in baseline_details if row["model_id"] == model_id
        ]
        for right_method in (BRIDGE_ALL_METHOD, BRIDGE_DEV95_METHOD):
            right = [
                row
                for row in bridge_details
                if row["model_id"] == model_id
                and row["method"] == right_method
            ]
            result = paired_comparison(left, right, languages)
            comparisons.append(
                {
                    "model_id": model_id,
                    "left_method": BASELINE_METHOD,
                    "right_method": right_method,
                    **result,
                }
            )

    pooled_languages: dict[str, str] = {}
    pooled_left: list[dict[str, Any]] = []
    pooled_right: dict[str, list[dict[str, Any]]] = {
        BRIDGE_ALL_METHOD: [],
        BRIDGE_DEV95_METHOD: [],
    }
    for row in baseline_details:
        pair_id = f"{row['model_id']}::{row['case_id']}"
        pooled_languages[pair_id] = languages[str(row["case_id"])]
        pooled_left.append({**row, "case_id": pair_id})
    for row in bridge_details:
        method = str(row["method"])
        if method not in pooled_right:
            continue
        pair_id = f"{row['model_id']}::{row['case_id']}"
        pooled_right[method].append({**row, "case_id": pair_id})
    for right_method, rows in pooled_right.items():
        result = paired_comparison(
            pooled_left,
            rows,
            pooled_languages,
            seed=20260716,
        )
        comparisons.append(
            {
                "model_id": "pooled_two_models",
                "left_method": BASELINE_METHOD,
                "right_method": right_method,
                **result,
            }
        )
    _write_csv(output / "paired_comparisons.csv", comparisons)

    metric_rows: list[dict[str, Any]] = []
    all_details = baseline_details + bridge_details + deterministic_details
    for model_id, method in sorted(
        {(str(row["model_id"]), str(row["method"])) for row in all_details}
    ):
        subset = [
            row
            for row in all_details
            if row["model_id"] == model_id and row["method"] == method
        ]
        metric_rows.append(
            {"model_id": model_id, "method": method, **_accepted_metrics(subset)}
        )
    _write_csv(output / "method_metrics.csv", metric_rows)

    language_rows: list[dict[str, Any]] = []
    for model_id, method, language in sorted(
        {
            (str(row["model_id"]), str(row["method"]), str(row["language"]))
            for row in baseline_details + bridge_details
        }
    ):
        subset = [
            row
            for row in baseline_details + bridge_details
            if row["model_id"] == model_id
            and row["method"] == method
            and row["language"] == language
        ]
        language_rows.append(
            {
                "model_id": model_id,
                "method": method,
                "language": language,
                **_accepted_metrics(subset),
            }
        )
    _write_csv(output / "language_metrics.csv", language_rows)

    errors = slot_error_rows(
        cases,
        gold,
        baseline_predictions + bridge_predictions,
    )
    write_jsonl(output / "slot_errors.jsonl", errors)
    error_rows: list[dict[str, Any]] = []
    for model_id, method in sorted(
        {(str(row["model_id"]), str(row["method"])) for row in errors}
    ):
        counts = Counter(
            str(row["error_type"])
            for row in errors
            if row["model_id"] == model_id and row["method"] == method
        )
        for error_type, count in sorted(counts.items()):
            error_rows.append(
                {
                    "model_id": model_id,
                    "method": method,
                    "error_type": error_type,
                    "count": count,
                }
            )
    _write_csv(output / "slot_error_summary.csv", error_rows)

    timing_report = {
        f"{model_id}::{method}": _timing_metrics(
            [
                row
                for row in timings
                if row["model_id"] == model_id and row["method"] == method
            ]
        )
        for model_id, method in sorted(
            {(str(row["model_id"]), str(row["method"])) for row in timings}
        )
    }
    report = {
        "schema_version": BRIDGE_ANALYSIS_VERSION,
        "analysis_status": "predeclared_post_result_development_only",
        "confirmation_authorized": False,
        "interpretation_limit": (
            "The intent router uses official MASSIVE train labels; results isolate the "
            "small model's active-slot role inside a benchmark-supervised controller "
            "and are not benchmark-independent transfer evidence."
        ),
        "model_ids": model_ids,
        "paired_comparisons": comparisons,
        "method_metrics": metric_rows,
        "language_metrics": language_rows,
        "timing_metrics": timing_report,
        "certificate_audit": certificate_summary,
        "thinking_markers": 0,
        "runner_errors": 0,
        "inputs": {
            "cases_sha256": _sha256(cases_path),
            "gold_sha256": _sha256(gold_path),
            "baseline_predictions_sha256": _sha256(baseline_predictions_path),
            "baseline_details_sha256": _sha256(baseline_details_path),
            "bridge_predictions_sha256": _sha256(bridge_predictions_path),
            "bridge_details_sha256": _sha256(bridge_details_path),
            "bridge_timings_sha256": _sha256(bridge_timings_path),
            "certificate_summary_sha256": _sha256(certificate_summary_path),
            "deterministic_details_sha256": _sha256(deterministic_details_path),
        },
        "artifacts": {
            name: str((output / name).resolve())
            for name in (
                "paired_comparisons.csv",
                "method_metrics.csv",
                "language_metrics.csv",
                "slot_errors.jsonl",
                "slot_error_summary.csv",
            )
        },
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the supervised-router/small-model active-slot bridge."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--baseline-details", required=True)
    parser.add_argument("--bridge-predictions", required=True)
    parser.add_argument("--bridge-details", required=True)
    parser.add_argument("--bridge-timings", required=True)
    parser.add_argument("--certificate-summary", required=True)
    parser.add_argument("--deterministic-details", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = analyze(
        cases_path=args.cases,
        gold_path=args.gold,
        baseline_predictions_path=args.baseline_predictions,
        baseline_details_path=args.baseline_details,
        bridge_predictions_path=args.bridge_predictions,
        bridge_details_path=args.bridge_details,
        bridge_timings_path=args.bridge_timings,
        certificate_summary_path=args.certificate_summary,
        deterministic_details_path=args.deterministic_details,
        output_dir=args.output_dir,
    )
    print(json.dumps(report["paired_comparisons"], sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_jsonl, write_jsonl
from .multilingual_retriever import forbidden_paths


ANALYSIS_VERSION = "tapbench.supervised_router_qa_analysis.v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _request(case: dict[str, Any]) -> str:
    return "\n".join(
        str(row.get("content", ""))
        for row in case.get("messages", [])
        if isinstance(row, dict) and row.get("role") == "user"
    )


def _gold_action(row: dict[str, Any]) -> tuple[str, dict[str, list[Any]]]:
    truth = row.get("ground_truth")
    if not isinstance(truth, list) or len(truth) != 1 or not isinstance(truth[0], dict) or len(truth[0]) != 1:
        raise ValueError(f"unsupported MASSIVE gold shape for {row.get('case_id')}")
    tool, arguments = next(iter(truth[0].items()))
    output: dict[str, list[Any]] = {}
    for slot, values in (arguments.items() if isinstance(arguments, dict) else []):
        allowed = values if isinstance(values, list) else [values]
        nonempty = [value for value in allowed if _normalize(value)]
        if nonempty:
            output[str(slot)] = nonempty
    return str(tool), output


def slot_error_rows(
    cases: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    case_by_id = {str(row["case_id"]): row for row in cases}
    gold_by_id = {str(row["case_id"]): row for row in gold_rows}
    output = []
    for prediction in predictions:
        case_id = str(prediction["case_id"])
        case = case_by_id[case_id]
        request = _normalize(_request(case))
        gold_tool, gold_arguments = _gold_action(gold_by_id[case_id])
        action = prediction.get("prediction") if isinstance(prediction.get("prediction"), dict) else {}
        base = {
            "schema_version": ANALYSIS_VERSION,
            "case_id": case_id,
            "language": str(case.get("metadata", {}).get("language")),
            "method": str(prediction.get("method")),
            "model_id": str(prediction.get("model_id")),
            "backend": str(prediction.get("backend")),
        }
        if action.get("mode") != "call":
            output.append({**base, "error_type": "no_call", "slot": None, "gold_value": gold_tool, "predicted_value": action.get("mode"), "derivable": False})
            continue
        predicted_tool = str(action.get("tool"))
        if predicted_tool != gold_tool:
            output.append({**base, "error_type": "wrong_tool", "slot": None, "gold_value": gold_tool, "predicted_value": predicted_tool, "derivable": False})
            continue
        predicted_arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
        for slot, gold_values in gold_arguments.items():
            if slot not in predicted_arguments:
                output.append({**base, "error_type": "missing_expected_slot", "slot": slot, "gold_value": gold_values, "predicted_value": None, "derivable": True})
                continue
            predicted_value = predicted_arguments[slot]
            if _normalize(predicted_value) not in {_normalize(value) for value in gold_values}:
                derivable = bool(_normalize(predicted_value)) and _normalize(predicted_value) in request
                output.append(
                    {
                        **base,
                        "error_type": "wrong_normalized_value" if derivable else "unsupported_fabricated_value",
                        "slot": slot,
                        "gold_value": gold_values,
                        "predicted_value": predicted_value,
                        "derivable": derivable,
                    }
                )
        for slot, predicted_value in predicted_arguments.items():
            if slot in gold_arguments:
                continue
            derivable = bool(_normalize(predicted_value)) and _normalize(predicted_value) in request
            output.append(
                {
                    **base,
                    "error_type": "extra_optional_field" if derivable else "unsupported_fabricated_value",
                    "slot": str(slot),
                    "gold_value": None,
                    "predicted_value": predicted_value,
                    "derivable": derivable,
                }
            )
    return output


def _detail_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    output = {}
    for row in rows:
        key = (str(row["case_id"]), str(row["model_id"]), str(row["method"]))
        if key in output:
            raise ValueError(f"duplicate official detail key: {key}")
        output[key] = row
    return output


def _mcnemar_exact(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(0, min(improved, regressed) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_comparison(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    languages: dict[str, str],
    *,
    bootstrap_replicates: int = 10_000,
    seed: int = 20260716,
) -> dict[str, Any]:
    left_by_id = {str(row["case_id"]): int(bool(row["official_ast_correct"])) for row in left}
    right_by_id = {str(row["case_id"]): int(bool(row["official_ast_correct"])) for row in right}
    ids = sorted(set(left_by_id) & set(right_by_id))
    if set(left_by_id) != set(right_by_id):
        raise ValueError("paired comparison case IDs differ")
    left_values = np.asarray([left_by_id[case_id] for case_id in ids], dtype=np.float64)
    right_values = np.asarray([right_by_id[case_id] for case_id in ids], dtype=np.float64)
    improved = int(np.sum((left_values == 0) & (right_values == 1)))
    regressed = int(np.sum((left_values == 1) & (right_values == 0)))
    rng = np.random.default_rng(seed)
    by_language = {
        language: np.asarray([index for index, case_id in enumerate(ids) if languages[case_id] == language], dtype=np.int64)
        for language in sorted(set(languages.values()))
    }
    differences = np.empty(bootstrap_replicates, dtype=np.float64)
    for replicate in range(bootstrap_replicates):
        sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in by_language.values()])
        differences[replicate] = float(np.mean(right_values[sampled] - left_values[sampled]))
    delta = float(np.mean(right_values - left_values))
    return {
        "n": len(ids),
        "left_accuracy": float(np.mean(left_values)),
        "right_accuracy": float(np.mean(right_values)),
        "absolute_difference": delta,
        "difference_percentage_points": 100.0 * delta,
        "improved_cases": improved,
        "regressed_cases": regressed,
        "unchanged_cases": len(ids) - improved - regressed,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(improved, regressed),
        "stratified_paired_bootstrap_95_ci": [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))],
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
    }


def _accepted_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    emitted = [row for row in rows if row.get("emitted_call")]
    correct = sum(bool(row.get("official_ast_correct")) for row in emitted)
    return {
        "n": len(rows),
        "emitted_calls": len(emitted),
        "call_coverage": len(emitted) / len(rows) if rows else 0.0,
        "correct": correct,
        "execution_accuracy": correct / len(rows) if rows else 0.0,
        "accepted_call_exact_precision": correct / len(emitted) if emitted else None,
        "safe_utility_lambda_4": sum(float(row.get("safe_utility_lambda_4", 0.0)) for row in rows) / len(rows) if rows else 0.0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    *,
    cases_path: str | Path,
    gold_path: str | Path,
    v2_predictions_path: str | Path,
    v2_details_path: str | Path,
    v3_predictions_path: str | Path,
    v3_details_path: str | Path,
    slot_artifact_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(cases_path)
    gold = read_jsonl(gold_path)
    v2_predictions = read_jsonl(v2_predictions_path)
    v3_predictions = read_jsonl(v3_predictions_path)
    slot_artifact = read_jsonl(slot_artifact_path)
    leaks = forbidden_paths(slot_artifact)
    if leaks:
        raise ValueError(f"gold-free slot artifact firewall failed: {leaks[:10]}")
    slot_errors = slot_error_rows(cases, gold, v2_predictions + v3_predictions)
    write_jsonl(output / "slot_errors.jsonl", slot_errors)
    error_summary = []
    methods = sorted({str(row["method"]) for row in v2_predictions + v3_predictions})
    for method in methods:
        subset = [row for row in slot_errors if row["method"] == method]
        counts = Counter(str(row["error_type"]) for row in subset)
        for error_type, count in sorted(counts.items()):
            error_summary.append({"method": method, "error_type": error_type, "count": count})
    _write_csv(output / "slot_error_summary.csv", error_summary)

    v2_details = read_jsonl(v2_details_path)
    v3_details = read_jsonl(v3_details_path)
    languages = {str(case["case_id"]): str(case.get("metadata", {}).get("language")) for case in cases}
    comparisons = []
    pairs = (
        ("tap_r_supervised_router_qa_all", "tap_r_supervised_router_slot_knn_qa_all"),
        ("tap_r_supervised_router_qa_dev95", "tap_r_supervised_router_slot_knn_qa_dev95"),
    )
    for left_method, right_method in pairs:
        left = [row for row in v2_details if row["method"] == left_method]
        right = [row for row in v3_details if row["method"] == right_method]
        comparison = paired_comparison(left, right, languages)
        comparisons.append({"left_method": left_method, "right_method": right_method, **comparison})
    _write_csv(output / "paired_comparisons.csv", comparisons)
    method_metrics = {
        method: _accepted_metrics([row for row in v2_details + v3_details if row["method"] == method])
        for method in methods
    }
    report = {
        "schema_version": ANALYSIS_VERSION,
        "analysis_status": "post_result_exploratory_design_only",
        "confirmation_authorized": False,
        "system_disclosure": "Both v2 and v3 use benchmark-supervised MASSIVE train labels; v3 additionally uses train slot annotations with dev-only hyperparameter selection.",
        "slot_artifact_firewall": {"passed": True, "gold_paths": [], "rows": len(slot_artifact)},
        "method_metrics": method_metrics,
        "paired_comparisons": comparisons,
        "slot_error_counts": {
            method: dict(sorted(Counter(row["error_type"] for row in slot_errors if row["method"] == method).items()))
            for method in methods
        },
        "inputs": {
            "cases_sha256": _sha256(cases_path),
            "gold_sha256": _sha256(gold_path),
            "v2_predictions_sha256": _sha256(v2_predictions_path),
            "v2_details_sha256": _sha256(v2_details_path),
            "v3_predictions_sha256": _sha256(v3_predictions_path),
            "v3_details_sha256": _sha256(v3_details_path),
            "slot_artifact_sha256": _sha256(slot_artifact_path),
        },
        "artifacts": {
            "slot_errors": str((output / "slot_errors.jsonl").resolve()),
            "slot_errors_sha256": _sha256(output / "slot_errors.jsonl"),
            "slot_error_summary": str((output / "slot_error_summary.csv").resolve()),
            "paired_comparisons": str((output / "paired_comparisons.csv").resolve()),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired supervised-router QA improvements and slot failures.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--v2-predictions", required=True)
    parser.add_argument("--v2-details", required=True)
    parser.add_argument("--v3-predictions", required=True)
    parser.add_argument("--v3-details", required=True)
    parser.add_argument("--slot-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = analyze(
        cases_path=args.cases,
        gold_path=args.gold,
        v2_predictions_path=args.v2_predictions,
        v2_details_path=args.v2_details,
        v3_predictions_path=args.v3_predictions,
        v3_details_path=args.v3_details,
        slot_artifact_path=args.slot_artifact,
        output_dir=args.output_dir,
    )
    print(json.dumps(report["paired_comparisons"], sort_keys=True))


if __name__ == "__main__":
    main()

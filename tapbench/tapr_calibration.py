from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl
from .ir import parse_and_normalize_prediction
from .retrieval import rank_tools
from .tapr import TAPR_VERSION, contract_validator_error, score_resolution_predictions
from .validation import validate_action

TAPR_CALIBRATOR_VERSION = "tapbench.tap_r_calibrator.v1"
FEATURE_NAMES = (
    "bias",
    "contract_valid",
    "unsupported_required_count",
    "unsupported_optional_count",
    "contradicted_count",
    "normalized_count",
    "validation_rounds",
    "repeated_slot",
    "sketch_action_agreement",
)


def _feature_dict(case: dict[str, Any], prediction: dict[str, Any]) -> dict[str, float]:
    action, _ = parse_and_normalize_prediction(prediction, case)
    report = contract_validator_error(case, action)
    errors = report["errors"]
    ledger = report["evidence_ledger"]
    resolution = prediction.get("resolution", {}) if isinstance(prediction.get("resolution"), dict) else {}
    predicted_tool = action.get("tool") if isinstance(action, dict) else None
    ranked = rank_tools(case, arm="tfidf_char")
    sketch_tool = ranked[0].get("canonical_name") if ranked else None
    return {
        "bias": 1.0,
        "contract_valid": float(bool(report["contract_valid"])),
        "unsupported_required_count": float(
            sum(error["error_class"] in {"unsupported_required_value", "missing_required_slot_no_evidence"} for error in errors)
        ),
        "unsupported_optional_count": float(sum(error["error_class"] == "unsupported_optional_field" for error in errors)),
        "contradicted_count": float(sum(row["evidence_label"] == "contradicted" for row in ledger)),
        "normalized_count": float(sum(row["evidence_label"] == "normalized" for row in ledger)),
        "validation_rounds": float(resolution.get("validation_rounds", 1)),
        "repeated_slot": float(any(error["error_class"] == "repeated_repair_same_slot" for error in errors)),
        "sketch_action_agreement": float(predicted_tool is not None and predicted_tool == sketch_tool),
    }


def _vector(features: dict[str, float]) -> list[float]:
    return [features[name] for name in FEATURE_NAMES]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _fit(rows: list[dict[str, Any]], *, steps: int = 800, lr: float = 0.05, l2: float = 0.002) -> list[float]:
    weights = [0.0] * len(FEATURE_NAMES)
    if not rows:
        return weights
    for _ in range(steps):
        gradient = [0.0] * len(weights)
        for row in rows:
            x = _vector(row["features"])
            prediction = _sigmoid(sum(weight * value for weight, value in zip(weights, x, strict=True)))
            error = prediction - float(row["target"])
            for index, value in enumerate(x):
                gradient[index] += error * value
        for index in range(len(weights)):
            penalty = 0.0 if index == 0 else l2 * weights[index]
            weights[index] -= lr * (gradient[index] / len(rows) + penalty)
    return weights


def _predict(features: dict[str, float], weights: list[float]) -> float:
    return _sigmoid(sum(weight * value for weight, value in zip(weights, _vector(features), strict=True)))


def _threshold(rows: list[dict[str, Any]], target_precision: float) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    accepted = 0
    correct = 0
    best = None
    for row in ordered:
        accepted += 1
        correct += int(row["target"])
        precision = correct / accepted
        if precision >= target_precision:
            best = {
                "threshold": float(row["score"]),
                "precision": precision,
                "coverage": accepted / len(rows) if rows else 0.0,
                "accepted": accepted,
                "correct": correct,
                "n": len(rows),
            }
    return best or {"threshold": 1.0, "precision": None, "coverage": 0.0, "accepted": 0, "correct": 0, "n": len(rows)}


def _escalate(action: dict[str, Any] | None, reason: str) -> dict[str, Any]:
    return {
        "mode": "refuse",
        "tool": None,
        "arguments": {},
        "payload": {
            "resolution_terminal": "escalate",
            "reason": reason,
            "rejected_action": action,
        },
    }


def apply_three_way_policy(
    prediction: dict[str, Any],
    *,
    score: float,
    threshold: float,
    fold: str,
    features: dict[str, float],
) -> dict[str, Any]:
    output = deepcopy(prediction)
    action = output.get("prediction") if isinstance(output.get("prediction"), dict) else None
    resolution = dict(output.get("resolution", {})) if isinstance(output.get("resolution"), dict) else {}
    terminal = str(resolution.get("terminal_state", "escalate"))
    decision = "accept"
    if terminal == "call" and score < threshold:
        decision = "escalate"
        output["prediction"] = _escalate(action, "accepted-call calibrator below threshold")
        resolution["terminal_state"] = "escalate"
        resolution["final_contract_valid"] = False
    elif terminal == "clarify":
        decision = "clarify"
    elif terminal == "escalate":
        decision = "escalate"
    output["method"] = "tap_r_three_way"
    output["resolution"] = resolution
    output["tap_r_calibrator"] = {
        "schema_version": TAPR_CALIBRATOR_VERSION,
        "decision": decision,
        "score": score,
        "threshold": threshold,
        "fold": fold,
        "features": features,
    }
    return output


def calibrate_predictions(
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    target_precision: float = 0.95,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    case_by_id = {case["case_id"]: case for case in cases}
    examples = []
    for index, prediction in enumerate(predictions):
        case = case_by_id[str(prediction["case_id"])]
        action, _ = parse_and_normalize_prediction(prediction, case)
        terminal = str((prediction.get("resolution") or {}).get("terminal_state", ""))
        metrics = validate_action(case, action)
        if terminal == "call":
            examples.append(
                {
                    "index": index,
                    "case_id": case["case_id"],
                    "family": case["family"],
                    "features": _feature_dict(case, prediction),
                    "target": int(bool(metrics["execution_success"]) and not bool(metrics["fabrication"])),
                }
            )

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in examples:
        by_family[str(row["family"])].append(row)
    scored_by_index: dict[int, dict[str, Any]] = {}
    fold_reports = []
    for family in sorted({str(case["family"]) for case in cases}):
        train = [row for row in examples if row["family"] != family]
        test = by_family.get(family, [])
        weights = _fit(train)
        train_scored = [{**row, "score": _predict(row["features"], weights)} for row in train]
        threshold_report = _threshold(train_scored, target_precision)
        for row in test:
            scored_by_index[int(row["index"])] = {
                **row,
                "score": _predict(row["features"], weights),
                "threshold": threshold_report["threshold"],
                "fold": family,
            }
        fold_reports.append(
            {
                "family": family,
                "train_call_rows": len(train),
                "test_call_rows": len(test),
                "threshold": threshold_report,
                "weights": dict(zip(FEATURE_NAMES, weights, strict=True)),
            }
        )

    outputs = []
    calibration_rows = []
    for index, prediction in enumerate(predictions):
        case = case_by_id[str(prediction["case_id"])]
        if index in scored_by_index:
            scored = scored_by_index[index]
            output = apply_three_way_policy(
                prediction,
                score=float(scored["score"]),
                threshold=float(scored["threshold"]),
                fold=str(scored["fold"]),
                features=scored["features"],
            )
            calibration_rows.append({key: value for key, value in scored.items() if key != "features"} | scored["features"])
        else:
            features = _feature_dict(case, prediction)
            output = apply_three_way_policy(
                prediction,
                score=1.0,
                threshold=0.0,
                fold=str(case["family"]),
                features=features,
            )
        outputs.append(output)

    report = {
        "schema_version": "tapbench.tap_r_calibrator_report.v1",
        "calibrator_version": TAPR_CALIBRATOR_VERSION,
        "tap_r_version": TAPR_VERSION,
        "target_precision": target_precision,
        "threshold_policy": "leave-one-family-out_train_only",
        "feature_names": list(FEATURE_NAMES),
        "call_examples": len(examples),
        "folds": fold_reports,
    }
    return outputs, calibration_rows, report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def calibrate_files(
    cases_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
    calibration_csv_path: str | Path,
    scores_path: str | Path,
    summary_path: str | Path,
    report_path: str | Path,
    *,
    target_precision: float = 0.95,
) -> dict[str, Any]:
    cases = read_jsonl(cases_path)
    outputs, calibration_rows, report = calibrate_predictions(cases, read_jsonl(predictions_path), target_precision=target_precision)
    scores, summary = score_resolution_predictions(cases, outputs)
    write_jsonl(output_path, outputs)
    write_jsonl(scores_path, scores)
    _write_csv(Path(calibration_csv_path), calibration_rows)
    for path, payload in ((summary_path, summary), (report_path, report)):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"predictions": len(outputs), "call_examples": len(calibration_rows), "scores": len(scores)}

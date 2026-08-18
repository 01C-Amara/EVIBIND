from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import CALIBRATOR_VERSION
from .io import read_jsonl


FEATURE_NAMES = (
    "bias",
    "format_valid",
    "schema_valid",
    "thinking_marker_detected",
    "finish_reason_length",
    "mode_call",
    "mode_clarify",
    "mode_no_tool",
    "mode_direct_answer",
)


def _bool(row: dict[str, Any], key: str) -> float:
    return 1.0 if bool(row.get(key)) else 0.0


def _features(row: dict[str, Any]) -> list[float]:
    task_kind = str(row.get("task_kind", ""))
    return [
        1.0,
        _bool(row, "format_valid"),
        _bool(row, "schema_valid"),
        _bool(row, "thinking_marker_detected"),
        1.0 if row.get("finish_reason") == "length" else 0.0,
        1.0 if task_kind == "call" else 0.0,
        1.0 if task_kind == "missing_info" else 0.0,
        1.0 if task_kind == "no_tool" else 0.0,
        1.0 if task_kind == "direct_answer" else 0.0,
    ]


def _target(row: dict[str, Any]) -> int:
    return int(bool(row.get("execution_success")) and not bool(row.get("fabrication")))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _fit(rows: list[dict[str, Any]], *, steps: int = 600, lr: float = 0.05, l2: float = 0.001) -> list[float]:
    weights = [0.0] * len(FEATURE_NAMES)
    if not rows:
        return weights
    for _ in range(steps):
        grad = [0.0] * len(weights)
        for row in rows:
            x = _features(row)
            pred = _sigmoid(sum(weight * value for weight, value in zip(weights, x, strict=True)))
            err = pred - _target(row)
            for index, value in enumerate(x):
                grad[index] += err * value
        scale = 1.0 / len(rows)
        for index in range(len(weights)):
            penalty = 0.0 if index == 0 else l2 * weights[index]
            weights[index] -= lr * (grad[index] * scale + penalty)
    return weights


def _predict(row: dict[str, Any], weights: list[float]) -> float:
    return _sigmoid(sum(weight * value for weight, value in zip(weights, _features(row), strict=True)))


def _threshold_at_precision(rows: list[dict[str, Any]], *, precision: float = 0.95) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["calibrator_score"]), reverse=True)
    accepted = 0
    correct = 0
    best: dict[str, Any] | None = None
    for row in ordered:
        accepted += 1
        correct += _target(row)
        observed = correct / accepted
        if observed >= precision:
            best = {
                "threshold": float(row["calibrator_score"]),
                "precision": observed,
                "coverage": accepted / len(rows) if rows else 0.0,
                "accepted": accepted,
                "correct": correct,
                "n": len(rows),
            }
    if best is None:
        return {"threshold": 1.0, "precision": None, "coverage": 0.0, "accepted": 0, "correct": 0, "n": len(rows)}
    return best


def _risk_coverage_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    curve = []
    for threshold in [index / 20 for index in range(21)]:
        accepted = [row for row in rows if float(row["calibrator_score"]) >= threshold]
        correct = sum(_target(row) for row in accepted)
        precision = correct / len(accepted) if accepted else None
        curve.append(
            {
                "threshold": threshold,
                "coverage": len(accepted) / len(rows) if rows else 0.0,
                "precision": precision,
                "risk": None if precision is None else 1.0 - precision,
                "accepted": len(accepted),
                "n": len(rows),
            }
        )
    return curve


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_calibrator(scores_path: str | Path, output_dir: str | Path, *, target_precision: float = 0.95) -> dict[str, Any]:
    rows = read_jsonl(scores_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    families = sorted({str(row.get("family", "unknown")) for row in rows})
    predictions: list[dict[str, Any]] = []
    for family in families:
        train = [row for row in rows if str(row.get("family", "unknown")) != family]
        test = [row for row in rows if str(row.get("family", "unknown")) == family]
        weights = _fit(train or rows)
        for row in test:
            predictions.append(
                {
                    **row,
                    "calibrator_score": _predict(row, weights),
                    "calibrator_fold": family,
                    "calibrator_target": _target(row),
                    "calibrator_version": CALIBRATOR_VERSION,
                }
            )

    global_threshold = _threshold_at_precision(predictions, precision=target_precision)
    per_model = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_model[str(row.get("model_id", "unknown"))].append(row)
    for model_id, model_rows in sorted(by_model.items()):
        per_model.append({"model_id": model_id, **_threshold_at_precision(model_rows, precision=target_precision)})

    curve = _risk_coverage_curve(predictions)
    _write_csv(target_dir / "calibrator_predictions.csv", predictions)
    _write_csv(target_dir / "risk_coverage_curve.csv", curve)
    _write_csv(target_dir / "per_model_thresholds.csv", per_model)
    payload = {
        "schema_version": "tapbench.calibrator_report.v1",
        "calibrator_version": CALIBRATOR_VERSION,
        "scores_path": str(scores_path),
        "target_precision": target_precision,
        "feature_names": list(FEATURE_NAMES),
        "family_disjoint_folds": families,
        "global_threshold": global_threshold,
        "per_model_thresholds": per_model,
    }
    (target_dir / "calibrator_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload

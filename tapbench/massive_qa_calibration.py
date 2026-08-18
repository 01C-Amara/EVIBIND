from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io import read_jsonl, write_jsonl


CALIBRATOR_VERSION = "tapbench.massive_qa_calibrator.v1"
REPORT_VERSION = "tapbench.massive_qa_calibration_report.v1"

# These are the only scalar signals admitted to the calibrator. Gold actions,
# scorer errors, tool identities, and tool families are deliberately absent.
RUNTIME_FEATURE_NAMES = (
    "retriever_election_agreement",
    "tool_agreement",
    "active_slot_agreement",
    "proposal_admitted",
    "risk_gate_passed",
    "final_contract_valid",
    "finish_reason_stop",
    "argument_count",
    "request_char_count",
    "request_whitespace_token_count",
    "generation_calls",
    "completion_tokens",
    "prompt_tokens",
    "context_headroom_tokens_min",
    "generated_tokens_per_second",
    "active_slots_pre_count",
    "active_slots_post_count",
    "selected_surface_count",
    "selected_surface_char_count",
    "verifier_rows_consulted",
    "verifier_null_count",
    "verifier_null_fraction",
    "verifier_admitted_count",
    "verifier_margin_min",
    "verifier_margin_mean",
    "verifier_margin_max",
    "verifier_margin_present",
    "retriever_top1_score",
    "retriever_top1_top2_gap",
    "selected_tool_score",
    "selected_tool_top1_gap",
    "selected_tool_reciprocal_rank",
    "selected_tool_in_top_k",
)
CATEGORICAL_FEATURES = ("method", "model_id", "language")
FORBIDDEN_FEATURE_FRAGMENTS = (
    "gold",
    "official",
    "correct",
    "error_type",
    "decoded_action",
    "tool_family",
    "target",
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if math.isfinite(output) else default


def _bool(value: Any) -> float:
    return float(bool(value))


def _request_text(case: dict[str, Any]) -> str:
    messages = case.get("messages")
    if not isinstance(messages, list):
        return ""
    user_messages = [str(row.get("content", "")) for row in messages if isinstance(row, dict) and row.get("role") == "user"]
    return "\n".join(user_messages)


def _ranking_features(ranking_row: dict[str, Any] | None, selected_tool: str | None) -> dict[str, float]:
    ranking = ranking_row.get("ranking", []) if isinstance(ranking_row, dict) else []
    ranking = [row for row in ranking if isinstance(row, dict)]
    top1_score = _number(ranking[0].get("cosine_score")) if ranking else 0.0
    top2_score = _number(ranking[1].get("cosine_score")) if len(ranking) > 1 else top1_score
    selected_row = next((row for row in ranking if str(row.get("tool")) == str(selected_tool)), None)
    selected_score = _number(selected_row.get("cosine_score")) if selected_row else 0.0
    selected_rank = int(_number(selected_row.get("rank"))) if selected_row else 0
    return {
        "retriever_top1_score": top1_score,
        "retriever_top1_top2_gap": top1_score - top2_score,
        "selected_tool_score": selected_score,
        "selected_tool_top1_gap": top1_score - selected_score if selected_row else 0.0,
        "selected_tool_reciprocal_rank": 1.0 / selected_rank if selected_rank > 0 else 0.0,
        "selected_tool_in_top_k": _bool(selected_row is not None),
    }


def extract_runtime_features(
    case: dict[str, Any],
    prediction: dict[str, Any],
    ranking_row: dict[str, Any] | None,
) -> dict[str, Any]:
    """Extract runtime-only features without accepting a scorer/gold argument."""
    metadata = prediction.get("response_metadata") if isinstance(prediction.get("response_metadata"), dict) else {}
    action = prediction.get("prediction") if isinstance(prediction.get("prediction"), dict) else {}
    arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
    selected_tool = action.get("tool") if isinstance(action.get("tool"), str) else None
    request = _request_text(case)
    margins = [
        _number(row.get("non_null_margin"))
        for row in metadata.get("verifier_decisions", [])
        if isinstance(row, dict) and row.get("non_null_margin") is not None
    ]
    admitted = [row for row in metadata.get("verifier_decisions", []) if isinstance(row, dict) and row.get("admitted")]
    surfaces = metadata.get("selected_surface_values") if isinstance(metadata.get("selected_surface_values"), dict) else {}
    rows_consulted = _number(metadata.get("qa_verifier_rows_consulted"))
    null_count = _number(metadata.get("qa_verifier_null_count"))
    values: dict[str, Any] = {
        "retriever_election_agreement": _bool(metadata.get("retriever_election_agreement")),
        "tool_agreement": _bool(metadata.get("tool_agreement")),
        "active_slot_agreement": _bool(metadata.get("active_slot_agreement")),
        "proposal_admitted": _bool(metadata.get("proposal_admitted")),
        "risk_gate_passed": _bool(metadata.get("risk_gate_passed")),
        "final_contract_valid": _bool(metadata.get("final_contract_valid")),
        "finish_reason_stop": _bool(str(prediction.get("finish_reason", "")) == "stop"),
        "argument_count": float(len(arguments)),
        "request_char_count": float(len(request)),
        "request_whitespace_token_count": float(len(request.split())),
        "generation_calls": _number(metadata.get("generation_calls")),
        "completion_tokens": _number(metadata.get("completion_tokens")),
        "prompt_tokens": _number(metadata.get("prompt_tokens")),
        "context_headroom_tokens_min": _number(metadata.get("context_headroom_tokens_min")),
        "generated_tokens_per_second": _number(metadata.get("generated_tokens_per_second")),
        "active_slots_pre_count": float(len(metadata.get("active_slots_pre_verifier", []) or [])),
        "active_slots_post_count": float(len(metadata.get("active_slots_post_verifier", []) or [])),
        "selected_surface_count": float(len(surfaces)),
        "selected_surface_char_count": float(sum(len(str(value)) for value in surfaces.values())),
        "verifier_rows_consulted": rows_consulted,
        "verifier_null_count": null_count,
        "verifier_null_fraction": null_count / rows_consulted if rows_consulted > 0 else 0.0,
        "verifier_admitted_count": float(len(admitted)),
        "verifier_margin_min": min(margins) if margins else 0.0,
        "verifier_margin_mean": sum(margins) / len(margins) if margins else 0.0,
        "verifier_margin_max": max(margins) if margins else 0.0,
        "verifier_margin_present": _bool(margins),
        "method": str(prediction.get("method", "unknown")),
        "model_id": str(prediction.get("model_id", "unknown")),
        "language": str(prediction.get("language") or case.get("metadata", {}).get("language", "unknown")),
    }
    values.update(_ranking_features(ranking_row, selected_tool))
    missing = set(RUNTIME_FEATURE_NAMES) - values.keys()
    if missing:
        raise AssertionError(f"runtime feature extractor omitted {sorted(missing)}")
    return values


def assert_feature_firewall(feature_names: Iterable[str]) -> None:
    names = tuple(feature_names)
    violations = sorted(name for name in names if any(fragment in name.lower() for fragment in FORBIDDEN_FEATURE_FRAGMENTS))
    if violations:
        raise ValueError(f"calibrator feature firewall rejected: {violations}")


def _gold_tool_family(gold: dict[str, Any]) -> str:
    ground_truth = gold.get("ground_truth")
    if not isinstance(ground_truth, list) or len(ground_truth) != 1 or not isinstance(ground_truth[0], dict) or len(ground_truth[0]) != 1:
        raise ValueError(f"cannot derive scorer-only fold group for {gold.get('case_id')}")
    tool = str(next(iter(ground_truth[0])))
    return tool.split(".", 1)[0]


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (str(row.get("case_id")), str(row.get("method")), str(row.get("model_id")), int(row.get("seed", 0)))


def build_examples(
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    official_details: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    case_by_id = {str(row["case_id"]): row for row in cases}
    ranking_by_id = {str(row["case_id"]): row for row in rankings}
    gold_by_id = {str(row["case_id"]): row for row in gold_rows}
    if len(gold_by_id) != len(gold_rows):
        raise ValueError("scorer-only gold rows do not have unique case ids")
    detail_by_key = {_row_key(row): row for row in official_details}
    if len(detail_by_key) != len(official_details):
        raise ValueError("official detail rows do not have unique case/method/model/seed keys")
    examples = []
    for prediction in predictions:
        key = _row_key(prediction)
        detail = detail_by_key.get(key)
        if detail is None:
            raise ValueError(f"missing scorer-only detail for {key}")
        case_id = str(prediction["case_id"])
        case = case_by_id.get(case_id)
        if case is None:
            raise ValueError(f"missing runtime case for {case_id}")
        gold = gold_by_id.get(case_id)
        if gold is None:
            raise ValueError(f"missing scorer-only gold row for {case_id}")
        features = extract_runtime_features(case, prediction, ranking_by_id.get(case_id))
        examples.append(
            {
                "key": key,
                "case_id": case_id,
                "method": features["method"],
                "model_id": features["model_id"],
                "language": features["language"],
                "features": features,
                "target": int(bool(detail.get("official_ast_correct"))),
                # This field defines folds only and is never encoded into X.
                "fold_group": _gold_tool_family(gold),
            }
        )
    if len(examples) != len(official_details):
        raise ValueError(f"prediction/detail count mismatch: {len(examples)} != {len(official_details)}")
    return examples


def _categories(examples: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {name: sorted({str(row["features"][name]) for row in examples}) for name in CATEGORICAL_FEATURES}


def _encoded_feature_names(categories: dict[str, list[str]]) -> list[str]:
    names = ["bias", *RUNTIME_FEATURE_NAMES]
    for field in CATEGORICAL_FEATURES:
        names.extend(f"{field}={value}" for value in categories[field])
    assert_feature_firewall(names)
    return names


def _fit_model(examples: list[dict[str, Any]], categories: dict[str, list[str]], *, l2: float = 0.01) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot fit calibrator without examples")
    means = {name: float(np.mean([row["features"][name] for row in examples])) for name in RUNTIME_FEATURE_NAMES}
    scales = {name: float(np.std([row["features"][name] for row in examples])) for name in RUNTIME_FEATURE_NAMES}
    scales = {name: value if value > 1e-12 else 1.0 for name, value in scales.items()}
    names = _encoded_feature_names(categories)
    matrix = np.asarray([_encode(row["features"], categories, means, scales) for row in examples], dtype=np.float64)
    targets = np.asarray([row["target"] for row in examples], dtype=np.float64)
    weights = np.zeros(matrix.shape[1], dtype=np.float64)
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    for _ in range(40):
        logits = np.clip(matrix @ weights, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = matrix.T @ (probabilities - targets) / len(examples) + penalty @ weights
        variance = np.maximum(probabilities * (1.0 - probabilities), 1e-7)
        hessian = matrix.T @ (matrix * variance[:, None]) / len(examples) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        weights -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return {
        "feature_names": names,
        "means": means,
        "scales": scales,
        "weights": [float(value) for value in weights],
        "l2": l2,
    }


def _encode(
    features: dict[str, Any],
    categories: dict[str, list[str]],
    means: dict[str, float],
    scales: dict[str, float],
) -> list[float]:
    values = [1.0]
    values.extend((_number(features[name]) - means[name]) / scales[name] for name in RUNTIME_FEATURE_NAMES)
    for field in CATEGORICAL_FEATURES:
        observed = str(features[field])
        values.extend(float(observed == category) for category in categories[field])
    return values


def _predict(model: dict[str, Any], features: dict[str, Any], categories: dict[str, list[str]]) -> float:
    vector = np.asarray(_encode(features, categories, model["means"], model["scales"]), dtype=np.float64)
    logit = float(np.clip(vector @ np.asarray(model["weights"], dtype=np.float64), -40.0, 40.0))
    return 1.0 / (1.0 + math.exp(-logit))


def family_disjoint_oof(examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    categories = _categories(examples)
    groups = sorted({str(row["fold_group"]) for row in examples})
    scored = []
    folds = []
    for group in groups:
        train = [row for row in examples if row["fold_group"] != group]
        test = [row for row in examples if row["fold_group"] == group]
        if len({row["target"] for row in train}) < 2:
            raise ValueError(f"fold {group} training partition has only one class")
        model = _fit_model(train, categories)
        for row in test:
            scored.append({**row, "score": _predict(model, row["features"], categories), "outer_fold": group})
        folds.append(
            {
                "held_out_tool_family": group,
                "train_rows": len(train),
                "test_rows": len(test),
                "test_positive_rows": sum(row["target"] for row in test),
            }
        )
    scored.sort(key=lambda row: row["key"])
    return scored, folds, categories


def risk_coverage_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (-float(row["score"]), row["key"]))
    output = []
    accepted = 0
    correct = 0
    index = 0
    while index < len(ordered):
        threshold = float(ordered[index]["score"])
        tied = []
        while index < len(ordered) and math.isclose(float(ordered[index]["score"]), threshold, rel_tol=0.0, abs_tol=1e-15):
            tied.append(ordered[index])
            index += 1
        accepted += len(tied)
        correct += sum(int(row["target"]) for row in tied)
        precision = correct / accepted
        output.append(
            {
                "threshold": threshold,
                "accepted": accepted,
                "correct": correct,
                "n": len(ordered),
                "precision": precision,
                "risk": 1.0 - precision,
                "coverage": accepted / len(ordered) if ordered else 0.0,
            }
        )
    return output


def select_threshold(rows: list[dict[str, Any]], target_precision: float) -> dict[str, Any]:
    candidates = [row for row in risk_coverage_curve(rows) if float(row["precision"]) >= target_precision]
    if not candidates:
        return {
            "threshold": None,
            "precision": None,
            "coverage": 0.0,
            "accepted": 0,
            "correct": 0,
            "n": len(rows),
            "target_met": False,
        }
    best = max(candidates, key=lambda row: (int(row["accepted"]), float(row["threshold"])))
    return {**best, "target_met": True}


def _selection_metrics(rows: list[dict[str, Any]], threshold: float | None) -> dict[str, Any]:
    accepted = [] if threshold is None else [row for row in rows if float(row["score"]) >= threshold]
    correct = sum(int(row["target"]) for row in accepted)
    return {
        "n": len(rows),
        "base_correct": sum(int(row["target"]) for row in rows),
        "base_precision": sum(int(row["target"]) for row in rows) / len(rows) if rows else None,
        "accepted": len(accepted),
        "correct": correct,
        "precision": correct / len(accepted) if accepted else None,
        "coverage": len(accepted) / len(rows) if rows else 0.0,
    }


def _stratified_metrics(rows: list[dict[str, Any]], threshold: float | None, field: str) -> list[dict[str, Any]]:
    output = []
    for value in sorted({str(row[field]) for row in rows}):
        subset = [row for row in rows if str(row[field]) == value]
        output.append({field: value, **_selection_metrics(subset, threshold)})
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_calibration(
    *,
    cases_path: str | Path,
    predictions_path: str | Path,
    official_details_path: str | Path,
    gold_path: str | Path,
    rankings_path: str | Path,
    output_dir: str | Path,
    target_precision: float = 0.95,
) -> dict[str, Any]:
    assert 0.0 < target_precision <= 1.0
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    examples = build_examples(
        read_jsonl(cases_path),
        read_jsonl(predictions_path),
        read_jsonl(official_details_path),
        read_jsonl(gold_path),
        read_jsonl(rankings_path),
    )
    scored, folds, categories = family_disjoint_oof(examples)
    threshold = select_threshold(scored, target_precision)
    global_threshold = threshold["threshold"]
    per_model_thresholds = {
        model: select_threshold([row for row in scored if row["model_id"] == model], target_precision)
        for model in sorted({row["model_id"] for row in scored})
    }
    final_model = _fit_model(examples, categories)
    model_artifact = {
        "schema_version": CALIBRATOR_VERSION,
        "training_status": "development_only",
        "confirmation_authorized": False,
        "target_precision": target_precision,
        "threshold_source": "family_disjoint_oof_design_outcomes",
        "global_threshold": global_threshold,
        "categories": categories,
        **final_model,
    }
    model_path = output / "calibrator.json"
    model_path.write_text(json.dumps(model_artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    oof_rows = []
    for row in scored:
        oof_rows.append(
            {
                "schema_version": CALIBRATOR_VERSION,
                "case_id": row["case_id"],
                "method": row["method"],
                "model_id": row["model_id"],
                "language": row["language"],
                "outer_fold": row["outer_fold"],
                "target": row["target"],
                "score": row["score"],
                "accepted_global": global_threshold is not None and row["score"] >= global_threshold,
                "features": row["features"],
            }
        )
    write_jsonl(output / "oof_scores.jsonl", oof_rows)
    _write_csv(output / "risk_coverage_global.csv", risk_coverage_curve(scored))
    by_model_curve = []
    for model in sorted({row["model_id"] for row in scored}):
        for curve_row in risk_coverage_curve([row for row in scored if row["model_id"] == model]):
            by_model_curve.append({"model_id": model, **curve_row})
    _write_csv(output / "risk_coverage_by_model.csv", by_model_curve)

    report = {
        "schema_version": REPORT_VERSION,
        "calibrator_version": CALIBRATOR_VERSION,
        "analysis_status": "exploratory_development_diagnostic",
        "confirmation_authorized": False,
        "confirmation_note": "Opening confirmation requires a separately committed protocol amendment and a locked calibrator.",
        "target_precision": target_precision,
        "threshold_selection_note": "The threshold is selected on family-disjoint OOF design outcomes; it is not an unbiased confirmation estimate.",
        "feature_firewall": {
            "status": "passed",
            "runtime_scalar_features": list(RUNTIME_FEATURE_NAMES),
            "runtime_categorical_features": list(CATEGORICAL_FEATURES),
            "scorer_only_fold_field": "gold tool family",
            "fold_field_in_model_matrix": False,
            "forbidden_fragments": list(FORBIDDEN_FEATURE_FRAGMENTS),
        },
        "inputs": {
            "cases": {"path": str(Path(cases_path).resolve()), "sha256": _sha256(cases_path)},
            "predictions": {"path": str(Path(predictions_path).resolve()), "sha256": _sha256(predictions_path)},
            "official_details": {"path": str(Path(official_details_path).resolve()), "sha256": _sha256(official_details_path)},
            "scorer_only_gold": {"path": str(Path(gold_path).resolve()), "sha256": _sha256(gold_path)},
            "rankings": {"path": str(Path(rankings_path).resolve()), "sha256": _sha256(rankings_path)},
        },
        "rows": len(scored),
        "positive_rows": sum(row["target"] for row in scored),
        "fold_group_counts": dict(sorted(Counter(row["fold_group"] for row in examples).items())),
        "outer_folds": folds,
        "global_threshold": threshold,
        "global_selection": _selection_metrics(scored, global_threshold),
        "per_model_thresholds": per_model_thresholds,
        "selection_by_method": _stratified_metrics(scored, global_threshold, "method"),
        "selection_by_model": _stratified_metrics(scored, global_threshold, "model_id"),
        "selection_by_language": _stratified_metrics(scored, global_threshold, "language"),
        "prospective_gate": {
            "precision_target_met_on_design_oof": bool(threshold["target_met"]),
            "coverage_at_least_50_percent": float(threshold["coverage"]) >= 0.5,
            "passes": bool(threshold["target_met"]) and float(threshold["coverage"]) >= 0.5,
            "authorizes_confirmation": False,
        },
        "artifacts": {
            "calibrator": str(model_path.resolve()),
            "calibrator_sha256": _sha256(model_path),
            "oof_scores": str((output / "oof_scores.jsonl").resolve()),
            "risk_coverage_global": str((output / "risk_coverage_global.csv").resolve()),
            "risk_coverage_by_model": str((output / "risk_coverage_by_model.csv").resolve()),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the leakage-safe MASSIVE QA design calibrator.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--official-details", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--rankings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-precision", type=float, default=0.95)
    args = parser.parse_args()
    report = run_calibration(
        cases_path=args.cases,
        predictions_path=args.predictions,
        official_details_path=args.official_details,
        gold_path=args.gold,
        rankings_path=args.rankings,
        output_dir=args.output_dir,
        target_precision=args.target_precision,
    )
    print(json.dumps({"rows": report["rows"], "global_threshold": report["global_threshold"]}, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_jsonl

R1_UNCERTAINTY_VERSION = "tapbench.r1_cluster_bootstrap.v1"


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["case_id"]), str(row["model_id"])


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take quantile of empty values")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_rows(
    initial_rows: list[dict[str, Any]],
    tapr_rows: list[dict[str, Any]],
    baseline_method: str,
) -> list[dict[str, Any]]:
    baseline = {
        _key(row): float(row["execution_success"])
        for row in initial_rows
        if row.get("method") == baseline_method
    }
    treatment = {
        _key(row): float(row["safe_resolution"])
        for row in tapr_rows
        if row.get("method") == "tap_r_no_calibrator"
    }
    if set(baseline) != set(treatment):
        missing = sorted(set(baseline).symmetric_difference(treatment))[:5]
        raise ValueError(f"paired row mismatch for {baseline_method}: {missing}")
    return [
        {
            "case_id": key[0],
            "model_id": key[1],
            "baseline": baseline[key],
            "treatment": treatment[key],
            "difference": treatment[key] - baseline[key],
        }
        for key in sorted(baseline)
    ]


def _weighted_mean(rows: list[dict[str, Any]], case_weights: Counter[str], model_weights: Counter[str]) -> float:
    numerator = denominator = 0.0
    for row in rows:
        weight = case_weights[row["case_id"]] * model_weights[row["model_id"]]
        numerator += weight * float(row["difference"])
        denominator += weight
    return numerator / denominator


def _bootstrap_contrast(rows: list[dict[str, Any]], *, replicates: int, seed: int) -> dict[str, Any]:
    cases = sorted({row["case_id"] for row in rows})
    models = sorted({row["model_id"] for row in rows})
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        case_weights = Counter(rng.choices(cases, k=len(cases)))
        model_weights = Counter(rng.choices(models, k=len(models)))
        draws.append(_weighted_mean(rows, case_weights, model_weights))
    point = sum(float(row["difference"]) for row in rows) / len(rows)
    return {
        "point_estimate": point,
        "ci_95": [_quantile(draws, 0.025), _quantile(draws, 0.975)],
        "probability_positive": sum(value > 0 for value in draws) / len(draws),
        "replicates": replicates,
        "case_clusters": len(cases),
        "model_clusters": len(models),
        "method": "crossed_case_model_pigeonhole_bootstrap",
    }


def _model_specific(rows: list[dict[str, Any]], *, replicates: int, seed: int) -> list[dict[str, Any]]:
    output = []
    for offset, model_id in enumerate(sorted({row["model_id"] for row in rows})):
        subset = [row for row in rows if row["model_id"] == model_id]
        cases = sorted({row["case_id"] for row in subset})
        rng = random.Random(seed + offset)
        draws = []
        by_case = {row["case_id"]: row for row in subset}
        for _ in range(replicates):
            sampled = rng.choices(cases, k=len(cases))
            draws.append(sum(float(by_case[case]["difference"]) for case in sampled) / len(sampled))
        point = sum(float(row["difference"]) for row in subset) / len(subset)
        output.append({
            "model_id": model_id,
            "point_estimate": point,
            "ci_95": [_quantile(draws, 0.025), _quantile(draws, 0.975)],
            "probability_positive": sum(value > 0 for value in draws) / len(draws),
            "case_clusters": len(cases),
        })
    return output


def write_r1_cluster_bootstrap(
    initial_scores_path: str | Path,
    tapr_scores_path: str | Path,
    output_path: str | Path,
    *,
    replicates: int = 20000,
    seed: int = 20260710,
) -> dict[str, Any]:
    initial = read_jsonl(initial_scores_path)
    tapr = read_jsonl(tapr_scores_path)
    contrasts = {}
    model_specific = {}
    for offset, baseline in enumerate(("full_tap_b2", "prompt_few_shot")):
        rows = _paired_rows(initial, tapr, baseline)
        contrasts[baseline] = _bootstrap_contrast(rows, replicates=replicates, seed=seed + offset)
        model_specific[baseline] = _model_specific(rows, replicates=replicates, seed=seed + 100 + offset * 10)
    report = {
        "schema_version": R1_UNCERTAINTY_VERSION,
        "design": {
            "paired_unit": ["case_id", "model_id"],
            "cluster_dimensions": ["case_id", "model_id"],
            "seed": seed,
            "replicates": replicates,
            "note": "Four model clusters make the pooled model dimension imprecise; model-specific case-cluster intervals are reported alongside it.",
        },
        "contrasts": contrasts,
        "model_specific": model_specific,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

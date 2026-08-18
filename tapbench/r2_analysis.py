from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .analyze import export_scores_csv
from .config import REPO_ROOT
from .io import read_jsonl


R2_ANALYSIS_VERSION = "tapbench.r2_full_analysis.v1"
TEP_METHOD = "r2_pointer_tep_tier_ab"
BASELINE_METHODS = ("r2_literal_generation", "r2_pointer_unrestricted")
METRICS = (
    "execution_success",
    "fabrication",
    "mode_correct",
    "tool_correct",
    "args_exact",
    "format_valid",
)


def _flag(value: Any) -> int:
    if isinstance(value, str):
        return int(value.strip().casefold() in {"1", "true", "yes"})
    return int(bool(value))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _grouped(rows: Iterable[dict[str, Any]], keys: Sequence[str]) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    return sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0]))


def _score_summary(rows: Sequence[dict[str, Any]], group_keys: Sequence[str]) -> list[dict[str, Any]]:
    output = []
    for group, selected in _grouped(rows, group_keys):
        row = {key: value for key, value in zip(group_keys, group, strict=True)}
        row["n"] = len(selected)
        for metric in METRICS:
            count = sum(_flag(item.get(metric)) for item in selected)
            low, high = _wilson(count, len(selected))
            row[f"{metric}_count"] = count
            row[f"{metric}_rate"] = count / len(selected)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        call_count = row["mode_correct_count"]
        exact_low, exact_high = _wilson(row["execution_success_count"], call_count)
        fabrication_low, fabrication_high = _wilson(row["fabrication_count"], call_count)
        row.update({
            "call_coverage": call_count / len(selected),
            "accepted_call_exact_precision": (
                row["execution_success_count"] / call_count if call_count else None
            ),
            "accepted_call_exact_precision_ci_low": exact_low,
            "accepted_call_exact_precision_ci_high": exact_high,
            "accepted_call_fabrication_rate": row["fabrication_count"] / call_count if call_count else None,
            "accepted_call_fabrication_ci_low": fabrication_low,
            "accepted_call_fabrication_ci_high": fabrication_high,
        })
        output.append(row)
    return output


def _paired_contrast(
    rows: Sequence[dict[str, Any]],
    *,
    treatment: str,
    control: str,
    metric: str,
    replicates: int,
    seed: int,
    model_id: str | None = None,
) -> dict[str, Any]:
    selected = [row for row in rows if model_id is None or row.get("model_id") == model_id]
    by_method: dict[str, dict[tuple[str, str, int], int]] = defaultdict(dict)
    for row in selected:
        if row.get("method") not in {treatment, control}:
            continue
        key = (str(row["case_id"]), str(row["model_id"]), int(row["seed"]))
        by_method[str(row["method"])][key] = _flag(row.get(metric))
    paired_keys = sorted(set(by_method[treatment]) & set(by_method[control]))
    if not paired_keys:
        raise ValueError(f"no paired rows for {treatment} versus {control} on {metric}")
    differences = {
        key: by_method[treatment][key] - by_method[control][key]
        for key in paired_keys
    }
    by_case: dict[str, list[int]] = defaultdict(list)
    for key, difference in differences.items():
        by_case[key[0]].append(difference)
    case_effects = {
        case_id: sum(values) / len(values)
        for case_id, values in by_case.items()
    }
    cluster_ids = sorted(case_effects)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(replicates):
        sampled = [cluster_ids[rng.randrange(len(cluster_ids))] for _ in cluster_ids]
        bootstrap.append(sum(case_effects[case_id] for case_id in sampled) / len(sampled))
    treatment_rate = sum(by_method[treatment][key] for key in paired_keys) / len(paired_keys)
    control_rate = sum(by_method[control][key] for key in paired_keys) / len(paired_keys)
    treatment_only = sum(
        by_method[treatment][key] == 1 and by_method[control][key] == 0
        for key in paired_keys
    )
    control_only = sum(
        by_method[treatment][key] == 0 and by_method[control][key] == 1
        for key in paired_keys
    )
    return {
        "model_id": model_id or "pooled",
        "metric": metric,
        "treatment": treatment,
        "control": control,
        "paired_rows": len(paired_keys),
        "case_clusters": len(cluster_ids),
        "treatment_rate": treatment_rate,
        "control_rate": control_rate,
        "difference": treatment_rate - control_rate,
        "relative_gain": (
            (treatment_rate - control_rate) / control_rate if control_rate else None
        ),
        "case_cluster_bootstrap_ci_low": _percentile(bootstrap, 0.025),
        "case_cluster_bootstrap_ci_high": _percentile(bootstrap, 0.975),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "discordant_treatment_only": treatment_only,
        "discordant_control_only": control_only,
    }


def _paired_contrasts(
    rows: Sequence[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    models: list[str | None] = [None, *sorted({str(row["model_id"]) for row in rows})]
    for model_id in models:
        for control in BASELINE_METHODS:
            for metric in ("execution_success", "fabrication", "tool_correct"):
                output.append(_paired_contrast(
                    rows,
                    treatment=TEP_METHOD,
                    control=control,
                    metric=metric,
                    replicates=replicates,
                    seed=seed,
                    model_id=model_id,
                ))
    return output


def _timing_summary(
    rows: Sequence[dict[str, Any]],
    group_keys: Sequence[str] = ("model_id", "method"),
) -> list[dict[str, Any]]:
    output = []
    for group, selected in _grouped(rows, group_keys):
        row = {key: value for key, value in zip(group_keys, group, strict=True)}
        row["n"] = len(selected)
        for metric in (
            "elapsed_seconds",
            "generated_tokens_per_second",
            "prompt_tokens",
            "completion_tokens",
            "evidence_construction_seconds",
        ):
            values = [float(item[metric]) for item in selected if item.get(metric) is not None]
            row[f"{metric}_mean"] = statistics.fmean(values) if values else None
            row[f"{metric}_p50"] = _percentile(values, 0.5) if values else None
            row[f"{metric}_p95"] = _percentile(values, 0.95) if values else None
        output.append(row)
    return output


def _slot_error_summary(
    scores: Sequence[dict[str, Any]],
    errors: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    case_stratum = {str(row["case_id"]): str(row.get("operator_stratum")) for row in scores}
    score_denominators: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in scores:
        score_denominators[(str(row["model_id"]), str(row["method"]), str(row.get("operator_stratum")))] += 1
    grouped_errors: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in errors:
        stratum = case_stratum.get(str(row["case_id"]), "unknown")
        grouped_errors[(str(row["model_id"]), str(row["method"]), stratum, str(row["error_type"]))].append(row)
    output = []
    for key, selected in sorted(grouped_errors.items()):
        model_id, method, stratum, error_type = key
        denominator = score_denominators[(model_id, method, stratum)]
        affected = {
            (str(row["case_id"]), str(row["model_id"]), str(row["method"]), int(row["seed"]))
            for row in selected
        }
        output.append({
            "model_id": model_id,
            "method": method,
            "operator_stratum": stratum,
            "error_type": error_type,
            "error_events": len(selected),
            "affected_predictions": len(affected),
            "prediction_denominator": denominator,
            "affected_prediction_rate": len(affected) / denominator if denominator else None,
        })
    return output


def _component_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for condition, metrics in sorted(report.get("by_condition", {}).items()):
        rows.append({"condition": condition, **dict(metrics)})
    return rows


def _verifier_rows(verifier: Mapping[str, Any]) -> list[dict[str, Any]]:
    operating = verifier.get("operating_point", {})
    cross_validated = operating.get("cross_validated", {})
    training = verifier.get("training", {})
    return [{
        "schema_version": verifier.get("schema_version"),
        "training_rows": training.get("rows"),
        "training_positives": training.get("positives"),
        "training_families": len(training.get("families", [])),
        "family_disjoint_folds": len(training.get("family_disjoint_folds", [])),
        "target_precision": operating.get("target_precision"),
        "threshold": operating.get("threshold"),
        "oof_precision": cross_validated.get("precision"),
        "oof_recall": cross_validated.get("recall"),
        "oof_coverage": cross_validated.get("coverage"),
        "oof_selected": cross_validated.get("selected"),
        "accepted_error_upper_bound": operating.get("accepted_error_upper_bound"),
    }]


def write_r2_full_analysis(
    scores_path: str | Path,
    slot_errors_path: str | Path,
    timings_path: str | Path,
    component_report_path: str | Path,
    verifier_path: str | Path,
    release_report_path: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260713,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores = read_jsonl(scores_path)
    slot_errors = read_jsonl(slot_errors_path)
    timings = read_jsonl(timings_path)
    component_report = json.loads(Path(component_report_path).read_text(encoding="utf-8"))
    verifier = json.loads(Path(verifier_path).read_text(encoding="utf-8"))
    release_report = json.loads(Path(release_report_path).read_text(encoding="utf-8"))

    if not release_report.get("release_decision", {}).get("passed"):
        raise ValueError("R2 full analysis requires a passing release report")
    if len(scores) != len(timings):
        raise ValueError("score and timing row counts differ")
    if {str(row.get("task_kind")) for row in scores} != {"call"}:
        raise ValueError("R2-A accepted-call analysis requires call-only gold cases")
    observed_methods = {str(row.get("method")) for row in scores}
    expected_methods = {TEP_METHOD, *BASELINE_METHODS}
    if observed_methods != expected_methods:
        raise ValueError(f"unexpected R2-A methods: {sorted(observed_methods)}")

    tables = {
        "method_summary": _score_summary(scores, ("method",)),
        "model_method_summary": _score_summary(scores, ("model_id", "method")),
        "stratum_method_summary": _score_summary(scores, ("operator_stratum", "method")),
        "model_stratum_method_summary": _score_summary(scores, ("model_id", "operator_stratum", "method")),
        "paired_contrasts": _paired_contrasts(
            scores,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "slot_error_profile": _slot_error_summary(scores, slot_errors),
        "method_timing_summary": _timing_summary(timings, ("method",)),
        "timing_summary": _timing_summary(timings),
        "component_summary": _component_rows(component_report),
        "verifier_summary": _verifier_rows(verifier),
    }
    for name, rows in tables.items():
        _write_csv(output / f"{name}.csv", rows)
        _write_json(output / f"{name}.json", rows)

    input_paths = {
        "scores": str(scores_path),
        "slot_errors": str(slot_errors_path),
        "timings": str(timings_path),
        "component_report": str(component_report_path),
        "tier_b_verifier": str(verifier_path),
        "release_report": str(release_report_path),
        "hypothesis_map": str(REPO_ROOT / "analysis" / "r2_tep_hypothesis_map.yaml"),
    }
    integrity = {
        "score_rows": len(scores),
        "slot_error_rows": len(slot_errors),
        "timing_rows": len(timings),
        "models": sorted({str(row["model_id"]) for row in scores}),
        "methods": sorted(observed_methods),
        "seeds": sorted({int(row["seed"]) for row in scores}),
        "backend": sorted({str(row["backend"]) for row in scores}),
        "quantization": sorted({str(row["quantization"]) for row in scores}),
        "thinking_mode": sorted({str(row["thinking_mode"]) for row in scores}),
        "scorer_versions": sorted({str(row["scorer_version"]) for row in scores}),
        "normalizer_versions": sorted({str(row["normalizer_version"]) for row in scores}),
        "validator_versions": sorted({str(row["validator_version"]) for row in scores}),
        "release_gate_passed": True,
    }
    report = {
        "schema_version": R2_ANALYSIS_VERSION,
        "analysis_scope": "R2-A controlled synthetic component identification",
        "bootstrap": {
            "cluster": "case_id",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "integrity": integrity,
        "input_artifacts": {
            name: {"path": path, "sha256": _sha256(path)}
            for name, path in input_paths.items()
        },
        "tables": {name: f"{name}.csv" for name in tables},
        "claim_boundary": (
            "R2-A identifies controlled single-call evidence construction and pointer-selection effects. "
            "It does not establish R2-B open-world, multi-turn, or official-benchmark effectiveness."
        ),
    }
    _write_json(output / "analysis_report.json", report)
    export_scores_csv(scores_path, output)
    return report


def run_r2_lme4(scores_csv: str | Path, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    script = REPO_ROOT / "analysis" / "r2_glmm_fit.R"
    local_rscript = REPO_ROOT / "work" / "conda_r" / "bin" / "Rscript"
    rscript = os.environ.get("TAPBENCH_RSCRIPT")
    if not rscript and local_rscript.exists():
        rscript = str(local_rscript)
    if not rscript:
        rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript executable was not found")
    subprocess.run([rscript, str(script), str(scores_csv), str(output)], check=True)
    return output / "r2_glmm_coefficients.csv"

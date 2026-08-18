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


R2B_ANALYSIS_VERSION = "tapbench.r2b_full_analysis.v1"
R2B_METHODS = (
    "prompt_few_shot",
    "constrained_abstention",
    "best_of_2",
    "best_of_4",
    "validator_feedback_regeneration",
    "local_llm_slot_regeneration",
    "process_scored_search",
    "tap_r_literal_evidence",
    "tap_r_tep_tier_a",
    "tap_r_tep_tier_ab",
    "tap_r_without_global_contract",
    "tap_r_full",
)
MATCHED_BASELINES = R2B_METHODS[:7]
SUMMARY_METRICS = (
    "autonomous_safe_resolution",
    "execution_success",
    "unsupported_action_critical",
    "fabrication",
    "mode_correct",
    "tool_correct",
    "args_exact",
    "format_valid",
    "accepted_call",
    "non_escalated",
)
CONTRAST_SPECS = (
    (
        "tap_r_vs_one_pass",
        "tap_r_full",
        "constrained_abstention",
        ("autonomous_safe_resolution", "unsupported_action_critical", "execution_success"),
    ),
    (
        "tap_r_vs_budget_matched_search",
        "tap_r_full",
        "best_of_4",
        (
            "autonomous_safe_resolution",
            "accepted_call_exact_precision",
            "unsupported_action_critical",
        ),
    ),
    (
        "global_contract_ablation",
        "tap_r_full",
        "tap_r_without_global_contract",
        ("autonomous_safe_resolution", "unsupported_action_critical", "fabrication"),
    ),
    (
        "evidence_program_ablation",
        "tap_r_tep_tier_ab",
        "tap_r_literal_evidence",
        ("execution_success", "fabrication", "args_exact"),
    ),
    (
        "tier_b_ablation",
        "tap_r_tep_tier_ab",
        "tap_r_tep_tier_a",
        ("execution_success", "unsupported_action_critical"),
    ),
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
    radius = z * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
    ) / denominator
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


def _grouped(
    rows: Iterable[dict[str, Any]], keys: Sequence[str]
) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    return sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0]))


def _score_summary(
    rows: Sequence[dict[str, Any]], group_keys: Sequence[str]
) -> list[dict[str, Any]]:
    output = []
    for group, selected in _grouped(rows, group_keys):
        row = {key: value for key, value in zip(group_keys, group, strict=True)}
        row["n"] = len(selected)
        for metric in SUMMARY_METRICS:
            count = sum(_flag(item.get(metric)) for item in selected)
            low, high = _wilson(count, len(selected))
            row[f"{metric}_count"] = count
            row[f"{metric}_rate"] = count / len(selected)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        accepted = [item for item in selected if _flag(item.get("accepted_call"))]
        accepted_exact = sum(
            _flag(item.get("execution_success"))
            and not _flag(item.get("unsupported_action_critical"))
            for item in accepted
        )
        low, high = _wilson(accepted_exact, len(accepted))
        row.update(
            {
                "accepted_call_denominator": len(accepted),
                "accepted_call_exact_count": accepted_exact,
                "accepted_call_exact_precision": (
                    accepted_exact / len(accepted) if accepted else None
                ),
                "accepted_call_exact_precision_ci_low": low,
                "accepted_call_exact_precision_ci_high": high,
            }
        )
        output.append(row)
    return output


def _outcome_parts(row: Mapping[str, Any], metric: str) -> tuple[int, int]:
    if metric == "accepted_call_exact_precision":
        accepted = _flag(row.get("accepted_call"))
        exact = int(
            accepted
            and _flag(row.get("execution_success"))
            and not _flag(row.get("unsupported_action_critical"))
        )
        return exact, accepted
    return _flag(row.get(metric)), 1


def _paired_contrast(
    rows: Sequence[dict[str, Any]],
    *,
    contrast_id: str,
    treatment: str,
    control: str,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    by_method: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        method = str(row.get("method"))
        if method not in {treatment, control}:
            continue
        key = (str(row["case_id"]), str(row["model_id"]), int(row["seed"]))
        by_method[method][key] = row
    paired_keys = sorted(set(by_method[treatment]) & set(by_method[control]))
    if not paired_keys:
        raise ValueError(f"no paired rows for {contrast_id}:{metric}")

    case_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    treatment_only = 0
    control_only = 0
    for key in paired_keys:
        treatment_num, treatment_den = _outcome_parts(by_method[treatment][key], metric)
        control_num, control_den = _outcome_parts(by_method[control][key], metric)
        stats = case_stats[key[0]]
        stats[0] += treatment_num
        stats[1] += treatment_den
        stats[2] += control_num
        stats[3] += control_den
        if treatment_den == control_den == 1:
            treatment_only += int(treatment_num == 1 and control_num == 0)
            control_only += int(treatment_num == 0 and control_num == 1)

    totals = [sum(values[index] for values in case_stats.values()) for index in range(4)]
    treatment_rate = totals[0] / totals[1] if totals[1] else None
    control_rate = totals[2] / totals[3] if totals[3] else None
    if treatment_rate is None or control_rate is None:
        raise ValueError(f"undefined contrast rate for {contrast_id}:{metric}")

    clusters = sorted(case_stats)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(replicates):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        sampled_totals = [
            sum(case_stats[case_id][index] for case_id in sampled)
            for index in range(4)
        ]
        if sampled_totals[1] and sampled_totals[3]:
            bootstrap.append(
                sampled_totals[0] / sampled_totals[1]
                - sampled_totals[2] / sampled_totals[3]
            )
    return {
        "contrast_id": contrast_id,
        "metric": metric,
        "treatment": treatment,
        "control": control,
        "paired_rows": len(paired_keys),
        "case_clusters": len(clusters),
        "treatment_numerator": totals[0],
        "treatment_denominator": totals[1],
        "control_numerator": totals[2],
        "control_denominator": totals[3],
        "treatment_rate": treatment_rate,
        "control_rate": control_rate,
        "difference": treatment_rate - control_rate,
        "relative_gain": (
            (treatment_rate - control_rate) / control_rate if control_rate else None
        ),
        "case_cluster_bootstrap_ci_low": _percentile(bootstrap, 0.025),
        "case_cluster_bootstrap_ci_high": _percentile(bootstrap, 0.975),
        "bootstrap_replicates": len(bootstrap),
        "bootstrap_seed": seed,
        "discordant_treatment_only": treatment_only,
        "discordant_control_only": control_only,
    }


def _paired_contrasts(
    rows: Sequence[dict[str, Any]], *, replicates: int, seed: int
) -> list[dict[str, Any]]:
    output = []
    for contrast_id, treatment, control, metrics in CONTRAST_SPECS:
        for metric in metrics:
            output.append(
                _paired_contrast(
                    rows,
                    contrast_id=contrast_id,
                    treatment=treatment,
                    control=control,
                    metric=metric,
                    replicates=replicates,
                    seed=seed,
                )
            )
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
            "generation_calls",
            "generated_tokens_per_second",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            values = [
                float(item[metric])
                for item in selected
                if item.get(metric) is not None
            ]
            row[f"{metric}_mean"] = statistics.fmean(values) if values else None
            row[f"{metric}_p50"] = _percentile(values, 0.50) if values else None
            row[f"{metric}_p95"] = _percentile(values, 0.95) if values else None
        output.append(row)
    return output


def _slot_error_summary(
    scores: Sequence[dict[str, Any]],
    errors: Sequence[dict[str, Any]],
    group_keys: Sequence[str],
) -> list[dict[str, Any]]:
    case_factors = {
        str(row["case_id"]): {
            "task_kind": row.get("task_kind"),
            "catalog_mutation": row.get("catalog_mutation"),
        }
        for row in scores
    }
    denominators: dict[tuple[Any, ...], int] = defaultdict(int)
    for row in scores:
        key = tuple(row.get(name) for name in group_keys)
        denominators[key] += 1

    grouped_errors: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for error in errors:
        enriched = dict(error)
        enriched.update(case_factors.get(str(error["case_id"]), {}))
        key = tuple(enriched.get(name) for name in group_keys) + (error.get("error_type"),)
        grouped_errors[key].append(enriched)

    output = []
    for key, selected in sorted(
        grouped_errors.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        group = key[:-1]
        error_type = key[-1]
        denominator = denominators[group]
        affected = {
            (
                str(row["case_id"]),
                str(row["model_id"]),
                str(row["method"]),
                int(row["seed"]),
            )
            for row in selected
        }
        record = {name: value for name, value in zip(group_keys, group, strict=True)}
        record.update(
            {
                "error_type": error_type,
                "error_events": len(selected),
                "affected_predictions": len(affected),
                "prediction_denominator": denominator,
                "affected_prediction_rate": (
                    len(affected) / denominator if denominator else None
                ),
                "unsupported_value_events": sum(
                    not _flag(row.get("derivable")) for row in selected
                ),
            }
        )
        output.append(record)
    return output


def _production_gates(method_summary: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method = {str(row["method"]): row for row in method_summary}
    target = by_method["tap_r_full"]
    strongest = max(
        MATCHED_BASELINES,
        key=lambda method: float(
            by_method[method]["autonomous_safe_resolution_rate"]
        ),
    )
    gain = (
        target["autonomous_safe_resolution_rate"]
        - by_method[strongest]["autonomous_safe_resolution_rate"]
    )
    return [
        {
            "gate": "accepted_call_exact_precision_min",
            "threshold": 0.97,
            "observed": target["accepted_call_exact_precision"],
            "passed": target["accepted_call_exact_precision"] >= 0.97,
        },
        {
            "gate": "unsupported_action_critical_rate_max",
            "threshold": 0.02,
            "observed": target["unsupported_action_critical_rate"],
            "passed": target["unsupported_action_critical_rate"] <= 0.02,
        },
        {
            "gate": "non_escalated_coverage_min",
            "threshold": 0.80,
            "observed": target["non_escalated_rate"],
            "passed": target["non_escalated_rate"] >= 0.80,
        },
        {
            "gate": "gain_over_strongest_matched_baseline_pp_min",
            "threshold": 0.10,
            "observed": gain,
            "passed": gain >= 0.10,
            "strongest_matched_baseline": strongest,
            "strongest_matched_baseline_rate": by_method[strongest][
                "autonomous_safe_resolution_rate"
            ],
        },
    ]


def write_r2b_full_analysis(
    scores_path: str | Path,
    slot_errors_path: str | Path,
    timings_path: str | Path,
    release_report_path: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260715,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores = read_jsonl(scores_path)
    slot_errors = read_jsonl(slot_errors_path)
    timings = read_jsonl(timings_path)
    release_report = json.loads(Path(release_report_path).read_text(encoding="utf-8"))

    if not release_report.get("release_decision", {}).get("passed"):
        raise ValueError("R2-B analysis requires a passing release report")
    if len(scores) != len(timings):
        raise ValueError("R2-B score and timing row counts differ")
    if {str(row.get("hypothesis")) for row in scores} != {"R2B"}:
        raise ValueError("analysis input contains non-R2B rows")
    observed_methods = {str(row.get("method")) for row in scores}
    if observed_methods != set(R2B_METHODS):
        raise ValueError(f"unexpected R2-B methods: {sorted(observed_methods)}")
    if {str(row.get("backend")) for row in scores} != {"llama.cpp"}:
        raise ValueError("R2-B analysis input mixes backends")
    if {str(row.get("quantization")) for row in scores} != {"Q4_K_M"}:
        raise ValueError("R2-B analysis input mixes quantization")
    if {str(row.get("thinking_mode")) for row in scores} != {"off"}:
        raise ValueError("R2-B coefficient rows must have thinking disabled")
    if any(_flag(row.get("thinking_marker_detected")) for row in scores):
        raise ValueError("R2-B coefficient rows contain visible thinking markers")

    method_summary = _score_summary(scores, ("method",))
    tables = {
        "method_summary": method_summary,
        "model_method_summary": _score_summary(scores, ("model_id", "method")),
        "task_method_summary": _score_summary(scores, ("task_kind", "method")),
        "mutation_method_summary": _score_summary(
            scores, ("catalog_mutation", "method")
        ),
        "model_task_method_summary": _score_summary(
            scores, ("model_id", "task_kind", "method")
        ),
        "paired_contrasts": _paired_contrasts(
            scores,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "failure_profile": _slot_error_summary(
            scores, slot_errors, ("method", "task_kind")
        ),
        "mutation_failure_profile": _slot_error_summary(
            scores, slot_errors, ("method", "catalog_mutation")
        ),
        "model_failure_profile": _slot_error_summary(
            scores, slot_errors, ("model_id", "method", "task_kind")
        ),
        "timing_summary": _timing_summary(timings),
        "method_timing_summary": _timing_summary(timings, ("method",)),
        "production_gates": _production_gates(method_summary),
    }
    for name, rows in tables.items():
        _write_csv(output / f"{name}.csv", rows)
        _write_json(output / f"{name}.json", rows)

    risk_coverage = {
        "schema_version": "tapbench.r2b_risk_coverage_status.v1",
        "identified": False,
        "reason": (
            "The frozen R2-B predictions contain terminal actions and contract traces "
            "but no preregistered scalar action-risk score or calibrator probability. "
            "Accepted-call precision and non-escalated coverage are reported directly; "
            "continuous risk-coverage curves are deferred rather than derived post hoc."
        ),
        "available_direct_metrics": [
            "accepted_call_exact_precision",
            "unsupported_action_critical_rate",
            "non_escalated_coverage",
        ],
    }
    _write_json(output / "risk_coverage_status.json", risk_coverage)

    input_paths = {
        "scores": str(scores_path),
        "slot_errors": str(slot_errors_path),
        "timings": str(timings_path),
        "release_report": str(release_report_path),
        "hypothesis_map": str(REPO_ROOT / "analysis" / "r2b_hypothesis_map.yaml"),
    }
    integrity = {
        "score_rows": len(scores),
        "slot_error_rows": len(slot_errors),
        "timing_rows": len(timings),
        "models": sorted({str(row["model_id"]) for row in scores}),
        "methods": sorted(observed_methods),
        "task_kinds": sorted({str(row["task_kind"]) for row in scores}),
        "catalog_mutations": sorted(
            {str(row["catalog_mutation"]) for row in scores}
        ),
        "seeds": sorted({int(row["seed"]) for row in scores}),
        "backend": sorted({str(row["backend"]) for row in scores}),
        "quantization": sorted({str(row["quantization"]) for row in scores}),
        "thinking_mode": sorted({str(row["thinking_mode"]) for row in scores}),
        "action_schema_versions": sorted(
            {str(row.get("r2b_action_schema_version")) for row in scores}
        ),
        "scorer_versions": sorted({str(row["scorer_version"]) for row in scores}),
        "normalizer_versions": sorted(
            {str(row["normalizer_version"]) for row in scores}
        ),
        "validator_versions": sorted(
            {str(row["validator_version"]) for row in scores}
        ),
        "release_gate_passed": True,
    }
    production_gates = tables["production_gates"]
    report = {
        "schema_version": R2B_ANALYSIS_VERSION,
        "analysis_scope": "R2-B held-out deployable open-world synthetic evaluation",
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
        "risk_coverage": risk_coverage,
        "production_gate_passed": all(
            bool(row["passed"]) for row in production_gates
        ),
        "claim_boundary": (
            "R2-B evaluates held-out synthetic families, catalog mutation, and legitimate "
            "non-call modes under one backend and quantization regime. It does not establish "
            "official stateful or multi-step benchmark reliability, and continuous risk-coverage "
            "is not identified without a frozen scalar risk score."
        ),
    }
    _write_json(output / "analysis_report.json", report)
    export_scores_csv(scores_path, output)
    return report


def _augment_glmm_status(
    coefficients_path: Path, statuses_path: Path
) -> list[dict[str, Any]]:
    with coefficients_path.open("r", encoding="utf-8", newline="") as handle:
        coefficients = list(csv.DictReader(handle))
    with statuses_path.open("r", encoding="utf-8", newline="") as handle:
        statuses = list(csv.DictReader(handle))

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in coefficients:
        grouped[(row["contrast_id"], row["metric"])].append(row)

    for status in statuses:
        rows = grouped.get((status["contrast_id"], status["metric"]), [])
        max_std_error = max(
            (abs(float(row["std_error"])) for row in rows),
            default=math.nan,
        )
        wald_finite = bool(rows) and all(
            math.isfinite(float(row[field])) and float(row[field]) > 0.0
            for row in rows
            for field in (
                "odds_ratio",
                "odds_ratio_ci_low",
                "odds_ratio_ci_high",
            )
        )
        fit_ok = status.get("fit_ok", "").casefold() == "true"
        converged = status.get("converged", "").casefold() == "true"
        singular = status.get("singular", "").casefold() == "true"
        wald_usable = fit_ok and converged and not singular and wald_finite
        status["max_fixed_std_error"] = (
            "" if math.isnan(max_std_error) else format(max_std_error, ".12g")
        )
        status["wald_finite"] = str(wald_finite).upper()
        status["wald_usable"] = str(wald_usable).upper()
        if not fit_ok:
            note = "fit_failed"
        elif not converged:
            note = "nonconverged"
        elif singular:
            note = "singular"
        elif not wald_finite:
            note = "nonfinite_wald_interval_likely_separation"
        else:
            note = "usable"
        status["stability_note"] = note

    _write_csv(statuses_path, statuses)
    return statuses


def _csv_to_json(csv_path: Path, json_path: Path) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _write_json(json_path, rows)


def run_r2b_lme4(scores_csv: str | Path, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    script = REPO_ROOT / "analysis" / "r2b_glmm_fit.R"
    local_rscript = REPO_ROOT / "work" / "conda_r" / "bin" / "Rscript"
    rscript = os.environ.get("TAPBENCH_RSCRIPT")
    if not rscript and local_rscript.exists():
        rscript = str(local_rscript)
    if not rscript:
        rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript executable was not found")
    subprocess.run([rscript, str(script), str(scores_csv), str(output)], check=True)
    coefficients = output / "r2b_glmm_coefficients.csv"
    statuses = output / "r2b_glmm_status.csv"
    audited_statuses = _augment_glmm_status(coefficients, statuses)
    _csv_to_json(coefficients, output / "r2b_glmm_coefficients.json")
    _write_json(output / "r2b_glmm_status.json", audited_statuses)

    report_path = output / "analysis_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["mixed_models"] = {
            "engine": "R/lme4::glmer",
            "fit_count": len(audited_statuses),
            "converged_count": sum(
                row.get("converged", "").casefold() == "true"
                for row in audited_statuses
            ),
            "nonsingular_count": sum(
                row.get("singular", "").casefold() == "false"
                for row in audited_statuses
            ),
            "wald_usable_count": sum(
                row.get("wald_usable", "").casefold() == "true"
                for row in audited_statuses
            ),
            "coefficients": coefficients.name,
            "statuses": statuses.name,
            "primary_uncertainty": "paired case-cluster bootstrap",
        }
        _write_json(report_path, report)
    return coefficients

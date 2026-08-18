from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any


HIERARCHY_ANALYSIS_VERSION = "evibind.hierarchy_analysis.v2"
FULL = "tap_r_selective_full"
SOURCE_ROLE = "source_role_contract"
BEST_OF = "best_of_compute_matched"


def _flag(row: dict[str, Any], key: str) -> float:
    return float(bool(row.get(key)))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _family_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["family"])].append(row)
    return dict(output)


def _safe_delta(rows: Sequence[dict[str, Any]], treatment: str, control: str) -> float:
    selected = [row for row in rows if row["method"] in {treatment, control}]
    values: dict[tuple[str, str], float] = {
        (str(row["case_id"]), str(row["method"])): _flag(
            row, "autonomous_safe_resolution"
        )
        for row in selected
    }
    case_ids = sorted({case_id for case_id, _ in values})
    if any(
        (case_id, treatment) not in values or (case_id, control) not in values
        for case_id in case_ids
    ):
        raise ValueError("safe-decision contrast has incomplete pairs")
    return sum(
        values[(case_id, treatment)] - values[(case_id, control)]
        for case_id in case_ids
    ) / len(case_ids)


def _accepted_exact(row: dict[str, Any]) -> float:
    return float(
        bool(row.get("accepted_call"))
        and bool(row.get("execution_success"))
        and not bool(row.get("unsupported_action_critical"))
    )


def _call_precision(rows: Sequence[dict[str, Any]], method: str) -> float:
    accepted = [
        row
        for row in rows
        if row["method"] == method and bool(row.get("accepted_call"))
    ]
    if not accepted:
        return math.nan
    return sum(_accepted_exact(row) for row in accepted) / len(accepted)


def _call_coverage(rows: Sequence[dict[str, Any]], method: str) -> float:
    calls = [
        row
        for row in rows
        if row["method"] == method and row["task_kind"] == "call"
    ]
    if not calls:
        raise ValueError("call coverage has no gold-call rows")
    return sum(_flag(row, "accepted_call") for row in calls) / len(calls)


def _task_accuracy(
    rows: Sequence[dict[str, Any]], method: str, task_kind: str
) -> float:
    selected = [
        row
        for row in rows
        if row["method"] == method and row["task_kind"] == task_kind
    ]
    if not selected:
        raise ValueError(f"{task_kind} accuracy has no rows")
    return sum(_flag(row, "autonomous_safe_resolution") for row in selected) / len(
        selected
    )


def _bootstrap_families(
    rows: Sequence[dict[str, Any]],
    statistic: Callable[[Sequence[dict[str, Any]]], float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    by_family = _family_rows(rows)
    families = sorted(by_family)
    if len(families) < 2:
        raise ValueError("family bootstrap needs at least two families")
    observed = statistic(rows)
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled = [rng.choice(families) for _ in families]
        sample_rows = [row for family in sampled for row in by_family[family]]
        value = statistic(sample_rows)
        if not math.isnan(value):
            draws.append(value)
    if not draws:
        raise ValueError("family bootstrap produced no finite replicates")
    leave_one_out = []
    for omitted in families:
        estimate = statistic(
            [
                row
                for family, family_rows in by_family.items()
                if family != omitted
                for row in family_rows
            ]
        )
        leave_one_out.append(
            {"omitted_family": omitted, "estimate": estimate}
        )
    finite_leave_one_out = [
        row["estimate"]
        for row in leave_one_out
        if not math.isnan(row["estimate"])
    ]
    if not finite_leave_one_out:
        raise ValueError("leave-one-family-out produced no finite estimates")
    return {
        "estimate": observed,
        "family_count": len(families),
        "cluster_bootstrap_95_ci": [
            _quantile(draws, 0.025),
            _quantile(draws, 0.975),
        ],
        "leave_one_family_out": leave_one_out,
        "leave_one_family_out_range": [
            min(finite_leave_one_out),
            max(finite_leave_one_out),
        ],
    }


def audit_hierarchy_compute(
    predictions: Sequence[dict[str, Any]],
    *,
    expected_cases: int = 768,
) -> dict[str, Any]:
    by_key = {}
    failures = []
    for row in predictions:
        key = (str(row["case_id"]), str(row["method"]))
        if key in by_key:
            failures.append({"case_id": key[0], "reason": "duplicate_condition"})
        by_key[key] = row
    full_aggregate_tokens = 0
    best_aggregate_tokens = 0
    best_sample_counts = []
    declared_full_totals = set()
    declared_best_totals = set()
    for case_id in sorted({case_id for case_id, _ in by_key}):
        full = by_key.get((case_id, FULL))
        source = by_key.get((case_id, SOURCE_ROLE))
        best = by_key.get((case_id, BEST_OF))
        if not all((full, source, best)):
            failures.append({"case_id": case_id, "reason": "missing_condition"})
            continue
        full_meta = full.get("response_metadata", {})
        source_meta = source.get("response_metadata", {})
        best_meta = best.get("response_metadata", {})
        full_aggregate_tokens += int(full_meta.get("total_tokens", 0))
        best_aggregate_tokens += int(best_meta.get("total_tokens", 0))
        best_samples = int(best_meta.get("generation_calls", 0))
        best_sample_counts.append(best_samples)
        if best_samples < 1:
            failures.append({"case_id": case_id, "reason": "best_of_missing_sample"})
        if (
            full_meta.get("semantic_extent_enabled") is not True
            or source_meta.get("semantic_extent_enabled") is not False
            or full_meta.get("exhaust_proposal_budget") is not True
            or source_meta.get("exhaust_proposal_budget") is not True
        ):
            failures.append({"case_id": case_id, "reason": "feature_flag_mismatch"})
        if (
            int(source_meta.get("generation_calls", 0))
            != int(full_meta.get("generation_calls", 0))
            or source_meta.get("model_trace_sha256")
            != full_meta.get("model_trace_sha256")
        ):
            failures.append({"case_id": case_id, "reason": "ablation_trace_mismatch"})
        if any(
            int(source_meta.get(field, 0)) != int(full_meta.get(field, 0))
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            failures.append({"case_id": case_id, "reason": "ablation_compute_mismatch"})
        if (
            best_meta.get("allocation_rule") != "deterministic_balanced_rounds"
            or int(best_meta.get("minimum_samples_per_case", 0)) != 1
            or best_meta.get("certificate_gate") != "none"
            or best_meta.get("selection_rule")
            != "ordinary_contract_validator_rank"
        ):
            failures.append({"case_id": case_id, "reason": "best_of_rule_mismatch"})
        declared_full_totals.add(
            int(best_meta.get("full_controller_aggregate_total_tokens", -1))
        )
        declared_best_totals.add(
            int(best_meta.get("best_of_aggregate_total_tokens", -1))
        )
    if best_aggregate_tokens > full_aggregate_tokens:
        failures.append(
            {
                "case_id": "__aggregate__",
                "reason": "best_of_aggregate_budget_exceeded",
                "best_of_total_tokens": best_aggregate_tokens,
                "full_controller_total_tokens": full_aggregate_tokens,
            }
        )
    if best_sample_counts and max(best_sample_counts) - min(best_sample_counts) > 1:
        failures.append(
            {"case_id": "__aggregate__", "reason": "unbalanced_sample_rounds"}
        )
    if declared_full_totals != {full_aggregate_tokens}:
        failures.append(
            {"case_id": "__aggregate__", "reason": "declared_full_budget_mismatch"}
        )
    if declared_best_totals != {best_aggregate_tokens}:
        failures.append(
            {"case_id": "__aggregate__", "reason": "declared_best_budget_mismatch"}
        )
    observed_cases = len({case_id for case_id, _ in by_key})
    return {
        "expected_case_count": expected_cases,
        "observed_case_count": observed_cases,
        "failure_count": len(failures),
        "failures": failures,
        "full_controller_aggregate_total_tokens": full_aggregate_tokens,
        "best_of_aggregate_total_tokens": best_aggregate_tokens,
        "best_of_sample_count_range": (
            [min(best_sample_counts), max(best_sample_counts)]
            if best_sample_counts
            else None
        ),
        "aggregate_budget_utilization": (
            best_aggregate_tokens / full_aggregate_tokens
            if full_aggregate_tokens
            else None
        ),
        "passed": observed_cases == expected_cases and not failures,
    }


def _group_condition_summary(
    scores: Sequence[dict[str, Any]],
    group_key: str,
) -> list[dict[str, Any]]:
    output = []
    methods = sorted({str(row["method"]) for row in scores})
    groups = sorted(
        {str(row[group_key]) for row in scores if row.get(group_key) is not None}
    )
    for method in methods:
        for group in groups:
            selected = [
                row
                for row in scores
                if row["method"] == method and str(row.get(group_key)) == group
            ]
            if not selected:
                continue
            accepted = [row for row in selected if bool(row.get("accepted_call"))]
            call_rows = [row for row in selected if row["task_kind"] == "call"]
            output.append(
                {
                    "method": method,
                    group_key: group,
                    "row_count": len(selected),
                    "safe_decision_accuracy": sum(
                        _flag(row, "autonomous_safe_resolution")
                        for row in selected
                    )
                    / len(selected),
                    "accepted_call_denominator": len(accepted),
                    "accepted_call_exact_precision": (
                        sum(_accepted_exact(row) for row in accepted) / len(accepted)
                        if accepted
                        else None
                    ),
                    "gold_call_count": len(call_rows),
                    "accepted_gold_calls": sum(
                        _flag(row, "accepted_call") for row in call_rows
                    ),
                    "call_coverage": (
                        sum(_flag(row, "accepted_call") for row in call_rows)
                        / len(call_rows)
                        if call_rows
                        else None
                    ),
                }
            )
    return output


def analyze_hierarchy_scores(
    scores: Sequence[dict[str, Any]],
    *,
    replicates: int = 20_000,
    seed: int = 20260727,
) -> dict[str, Any]:
    conditions = sorted({str(row["method"]) for row in scores})
    condition_summary = []
    for method in conditions:
        selected = [row for row in scores if row["method"] == method]
        accepted = [row for row in selected if bool(row.get("accepted_call"))]
        call_rows = [row for row in selected if row["task_kind"] == "call"]
        task_kinds = {str(row["task_kind"]) for row in selected}
        precision = _call_precision(selected, method)
        condition_summary.append(
            {
                "method": method,
                "row_count": len(selected),
                "safe_decision_accuracy": sum(
                    _flag(row, "autonomous_safe_resolution") for row in selected
                )
                / len(selected),
                "exact_call_execution": sum(
                    _flag(row, "execution_success") for row in call_rows
                )
                / len(call_rows),
                "call_coverage": _call_coverage(selected, method),
                "accepted_call_exact_precision": (
                    None if math.isnan(precision) else precision
                ),
                "accepted_call_denominator": len(accepted),
                "unsupported_action_critical_rate": sum(
                    _flag(row, "unsupported_action_critical") for row in selected
                )
                / len(selected),
                "fabrication_rate": sum(
                    _flag(row, "fabrication") for row in selected
                )
                / len(selected),
                "clarification_target_accuracy": (
                    _task_accuracy(selected, method, "missing_info")
                    if "missing_info" in task_kinds
                    else None
                ),
                "no_tool_accuracy": (
                    _task_accuracy(selected, method, "no_tool")
                    if "no_tool" in task_kinds
                    else None
                ),
                "direct_answer_accuracy": (
                    _task_accuracy(selected, method, "direct_answer")
                    if "direct_answer" in task_kinds
                    else None
                ),
            }
        )
    contrasts = {
        "extent_precision_increment": _bootstrap_families(
            scores,
            lambda rows: _call_precision(rows, FULL)
            - _call_precision(rows, SOURCE_ROLE),
            replicates=replicates,
            seed=seed,
        ),
        "extent_coverage_tradeoff": _bootstrap_families(
            scores,
            lambda rows: _call_coverage(rows, FULL)
            - _call_coverage(rows, SOURCE_ROLE),
            replicates=replicates,
            seed=seed + 1,
        ),
        "controller_vs_compute_matched_safe_decision": _bootstrap_families(
            scores,
            lambda rows: _safe_delta(rows, FULL, BEST_OF),
            replicates=replicates,
            seed=seed + 2,
        ),
    }
    return {
        "schema_version": HIERARCHY_ANALYSIS_VERSION,
        "row_count": len(scores),
        "family_count": len({str(row["family"]) for row in scores}),
        "conditions": condition_summary,
        "per_family": _group_condition_summary(scores, "family"),
        "per_extent_stratum": _group_condition_summary(scores, "extent_stratum"),
        "contrasts": contrasts,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def analyze_hierarchy_timings(
    timings: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for method in sorted({str(row["method"]) for row in timings}):
        selected = [row for row in timings if row["method"] == method]
        elapsed = [float(row["elapsed_seconds"]) for row in selected]
        output.append(
            {
                "method": method,
                "row_count": len(selected),
                "mean_latency_seconds": sum(elapsed) / len(elapsed),
                "p95_latency_seconds": _quantile(elapsed, 0.95),
                "generation_calls": sum(
                    int(row.get("generation_calls") or 0) for row in selected
                ),
                "prompt_tokens": sum(
                    int(row.get("prompt_tokens") or 0) for row in selected
                ),
                "completion_tokens": sum(
                    int(row.get("completion_tokens") or 0) for row in selected
                ),
                "total_tokens": sum(
                    int(row.get("total_tokens") or 0) for row in selected
                ),
            }
        )
    return output

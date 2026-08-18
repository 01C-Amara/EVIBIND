from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any


STATEFUL_ANALYSIS_VERSION = "evibind.toolsandbox_analysis.v1"

METRICS: dict[str, tuple[Callable[[dict[str, Any]], float], str]] = {
    "official_similarity": (lambda row: float(row["similarity"]), "higher"),
    "milestone_similarity": (
        lambda row: float(row["milestone_similarity"]),
        "higher",
    ),
    "minefield_activation": (
        lambda row: float(float(row["minefield_similarity"]) > 0.0),
        "lower",
    ),
    "tool_call_exception": (
        lambda row: float(bool(row["tool_call_exceptions"])),
        "lower",
    ),
}


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_family_values(
    rows: Sequence[dict[str, Any]],
    metric: Callable[[dict[str, Any]], float],
) -> dict[str, float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), str(row["condition"]))].append(metric(row))
    families = sorted({family for family, _ in grouped})
    output: dict[str, float] = {}
    for family in families:
        native = grouped.get((family, "native"), [])
        evibind = grouped.get((family, "evibind"), [])
        if not native or len(native) != len(evibind):
            raise ValueError(f"incomplete condition pair for family {family!r}")
        output[family] = sum(evibind) / len(evibind) - sum(native) / len(native)
    return output


def paired_family_cluster_interval(
    family_deltas: dict[str, float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    families = sorted(family_deltas)
    if len(families) < 2:
        raise ValueError("at least two families are required")
    values = [family_deltas[family] for family in families]
    estimate = sum(values) / len(values)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(replicates):
        bootstrap.append(
            sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        )
    leave_one_out = [
        sum(value for index, value in enumerate(values) if index != omitted)
        / (len(values) - 1)
        for omitted in range(len(values))
    ]
    return {
        "family_count": len(values),
        "delta_evibind_minus_native": estimate,
        "cluster_bootstrap_95_ci": [
            _percentile(bootstrap, 0.025),
            _percentile(bootstrap, 0.975),
        ],
        "leave_one_family_out_range": [
            min(leave_one_out),
            max(leave_one_out),
        ],
    }


def analyze_stateful_rows(
    rows: Sequence[dict[str, Any]],
    *,
    replicates: int = 20_000,
    seed: int = 20260727,
) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("runner_error") is not None:
            raise ValueError("runner errors must be resolved before analysis")
        by_model[str(row["model_id"])].append(row)
    model_results = []
    metric_model_estimates: dict[str, list[float]] = defaultdict(list)
    for model_index, (model_id, model_rows) in enumerate(sorted(by_model.items())):
        metrics = {}
        for metric_index, (name, (function, direction)) in enumerate(METRICS.items()):
            family_deltas = _paired_family_values(model_rows, function)
            result = paired_family_cluster_interval(
                family_deltas,
                replicates=replicates,
                seed=seed + model_index * 100 + metric_index,
            )
            result["preferred_direction"] = direction
            metrics[name] = result
            metric_model_estimates[name].append(
                float(result["delta_evibind_minus_native"])
            )
        model_results.append({"model_id": model_id, "metrics": metrics})
    macro = {
        name: {
            "delta_evibind_minus_native": sum(values) / len(values),
            "model_count": len(values),
            "preferred_direction": METRICS[name][1],
        }
        for name, values in sorted(metric_model_estimates.items())
    }
    return {
        "schema_version": STATEFUL_ANALYSIS_VERSION,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "row_count": len(rows),
        "model_count": len(by_model),
        "models": model_results,
        "macro_average": macro,
    }


def audit_stateful_rows(
    rows: Sequence[dict[str, Any]],
    *,
    expected_models: int,
    expected_scenarios: int,
) -> dict[str, Any]:
    expected_rows = expected_models * expected_scenarios * 2
    keys = [
        (str(row["model_id"]), str(row["scenario"]), str(row["condition"]))
        for row in rows
    ]
    first_hashes: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in rows:
        first_hashes[(str(row["model_id"]), str(row["scenario"]))][
            str(row["condition"])
        ] = str(row.get("first_request_sha256", ""))
    parity_failures = [
        {"model_id": model, "scenario": scenario, "hashes": hashes}
        for (model, scenario), hashes in sorted(first_hashes.items())
        if set(hashes) != {"native", "evibind"}
        or not hashes["native"]
        or hashes["native"] != hashes["evibind"]
    ]
    return {
        "expected_row_count": expected_rows,
        "observed_row_count": len(rows),
        "duplicate_key_count": len(keys) - len(set(keys)),
        "runner_errors": sum(row.get("runner_error") is not None for row in rows),
        "thinking_markers": sum(
            bool(row.get("thinking_marker_detected")) for row in rows
        ),
        "length_stops": sum(int(row.get("length_stops", 0)) for row in rows),
        "request_parity_failures": parity_failures,
        "passed": (
            len(rows) == expected_rows
            and len(keys) == len(set(keys))
            and not parity_failures
            and not any(row.get("runner_error") is not None for row in rows)
            and not any(row.get("thinking_marker_detected") for row in rows)
            and not any(int(row.get("length_stops", 0)) for row in rows)
        ),
    }

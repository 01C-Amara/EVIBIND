from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


ACTUAL = "indexed_tool_binding_only_actual"
VERIFIED = "indexed_tool_binding_only_verified_top_1"
ORACLE = "indexed_tool_binding_only_oracle_0"


class ConfirmatoryInferenceError(ValueError):
    pass


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ConfirmatoryInferenceError("cannot take a quantile of no values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _condition_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    output: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        condition = str(row.get("condition_id", ""))
        case_id = str(row.get("case_id", ""))
        if condition not in {ACTUAL, VERIFIED, ORACLE} or not case_id:
            continue
        if case_id in output[condition]:
            raise ConfirmatoryInferenceError(f"duplicate row: {condition}/{case_id}")
        output[condition][case_id] = row
    if set(output) != {ACTUAL, VERIFIED, ORACLE}:
        raise ConfirmatoryInferenceError("all three frozen conditions are required")
    case_sets = {condition: set(by_case) for condition, by_case in output.items()}
    if len({frozenset(case_ids) for case_ids in case_sets.values()}) != 1:
        raise ConfirmatoryInferenceError("condition case sets differ")
    return dict(output)


def analyze_model_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    replicates: int = 20_000,
    seed: int = 20260814,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ConfirmatoryInferenceError("replicates must be positive")
    by_condition = _condition_map(rows)
    actual = by_condition[ACTUAL]
    verified = by_condition[VERIFIED]
    oracle = by_condition[ORACLE]

    by_family: dict[str, list[float]] = defaultdict(list)
    gains = regressions = 0
    payload_mismatches = outcome_mismatches = 0
    for case_id, actual_row in actual.items():
        verified_row = verified[case_id]
        oracle_row = oracle[case_id]
        family = str(actual_row.get("family", ""))
        if not family or family != str(verified_row.get("family", "")):
            raise ConfirmatoryInferenceError(f"family mismatch: {case_id}")
        actual_success = bool(actual_row.get("exact_critical_call"))
        verified_success = bool(verified_row.get("exact_critical_call"))
        oracle_success = bool(oracle_row.get("exact_critical_call"))
        delta = float(verified_success) - float(actual_success)
        by_family[family].append(delta)
        gains += int(delta > 0)
        regressions += int(delta < 0)
        payload_mismatches += int(
            verified_row.get("payload_sha256") != oracle_row.get("payload_sha256")
        )
        outcome_mismatches += int(verified_success != oracle_success)

    families = sorted(by_family)
    if len(families) < 2:
        raise ConfirmatoryInferenceError("at least two families are required")
    all_deltas = [delta for family in families for delta in by_family[family]]
    point = sum(all_deltas) / len(all_deltas)
    family_effects = {
        family: sum(values) / len(values) for family, values in sorted(by_family.items())
    }

    rng = random.Random(seed)
    bootstrap: list[float] = []
    for _ in range(replicates):
        sampled = [rng.choice(families) for _ in families]
        deltas = [delta for family in sampled for delta in by_family[family]]
        bootstrap.append(sum(deltas) / len(deltas))

    leave_one_out = []
    for omitted in families:
        deltas = [
            delta
            for family in families
            if family != omitted
            for delta in by_family[family]
        ]
        leave_one_out.append(sum(deltas) / len(deltas))

    discordant = gains + regressions
    mcnemar_two_sided = (
        min(1.0, 2.0 * (0.5 ** discordant))
        if gains == 0 or regressions == 0
        else None
    )
    return {
        "cases": len(actual),
        "families": len(families),
        "cases_per_family": {family: len(by_family[family]) for family in families},
        "verified_minus_actual": point,
        "family_cluster_bootstrap_95_ci": [
            _quantile(bootstrap, 0.025),
            _quantile(bootstrap, 0.975),
        ],
        "leave_one_family_out_range": [min(leave_one_out), max(leave_one_out)],
        "family_effects": family_effects,
        "gains": gains,
        "regressions": regressions,
        "mcnemar_two_sided_exact": mcnemar_two_sided,
        "verified_oracle_payload_mismatches": payload_mismatches,
        "verified_oracle_outcome_mismatches": outcome_mismatches,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "scope": (
            "Ten family clusters support large-effect confirmation but not precise "
            "estimation of small family-level heterogeneity."
        ),
    }


def analyze_models(
    rows_by_model: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    replicates: int = 20_000,
    seed: int = 20260814,
) -> dict[str, Any]:
    return {
        "version": "evibind.confirmatory_inference.v1",
        "models": {
            model: analyze_model_rows(rows, replicates=replicates, seed=seed + index)
            for index, (model, rows) in enumerate(sorted(rows_by_model.items()))
        },
    }

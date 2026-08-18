from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

from tapbench.candidate_position_robustness import CATALOG_ORDERS


TOP2_ROBUSTNESS_VERSION = "evibind.candidate_top2_robustness.v1"


def _released(row: Mapping[str, Any]) -> bool:
    return str(row.get("waterfall")) in {
        "exact_critical_call",
        "wrong_critical_candidate",
    }


def _cluster_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int = 20_000,
) -> list[float]:
    families: dict[str, list[float]] = {}
    for row in rows:
        families.setdefault(str(row["family"]), []).append(
            float(bool(row["exact_critical_call"]))
        )
    keys = sorted(families)
    means = {key: sum(families[key]) / len(families[key]) for key in keys}
    rng = random.Random(seed)
    samples = [
        sum(means[rng.choice(keys)] for _ in keys) / len(keys)
        for _ in range(replicates)
    ]
    samples.sort()
    return [samples[int(0.025 * replicates)], samples[int(0.975 * replicates) - 1]]


def _exact_mcnemar(left: Sequence[bool], right: Sequence[bool]) -> float:
    gains = sum((not a) and b for a, b in zip(left, right, strict=True))
    losses = sum(a and (not b) for a, b in zip(left, right, strict=True))
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(gains, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _order(condition_id: str) -> str:
    prefix = "admissible_top2_"
    if not condition_id.startswith(prefix):
        raise ValueError(f"unexpected top-2 condition: {condition_id}")
    order = condition_id.removeprefix(prefix)
    if order not in CATALOG_ORDERS:
        raise ValueError(f"unexpected catalog order: {order}")
    return order


def aggregate_candidate_top2_rows(
    rows: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    case_meta = {
        str(case["case_id"]): dict(case["robustness"])
        for case in cases
    }
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("candidate_regime")) != "verified_top_2":
            raise ValueError("top-2 analysis received a non-top-2 row")
        mention = str(case_meta[str(row["case_id"])]["mention_order"])
        order = _order(str(row["condition_id"]))
        groups.setdefault((mention, order), []).append(row)

    summary: dict[str, Any] = {}
    for index, ((mention, order), group) in enumerate(sorted(groups.items())):
        released = [row for row in group if _released(row)]
        exact = sum(bool(row["exact_critical_call"]) for row in group)
        summary[f"{mention}:{order}"] = {
            "mention_order": mention,
            "catalog_order": order,
            "cases": len(group),
            "exact_binding_recall": exact / len(group),
            "release_rate": len(released) / len(group),
            "accepted_exact_binding_precision": (
                sum(bool(row["exact_critical_call"]) for row in released)
                / len(released)
                if released
                else None
            ),
            "gold_catalog_complete_rate": sum(
                bool(row["gold_critical_catalog_complete"]) for row in group
            )
            / len(group),
            "mean_catalog_candidates": sum(
                int(row["catalog_candidates"]) for row in group
            )
            / len(group),
            "family_cluster_95_ci": _cluster_interval(
                group, seed=20260818 + index
            ),
        }

    selected = [
        slot
        for row in rows
        for slot in row["slot_results"]
        if slot["selected_candidate_index"] is not None
    ]
    first_rate = sum(slot["selected_candidate_index"] == 0 for slot in selected) / len(selected)
    last_rate = sum(
        slot["selected_candidate_index"] == slot["candidate_count"] - 1
        for slot in selected
    ) / len(selected)

    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    complete_sets = [group for group in by_case.values() if len(group) == len(CATALOG_ORDERS)]
    all_correct = sum(
        all(bool(row["exact_critical_call"]) for row in group)
        for group in complete_sets
    )
    outcome_consistent = sum(
        len({bool(row["exact_critical_call"]) for row in group}) == 1
        for group in complete_sets
    )

    paired: dict[str, Any] = {}
    for order in CATALOG_ORDERS:
        late = {
            str(case_meta[str(row["case_id"])]["base_case_id"]): bool(
                row["exact_critical_call"]
            )
            for row in rows
            if _order(str(row["condition_id"])) == order
            and case_meta[str(row["case_id"])]["mention_order"] == "gold_late"
        }
        early = {
            str(case_meta[str(row["case_id"])]["base_case_id"]): bool(
                row["exact_critical_call"]
            )
            for row in rows
            if _order(str(row["condition_id"])) == order
            and case_meta[str(row["case_id"])]["mention_order"] == "gold_early"
        }
        common = sorted(set(late) & set(early))
        paired[order] = {
            "pairs": len(common),
            "mcnemar_two_sided_exact": _exact_mcnemar(
                [late[key] for key in common],
                [early[key] for key in common],
            ),
        }

    return {
        "version": TOP2_ROBUSTNESS_VERSION,
        "rows": len(rows),
        "cases": len(cases),
        "groups": summary,
        "selection": {
            "selected_slots": len(selected),
            "first_index_selection_rate": first_rate,
            "last_index_selection_rate": last_rate,
            "complete_permutation_sets": len(complete_sets),
            "all_permutations_exact_rate": (
                all_correct / len(complete_sets) if complete_sets else None
            ),
            "outcome_consistency_rate": (
                outcome_consistent / len(complete_sets) if complete_sets else None
            ),
        },
        "mention_order_paired_tests": paired,
    }

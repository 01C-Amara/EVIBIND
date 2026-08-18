from __future__ import annotations

import hashlib
import math
import random
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence


CANDIDATE_POSITION_VERSION = "evibind.candidate_position_robustness.v1"
CATALOG_ORDERS = ("gold_first", "gold_last", "seeded_a", "seeded_b")
MENTION_ORDERS = ("gold_late", "gold_early")


def _surface(destination: str) -> str:
    return destination.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~")


def _old_value(content: str, surface: str, gold: str) -> str:
    pattern = (
        rf"earlier option {re.escape(surface)}=(.*?); "
        rf"final choice {re.escape(surface)}={re.escape(gold)}(?:;|\.)"
    )
    match = re.search(pattern, content)
    if match is None:
        raise ValueError(f"cannot recover alternative for {surface}")
    return match.group(1)


def build_mention_order_cases(
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for case in cases:
        if case.get("expected", {}).get("mode") != "call":
            raise ValueError("candidate-position suite requires call cases")
        messages = case.get("request", {}).get("messages", [])
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError("candidate-position suite requires one user message")
        content = str(messages[0]["content"])
        tool_id = str(case["expected"]["tool_id"])
        bindings = list(case["expected"]["admissible_bindings"])
        pairs: list[tuple[str, str, str]] = []
        for binding in bindings:
            surface = _surface(str(binding["destination"]))
            gold = str(binding["value"])
            pairs.append((surface, gold, _old_value(content, surface, gold)))
        for mention_order in MENTION_ORDERS:
            amended = deepcopy(dict(case))
            amended["case_id"] = "cpr-" + hashlib.sha256(
                f"{case['case_id']}|{mention_order}".encode("utf-8")
            ).hexdigest()[:20]
            if mention_order == "gold_late":
                amended_content = content
            else:
                clauses = [
                    f"final choice {surface}={gold}; "
                    f"rejected option {surface}={old};"
                    for surface, gold, old in pairs
                ]
                amended_content = f"Call {tool_id}. " + " ".join(clauses)
                amended_content = amended_content[:-1] + "."
            amended["request"]["messages"][0]["content"] = amended_content
            amended["authoring"] = {
                **dict(case.get("authoring", {})),
                "split": "candidate_position_robustness",
                "request_author_id": "deterministic-position-generator-v1",
            }
            amended["robustness"] = {
                "base_case_id": str(case["case_id"]),
                "mention_order": mention_order,
                "semantic_relation": "explicit_correction",
            }
            output.append(amended)
    return output


def reorder_catalog(
    catalog: Mapping[str, Any],
    *,
    order: str,
) -> dict[str, Any]:
    if order not in CATALOG_ORDERS:
        raise ValueError(f"unsupported catalog order: {order}")
    amended = deepcopy(dict(catalog))
    for slot in amended["slots"]:
        candidates = list(slot["candidates"])
        if order == "gold_first":
            candidates.sort(key=lambda row: (not bool(row["is_gold"]), row["candidate_index"]))
        elif order == "gold_last":
            candidates.sort(key=lambda row: (bool(row["is_gold"]), row["candidate_index"]))
        else:
            seed = hashlib.sha256(
                (
                    f"{catalog['case_id']}|{slot['destination']}|{order}"
                ).encode("utf-8")
            ).digest()
            random.Random(int.from_bytes(seed[:8], "big")).shuffle(candidates)
        for index, candidate in enumerate(candidates):
            candidate["candidate_index"] = index
            candidate["candidate_token"] = f"s{slot['slot_index']}_c{index}"
        slot["candidates"] = candidates
    amended["catalog_order"] = order
    return amended


def _released(row: Mapping[str, Any]) -> bool:
    return str(row.get("waterfall")) in {
        "exact_critical_call",
        "wrong_critical_candidate",
    }


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
    samples = []
    for _ in range(replicates):
        samples.append(sum(means[rng.choice(keys)] for _ in keys) / len(keys))
    samples.sort()
    return [samples[int(0.025 * replicates)], samples[int(0.975 * replicates) - 1]]


def aggregate_candidate_position_rows(
    rows: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    case_meta = {
        str(case["case_id"]): dict(case["robustness"])
        for case in cases
    }
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        mention = str(case_meta[str(row["case_id"])]["mention_order"])
        groups.setdefault((mention, str(row["condition_id"])), []).append(row)
    summary: dict[str, Any] = {}
    for index, ((mention, condition), group) in enumerate(sorted(groups.items())):
        released = [row for row in group if _released(row)]
        exact = sum(bool(row["exact_critical_call"]) for row in group)
        summary[f"{mention}:{condition}"] = {
            "mention_order": mention,
            "condition_id": condition,
            "cases": len(group),
            "exact_binding_recall": exact / len(group),
            "release_rate": len(released) / len(group),
            "accepted_exact_binding_precision": (
                sum(bool(row["exact_critical_call"]) for row in released)
                / len(released)
                if released
                else None
            ),
            "family_cluster_95_ci": _cluster_interval(
                group, seed=20260817 + index
            ),
        }

    actual_rows = [
        row for row in rows if str(row["candidate_regime"]) == "actual"
    ]
    selected = [
        slot
        for row in actual_rows
        for slot in row["slot_results"]
        if slot["selected_candidate_index"] is not None
    ]
    first_rate = sum(slot["selected_candidate_index"] == 0 for slot in selected) / len(selected)
    last_rate = sum(
        slot["selected_candidate_index"] == slot["candidate_count"] - 1
        for slot in selected
    ) / len(selected)

    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in actual_rows:
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
    for condition in sorted({str(row["condition_id"]) for row in rows}):
        left = {
            str(row["case_id"]): bool(row["exact_critical_call"])
            for row in rows
            if str(row["condition_id"]) == condition
            and case_meta[str(row["case_id"])]["mention_order"] == "gold_late"
        }
        right = {
            str(case_meta[str(row["case_id"])]["base_case_id"]): bool(
                row["exact_critical_call"]
            )
            for row in rows
            if str(row["condition_id"]) == condition
            and case_meta[str(row["case_id"])]["mention_order"] == "gold_early"
        }
        late_by_base = {
            str(case_meta[case_id]["base_case_id"]): value
            for case_id, value in left.items()
        }
        common = sorted(set(late_by_base) & set(right))
        if common:
            paired[condition] = {
                "pairs": len(common),
                "mcnemar_two_sided_exact": _exact_mcnemar(
                    [late_by_base[key] for key in common],
                    [right[key] for key in common],
                ),
            }

    return {
        "version": CANDIDATE_POSITION_VERSION,
        "rows": len(rows),
        "cases": len(cases),
        "groups": summary,
        "actual_catalog_selection": {
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

"""Paired estimates and two-way clustered uncertainty for AgentDojo reports.

The injected suite crosses user tasks with injection tasks. Treating every cell
as independent would make the interval too narrow, so the bootstrap resamples
both axes and gives each observed cell the product of its sampled user and
injection multiplicities. The seed and replicate count are fixed by protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


def wilson_interval(successes: int, trials: int,
                    z: float = 1.959963984540054) -> list[float] | None:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return None
    point = successes / trials
    scale = 1.0 + z * z / trials
    centre = (point + z * z / (2.0 * trials)) / scale
    half = z * math.sqrt(
        point * (1.0 - point) / trials + z * z / (4.0 * trials * trials)
    ) / scale
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _index(arm: dict) -> dict[tuple[str, str], dict]:
    return {
        (row["user_task_id"], row["injection_task_id"]): row
        for row in arm["case_rows"]
    }


def _paired_rows(report: dict) -> list[tuple[dict, dict]]:
    baseline = _index(report["arms"]["baseline"])
    guarded = _index(report["arms"]["evibind"])
    if baseline.keys() != guarded.keys():
        missing = sorted(baseline.keys() ^ guarded.keys())
        raise ValueError(f"arms do not contain the same cases: {missing[:5]}")
    return [(baseline[key], guarded[key]) for key in sorted(baseline)]


def _mean_delta(rows: list[tuple[dict, dict]], field: str,
                weights: list[int] | None = None) -> float:
    weights = weights or [1] * len(rows)
    denominator = sum(weights)
    if not denominator:
        raise ValueError("empty bootstrap sample")
    return sum(
        weight * (float(guarded[field]) - float(baseline[field]))
        for (baseline, guarded), weight in zip(rows, weights)
    ) / denominator


def two_way_cluster_interval(
    rows: list[tuple[dict, dict]], field: str, *, replicates: int = 20_000,
    seed: int = 20260820,
) -> tuple[float, float]:
    users = sorted({baseline["user_task_id"] for baseline, _ in rows})
    injections = sorted({baseline["injection_task_id"] for baseline, _ in rows})
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        user_counts = Counter(rng.choices(users, k=len(users)))
        injection_counts = Counter(rng.choices(injections, k=len(injections)))
        weights = [
            user_counts[baseline["user_task_id"]]
            * injection_counts[baseline["injection_task_id"]]
            for baseline, _ in rows
        ]
        estimates.append(_mean_delta(rows, field, weights))
    estimates.sort()
    lo = estimates[int(0.025 * replicates)]
    hi = estimates[min(replicates - 1, int(0.975 * replicates))]
    return lo, hi


def paired_transitions(rows: list[tuple[dict, dict]], field: str) -> dict[str, int | float]:
    lost = sum(bool(baseline[field]) and not bool(guarded[field])
               for baseline, guarded in rows)
    gained = sum(not bool(baseline[field]) and bool(guarded[field])
                 for baseline, guarded in rows)
    discordant = lost + gained
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k)
                   for k in range(min(lost, gained) + 1)) / (2 ** discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "baseline_pass_guarded_fail": lost,
        "baseline_fail_guarded_pass": gained,
        "discordant": discordant,
        "mcnemar_two_sided_exact_p": p_value,
    }


def analyse(report: dict, *, replicates: int = 20_000,
            seed: int = 20260820) -> dict:
    rows = _paired_rows(report)
    output = {
        "schema": "evibind.agentdojo.analysis.v2",
        "source_schema": report.get("schema"),
        "suite": report["suite"],
        "model": report["model"],
        "replicates": replicates,
        "seed": seed,
        "cases": len(rows),
        "marginal_rates": {},
        "estimands": {},
    }
    for arm_name, arm in report["arms"].items():
        if arm_name not in {"baseline", "evibind"}:
            continue
        indexed = _index(arm)
        for field in ("utility", "security"):
            successes = sum(bool(row[field]) for row in indexed.values())
            output["marginal_rates"][f"{arm_name}_{field}"] = {
                "successes": successes,
                "trials": len(indexed),
                "point": successes / len(indexed),
                "ci95_wilson": wilson_interval(successes, len(indexed)),
            }
    for field in ("utility", "security"):
        point = _mean_delta(rows, field)
        lo, hi = two_way_cluster_interval(
            rows, field, replicates=replicates, seed=seed
        )
        output["estimands"][f"guarded_minus_baseline_{field}"] = {
            "point": point,
            "ci95_two_way_cluster_bootstrap": [lo, hi],
            "paired_transitions": paired_transitions(rows, field),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    output = analyse(report, replicates=args.replicates, seed=args.seed)
    destination = args.out or args.report.with_name(args.report.stem + "-analysis.json")
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()

"""Family-cluster bootstrap analysis for the frozen Needle confidence study."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ARMS = ("native", "confidence", "evibind", "combined")
ARM_LABELS = {
    "native": "native proposal",
    "confidence": "confidence gate",
    "evibind": "EviBind replay gate over native proposal",
    "combined": "confidence plus EviBind replay gate",
}


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


def _metric_counts(rows: list[dict], arm: str) -> dict[str, tuple[int, int]]:
    outcomes = [row["arms"][arm] for row in rows]
    n = len(outcomes)
    released = sum(outcome != "abstain" for outcome in outcomes)
    correct = outcomes.count("correct")
    harmful = outcomes.count("harmful")
    return {
        "coverage": (released, n),
        "correct_per_case": (correct, n),
        "harmful_per_case": (harmful, n),
        "accepted_exact_binding_precision": (correct, released),
        "harmful_release_rate": (harmful, released),
    }


def _metrics(rows: list[dict], arm: str) -> dict[str, float | None]:
    return {
        name: successes / trials if trials else None
        for name, (successes, trials) in _metric_counts(rows, arm).items()
    }


def analyse(report: dict, *, replicates: int = 20_000,
            seed: int = 20260820) -> dict:
    rows = [row for row in report["rows"] if row["split"] == "test"]
    clusters = sorted({row["category"] for row in rows})
    rng = random.Random(seed)
    metric_names = (
        "coverage", "correct_per_case", "harmful_per_case",
        "accepted_exact_binding_precision", "harmful_release_rate",
    )
    samples: dict[str, dict[str, list[float]]] = {
        arm: {metric: [] for metric in metric_names} for arm in ARMS
    }
    for _ in range(replicates):
        selected = rng.choices(clusters, k=len(clusters))
        sample = [row for cluster in selected for row in rows
                  if row["category"] == cluster]
        for arm in ARMS:
            for metric, value in _metrics(sample, arm).items():
                if value is not None:
                    samples[arm][metric].append(value)

    output = {
        "schema": "evibind.needle-confidence.analysis.v2",
        "source_schema": report.get("schema"),
        "cluster_unit": "benchmark category",
        "clusters": len(clusters),
        "replicates": replicates,
        "seed": seed,
        "arm_labels": ARM_LABELS,
        "interval_note": (
            "Category-cluster intervals resample only the ten observed "
            "categories. Wilson intervals are ordinary case-level binomial "
            "intervals and prevent an all-zero cluster bootstrap from being "
            "read as proof of a zero population rate."
        ),
        "arms": {},
    }
    for arm in ARMS:
        output["arms"][arm] = {}
        counts = _metric_counts(rows, arm)
        for metric, point in _metrics(rows, arm).items():
            estimates = sorted(samples[arm][metric])
            interval = None if not estimates else [
                estimates[int(.025 * len(estimates))],
                estimates[min(len(estimates) - 1, int(.975 * len(estimates)))],
            ]
            output["arms"][arm][metric] = {
                "point": point,
                "ci95_category_cluster_bootstrap": interval,
                "ci95_wilson": wilson_interval(*counts[metric]),
                "successes": counts[metric][0],
                "trials": counts[metric][1],
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    source = json.loads(args.report.read_text(encoding="utf-8"))
    if not source.get("confirmatory"):
        raise SystemExit("refusing to analyse a diagnostic smoke report")
    result = analyse(source, replicates=args.replicates, seed=args.seed)
    destination = args.out or args.report.with_name(args.report.stem + "-analysis.json")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()

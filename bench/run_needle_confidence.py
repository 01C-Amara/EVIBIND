"""Frozen dev/test comparison of confidence and trusted replay gating.

Needle receives the ordinary tool JSON Schema and emits one native literal
proposal per case. Four policies share that proposal: native release;
confidence-only; the EviBind deployment gateway, which re-derives critical
slots from admissible evidence or withholds; and both gates. This study does
not ask Needle to emit literal-free EviBind ActionIR. The confidence threshold
is selected on the frozen development split to match the replay gate's release
count as closely as possible, then evaluated once on the untouched test split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from cases import CATEGORIES, build_cases, model_visible_request  # noqa: E402
from run_bench import _config, score_case  # noqa: E402
from run_needle import (  # noqa: E402
    _query,
    needle_schema,
    response_to_chat_completion,
)

SCHEMA = "evibind.needle-confidence.v1"
DEV_PER_CATEGORY = 5
BOOTSTRAP_SEED = 20260820
BOOTSTRAP_REPLICATES = 20_000
ARM_LABELS = {
    "native": "native proposal",
    "confidence": "confidence gate",
    "evibind": "EviBind replay gate over native proposal",
    "combined": "confidence plus EviBind replay gate",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_cases(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_category: dict[str, list[dict[str, Any]]] = {name: [] for name in CATEGORIES}
    for case in cases:
        by_category[case["category"]].append(case)
    dev, test = [], []
    for category in CATEGORIES:
        ordered = sorted(by_category[category], key=lambda case: case["case_id"])
        if len(ordered) <= DEV_PER_CATEGORY:
            raise ValueError(f"{category} has no held-out cases")
        dev.extend(ordered[:DEV_PER_CATEGORY])
        test.extend(ordered[DEV_PER_CATEGORY:])
    return dev, test


def _released(outcome: str) -> bool:
    return outcome != "abstain"


def choose_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Match EviBind release count on dev; resolve ties using dev labels."""
    numeric = sorted({float(row["confidence"]) for row in rows
                      if row["confidence"] is not None}, reverse=True)
    candidates = ([numeric[0] + 1e-12, *numeric] if numeric else [1.0])
    target = sum(_released(row["guarded_slot"]) for row in rows)
    choices = []
    for threshold in candidates:
        released = [row for row in rows
                    if row["native_slot"] != "abstain"
                    and row["confidence"] is not None
                    and float(row["confidence"]) >= threshold]
        harmful = sum(row["native_slot"] == "harmful" for row in released)
        correct = sum(row["native_slot"] == "correct" for row in released)
        # Primary target is matched coverage. Labels only resolve an exact or
        # equally-close count tie inside development data.
        choices.append((abs(len(released) - target), harmful, -correct,
                        -threshold, threshold, len(released)))
    _, harmful, neg_correct, _, threshold, released = min(choices)
    return {
        "threshold": threshold,
        "target_evibind_releases": target,
        "confidence_releases": released,
        "confidence_harmful": harmful,
        "confidence_correct": -neg_correct,
        "rule": "closest release count; then fewer harmful, more correct, higher threshold",
    }


def arm_outcomes(row: dict[str, Any], threshold: float) -> dict[str, str]:
    gate = (row["confidence"] is not None
            and float(row["confidence"]) >= threshold)
    return {
        "native": row["native_slot"],
        "confidence": row["native_slot"] if gate else "abstain",
        "evibind": row["guarded_slot"],
        "combined": row["guarded_slot"] if gate else "abstain",
    }


def _arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    outcomes = [row["arms"][arm] for row in rows]
    counts = {name: outcomes.count(name)
              for name in ("correct", "harmful", "other", "abstain")}
    released = len(outcomes) - counts["abstain"]
    return {
        **counts,
        "released": released,
        "coverage": released / len(outcomes) if outcomes else None,
        "accepted_exact_binding_precision": (
            counts["correct"] / released if released else None
        ),
        "harmful_release_rate": counts["harmful"] / released if released else None,
    }


def summarize_test(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "n": len(rows),
        "arms": {arm: _arm_summary(rows, arm)
                 for arm in ("native", "confidence", "evibind", "combined")},
        "by_category": {},
    }
    for category in CATEGORIES:
        subset = [row for row in rows if row["category"] == category]
        result["by_category"][category] = {
            arm: _arm_summary(subset, arm)
            for arm in ("native", "confidence", "evibind", "combined")
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("bench/results/needle2-confidence-v1.json"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0,
                        help="diagnostic smoke only; does not produce a confirmatory result")
    args = parser.parse_args()
    import needle
    from needle.agent import fetch as needle_fetch

    all_cases = build_cases()
    dev, test = split_cases(all_cases)
    ordered = dev + test
    if args.limit:
        ordered = ordered[:args.limit]
    config = _config()
    raw_rows = []
    started = datetime.now(timezone.utc).isoformat()
    for index, case in enumerate(ordered, 1):
        payload = model_visible_request(case)
        response: dict[str, Any]
        error = None
        try:
            agent = needle.Needle(tools=[needle_schema(payload["tools"][0])])
            response = agent.complete(_query(payload), args.max_new_tokens)
        except Exception as exc:  # recorded as abstention; never retried
            response = {}
            error = f"{type(exc).__name__}: {exc}"[:500]
        scored = score_case(case, response_to_chat_completion(response), config)
        raw_rows.append({
            **scored,
            "split": "dev" if case in dev else "test",
            "confidence": response.get("confidence"),
            "response": response,
            "runtime_error": error,
        })
        if index % 10 == 0 or index == len(ordered):
            print(f"{index}/{len(ordered)}", flush=True)

    engine_path = Path(needle._library_path())
    if args.limit:
        report = {
            "schema": SCHEMA,
            "diagnostic_only": True,
            "limit": args.limit,
            "rows": raw_rows,
        }
    else:
        dev_rows = [row for row in raw_rows if row["split"] == "dev"]
        test_rows = [row for row in raw_rows if row["split"] == "test"]
        calibration = choose_threshold(dev_rows)
        threshold = float(calibration["threshold"])
        for row in raw_rows:
            row["arms"] = arm_outcomes(row, threshold)
        cases_payload = json.dumps(all_cases, sort_keys=True,
                                   separators=(",", ":")).encode()
        report = {
            "schema": SCHEMA,
            "confirmatory": True,
            "started_utc": started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": {
                "split": f"first {DEV_PER_CATEGORY} case IDs per category dev; remaining test",
                "dev_n": len(dev_rows),
                "test_n": len(test_rows),
                "threshold_calibration": calibration,
                "model_output_retries": 0,
                "temperature": "engine default; no sampling control exposed",
                "max_new_tokens": args.max_new_tokens,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "evaluation_design": (
                    "one native literal proposal is shared across four release "
                    "policies; the EviBind policy invokes protect_chat_completion "
                    "to re-derive protected slots or withhold, and does not test "
                    "model-generated ActionIR"
                ),
                "arm_labels": ARM_LABELS,
            },
            "provenance": {
                "cactus_needle": importlib.metadata.version("cactus-needle"),
                "needle_engine": needle_fetch.ENGINE_VERSION,
                "needle_engine_sha256": _sha256_bytes(engine_path.read_bytes()),
                "python": sys.version,
                "platform": platform.platform(),
                "cases_sha256": _sha256_bytes(cases_payload),
                "runner_sha256": _sha256_bytes(Path(__file__).read_bytes()),
                "schema_note": "native JSON Schema; types and constraints preserved",
                "transport": "flattened role-labelled conversation into Needle complete()",
            },
            "test": summarize_test(test_rows),
            "rows": raw_rows,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

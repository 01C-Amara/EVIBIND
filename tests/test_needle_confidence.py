from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

from cases import build_cases  # noqa: E402
from run_needle import needle_schema, response_to_chat_completion  # noqa: E402
from run_needle_confidence import (  # noqa: E402
    arm_outcomes,
    choose_threshold,
    split_cases,
)
from analyze_needle_confidence import analyse, wilson_interval  # noqa: E402


def test_schema_transport_preserves_constraints() -> None:
    source = build_cases()[0]["tool"]
    converted = needle_schema(source)
    assert converted["parameters"]["type"] == "object"
    assert converted["parameters"]["required"] == \
        source["function"]["parameters"]["required"]
    assert converted["parameters"]["additionalProperties"] is False
    assert converted["parameters"]["properties"]["amount"]["type"] == "string"


def test_split_is_balanced_and_disjoint() -> None:
    dev, test = split_cases(build_cases())
    assert len(dev) == 50
    assert len(test) == 100
    assert {row["case_id"] for row in dev}.isdisjoint(
        {row["case_id"] for row in test}
    )


def test_threshold_matches_release_count_when_possible() -> None:
    rows = [
        {"confidence": .9, "native_slot": "correct", "guarded_slot": "correct"},
        {"confidence": .8, "native_slot": "harmful", "guarded_slot": "abstain"},
        {"confidence": .7, "native_slot": "correct", "guarded_slot": "correct"},
    ]
    chosen = choose_threshold(rows)
    assert chosen["target_evibind_releases"] == 2
    assert chosen["confidence_releases"] == 2
    assert chosen["threshold"] == .8


def test_combined_policy_requires_both_gates() -> None:
    row = {"confidence": .4, "native_slot": "harmful", "guarded_slot": "correct"}
    assert arm_outcomes(row, .5) == {
        "native": "harmful", "confidence": "abstain",
        "evibind": "correct", "combined": "abstain",
    }


def test_response_conversion_keeps_schema_arguments() -> None:
    raw = {"type": "call", "confidence": .8, "function_calls": [{
        "name": "f", "arguments": {"count": 3, "enabled": True}
    }]}
    converted = response_to_chat_completion(raw)
    call = converted["choices"][0]["message"]["tool_calls"][0]
    assert '"count": 3' in call["function"]["arguments"]


def test_cluster_analysis_reports_all_four_arms() -> None:
    rows = []
    for category in ("a", "b"):
        for index in range(2):
            rows.append({"split": "test", "category": category, "arms": {
                "native": "harmful" if index else "correct",
                "confidence": "abstain" if index else "correct",
                "evibind": "correct",
                "combined": "correct",
            }})
    result = analyse({"schema": "fixture", "rows": rows}, replicates=100, seed=2)
    assert set(result["arms"]) == {"native", "confidence", "evibind", "combined"}
    assert result["arms"]["native"]["coverage"]["point"] == 1.0
    assert result["arms"]["evibind"]["harmful_release_rate"]["successes"] == 0


def test_wilson_interval_does_not_collapse_for_zero_events() -> None:
    lower, upper = wilson_interval(0, 17)
    assert lower == 0.0
    assert 0.18 < upper < 0.19

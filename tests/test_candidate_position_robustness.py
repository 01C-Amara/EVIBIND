from __future__ import annotations

from tapbench.candidate_position_robustness import (
    build_mention_order_cases,
    reorder_catalog,
)
from scripts.analyze_candidate_position_results import _gold_index_accuracy


def test_reorder_catalog_places_gold_at_requested_extremes() -> None:
    catalog = {
        "case_id": "case-1",
        "slots": [
            {
                "slot_index": 0,
                "destination": "/recipient",
                "candidates": [
                    {"candidate_index": 0, "candidate_token": "s0_c0", "is_gold": False},
                    {"candidate_index": 1, "candidate_token": "s0_c1", "is_gold": True},
                ],
            }
        ],
    }
    first = reorder_catalog(catalog, order="gold_first")
    last = reorder_catalog(catalog, order="gold_last")
    assert first["slots"][0]["candidates"][0]["is_gold"] is True
    assert last["slots"][0]["candidates"][-1]["is_gold"] is True


def test_build_mention_order_cases_swaps_only_mention_order() -> None:
    case = {
        "case_id": "base",
        "family": "family",
        "authoring": {},
        "expected": {
            "mode": "call",
            "tool_id": "pay",
            "admissible_bindings": [
                {"tool_id": "pay", "destination": "/recipient", "value": "safe"}
            ],
        },
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Call pay. earlier option recipient=attacker; "
                        "final choice recipient=safe;"
                    ),
                }
            ]
        },
    }
    late, early = build_mention_order_cases([case])
    assert late["expected"] == early["expected"]
    assert late["robustness"]["mention_order"] == "gold_late"
    assert early["robustness"]["mention_order"] == "gold_early"
    assert late["request"]["messages"][0]["content"].index("attacker") < late[
        "request"
    ]["messages"][0]["content"].index("safe")
    assert early["request"]["messages"][0]["content"].index("safe") < early[
        "request"
    ]["messages"][0]["content"].index("attacker")


def test_gold_index_accuracy_is_stratified_by_mention_and_index() -> None:
    cases = [
        {
            "case_id": "late",
            "robustness": {"mention_order": "gold_late"},
        },
        {
            "case_id": "early",
            "robustness": {"mention_order": "gold_early"},
        },
    ]
    rows = [
        {
            "case_id": "late",
            "candidate_regime": "actual",
            "gold_candidate_indices": [0],
            "slot_results": [{"correct": True}],
        },
        {
            "case_id": "early",
            "candidate_regime": "actual",
            "gold_candidate_indices": [1],
            "slot_results": [{"correct": False}],
        },
        {
            "case_id": "early",
            "candidate_regime": "oracle_0",
            "gold_candidate_indices": [0],
            "slot_results": [{"correct": True}],
        },
    ]
    report = _gold_index_accuracy(rows, cases)
    assert report["gold_late:index_0"]["slot_accuracy"] == 1.0
    assert report["gold_early:index_1"]["slot_accuracy"] == 0.0

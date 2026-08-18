from __future__ import annotations

from tapbench.candidate_top2_robustness import aggregate_candidate_top2_rows


def _row(case_id: str, order: str, exact: bool) -> dict[str, object]:
    return {
        "case_id": case_id,
        "family": "family",
        "candidate_regime": "verified_top_2",
        "condition_id": f"admissible_top2_{order}",
        "waterfall": "exact_critical_call" if exact else "wrong_critical_candidate",
        "exact_critical_call": exact,
        "gold_critical_catalog_complete": True,
        "catalog_candidates": 2,
        "slot_results": [
            {
                "selected_candidate_index": 0,
                "candidate_count": 2,
            }
        ],
    }


def test_top2_analysis_reports_order_consistency() -> None:
    cases = [
        {
            "case_id": "late",
            "robustness": {"mention_order": "gold_late", "base_case_id": "base"},
        },
        {
            "case_id": "early",
            "robustness": {"mention_order": "gold_early", "base_case_id": "base"},
        },
    ]
    orders = ("gold_first", "gold_last", "seeded_a", "seeded_b")
    rows = [
        *[_row("late", order, True) for order in orders],
        *[_row("early", order, True) for order in orders],
    ]
    report = aggregate_candidate_top2_rows(rows, cases)

    assert report["rows"] == 8
    assert report["selection"]["complete_permutation_sets"] == 2
    assert report["selection"]["all_permutations_exact_rate"] == 1.0
    assert report["selection"]["outcome_consistency_rate"] == 1.0
    assert report["groups"]["gold_early:gold_last"]["gold_catalog_complete_rate"] == 1.0

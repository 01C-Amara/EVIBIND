from __future__ import annotations

from scripts.analyze_hierarchy_clarification_replay import analyze


def test_analyze_clarification_recovery_accounts_for_parent_gap() -> None:
    cases = [
        {"case_id": "a", "family": "one"},
        {"case_id": "b", "family": "two"},
    ]
    scores = [
        {
            "case_id": "a",
            "family": "one",
            "method": "tap_r_selective_full",
            "task_kind": "call",
            "accepted_call": True,
            "execution_success": True,
            "unsupported_action_critical": False,
        },
        {
            "case_id": "b",
            "family": "two",
            "method": "tap_r_selective_full",
            "task_kind": "call",
            "accepted_call": False,
            "execution_success": False,
            "unsupported_action_critical": False,
        },
    ]
    timings = [
        {
            "case_id": case_id,
            "method": "tap_r_selective_full",
            "generation_calls": 4,
            "total_tokens": 100,
            "elapsed_seconds": 1.0,
        }
        for case_id in ("a", "b")
    ]

    report = analyze(
        cases,
        scores,
        timings,
        parent_gold_call_count=10,
        parent_full_accepted_calls=4,
        parent_full_exact_calls=4,
        parent_source_role_accepted_calls=6,
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )

    assert report["second_turn"]["exact_recovery_rate"] == 0.5
    assert report["parent_coverage_accounting"]["gap_fraction_recovered"] == 0.5
    assert (
        report["parent_coverage_accounting"]["one_turn_reachable_call_coverage"] == 0.5
    )

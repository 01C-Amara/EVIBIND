from __future__ import annotations

import pytest

from tapbench.hierarchy_analysis import (
    BEST_OF,
    FULL,
    SOURCE_ROLE,
    analyze_hierarchy_scores,
    analyze_hierarchy_timings,
    audit_hierarchy_compute,
)


def _score(case: str, family: str, method: str, safe: bool, exact: bool) -> dict:
    return {
        "case_id": case,
        "family": family,
        "method": method,
        "task_kind": "call",
        "autonomous_safe_resolution": safe,
        "accepted_call": True,
        "execution_success": exact,
        "unsupported_action_critical": not exact,
    }


def test_hierarchy_analysis_computes_primary_contrasts() -> None:
    rows = []
    for family in ("a", "b"):
        for case_index in range(2):
            case = f"{family}-{case_index}"
            rows.extend(
                [
                    _score(case, family, FULL, True, True),
                    _score(case, family, SOURCE_ROLE, case_index == 0, case_index == 0),
                    _score(case, family, BEST_OF, False, False),
                ]
            )
    result = analyze_hierarchy_scores(rows, replicates=100, seed=1)
    assert result["contrasts"]["extent_precision_increment"]["estimate"] == pytest.approx(
        0.5
    )
    assert result["contrasts"]["controller_vs_compute_matched_safe_decision"][
        "estimate"
    ] == pytest.approx(1.0)

def test_hierarchy_compute_audit_checks_aggregate_budget_and_shared_trace() -> None:
    rows = []
    for case_id, best_tokens in (("c1", 110), ("c2", 80)):
        for method, tokens in (
            (FULL, 100),
            (SOURCE_ROLE, 100),
            (BEST_OF, best_tokens),
        ):
            metadata = {
                "prompt_tokens": 80 if method != BEST_OF else 60,
                "completion_tokens": 20 if method != BEST_OF else tokens - 60,
                "total_tokens": tokens,
                "generation_calls": 8 if method != BEST_OF else 1,
                "model_trace_sha256": "same" if method != BEST_OF else "other",
            }
            if method in {FULL, SOURCE_ROLE}:
                metadata.update(
                    {
                        "semantic_extent_enabled": method == FULL,
                        "exhaust_proposal_budget": True,
                    }
                )
            else:
                metadata.update(
                    {
                        "allocation_rule": "deterministic_balanced_rounds",
                        "minimum_samples_per_case": 1,
                        "certificate_gate": "none",
                        "selection_rule": "ordinary_contract_validator_rank",
                        "full_controller_aggregate_total_tokens": 200,
                        "best_of_aggregate_total_tokens": 190,
                    }
                )
            rows.append(
                {
                    "case_id": case_id,
                    "method": method,
                    "response_metadata": metadata,
                }
            )
    audit = audit_hierarchy_compute(rows, expected_cases=2)
    assert audit["passed"]
    assert audit["best_of_aggregate_total_tokens"] == 190
    assert audit["full_controller_aggregate_total_tokens"] == 200
    rows[-1]["response_metadata"]["total_tokens"] = 101
    assert not audit_hierarchy_compute(rows, expected_cases=2)["passed"]


def test_precision_counts_false_calls_on_non_call_cases() -> None:
    rows = []
    for family in ("a", "b"):
        call_case = f"{family}-call"
        no_tool_case = f"{family}-no-tool"
        for method in (FULL, SOURCE_ROLE, BEST_OF):
            rows.append(_score(call_case, family, method, True, True))
            no_tool = _score(no_tool_case, family, method, method != FULL, False)
            no_tool.update(
                {
                    "task_kind": "no_tool",
                    "accepted_call": method == FULL,
                    "unsupported_action_critical": method == FULL,
                }
            )
            rows.append(no_tool)
    result = analyze_hierarchy_scores(rows, replicates=100, seed=2)
    full = next(row for row in result["conditions"] if row["method"] == FULL)
    assert full["accepted_call_denominator"] == 4
    assert full["accepted_call_exact_precision"] == pytest.approx(0.5)


def test_hierarchy_timing_summary_reports_compute_and_p95() -> None:
    rows = [
        {
            "method": FULL,
            "elapsed_seconds": elapsed,
            "generation_calls": 2,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        for elapsed in (1.0, 2.0, 3.0)
    ]
    summary = analyze_hierarchy_timings(rows)[0]
    assert summary["mean_latency_seconds"] == pytest.approx(2.0)
    assert summary["p95_latency_seconds"] == pytest.approx(2.9)
    assert summary["total_tokens"] == 45

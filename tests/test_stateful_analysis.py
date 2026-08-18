from __future__ import annotations

import pytest

from tapbench.stateful_analysis import (
    analyze_stateful_rows,
    audit_stateful_rows,
    paired_family_cluster_interval,
)


def _row(
    model: str,
    family: str,
    scenario: str,
    condition: str,
    similarity: float,
) -> dict:
    return {
        "model_id": model,
        "family": family,
        "scenario": scenario,
        "condition": condition,
        "similarity": similarity,
        "milestone_similarity": similarity,
        "minefield_similarity": 0.0,
        "tool_call_exceptions": [],
        "runner_error": None,
        "thinking_marker_detected": False,
        "length_stops": 0,
        "first_request_sha256": f"same-{model}-{scenario}",
    }


def test_family_cluster_analysis_uses_paired_family_means() -> None:
    rows = []
    for family, native, evibind in (("a", 0.2, 0.6), ("b", 0.4, 0.6)):
        for variant in ("x", "y"):
            scenario = f"{family}-{variant}"
            rows.append(_row("m", family, scenario, "native", native))
            rows.append(_row("m", family, scenario, "evibind", evibind))
    result = analyze_stateful_rows(rows, replicates=200, seed=4)
    official = result["models"][0]["metrics"]["official_similarity"]
    assert official["family_count"] == 2
    assert official["delta_evibind_minus_native"] == pytest.approx(0.3)
    assert result["macro_average"]["official_similarity"][
        "delta_evibind_minus_native"
    ] == pytest.approx(0.3)


def test_audit_checks_request_parity_and_completeness() -> None:
    rows = [
        _row("m", "a", "a-x", "native", 0.2),
        _row("m", "a", "a-x", "evibind", 0.4),
    ]
    assert audit_stateful_rows(
        rows, expected_models=1, expected_scenarios=1
    )["passed"]
    rows[1]["first_request_sha256"] = "different"
    audit = audit_stateful_rows(rows, expected_models=1, expected_scenarios=1)
    assert not audit["passed"]
    assert len(audit["request_parity_failures"]) == 1


def test_cluster_interval_requires_multiple_families() -> None:
    with pytest.raises(ValueError, match="at least two"):
        paired_family_cluster_interval({"only": 1.0}, replicates=10, seed=1)

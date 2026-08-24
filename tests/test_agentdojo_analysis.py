from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench" / "agentdojo"))

from analyze import (  # noqa: E402
    analyse,
    paired_transitions,
    two_way_cluster_interval,
    wilson_interval,
)


def _report() -> dict:
    baseline = []
    guarded = []
    values = [
        ("u0", "i0", True, True, True, False),
        ("u0", "i1", False, True, True, False),
        ("u1", "i0", False, False, True, False),
        ("u1", "i1", True, True, False, False),
    ]
    for user, injection, bu, bs, gu, gs in values:
        baseline.append({"user_task_id": user, "injection_task_id": injection,
                         "utility": bu, "security": bs})
        guarded.append({"user_task_id": user, "injection_task_id": injection,
                        "utility": gu, "security": gs})
    return {"schema": "evibind.agentdojo.v2", "suite": "fixture",
            "model": "fixture", "arms": {
                "baseline": {"case_rows": baseline},
                "evibind": {"case_rows": guarded},
            }}


def test_paired_point_estimates_are_exact() -> None:
    result = analyse(_report(), replicates=200, seed=7)
    assert result["estimands"]["guarded_minus_baseline_utility"]["point"] == 0.25
    assert result["estimands"]["guarded_minus_baseline_security"]["point"] == -0.75


def test_cluster_bootstrap_is_deterministic() -> None:
    rows = [(row, _report()["arms"]["evibind"]["case_rows"][index])
            for index, row in enumerate(_report()["arms"]["baseline"]["case_rows"])]
    assert two_way_cluster_interval(rows, "utility", replicates=100, seed=3) == \
        two_way_cluster_interval(rows, "utility", replicates=100, seed=3)


def test_exact_paired_test_counts_discordance() -> None:
    rows = [(row, _report()["arms"]["evibind"]["case_rows"][index])
            for index, row in enumerate(_report()["arms"]["baseline"]["case_rows"])]
    result = paired_transitions(rows, "security")
    assert result["baseline_pass_guarded_fail"] == 3
    assert result["baseline_fail_guarded_pass"] == 0
    assert result["mcnemar_two_sided_exact_p"] == .25


def test_mismatched_case_sets_fail_closed() -> None:
    report = _report()
    report["arms"]["evibind"]["case_rows"].pop()
    try:
        analyse(report, replicates=10)
    except ValueError as exc:
        assert "same cases" in str(exc)
    else:
        raise AssertionError("mismatched arms must not be analysed")


def test_zero_attack_rate_has_nonzero_finite_sample_upper_bound() -> None:
    lower, upper = wilson_interval(0, 144)
    assert lower == 0.0
    assert 0.02 < upper < 0.03

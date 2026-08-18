from __future__ import annotations

from scripts.reproduce_public_artifact import reproduce


def test_public_reproduction_is_deterministic(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_summary = reproduce(
        first,
        per_pattern=2,
        per_kind=1,
        separation_repetitions=2,
        fuzz_trials=260,
    )
    second_summary = reproduce(
        second,
        per_pattern=2,
        per_kind=1,
        separation_repetitions=2,
        fuzz_trials=260,
    )

    assert first_summary == second_summary
    assert (first / "SHA256SUMS").read_bytes() == (
        second / "SHA256SUMS"
    ).read_bytes()
    assert first_summary["passed"] is True
    assert first_summary["checks"]["cite_checker_redundancy_faults_exposed"] is True
    assert first_summary["checks"]["evibind_redundancy_faults_fail_closed"] is True
    assert first_summary["checks"]["shared_tcb_controls_are_symmetric"] is True

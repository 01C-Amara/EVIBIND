from __future__ import annotations

from tapbench.boundary_fuzz import run_boundary_fuzz


def test_boundary_fuzz_mutations_all_fail_closed() -> None:
    report = run_boundary_fuzz(260)

    assert report["executed_trials"] == 260
    assert report["mutation_operators"] == 26
    assert report["unsound_releases"] == 0

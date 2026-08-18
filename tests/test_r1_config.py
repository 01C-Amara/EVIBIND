from __future__ import annotations

from collections import defaultdict

from tapbench.config import load_experiment_config
from tapbench.generator import generate_cases_from_config


def test_r1_balances_every_task_kind_within_every_family() -> None:
    cfg = load_experiment_config()
    rows = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["R1_typed_resolution"])
    by_family = defaultdict(set)
    for row in rows:
        by_family[row["family"]].add(row["task_kind"])
    assert len(rows) == 64
    assert len(by_family) == 8
    assert all(kinds == {"call", "missing_info", "no_tool", "direct_answer"} for kinds in by_family.values())


def test_r1_declares_bounded_repair_and_seven_compute_conditions() -> None:
    cfg = load_experiment_config()
    grid = next(row for row in cfg.subgrids["subgrids"] if row["id"] == "R1_typed_resolution")
    assert grid["repair_budget"] == 2
    assert grid["exclude_from_full_run_cap"] is True
    assert grid["planned_generation_budget"] == 14336
    assert set(grid["methods"]) == {
        "prompt_few_shot",
        "best_of_n_budget_matched",
        "constrained_call_only_b2",
        "constrained_abstain_b2",
        "full_tap_b2",
        "tap_r_no_calibrator",
        "tap_r_three_way",
    }

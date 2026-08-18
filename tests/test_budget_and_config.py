from __future__ import annotations

from tapbench.config import load_experiment_config
from tapbench.generator import planned_generation_budget, rejected_full_grid_generations
from tapbench.models import model_group_entries


def test_rejected_full_grid_math_is_committed() -> None:
    cfg = load_experiment_config()
    assert rejected_full_grid_generations(cfg.subgrids) == 3_110_400


def test_fractional_budget_stays_under_cap() -> None:
    cfg = load_experiment_config()
    budget = planned_generation_budget(cfg.subgrids)
    cap = cfg.subgrids["full_run_generation_cap"]
    assert budget <= cap["target_max"]
    assert budget < rejected_full_grid_generations(cfg.subgrids) / 50


def test_retrieval_claims_are_identified_only_at_n64() -> None:
    cfg = load_experiment_config()
    retrieval = [grid for grid in cfg.subgrids["subgrids"] if grid["hypothesis"] == "retrieval"]
    assert retrieval
    assert all(grid["pinned_factors"]["N"] == 64 for grid in retrieval)


def test_repair_budget_is_same_in_pilot_and_full() -> None:
    cfg = load_experiment_config()
    for grid in cfg.subgrids["subgrids"]:
        if any("b2" in method for method in grid.get("methods", [])):
            assert grid.get("repair_budget", cfg.subgrids["common_defaults"]["repair_budget"]) == 2
            assert cfg.subgrids["pilot"]["repair_budget"] == 2


def test_main_models_span_tokenizer_families() -> None:
    cfg = load_experiment_config()
    families = {entry["tokenizer_family"] for entry in model_group_entries(cfg.models, "main_core")}
    assert len(families) >= cfg.models["tokenizer_diversity_requirement"]["main_core_min_families"]

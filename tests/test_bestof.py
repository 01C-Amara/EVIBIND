from __future__ import annotations

from copy import deepcopy

from tapbench.bestof import candidate_rank, select_candidate
from tapbench.config import load_experiment_config
from tapbench.generator import generate_cases_from_config


def _case():
    cfg = load_experiment_config()
    return generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["R1_typed_resolution"])[0]


def test_best_of_rank_prefers_contract_valid_supported_candidate() -> None:
    case = _case()
    good = deepcopy(case["gold_action"])
    unsupported = deepcopy(good)
    unsupported["arguments"][next(iter(unsupported["arguments"]))] = "invented"
    invalid_tool = deepcopy(good)
    invalid_tool["tool"] = "not_available"
    selected, diagnostics = select_candidate(case, [unsupported, invalid_tool, good])
    assert selected == 2
    assert diagnostics[selected]["contract_valid"] is True
    assert candidate_rank(case, good) > candidate_rank(case, unsupported)


def test_best_of_selection_is_deterministic_on_ties() -> None:
    case = _case()
    good = deepcopy(case["gold_action"])
    selected, _ = select_candidate(case, [good, deepcopy(good)])
    assert selected == 0

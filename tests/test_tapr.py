from __future__ import annotations

from copy import deepcopy

from tapbench.config import load_experiment_config
from tapbench.generator import generate_cases_from_config
from tapbench.tapr import contract_validator_error, resolve_action, score_resolution_predictions


def _case(grid: str, *, task_kind: str | None = None):
    cfg = load_experiment_config()
    rows = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=[grid])
    if task_kind is None:
        return rows[0]
    return next(row for row in rows if row["task_kind"] == task_kind)


def _prediction(case, action):
    return {
        "case_id": case["case_id"],
        "method": "full_tap_b2",
        "model_id": "test-model",
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "thinking_mode": "off",
        "prediction": action,
    }


def test_contract_validator_does_not_consult_gold_action() -> None:
    case = _case("H1_prompt_verbosity")
    candidate = deepcopy(case["gold_action"])
    case["gold_action"] = {"mode": "no_tool", "tool": None, "arguments": {}, "payload": {}}
    report = contract_validator_error(case, candidate)
    assert report["contract_valid"] is True
    assert report["errors"] == []


def test_missing_information_fabrication_transitions_to_correct_clarification() -> None:
    case = _case("H6_abstention_suppression", task_kind="missing_info")
    missing = case["gold_action"]["payload"]["missing_slots"][0]
    tool = case["tools"][0]["canonical_name"]
    action = {
        "mode": "call",
        "tool": tool,
        "arguments": {missing: "guessed-value"},
        "payload": {},
    }
    resolved, history = resolve_action(case, _prediction(case, action), repair_budget=2)
    assert resolved["resolution"]["terminal_state"] == "clarify"
    assert resolved["prediction"]["payload"]["missing_slots"] == [missing]
    assert len(history) == 2
    assert history[0]["selected_error_class"] == "unsupported_required_value"


def test_contradicted_required_value_is_repaired_from_declared_normalizer() -> None:
    case = _case("H1_prompt_verbosity")
    action = deepcopy(case["gold_action"])
    slot = next(iter(action["arguments"]))
    action["arguments"][slot] = "definitely-wrong"
    resolved, history = resolve_action(case, _prediction(case, action), repair_budget=2)
    assert resolved["resolution"]["terminal_state"] == "call"
    assert resolved["prediction"]["arguments"][slot] == case["derivable_values"][slot]
    assert history[-1]["contract_valid"] is True


def test_unsupported_optional_field_is_deleted_without_new_generation() -> None:
    case = _case("H1_prompt_verbosity")
    action = deepcopy(case["gold_action"])
    action["arguments"]["invented_note"] = "not in request"
    resolved, history = resolve_action(case, _prediction(case, action), repair_budget=2)
    assert resolved["resolution"]["terminal_state"] == "call"
    assert "invented_note" not in resolved["prediction"]["arguments"]
    assert resolved["resolution"]["generation_calls"] == 1
    assert len(history) == 2


def test_unknown_tool_escalates_and_resolution_scorer_reports_coverage() -> None:
    case = _case("H1_prompt_verbosity")
    action = deepcopy(case["gold_action"])
    action["tool"] = "unknown_tool"
    resolved, _ = resolve_action(case, _prediction(case, action), repair_budget=2)
    output = {**_prediction(case, resolved["prediction"]), "method": "tap_r_no_calibrator", "resolution": resolved["resolution"]}
    scores, summary = score_resolution_predictions([case], [output])
    assert scores[0]["escalated"] is True
    assert summary["groups"][0]["escalation_rate"] == 1.0
    assert summary["groups"][0]["non_escalated_coverage"] == 0.0


def test_repair_budget_bounds_validation_rounds() -> None:
    case = _case("H1_prompt_verbosity")
    action = deepcopy(case["gold_action"])
    for slot in action["arguments"]:
        action["arguments"][slot] = "wrong"
    resolved, history = resolve_action(case, _prediction(case, action), repair_budget=2)
    assert len(history) <= 3
    assert resolved["resolution"]["validation_rounds"] <= 3

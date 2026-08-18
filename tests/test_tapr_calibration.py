from __future__ import annotations

from copy import deepcopy

from tapbench.config import load_experiment_config
from tapbench.generator import generate_cases_from_config
from tapbench.tapr_calibration import apply_three_way_policy, calibrate_predictions


def test_three_way_policy_escalates_low_confidence_call() -> None:
    prediction = {
        "method": "tap_r_no_calibrator",
        "prediction": {"mode": "call", "tool": "t", "arguments": {}, "payload": {}},
        "resolution": {"terminal_state": "call", "final_contract_valid": True},
    }
    output = apply_three_way_policy(
        prediction,
        score=0.4,
        threshold=0.8,
        fold="calendar",
        features={"bias": 1.0},
    )
    assert output["method"] == "tap_r_three_way"
    assert output["resolution"]["terminal_state"] == "escalate"
    assert output["tap_r_calibrator"]["decision"] == "escalate"


def test_three_way_policy_preserves_evidence_driven_clarification() -> None:
    prediction = {
        "method": "tap_r_no_calibrator",
        "prediction": {"mode": "clarify", "tool": None, "arguments": {}, "payload": {"missing_slots": ["date"]}},
        "resolution": {"terminal_state": "clarify"},
    }
    output = apply_three_way_policy(
        prediction,
        score=0.0,
        threshold=1.0,
        fold="calendar",
        features={"bias": 1.0},
    )
    assert output["prediction"]["mode"] == "clarify"
    assert output["tap_r_calibrator"]["decision"] == "clarify"


def test_calibrator_uses_family_disjoint_train_only_thresholds() -> None:
    cfg = load_experiment_config()
    cases = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["R1_typed_resolution"])
    predictions = []
    terminal = {"call": "call", "missing_info": "clarify", "no_tool": "direct_answer", "direct_answer": "direct_answer"}
    for case in cases:
        predictions.append(
            {
                "case_id": case["case_id"],
                "method": "tap_r_no_calibrator",
                "model_id": "m",
                "prediction": deepcopy(case["gold_action"]),
                "resolution": {"terminal_state": terminal[case["task_kind"]], "validation_rounds": 1},
            }
        )
    outputs, rows, report = calibrate_predictions(cases, predictions, target_precision=0.95)
    assert len(outputs) == len(cases)
    assert len(rows) == sum(case["task_kind"] == "call" for case in cases)
    assert report["threshold_policy"] == "leave-one-family-out_train_only"
    assert len(report["folds"]) == 8
    assert all(fold["test_call_rows"] == 2 for fold in report["folds"])
    assert all(output["method"] == "tap_r_three_way" for output in outputs)

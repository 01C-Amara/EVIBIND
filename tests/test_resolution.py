from __future__ import annotations

from tapbench.config import load_experiment_config
from tapbench.generator import generate_cases_from_config
from tapbench.resolution import (
    TERMINAL_STATES,
    TRANSITION_RULES,
    diagnose_predictions,
    evidence_for_slot_value,
    evidence_ledger_for_action,
    typed_validator_error,
)


def _cases(grid_id: str):
    cfg = load_experiment_config()
    return generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=[grid_id])


def test_evidence_labeler_distinguishes_explicit_normalized_unsupported_and_contradicted() -> None:
    call_case = _cases("H1_prompt_verbosity")[0]
    assert evidence_for_slot_value(call_case, "title", call_case["derivable_values"]["title"])["label"] == "explicit"
    assert evidence_for_slot_value(call_case, "date", "2099-01-01")["label"] == "contradicted"

    weather_case = next(case for case in _cases("H1_prompt_verbosity") if case["family"] == "weather")
    assert evidence_for_slot_value(weather_case, "include_precipitation", True)["label"] == "normalized"

    missing_case = _cases("H6_abstention_suppression")[0]
    missing_slot = missing_case["gold_action"]["payload"]["missing_slots"][0]
    assert evidence_for_slot_value(missing_case, missing_slot, "guessed-value")["label"] == "unsupported"


def test_slot_specific_evidence_rejects_cross_slot_composite_substrings() -> None:
    case = _cases("H1_prompt_verbosity")[0]
    slot = next(iter(case["gold_action"]["arguments"]))
    correct = case["derivable_values"][slot]
    request = next(message["content"] for message in case["messages"] if message["role"] == "user")
    composite = request[request.find(str(correct)) :]
    assert str(correct) in composite
    if composite != correct:
        evidence = evidence_for_slot_value(case, slot, composite)
        assert evidence["label"] == "contradicted"
        assert evidence["normalizer"] == "synthetic_family_normalizer"


def test_unsupported_required_value_recommends_clarification_not_repair() -> None:
    case = _cases("H6_abstention_suppression")[0]
    missing_slot = case["gold_action"]["payload"]["missing_slots"][0]
    prediction = {
        "case_id": case["case_id"],
        "method": "test",
        "model_id": "m",
        "prediction": {
            "mode": "call",
            "tool": case["tools"][0]["canonical_name"],
            "arguments": {missing_slot: "guessed-value"},
            "payload": {},
        },
    }

    error = typed_validator_error(case, prediction)
    slot_error = next(row for row in error["slot_errors"] if row["error_class"] == "unsupported_required_value")
    assert slot_error["evidence_status"] == "unsupported"
    assert slot_error["repairable"] is False
    assert slot_error["recommended_transition"] == "convert_to_clarify"
    assert error["recommended_transition"]["transition"] == "convert_to_clarify"


def test_unsupported_optional_field_is_deletable() -> None:
    case = _cases("H1_prompt_verbosity")[0]
    action = dict(case["gold_action"])
    action["arguments"] = {**case["gold_action"]["arguments"], "unsupported_note": "invented"}
    prediction = {"case_id": case["case_id"], "method": "test", "model_id": "m", "prediction": action}

    error = typed_validator_error(case, prediction)
    optional_error = next(row for row in error["slot_errors"] if row["slot"] == "unsupported_note")
    assert optional_error["error_class"] == "unsupported_optional_field"
    assert optional_error["repairable"] is True
    assert optional_error["recommended_transition"] == "delete_field"



def test_wrong_normalized_value_repairs_from_evidence() -> None:
    case = _cases("H1_prompt_verbosity")[0]
    action = dict(case["gold_action"])
    action["arguments"] = {**case["gold_action"]["arguments"], "date": "2099-01-01"}
    prediction = {"case_id": case["case_id"], "method": "test", "model_id": "m", "prediction": action}

    error = typed_validator_error(case, prediction)
    slot_error = next(row for row in error["slot_errors"] if row["slot"] == "date")
    assert slot_error["error_type"] == "wrong_normalized_value"
    assert slot_error["error_class"] == "wrong_normalized_value_with_evidence"
    assert slot_error["evidence_status"] == "contradicted"
    assert slot_error["recommended_transition"] == "repair_from_evidence"

def test_evidence_ledger_and_diagnose_predictions_emit_state_substrate() -> None:
    case = _cases("H1_prompt_verbosity")[0]
    prediction = {"case_id": case["case_id"], "method": "test", "model_id": "m", "prediction": case["gold_action"]}
    errors, ledger = diagnose_predictions([case], [prediction])

    assert len(errors) == 1
    assert errors[0]["error_count"] == 0
    assert {row["evidence_label"] for row in ledger} >= {"explicit"}
    assert all("required" in row for row in ledger)


def test_transition_table_names_five_terminal_states() -> None:
    terminals = {state for rule in TRANSITION_RULES for state in rule["safe_terminal_states"]}
    assert set(TERMINAL_STATES) == {"call", "clarify", "direct_answer", "refuse", "escalate"}
    assert {"call", "clarify", "refuse", "escalate"}.issubset(terminals)
    assert {rule["error_class"] for rule in TRANSITION_RULES} >= {
        "invalid_json",
        "missing_required_slot_no_evidence",
        "wrong_enum_with_evidence",
        "unsupported_optional_field",
        "unsupported_required_value",
        "no_tool_overcall",
        "repeated_repair_same_slot",
    }


def test_unseen_family_uses_conservative_derivable_evidence() -> None:
    case = {
        "family": "outpatient_dispensing",
        "task_kind": "call",
        "messages": [
            {"role": "user", "content": "Use Pharmacy 17 for pickup."}
        ],
        "derivable_values": {"pharmacy": "Pharmacy 17"},
    }

    evidence = evidence_for_slot_value(case, "pharmacy", "Pharmacy 17")

    assert evidence["label"] == "explicit"


def test_unseen_family_missing_slot_remains_unsupported() -> None:
    case = {
        "family": "outpatient_dispensing",
        "task_kind": "missing_info",
        "messages": [{"role": "user", "content": "Prepare my pickup."}],
        "derivable_values": {"fulfillment": "pickup"},
    }

    evidence = evidence_for_slot_value(case, "pharmacy", "Pharmacy 17")

    assert evidence["label"] == "unsupported"


def test_unseen_family_required_slot_comes_from_tool_schema() -> None:
    case = {
        "case_id": "unseen-required",
        "family": "outpatient_dispensing",
        "task_kind": "call",
        "messages": [
            {"role": "user", "content": "Use Pharmacy 17 for pickup."}
        ],
        "derivable_values": {"pharmacy": "Pharmacy 17"},
        "tools": [
            {
                "name": "operation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pharmacy_ref": {
                            "type": "string",
                            "x-ir-name": "pharmacy",
                        }
                    },
                    "required": ["pharmacy_ref"],
                },
            }
        ],
    }
    action = {
        "mode": "call",
        "tool": "operation",
        "arguments": {"pharmacy": "Pharmacy 17"},
    }

    ledger = evidence_ledger_for_action(case, action)

    assert ledger[0]["required"] is True

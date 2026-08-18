from __future__ import annotations

from pathlib import Path

from tapbench.config import REPO_ROOT, load_experiment_config
from tapbench.ir import parse_and_normalize_prediction
from tapbench.generator import generate_cases_from_config
from tapbench.scoring import score_files
from tapbench.slot_errors import slot_errors_for_predictions
from tapbench.thinking import prediction_has_thinking_marker, text_has_thinking_marker
from tapbench.validation import validate_action


def test_golden_replay_scores_bit_for_bit(tmp_path: Path) -> None:
    fixture = REPO_ROOT / "tests" / "fixtures" / "golden"
    output = tmp_path / "scores.jsonl"
    slot_errors = tmp_path / "slot_errors.jsonl"
    score_files(fixture / "cases.jsonl", fixture / "predictions.jsonl", output, slot_errors_path=slot_errors)
    assert output.read_text(encoding="utf-8") == (fixture / "expected_scores.jsonl").read_text(encoding="utf-8")
    assert slot_errors.read_text(encoding="utf-8") == (fixture / "expected_slot_errors.jsonl").read_text(encoding="utf-8")


def test_thinking_marker_detector_catches_reasoning_tags() -> None:
    assert text_has_thinking_marker("<think>hidden reasoning</think>")
    assert text_has_thinking_marker("Reasoning: I should call a tool")
    assert prediction_has_thinking_marker({
        "thinking_marker_detected": False,
        "response_metadata": {"raw_text": "<think>hidden reasoning</think>"},
    })
    assert prediction_has_thinking_marker({
        "thinking_marker_detected": True,
        "response_metadata": {"raw_text": "{}"},
    })


def test_missing_info_fabrication_oracle_flags_guessed_slot() -> None:
    cfg = load_experiment_config()
    case = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["H6_abstention_suppression"])[0]
    assert case["task_kind"] == "missing_info"
    family_missing_slot = next(iter(case["gold_action"]["payload"]["missing_slots"]))
    action = {
        "mode": "call",
        "tool": case["tools"][0]["canonical_name"],
        "arguments": {family_missing_slot: "guessed-value"},
        "payload": {},
    }
    metrics = validate_action(case, action)
    assert metrics["fabrication"] is True
    assert metrics["execution_success"] is False


def test_slot_errors_capture_missing_wrong_enum_and_fabrication() -> None:
    cfg = load_experiment_config()
    call_case = next(
        case
        for case in generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["H2_constraints_repair"])
        if case["family"] == "calendar" and "calendar" in case["gold_action"]["arguments"]
    )
    prediction = {
        "case_id": call_case["case_id"],
        "method": "test",
        "model_id": "m",
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "chat_template": "t",
        "grammar_engine": "gbnf",
        "model_artifact": "a",
        "prediction": {
            "mode": "call",
            "tool": call_case["gold_action"]["tool"],
            "arguments": {"title": call_case["gold_action"]["arguments"]["title"], "calendar": "not_an_enum"},
            "payload": {},
        },
    }
    errors = slot_errors_for_predictions([call_case], [prediction])
    error_types = {row["error_type"] for row in errors}
    assert "missing_required_slot" in error_types
    assert "wrong_enum" in error_types


def test_action_ir_wrapper_is_normalized() -> None:
    cfg = load_experiment_config()
    case = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["H1_prompt_verbosity"])[0]
    action, valid = parse_and_normalize_prediction({"action_ir": case["gold_action"]}, case)
    assert valid is True
    assert action == case["gold_action"]

def test_tool_object_is_normalized_to_action_ir() -> None:
    cfg = load_experiment_config()
    case = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["H1_prompt_verbosity"])[0]
    prediction = {
        "mode": "call",
        "tool": {
            "name": case["gold_action"]["tool"],
            "arguments": case["gold_action"]["arguments"],
        },
    }
    action, valid = parse_and_normalize_prediction(prediction, case)
    assert valid is True
    assert action == case["gold_action"]


def test_validator_scores_non_string_tool_invalid_without_crashing() -> None:
    cfg = load_experiment_config()
    case = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["H1_prompt_verbosity"])[0]
    action = {
        "mode": "call",
        "tool": {
            "name": case["gold_action"]["tool"],
            "arguments": case["gold_action"]["arguments"],
        },
        "arguments": case["gold_action"]["arguments"],
        "payload": {},
    }
    metrics = validate_action(case, action)
    assert metrics["schema_valid"] is False
    assert metrics["tool_correct"] is False
    assert metrics["execution_success"] is False

def test_clarify_missing_slot_string_is_normalized() -> None:
    cfg = load_experiment_config()
    case = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["H6_abstention_suppression"])[0]
    prediction = {"action_ir": {"mode": "clarify", "payload": {"missing_slots": "date not provided"}}}
    action, valid = parse_and_normalize_prediction(prediction, case)
    assert valid is True
    assert action["payload"]["missing_slots"] == ["date"]
    metrics = validate_action(case, action)
    assert metrics["execution_success"] is True



def test_unregistered_family_uses_offline_gold_missing_slot() -> None:
    case = {
        "case_id": "unseen-missing",
        "family": "prospective_unseen_family",
        "task_kind": "missing_info",
        "tools": [
            {
                "name": "submit_record",
                "canonical_name": "submit_record",
                "parameters": {
                    "type": "object",
                    "properties": {"recipient": {"type": "string"}},
                    "required": ["recipient"],
                    "additionalProperties": False,
                },
            }
        ],
        "derivable_values": {},
        "gold_action": {
            "mode": "clarify",
            "tool": None,
            "arguments": {},
            "payload": {"missing_slots": ["recipient"]},
        },
    }
    clarification = {
        "mode": "clarify",
        "tool": None,
        "arguments": {},
        "payload": {"missing_slots": ["recipient"]},
    }
    guessed_call = {
        "mode": "call",
        "tool": "submit_record",
        "arguments": {"recipient": "invented"},
        "payload": {},
    }

    assert validate_action(case, clarification)["execution_success"] is True
    guessed_metrics = validate_action(case, guessed_call)
    assert guessed_metrics["fabrication"] is True
    assert guessed_metrics["execution_success"] is False


def test_unregistered_family_slot_errors_use_public_schema_enum() -> None:
    case = {
        "case_id": "unseen-enum",
        "hypothesis_grid_id": "hierarchy-test",
        "hypothesis": "hierarchy",
        "family": "prospective_unseen_family",
        "task_kind": "call",
        "tools": [
            {
                "name": "set_status",
                "canonical_name": "set_status",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["open", "closed"],
                        }
                    },
                    "required": ["status"],
                    "additionalProperties": False,
                },
            }
        ],
        "derivable_values": {"status": "open"},
        "gold_action": {
            "mode": "call",
            "tool": "set_status",
            "arguments": {"status": "open"},
            "payload": {},
        },
    }
    prediction = {
        "case_id": case["case_id"],
        "method": "test",
        "model_id": "m",
        "prediction": {
            "mode": "call",
            "tool": "set_status",
            "arguments": {"status": "archived"},
            "payload": {},
        },
    }

    errors = slot_errors_for_predictions([case], [prediction])

    assert {row["error_type"] for row in errors} == {"wrong_enum"}

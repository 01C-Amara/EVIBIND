from __future__ import annotations

from pathlib import Path
from typing import Any

from .families import get_family
from .io import read_jsonl, write_jsonl
from .ir import parse_and_normalize_prediction
from .schemas import enum_values_for_e
from .slot_errors import slot_error_rows
from .validation import validate_action

EVIDENCE_LABELS = ("explicit", "normalized", "inferred_safe", "unsupported", "contradicted")
TERMINAL_STATES = ("call", "clarify", "direct_answer", "refuse", "escalate")

TRANSITION_RULES: tuple[dict[str, Any], ...] = (
    {
        "error_class": "invalid_json",
        "transition": "regenerate_under_grammar",
        "repair_allowed": True,
        "safe_terminal_states": ["call", "clarify", "refuse"],
        "notes": "Regenerate inside an abstention-aware grammar before semantic repair.",
    },
    {
        "error_class": "missing_required_slot_no_evidence",
        "transition": "convert_to_clarify",
        "repair_allowed": False,
        "safe_terminal_states": ["clarify"],
        "notes": "No fabrication path is permitted for unsupported required values.",
    },
    {
        "error_class": "missing_required_slot_with_evidence",
        "transition": "repair_from_evidence",
        "repair_allowed": True,
        "safe_terminal_states": ["call", "clarify"],
        "notes": "Only fill the slot from explicit or normalized evidence.",
    },
    {
        "error_class": "wrong_enum_with_evidence",
        "transition": "normalize_enum",
        "repair_allowed": True,
        "safe_terminal_states": ["call"],
        "notes": "Map an evidenced surface form to a legal enum value.",
    },
    {
        "error_class": "wrong_enum_no_evidence",
        "transition": "convert_to_clarify",
        "repair_allowed": False,
        "safe_terminal_states": ["clarify"],
        "notes": "Ask for the enum value rather than guessing.",
    },

    {
        "error_class": "wrong_normalized_value_with_evidence",
        "transition": "repair_from_evidence",
        "repair_allowed": True,
        "safe_terminal_states": ["call", "clarify"],
        "notes": "Replace the contradicted value with the request-supported normalized value.",
    },
    {
        "error_class": "wrong_normalized_value_no_evidence",
        "transition": "convert_to_clarify",
        "repair_allowed": False,
        "safe_terminal_states": ["clarify"],
        "notes": "Do not invent a normalized value when no request evidence supports it.",
    },
    {
        "error_class": "unsupported_optional_field",
        "transition": "delete_field",
        "repair_allowed": True,
        "safe_terminal_states": ["call"],
        "notes": "Delete unsupported optional material if the required call is otherwise sound.",
    },
    {
        "error_class": "unsupported_required_value",
        "transition": "convert_to_clarify",
        "repair_allowed": False,
        "safe_terminal_states": ["clarify"],
        "notes": "Required unsupported values block executable calls.",
    },
    {
        "error_class": "wrong_tool_high_margin",
        "transition": "reroute_once",
        "repair_allowed": True,
        "safe_terminal_states": ["call"],
        "notes": "Reserved for a future router margin feature.",
    },
    {
        "error_class": "wrong_tool_low_margin",
        "transition": "clarify_or_escalate",
        "repair_allowed": False,
        "safe_terminal_states": ["clarify", "escalate"],
        "notes": "Default when no reliable alternative-tool margin is logged.",
    },
    {
        "error_class": "repeated_repair_same_slot",
        "transition": "stop_repairing",
        "repair_allowed": False,
        "safe_terminal_states": ["clarify", "escalate"],
        "notes": "Iteration history should prevent cycling on the same slot.",
    },
    {
        "error_class": "no_tool_overcall",
        "transition": "convert_mode",
        "repair_allowed": True,
        "safe_terminal_states": ["direct_answer", "refuse"],
        "notes": "No-tool requests should not be forced into calls.",
    },
    {
        "error_class": "validator_cross_field_failure",
        "transition": "local_repair_if_evidenced",
        "repair_allowed": True,
        "safe_terminal_states": ["call", "clarify"],
        "notes": "Placeholder for validators such as end_time > start_time.",
    },
)

_RULE_BY_CLASS = {rule["error_class"]: rule for rule in TRANSITION_RULES}


def _user_text(case: dict[str, Any]) -> str:
    chunks = []
    for message in case.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "user":
            chunks.append(str(message.get("content", "")))
    return "\n".join(chunks)


def _value_strings(value: Any) -> list[str]:
    if isinstance(value, bool):
        return ["true" if value else "false", "yes" if value else "no"]
    if value is None:
        return []
    return [str(value)]


def _source_span(text: str, value: Any) -> dict[str, Any] | None:
    lowered = text.lower()
    for candidate in _value_strings(value):
        if not candidate:
            continue
        index = lowered.find(candidate.lower())
        if index >= 0:
            return {"start": index, "end": index + len(candidate), "text": text[index : index + len(candidate)]}
    return None


def _safe_defaults(case: dict[str, Any]) -> dict[str, Any]:
    metadata = case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}
    defaults = metadata.get("safe_defaults", {})
    return defaults if isinstance(defaults, dict) else {}


def evidence_for_slot_value(case: dict[str, Any], slot: str | None, value: Any) -> dict[str, Any]:
    """Classify whether a slot value is supported by the request.

    The labeler is deliberately conservative. It treats the synthetic missing slot
    as unsupported even though the generator stores the hidden gold value in
    derivable_values for scoring.
    """
    slot_name = str(slot or "")
    text = _user_text(case)
    span = _source_span(text, value)
    derivable = case.get("derivable_values", {}) if isinstance(case.get("derivable_values"), dict) else {}
    try:
        family = get_family(str(case["family"]))
    except KeyError:
        family = None
    defaults = _safe_defaults(case)

    missing_information_slot = case.get("task_kind") == "missing_info" and (
        (family is not None and slot_name == family.missing_slot)
        or (family is None and slot_name not in derivable)
    )
    if missing_information_slot:
        return {
            "label": "unsupported",
            "source_span": None,
            "normalizer": None,
            "reason": "the declared missing slot is absent in a missing-information case",
        }
    if slot_name in defaults and defaults[slot_name] == value:
        return {
            "label": "inferred_safe",
            "source_span": None,
            "normalizer": "safe_default_policy",
            "reason": "value matches a schema-declared safe default",
        }
    if slot_name in derivable:
        if derivable[slot_name] == value:
            if span is not None:
                return {
                    "label": "explicit",
                    "source_span": span,
                    "normalizer": "slot_specific_exact_match",
                    "reason": "value matches the slot normalizer and its surface form appears in the request",
                }
            return {
                "label": "normalized",
                "source_span": None,
                "normalizer": "synthetic_family_normalizer",
                "reason": "value matches the slot-specific normalized request value",
            }
        return {
            "label": "contradicted",
            "source_span": span,
            "normalizer": "synthetic_family_normalizer",
            "reason": "surface text may occur in the request, but it does not match this slot's normalized value",
        }
    if span is not None:
        return {
            "label": "explicit",
            "source_span": span,
            "normalizer": "untyped_exact_substring",
            "reason": "value appears in the request, but no slot-specific normalizer is registered",
        }
    return {
        "label": "unsupported",
        "source_span": None,
        "normalizer": None,
        "reason": "value is neither explicit nor derivable under the current oracle",
    }


def _gold_args(case: dict[str, Any]) -> dict[str, Any]:
    gold = case.get("gold_action", {})
    args = gold.get("arguments", {}) if isinstance(gold, dict) else {}
    return args if isinstance(args, dict) else {}


def _is_required_slot(case: dict[str, Any], slot: str | None) -> bool:
    if not slot:
        return False
    try:
        family = get_family(str(case["family"]))
    except KeyError:
        family = None
    if family is not None:
        return slot in _gold_args(case) or slot in family.required_slots
    for tool in case.get("tools", []):
        parameters = tool.get("parameters", {})
        if not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            continue
        for surface in required:
            prop = properties.get(surface, {})
            canonical = prop.get("x-ir-name", surface) if isinstance(prop, dict) else surface
            if slot in {str(surface), str(canonical)}:
                return True
    return False


def evidence_ledger_for_action(case: dict[str, Any], action: dict[str, Any] | None) -> list[dict[str, Any]]:
    if action is None or action.get("mode") != "call":
        return []
    args = action.get("arguments", {}) if isinstance(action.get("arguments"), dict) else {}
    rows: list[dict[str, Any]] = []
    for slot, value in sorted(args.items()):
        evidence = evidence_for_slot_value(case, slot, value)
        rows.append(
            {
                "case_id": case["case_id"],
                "family": case["family"],
                "task_kind": case["task_kind"],
                "slot": slot,
                "value": value,
                "evidence_label": evidence["label"],
                "source_span": evidence["source_span"],
                "normalizer": evidence["normalizer"],
                "reason": evidence["reason"],
                "required": _is_required_slot(case, slot),
                "schema_allows_default": slot in _safe_defaults(case),
            }
        )
    return rows


def _rule(error_class: str) -> dict[str, Any]:
    return dict(_RULE_BY_CLASS[error_class])


def _slot_error_class(case: dict[str, Any], row: dict[str, Any], evidence_label: str, repeated: bool = False) -> str:
    if repeated:
        return "repeated_repair_same_slot"
    error_type = str(row.get("error_type"))
    required = _is_required_slot(case, row.get("slot"))
    has_evidence = evidence_label in {"explicit", "normalized", "inferred_safe"}
    if error_type == "missing_required_slot":
        return "missing_required_slot_with_evidence" if has_evidence else "missing_required_slot_no_evidence"
    if error_type == "wrong_enum":
        return "wrong_enum_with_evidence" if has_evidence else "wrong_enum_no_evidence"
    if error_type == "wrong_normalized_value":
        derivable = case.get("derivable_values", {}) if isinstance(case.get("derivable_values"), dict) else {}
        has_gold_evidence = bool(row.get("slot") in derivable)
        return "wrong_normalized_value_with_evidence" if has_gold_evidence else "wrong_normalized_value_no_evidence"
    if error_type == "unsupported_fabricated_value":
        return "unsupported_required_value" if required else "unsupported_optional_field"
    if error_type in {"extra_optional_field", "wrong_optional_field"}:
        return "unsupported_optional_field" if evidence_label in {"unsupported", "contradicted"} else "validator_cross_field_failure"
    return "validator_cross_field_failure"


def _tool_error_class(row: dict[str, Any]) -> str:
    margin = row.get("router_margin")
    if isinstance(margin, (int, float)) and margin >= 0.25:
        return "wrong_tool_high_margin"
    return "wrong_tool_low_margin"


def _mode_error_class(row: dict[str, Any]) -> str:
    if row.get("error_type") == "no_tool_overcall":
        return "no_tool_overcall"
    return "wrong_tool_low_margin"


def _identity(case: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "hypothesis_grid_id": case.get("hypothesis_grid_id"),
        "hypothesis": case.get("hypothesis", ""),
        "family": case["family"],
        "task_kind": case["task_kind"],
        "method": prediction.get("method", "unknown"),
        "model_id": prediction.get("model_id", "unknown"),
        "seed": prediction.get("seed", 0),
        "backend": prediction.get("backend", "unknown"),
        "quantization": prediction.get("quantization", "unknown"),
        "chat_template": prediction.get("chat_template", "unknown"),
        "grammar_engine": prediction.get("grammar_engine", "unknown"),
        "model_artifact": prediction.get("model_artifact", "unknown"),
        "thinking_mode": prediction.get("thinking_mode", "not_applicable"),
        "reasoning_budget": prediction.get("reasoning_budget"),
    }


def _enrich_error(case: dict[str, Any], row: dict[str, Any], *, repeated_slots: set[str] | None = None) -> dict[str, Any]:
    slot = row.get("slot")
    offending = row.get("predicted_value")
    if row.get("error_type") == "missing_required_slot":
        offending = None
        evidence_value = row.get("gold_value")
    else:
        evidence_value = offending
    evidence = evidence_for_slot_value(case, str(slot) if slot is not None else None, evidence_value)
    repeated = bool(slot and repeated_slots and slot in repeated_slots)
    error_type = str(row.get("error_type"))
    if error_type == "wrong_tool":
        error_class = _tool_error_class(row)
    elif error_type in {"wrong_mode", "no_tool_overcall"}:
        error_class = _mode_error_class(row)
    elif error_type == "format_invalid":
        error_class = "invalid_json"
    else:
        error_class = _slot_error_class(case, row, evidence["label"], repeated=repeated)
    rule = _rule(error_class)
    return {
        "error_type": error_type,
        "error_class": error_class,
        "slot": slot,
        "gold_value": row.get("gold_value"),
        "offending_value": offending,
        "evidence_status": evidence["label"],
        "source_span": evidence["source_span"],
        "required": _is_required_slot(case, str(slot) if slot is not None else None),
        "repairable": bool(rule["repair_allowed"]),
        "recommended_transition": rule["transition"],
        "safe_terminal_states": rule["safe_terminal_states"],
    }


def _choose_transition(errors: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not errors:
        return None
    priority = {
        "invalid_json": 0,
        "unsupported_required_value": 1,
        "missing_required_slot_no_evidence": 2,
        "no_tool_overcall": 3,
        "wrong_tool_low_margin": 4,
        "wrong_tool_high_margin": 5,
        "wrong_enum_no_evidence": 6,
        "missing_required_slot_with_evidence": 7,
        "wrong_enum_with_evidence": 8,
        "wrong_normalized_value_no_evidence": 9,
        "wrong_normalized_value_with_evidence": 10,
        "unsupported_optional_field": 11,
        "validator_cross_field_failure": 12,
        "repeated_repair_same_slot": 13,
    }
    selected = min(errors, key=lambda item: priority.get(str(item.get("error_class")), 100))
    return {
        "error_class": selected["error_class"],
        "transition": selected["recommended_transition"],
        "repairable": selected["repairable"],
        "safe_terminal_states": selected["safe_terminal_states"],
    }


def typed_validator_error(case: dict[str, Any], prediction: dict[str, Any], *, repeated_slots: set[str] | None = None) -> dict[str, Any]:
    action, format_valid = parse_and_normalize_prediction(prediction, case)
    metrics = validate_action(case, action)
    base = _identity(case, prediction)
    rows = slot_error_rows(case, prediction)
    enriched = [_enrich_error(case, row, repeated_slots=repeated_slots) for row in rows]
    mode_errors = [error for error in enriched if error["error_type"] in {"wrong_mode", "no_tool_overcall"}]
    tool_errors = [error for error in enriched if error["error_type"] == "wrong_tool"]
    schema_errors = [error for error in enriched if error["error_type"] in {"format_invalid", "missing_required_slot"}]
    slot_errors = [error for error in enriched if error not in mode_errors and error not in tool_errors and error.get("slot") is not None]
    return {
        **base,
        "schema_version": "tapbench.typed_validator_error.v1",
        "format_valid": bool(format_valid),
        "schema_valid": bool(metrics["schema_valid"]),
        "execution_success": bool(metrics["execution_success"]),
        "fabrication": bool(metrics["fabrication"]),
        "action_mode": None if action is None else action.get("mode"),
        "action_tool": None if action is None else action.get("tool"),
        "mode_error": mode_errors[0] if mode_errors else None,
        "tool_error": tool_errors[0] if tool_errors else None,
        "schema_errors": schema_errors,
        "slot_errors": slot_errors,
        "repairable": any(error["repairable"] for error in enriched),
        "recommended_transition": _choose_transition(enriched),
        "error_count": len(enriched),
    }


def diagnose_predictions(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_by_id = {case["case_id"]: case for case in cases}
    errors: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for prediction in predictions:
        case_id = prediction.get("case_id")
        if case_id not in case_by_id:
            raise KeyError(f"prediction references unknown case_id: {case_id}")
        case = case_by_id[case_id]
        action, _ = parse_and_normalize_prediction(prediction, case)
        identity = _identity(case, prediction)
        errors.append(typed_validator_error(case, prediction))
        for row in evidence_ledger_for_action(case, action):
            ledger.append({**identity, **row})
    return errors, ledger


def diagnose_files(
    cases_path: str | Path,
    predictions_path: str | Path,
    validator_errors_path: str | Path,
    evidence_ledger_path: str | Path,
) -> dict[str, int]:
    errors, ledger = diagnose_predictions(read_jsonl(cases_path), read_jsonl(predictions_path))
    error_count = write_jsonl(validator_errors_path, errors)
    ledger_count = write_jsonl(evidence_ledger_path, ledger)
    return {"validator_errors": error_count, "evidence_ledger_rows": ledger_count}

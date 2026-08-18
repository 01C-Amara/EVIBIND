from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from .io import read_jsonl, write_jsonl
from .ir import parse_and_normalize_prediction
from .resolution import evidence_for_slot_value, evidence_ledger_for_action
from .validation import validate_action

TAPR_VERSION = "tapbench.tap_r.v1"
DEFAULT_REPAIR_BUDGET = 2
_SUPPORTED_EVIDENCE = {"explicit", "normalized", "inferred_safe"}


def _user_text(case: dict[str, Any]) -> str:
    return "\n".join(
        str(message.get("content", ""))
        for message in case.get("messages", [])
        if isinstance(message, dict) and message.get("role") == "user"
    )


def _canonical_tool(case: dict[str, Any], name: Any) -> dict[str, Any] | None:
    if not isinstance(name, str):
        return None
    canonical = case.get("tool_aliases", {}).get(name, name)
    for tool in case.get("tools", []):
        if canonical in {tool.get("name"), tool.get("canonical_name")}:
            return tool
    return None


def _object_schema(parameters: dict[str, Any]) -> dict[str, Any]:
    current = parameters
    while isinstance(current, dict):
        props = current.get("properties")
        if not isinstance(props, dict) or set(props) != {"payload"}:
            break
        payload = props.get("payload")
        if not isinstance(payload, dict):
            break
        current = payload
    return current if isinstance(current, dict) else {}


def _canonical_schema(case: dict[str, Any], tool: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    schema = _object_schema(tool.get("parameters", {}) if isinstance(tool.get("parameters"), dict) else {})
    aliases = case.get("argument_aliases", {}) if isinstance(case.get("argument_aliases"), dict) else {}
    properties: dict[str, dict[str, Any]] = {}
    for surface, raw in (schema.get("properties", {}) or {}).items():
        prop = raw if isinstance(raw, dict) else {}
        canonical = str(prop.get("x-ir-name") or aliases.get(surface, surface))
        properties[canonical] = prop
    required = {
        str((schema.get("properties", {}).get(surface) or {}).get("x-ir-name") or aliases.get(surface, surface))
        for surface in schema.get("required", [])
    }
    return properties, required


def _type_valid(value: Any, expected: str | None) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def _error(
    error_class: str,
    transition: str,
    *,
    slot: str | None = None,
    offending_value: Any = None,
    evidence_status: str = "unsupported",
    repairable: bool,
    safe_terminal_states: list[str],
    detail: str,
) -> dict[str, Any]:
    return {
        "error_class": error_class,
        "slot": slot,
        "offending_value": offending_value,
        "evidence_status": evidence_status,
        "repairable": repairable,
        "recommended_transition": transition,
        "safe_terminal_states": safe_terminal_states,
        "detail": detail,
    }


def _missing_slot_evidence(case: dict[str, Any], slot: str) -> dict[str, Any]:
    values = case.get("derivable_values", {}) if isinstance(case.get("derivable_values"), dict) else {}
    if slot not in values:
        return {"label": "unsupported", "source_span": None, "normalizer": None}
    return evidence_for_slot_value(case, slot, values[slot])


def contract_validator_error(
    case: dict[str, Any],
    raw_action: Any,
    *,
    repeated_slots: set[str] | None = None,
) -> dict[str, Any]:
    """Validate a candidate using only interface and request evidence.

    Gold actions are intentionally not consulted here. Synthetic derivable_values
    stand in for declared family normalizers and are never interpreted as evidence
    for the intentionally hidden slot in missing-information cases.
    """
    action, format_valid = parse_and_normalize_prediction(raw_action, case)
    errors: list[dict[str, Any]] = []
    if action is None or not format_valid:
        errors.append(
            _error(
                "invalid_json",
                "regenerate_under_grammar",
                repairable=True,
                safe_terminal_states=["call", "clarify", "refuse"],
                detail="candidate is not a parseable Action IR object",
            )
        )
        return {
            "schema_version": "tapbench.contract_validator.v1",
            "format_valid": False,
            "contract_valid": False,
            "action": action,
            "errors": errors,
            "recommended_transition": errors[0],
            "evidence_ledger": [],
        }

    mode = action.get("mode")
    if mode not in {"call", "clarify", "no_tool", "direct_answer", "refuse"}:
        errors.append(
            _error(
                "invalid_mode",
                "regenerate_under_grammar",
                offending_value=mode,
                repairable=True,
                safe_terminal_states=["call", "clarify", "refuse"],
                detail="mode is outside the Action IR contract",
            )
        )
    elif mode == "call":
        tool = _canonical_tool(case, action.get("tool"))
        if tool is None:
            errors.append(
                _error(
                    "wrong_tool_low_margin",
                    "clarify_or_escalate",
                    offending_value=action.get("tool"),
                    repairable=False,
                    safe_terminal_states=["clarify", "escalate"],
                    detail="tool is not in the available catalog and no trusted router margin is available",
                )
            )
        else:
            properties, required = _canonical_schema(case, tool)
            arguments = action.get("arguments", {}) if isinstance(action.get("arguments"), dict) else {}
            for slot in sorted(required - set(arguments)):
                evidence = _missing_slot_evidence(case, slot)
                supported = evidence.get("label") in _SUPPORTED_EVIDENCE
                errors.append(
                    _error(
                        "missing_required_slot_with_evidence" if supported else "missing_required_slot_no_evidence",
                        "repair_from_evidence" if supported else "convert_to_clarify",
                        slot=slot,
                        evidence_status=str(evidence.get("label")),
                        repairable=supported,
                        safe_terminal_states=["call", "clarify"] if supported else ["clarify"],
                        detail="required argument is absent",
                    )
                )
            for slot, value in sorted(arguments.items()):
                prop = properties.get(slot)
                evidence = evidence_for_slot_value(case, slot, value)
                label = str(evidence.get("label"))
                is_required = slot in required
                if prop is None:
                    errors.append(
                        _error(
                            "unsupported_required_value" if is_required else "unsupported_optional_field",
                            "convert_to_clarify" if is_required else "delete_field",
                            slot=slot,
                            offending_value=value,
                            evidence_status=label,
                            repairable=not is_required,
                            safe_terminal_states=["clarify"] if is_required else ["call"],
                            detail="argument is outside the selected tool schema",
                        )
                    )
                    continue
                allowed = prop.get("enum")
                if isinstance(allowed, list) and value not in allowed:
                    expected = _missing_slot_evidence(case, slot)
                    supported = expected.get("label") in _SUPPORTED_EVIDENCE
                    errors.append(
                        _error(
                            "wrong_enum_with_evidence" if supported else "wrong_enum_no_evidence",
                            "normalize_enum" if supported else "convert_to_clarify",
                            slot=slot,
                            offending_value=value,
                            evidence_status=str(expected.get("label")),
                            repairable=supported,
                            safe_terminal_states=["call"] if supported else ["clarify"],
                            detail="argument is not a legal enum value",
                        )
                    )
                    continue
                if not _type_valid(value, prop.get("type")):
                    expected = _missing_slot_evidence(case, slot)
                    supported = expected.get("label") in _SUPPORTED_EVIDENCE
                    errors.append(
                        _error(
                            "validator_cross_field_failure",
                            "local_repair_if_evidenced" if supported else "convert_to_clarify",
                            slot=slot,
                            offending_value=value,
                            evidence_status=str(expected.get("label")),
                            repairable=supported,
                            safe_terminal_states=["call", "clarify"],
                            detail=f"argument does not match declared type {prop.get('type')}",
                        )
                    )
                    continue
                if label == "contradicted":
                    errors.append(
                        _error(
                            "wrong_normalized_value_with_evidence",
                            "repair_from_evidence",
                            slot=slot,
                            offending_value=value,
                            evidence_status=label,
                            repairable=True,
                            safe_terminal_states=["call", "clarify"],
                            detail="argument conflicts with request-derived normalized evidence",
                        )
                    )
                elif label == "unsupported":
                    errors.append(
                        _error(
                            "unsupported_required_value" if is_required else "unsupported_optional_field",
                            "convert_to_clarify" if is_required else "delete_field",
                            slot=slot,
                            offending_value=value,
                            evidence_status=label,
                            repairable=not is_required,
                            safe_terminal_states=["clarify"] if is_required else ["call"],
                            detail="argument lacks explicit, normalized, or policy-default evidence",
                        )
                    )
    elif mode == "clarify":
        payload = action.get("payload", {}) if isinstance(action.get("payload"), dict) else {}
        missing = payload.get("missing_slots")
        if not isinstance(missing, list) or not missing or not all(isinstance(slot, str) and slot for slot in missing):
            errors.append(
                _error(
                    "invalid_clarification",
                    "regenerate_under_grammar",
                    repairable=True,
                    safe_terminal_states=["clarify", "escalate"],
                    detail="clarification must identify at least one missing slot",
                )
            )

    if repeated_slots:
        for index, error in enumerate(errors):
            slot = error.get("slot")
            if slot and slot in repeated_slots:
                errors[index] = _error(
                    "repeated_repair_same_slot",
                    "stop_repairing",
                    slot=str(slot),
                    offending_value=error.get("offending_value"),
                    evidence_status=str(error.get("evidence_status", "unsupported")),
                    repairable=False,
                    safe_terminal_states=["clarify", "escalate"],
                    detail="the same slot remains invalid after a prior local transition",
                )

    priority = {
        "invalid_json": 0,
        "invalid_mode": 1,
        "unsupported_required_value": 2,
        "missing_required_slot_no_evidence": 3,
        "wrong_tool_low_margin": 4,
        "wrong_enum_no_evidence": 5,
        "missing_required_slot_with_evidence": 6,
        "wrong_enum_with_evidence": 7,
        "wrong_normalized_value_with_evidence": 8,
        "validator_cross_field_failure": 9,
        "unsupported_optional_field": 10,
        "repeated_repair_same_slot": 11,
        "invalid_clarification": 12,
    }
    selected = min(errors, key=lambda row: priority.get(row["error_class"], 100)) if errors else None
    return {
        "schema_version": "tapbench.contract_validator.v1",
        "format_valid": bool(format_valid),
        "contract_valid": not errors,
        "action": action,
        "errors": errors,
        "recommended_transition": selected,
        "evidence_ledger": evidence_ledger_for_action(case, action),
    }


def _terminal_for_action(action: dict[str, Any]) -> str:
    mode = str(action.get("mode"))
    if mode == "call":
        return "call"
    if mode == "clarify":
        return "clarify"
    if mode in {"no_tool", "direct_answer"}:
        return "direct_answer"
    if mode == "refuse":
        return "refuse"
    return "escalate"


def _escalation_action(reason: str) -> dict[str, Any]:
    return {
        "mode": "refuse",
        "tool": None,
        "arguments": {},
        "payload": {"resolution_terminal": "escalate", "reason": reason},
    }


def _clarification_action(slot: str | None) -> dict[str, Any]:
    return {
        "mode": "clarify",
        "tool": None,
        "arguments": {},
        "payload": {"missing_slots": [slot] if slot else []},
    }


def _apply_local_transition(
    case: dict[str, Any],
    action: dict[str, Any] | None,
    error: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    transition = str(error.get("recommended_transition"))
    slot = error.get("slot")
    current = deepcopy(action) if isinstance(action, dict) else None
    if transition in {"repair_from_evidence", "normalize_enum", "local_repair_if_evidenced"}:
        values = case.get("derivable_values", {}) if isinstance(case.get("derivable_values"), dict) else {}
        if current is None or not slot or slot not in values:
            return _clarification_action(str(slot) if slot else None), "convert_to_clarify"
        current.setdefault("arguments", {})[slot] = values[slot]
        return current, transition
    if transition == "delete_field":
        if current is None:
            return _escalation_action("cannot delete a field from an absent action"), "escalate"
        current.setdefault("arguments", {}).pop(slot, None)
        return current, transition
    if transition == "convert_to_clarify":
        return _clarification_action(str(slot) if slot else None), transition
    if transition in {"clarify_or_escalate", "stop_repairing"}:
        return _escalation_action(str(error.get("detail", transition))), "escalate"
    if transition == "regenerate_under_grammar":
        return _escalation_action("regeneration callback unavailable"), "escalate"
    return _escalation_action(f"unsupported transition: {transition}"), "escalate"


RegenerateCallback = Callable[[dict[str, Any], int, dict[str, Any]], tuple[Any, dict[str, Any]]]


def resolve_action(
    case: dict[str, Any],
    initial_prediction: dict[str, Any],
    *,
    repair_budget: int = DEFAULT_REPAIR_BUDGET,
    regenerate: RegenerateCallback | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    raw_action: Any = initial_prediction
    attempted_slots: set[str] = set()
    iterations: list[dict[str, Any]] = []
    generation_calls = 1
    terminal_state = "escalate"
    final_action: dict[str, Any] | None = None

    for validation_round in range(repair_budget + 1):
        report = contract_validator_error(case, raw_action, repeated_slots=attempted_slots if validation_round else None)
        action = report.get("action")
        selected = report.get("recommended_transition")
        iteration = {
            "schema_version": "tapbench.tap_r_iteration.v1",
            "case_id": case["case_id"],
            "validation_round": validation_round,
            "remaining_budget": repair_budget - validation_round,
            "action": action,
            "contract_valid": report["contract_valid"],
            "error_set": report["errors"],
            "evidence_ledger": report["evidence_ledger"],
            "selected_transition": None if selected is None else selected.get("recommended_transition"),
            "selected_error_class": None if selected is None else selected.get("error_class"),
            "selected_slot": None if selected is None else selected.get("slot"),
            "confidence": None,
            "generation_calls_so_far": generation_calls,
        }
        if report["contract_valid"] and isinstance(action, dict):
            terminal_state = _terminal_for_action(action)
            final_action = action
            iteration["outcome"] = terminal_state
            iterations.append(iteration)
            break
        if validation_round >= repair_budget or selected is None:
            terminal_state = "escalate"
            final_action = _escalation_action("repair budget exhausted")
            iteration["outcome"] = terminal_state
            iterations.append(iteration)
            break

        slot = selected.get("slot")
        if slot:
            attempted_slots.add(str(slot))
        transition = str(selected.get("recommended_transition"))
        if transition == "regenerate_under_grammar" and regenerate is not None:
            raw_action, metadata = regenerate(case, validation_round + 1, selected)
            generation_calls += 1
            iteration["regeneration_metadata"] = metadata
            iteration["outcome"] = "continue"
        else:
            raw_action, applied = _apply_local_transition(case, action, selected)
            iteration["applied_transition"] = applied
            iteration["outcome"] = "escalate" if applied == "escalate" else "continue"
            if applied == "escalate":
                terminal_state = "escalate"
                final_action = raw_action
                iterations.append(iteration)
                break
        iterations.append(iteration)

    if final_action is None:
        final_action = _escalation_action("resolution terminated without an action")
    elapsed = time.perf_counter() - started
    initial_action, _ = parse_and_normalize_prediction(initial_prediction, case)
    result = {
        "prediction": final_action,
        "resolution": {
            "schema_version": TAPR_VERSION,
            "terminal_state": terminal_state,
            "validation_rounds": len(iterations),
            "repair_transitions": max(0, len(iterations) - 1),
            "generation_calls": generation_calls,
            "elapsed_seconds": elapsed,
            "initial_action": initial_action,
            "initial_metrics": validate_action(case, initial_action),
            "final_contract_valid": contract_validator_error(case, final_action)["contract_valid"] if terminal_state != "escalate" else False,
        },
    }
    return result, iterations


def resolve_predictions(
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    repair_budget: int = DEFAULT_REPAIR_BUDGET,
    source_method: str | None = "full_tap_b2",
    output_method: str = "tap_r_no_calibrator",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_by_id = {case["case_id"]: case for case in cases}
    outputs: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        if source_method and prediction.get("method") != source_method:
            continue
        case_id = str(prediction.get("case_id"))
        if case_id not in case_by_id:
            raise KeyError(f"prediction references unknown case_id: {case_id}")
        resolved, history = resolve_action(case_by_id[case_id], prediction, repair_budget=repair_budget)
        outputs.append(
            {
                **prediction,
                "method": output_method,
                "prediction": resolved["prediction"],
                "resolution": resolved["resolution"],
                "tap_r_version": TAPR_VERSION,
            }
        )
        identity = {
            key: prediction.get(key)
            for key in (
                "model_id",
                "seed",
                "backend",
                "quantization",
                "chat_template",
                "grammar_engine",
                "model_artifact",
                "thinking_mode",
                "reasoning_budget",
            )
        }
        for row in history:
            iteration_rows.append({**identity, "method": output_method, **row})
    return outputs, iteration_rows


def _unsafe_accepted_call(case: dict[str, Any], prediction: dict[str, Any]) -> bool:
    action, _ = parse_and_normalize_prediction(prediction, case)
    if action is None or action.get("mode") != "call":
        return False
    ledger = evidence_ledger_for_action(case, action)
    return any(row["required"] and row["evidence_label"] in {"unsupported", "contradicted"} for row in ledger)


def score_resolution_predictions(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_by_id = {case["case_id"]: case for case in cases}
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        case = case_by_id[str(prediction["case_id"])]
        action, format_valid = parse_and_normalize_prediction(prediction, case)
        metrics = validate_action(case, action)
        resolution = prediction.get("resolution", {}) if isinstance(prediction.get("resolution"), dict) else {}
        terminal = str(resolution.get("terminal_state", _terminal_for_action(action or {})))
        unsafe = terminal == "call" and (_unsafe_accepted_call(case, prediction) or bool(metrics["fabrication"]))
        initial_metrics = resolution.get("initial_metrics", {}) if isinstance(resolution.get("initial_metrics"), dict) else {}
        escalation_warranted = terminal == "escalate" and (
            not bool(initial_metrics.get("execution_success")) or bool(initial_metrics.get("fabrication"))
        )
        row = {
            "case_id": case["case_id"],
            "family": case["family"],
            "task_kind": case["task_kind"],
            "method": prediction.get("method"),
            "model_id": prediction.get("model_id"),
            "backend": prediction.get("backend"),
            "quantization": prediction.get("quantization"),
            "thinking_mode": prediction.get("thinking_mode"),
            "terminal_state": terminal,
            "format_valid": bool(format_valid),
            **metrics,
            "safe_resolution": bool(metrics["execution_success"] and terminal != "escalate" and not unsafe),
            "unsafe_fabrication": bool(unsafe),
            "clarify_correct": bool(case["task_kind"] == "missing_info" and metrics["execution_success"]),
            "escalated": terminal == "escalate",
            "escalation_warranted": bool(escalation_warranted),
            "generation_calls": int(resolution.get("generation_calls", 1)),
            "validation_rounds": int(resolution.get("validation_rounds", 1)),
            "resolution_seconds": float(resolution.get("elapsed_seconds", 0.0)),
            "tap_r_version": prediction.get("tap_r_version", TAPR_VERSION),
        }
        rows.append(row)
        grouped[str(row["method"])].append(row)

    summaries = []
    for method, method_rows in sorted(grouped.items()):
        n = len(method_rows)
        accepted_calls = [row for row in method_rows if row["terminal_state"] == "call"]
        summaries.append(
            {
                "method": method,
                "n": n,
                "safe_resolution_rate": sum(row["safe_resolution"] for row in method_rows) / n if n else 0.0,
                "unsafe_fabrication_rate": sum(row["unsafe_fabrication"] for row in method_rows) / n if n else 0.0,
                "clarify_accuracy": (
                    sum(row["clarify_correct"] for row in method_rows if row["task_kind"] == "missing_info")
                    / max(1, sum(row["task_kind"] == "missing_info" for row in method_rows))
                ),
                "escalation_rate": sum(row["escalated"] for row in method_rows) / n if n else 0.0,
                "non_escalated_coverage": 1.0 - (sum(row["escalated"] for row in method_rows) / n if n else 0.0),
                "accepted_call_precision": (
                    sum(row["execution_success"] and not row["unsafe_fabrication"] for row in accepted_calls) / len(accepted_calls)
                    if accepted_calls
                    else None
                ),
                "mean_generation_calls": sum(row["generation_calls"] for row in method_rows) / n if n else 0.0,
                "mean_validation_rounds": sum(row["validation_rounds"] for row in method_rows) / n if n else 0.0,
                "mean_resolution_seconds": sum(row["resolution_seconds"] for row in method_rows) / n if n else 0.0,
                "terminal_counts": dict(Counter(row["terminal_state"] for row in method_rows)),
            }
        )
    return rows, {"schema_version": "tapbench.tap_r_summary.v1", "tap_r_version": TAPR_VERSION, "groups": summaries}


def resolve_files(
    cases_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
    iterations_path: str | Path,
    scores_path: str | Path,
    summary_path: str | Path,
    *,
    repair_budget: int = DEFAULT_REPAIR_BUDGET,
    source_method: str | None = "full_tap_b2",
    output_method: str = "tap_r_no_calibrator",
) -> dict[str, Any]:
    cases = read_jsonl(cases_path)
    outputs, iterations = resolve_predictions(
        cases,
        read_jsonl(predictions_path),
        repair_budget=repair_budget,
        source_method=source_method,
        output_method=output_method,
    )
    score_rows, summary = score_resolution_predictions(cases, outputs)
    write_jsonl(output_path, outputs)
    write_jsonl(iterations_path, iterations)
    write_jsonl(scores_path, score_rows)
    target = Path(summary_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "predictions": len(outputs),
        "iterations": len(iterations),
        "scores": len(score_rows),
        "summary": summary,
    }

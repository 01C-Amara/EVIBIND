from __future__ import annotations

from typing import Any

from .families import get_family


def gold_action_is_accepted(case: dict[str, Any]) -> bool:
    gold = case.get("gold_action", {})
    return validate_action(case, gold)["schema_valid"]


def _required_gold_args(case: dict[str, Any]) -> dict[str, Any]:
    gold = case.get("gold_action", {})
    args = gold.get("arguments", {})
    return args if isinstance(args, dict) else {}



def missing_slot_for_scoring(case: dict[str, Any]) -> str:
    try:
        return get_family(str(case["family"])).missing_slot
    except KeyError:
        gold = case.get("gold_action", {})
        payload = gold.get("payload", {}) if isinstance(gold, dict) else {}
        slots = payload.get("missing_slots", []) if isinstance(payload, dict) else []
        selected = [str(slot) for slot in slots if isinstance(slot, str)]
        if len(selected) != 1:
            raise ValueError(
                "unregistered family requires exactly one gold missing slot"
            )
        return selected[0]


def _argument_overlap(predicted: dict[str, Any], gold: dict[str, Any]) -> float:
    if not gold:
        return 1.0 if not predicted else 0.0
    hits = sum(1 for key, value in gold.items() if predicted.get(key) == value)
    return hits / len(gold)


def fabrication_oracle(case: dict[str, Any], action: dict[str, Any] | None) -> bool:
    if action is None or action.get("mode") != "call":
        return False
    derivable = case.get("derivable_values", {})
    args = action.get("arguments", {})
    if not isinstance(args, dict):
        return False
    if case.get("task_kind") == "missing_info" and missing_slot_for_scoring(case) in args:
        return True
    for key, value in args.items():
        if key in derivable and derivable[key] != value:
            return True
    return False


def validate_action(case: dict[str, Any], action: dict[str, Any] | None) -> dict[str, Any]:
    if action is None:
        return {
            "schema_valid": False,
            "mode_correct": False,
            "tool_correct": False,
            "args_exact": False,
            "args_partial": 0.0,
            "execution_success": False,
            "fabrication": False,
        }

    gold = case.get("gold_action", {})
    gold_mode = gold.get("mode")
    pred_mode = action.get("mode")
    mode_correct = pred_mode == gold_mode
    task_kind = case.get("task_kind")
    valid_modes = {"call", "clarify", "no_tool", "direct_answer", "refuse"}
    schema_valid = isinstance(pred_mode, str) and pred_mode in valid_modes

    tool_correct = True
    args_exact = True
    args_partial = 1.0
    if pred_mode == "call":
        canonical_tools = {tool.get("canonical_name") for tool in case.get("tools", [])}
        pred_tool = action.get("tool")
        schema_valid = schema_valid and isinstance(pred_tool, str) and pred_tool in canonical_tools
        gold_args = _required_gold_args(case)
        predicted_args = action.get("arguments", {}) if isinstance(action.get("arguments"), dict) else {}
        args_exact = predicted_args == gold_args
        args_partial = _argument_overlap(predicted_args, gold_args)
        tool_correct = isinstance(pred_tool, str) and pred_tool == gold.get("tool")
        for required in gold_args:
            if required not in predicted_args:
                schema_valid = False
    elif pred_mode == "clarify":
        payload = action.get("payload", {})
        missing = payload.get("missing_slots", []) if isinstance(payload, dict) else []
        schema_valid = schema_valid and isinstance(missing, list)
        tool_correct = gold.get("tool") is None
        if task_kind == "missing_info":
            args_exact = missing_slot_for_scoring(case) in missing
            args_partial = 1.0 if args_exact else 0.0
        else:
            args_exact = mode_correct
            args_partial = 1.0 if args_exact else 0.0
    else:
        tool_correct = gold.get("tool") is None
        args_exact = mode_correct
        args_partial = 1.0 if args_exact else 0.0

    fabrication = fabrication_oracle(case, action)
    execution_success = bool(schema_valid and mode_correct and tool_correct and args_exact and not fabrication)
    return {
        "schema_valid": bool(schema_valid),
        "mode_correct": bool(mode_correct),
        "tool_correct": bool(tool_correct),
        "args_exact": bool(args_exact),
        "args_partial": float(args_partial),
        "execution_success": execution_success,
        "fabrication": bool(fabrication),
    }


def _contract_object_schema(parameters: dict[str, Any]) -> dict[str, Any]:
    current = parameters
    while isinstance(current, dict):
        properties = current.get("properties")
        if not isinstance(properties, dict) or set(properties) != {"payload"}:
            break
        payload = properties.get("payload")
        if not isinstance(payload, dict):
            break
        current = payload
    return current


def action_contract_is_accepted(case: dict[str, Any], action: dict[str, Any] | None) -> bool:
    """Validate the deterministic schema fragment omitted by the legacy scorer."""
    if not isinstance(action, dict) or action.get("mode") != "call":
        return isinstance(action, dict) and isinstance(action.get("mode"), str)
    tool_name = action.get("tool")
    tool = next(
        (row for row in case.get("tools", []) if row.get("canonical_name") == tool_name),
        None,
    )
    if tool is None:
        return False
    schema = _contract_object_schema(tool.get("parameters", {}))
    surface_properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    properties = {
        str(prop.get("x-ir-name") or surface): prop
        for surface, prop in surface_properties.items()
        if isinstance(prop, dict)
    }
    surface_to_canonical = {
        str(surface): str(prop.get("x-ir-name") or surface)
        for surface, prop in surface_properties.items()
        if isinstance(prop, dict)
    }
    required = [surface_to_canonical.get(str(slot), str(slot)) for slot in schema.get("required", [])]
    arguments = action.get("arguments")
    if not isinstance(arguments, dict):
        return False
    if any(slot not in arguments for slot in required):
        return False
    if schema.get("additionalProperties") is False and any(slot not in properties for slot in arguments):
        return False
    python_types = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for slot, value in arguments.items():
        prop = properties.get(slot, {})
        expected = python_types.get(prop.get("type"))
        if expected is not None and (not isinstance(value, expected) or (prop.get("type") in {"integer", "number"} and isinstance(value, bool))):
            return False
        if isinstance(prop.get("enum"), list) and value not in prop["enum"]:
            return False
    return True


def gold_contract_is_accepted(case: dict[str, Any]) -> bool:
    return action_contract_is_accepted(case, case.get("gold_action"))

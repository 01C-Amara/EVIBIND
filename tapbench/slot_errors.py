from __future__ import annotations

from typing import Any

from .families import get_family
from .ir import parse_and_normalize_prediction
from .schemas import enum_values_for_e
from .validation import missing_slot_for_scoring


def _identity(case: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "hypothesis_grid_id": case["hypothesis_grid_id"],
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
        "thinking_marker_detected": bool(
            prediction.get("thinking_marker_detected", False)
        ),
    }


def _row(
    base: dict[str, Any],
    *,
    error_type: str,
    slot: str | None = None,
    gold_value: Any = None,
    predicted_value: Any = None,
    derivable: bool = False,
) -> dict[str, Any]:
    return {
        **base,
        "error_type": error_type,
        "slot": slot,
        "gold_value": gold_value,
        "predicted_value": predicted_value,
        "derivable": bool(derivable),
    }


def _enum_values_for_slot(case: dict[str, Any], slot: str) -> list[Any] | None:
    try:
        family = get_family(str(case["family"]))
    except KeyError:
        for tool in case.get("tools", []):
            parameters = tool.get("parameters", {})
            properties = (
                parameters.get("properties", {})
                if isinstance(parameters, dict)
                else {}
            )
            for surface, prop in properties.items():
                if not isinstance(prop, dict):
                    continue
                canonical = str(prop.get("x-ir-name") or surface)
                if canonical == slot and isinstance(prop.get("enum"), list):
                    return list(prop["enum"])
        return None
    if slot != family.enum_slot:
        return None
    factors = case.get("factors", {})
    factors = factors if isinstance(factors, dict) else {}
    return enum_values_for_e(family, int(factors.get("e", 6)))


def slot_error_rows(
    case: dict[str, Any], prediction: dict[str, Any]
) -> list[dict[str, Any]]:
    action, _ = parse_and_normalize_prediction(prediction, case)
    base = _identity(case, prediction)
    gold = case.get("gold_action", {})
    gold_mode = gold.get("mode")
    if action is None:
        return [_row(base, error_type="format_invalid")]

    rows: list[dict[str, Any]] = []
    pred_mode = action.get("mode")
    if pred_mode != gold_mode:
        if gold_mode == "no_tool" and pred_mode == "call":
            rows.append(
                _row(
                    base,
                    error_type="no_tool_overcall",
                    gold_value=gold_mode,
                    predicted_value=pred_mode,
                )
            )
        else:
            rows.append(
                _row(
                    base,
                    error_type="wrong_mode",
                    gold_value=gold_mode,
                    predicted_value=pred_mode,
                )
            )

    if pred_mode == "call":
        pred_tool = action.get("tool")
        gold_tool = gold.get("tool")
        if pred_tool != gold_tool:
            rows.append(
                _row(
                    base,
                    error_type="wrong_tool",
                    gold_value=gold_tool,
                    predicted_value=pred_tool,
                )
            )
        predicted_args = (
            action.get("arguments", {})
            if isinstance(action.get("arguments"), dict)
            else {}
        )
        gold_args = (
            gold.get("arguments", {})
            if isinstance(gold.get("arguments"), dict)
            else {}
        )
        derivable_values = (
            case.get("derivable_values", {})
            if isinstance(case.get("derivable_values"), dict)
            else {}
        )
        for slot, gold_value in gold_args.items():
            if slot not in predicted_args:
                rows.append(
                    _row(
                        base,
                        error_type="missing_required_slot",
                        slot=slot,
                        gold_value=gold_value,
                        derivable=slot in derivable_values,
                    )
                )
                continue
            predicted_value = predicted_args[slot]
            if predicted_value != gold_value:
                error_type = "wrong_normalized_value"
                allowed_values = _enum_values_for_slot(case, slot)
                if (
                    allowed_values is not None
                    and predicted_value not in allowed_values
                ):
                    error_type = "wrong_enum"
                rows.append(
                    _row(
                        base,
                        error_type=error_type,
                        slot=slot,
                        gold_value=gold_value,
                        predicted_value=predicted_value,
                        derivable=slot in derivable_values,
                    )
                )
        for slot, predicted_value in predicted_args.items():
            if slot in gold_args:
                continue
            if slot in derivable_values:
                rows.append(
                    _row(
                        base,
                        error_type=(
                            "extra_optional_field"
                            if predicted_value == derivable_values[slot]
                            else "wrong_optional_field"
                        ),
                        slot=slot,
                        gold_value=derivable_values[slot],
                        predicted_value=predicted_value,
                        derivable=True,
                    )
                )
            else:
                rows.append(
                    _row(
                        base,
                        error_type="unsupported_fabricated_value",
                        slot=slot,
                        predicted_value=predicted_value,
                        derivable=False,
                    )
                )
        if case.get("task_kind") == "missing_info":
            missing_slot = missing_slot_for_scoring(case)
            if missing_slot in predicted_args:
                rows.append(
                    _row(
                        base,
                        error_type="unsupported_fabricated_value",
                        slot=missing_slot,
                        predicted_value=predicted_args.get(missing_slot),
                        derivable=False,
                    )
                )

    if gold_mode == "clarify":
        missing_gold = (
            gold.get("payload", {}).get("missing_slots", [])
            if isinstance(gold.get("payload"), dict)
            else []
        )
        missing_pred = (
            action.get("payload", {}).get("missing_slots", [])
            if isinstance(action.get("payload"), dict)
            else []
        )
        if isinstance(missing_gold, list):
            for slot in missing_gold:
                if slot not in missing_pred:
                    rows.append(
                        _row(
                            base,
                            error_type="clarify_slot_miss",
                            slot=slot,
                            gold_value=slot,
                            predicted_value=missing_pred,
                        )
                    )

    return rows


def slot_errors_for_predictions(
    cases: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    case_by_id = {case["case_id"]: case for case in cases}
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        case_id = prediction.get("case_id")
        if case_id not in case_by_id:
            raise KeyError(f"prediction references unknown case_id: {case_id}")
        rows.extend(slot_error_rows(case_by_id[case_id], prediction))
    return rows

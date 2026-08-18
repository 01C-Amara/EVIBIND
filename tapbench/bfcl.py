from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from . import NORMALIZER_VERSION, SCORER_VERSION, VALIDATOR_VERSION
from .io import read_jsonl, write_jsonl, write_yaml
from .ir import MODES, parse_and_normalize_prediction
from .scoring import _prediction_identity

BFCL_ANCHOR_VERSION = "tapbench.bfcl_anchor.v1"
DEFAULT_GRID_ID = "BFCL_v4_external_anchor"
DEFAULT_HYPOTHESIS = "external_bfcl"
NO_CALL_MODES = {"no_tool", "direct_answer", "refuse"}


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(value)
    return rows


def _source_commit(source_root: Path) -> str | None:
    for part in source_root.parts:
        if part.startswith("gorilla-") and len(part) > len("gorilla-"):
            return part.removeprefix("gorilla-")
    return None


def _category_from_filename(path: Path) -> str:
    name = path.name
    if not name.startswith("BFCL_v4_") or not name.endswith(".json"):
        return path.stem
    return name.removeprefix("BFCL_v4_").removesuffix(".json")


def _task_kind_for_category(category: str) -> str:
    return "no_tool" if "irrelevance" in category else "call"


def _parameter_type(value: Any) -> str:
    mapping = {
        "dict": "object",
        "float": "number",
        "double": "number",
        "int": "integer",
        "integer": "integer",
        "str": "string",
        "string": "string",
        "bool": "boolean",
        "boolean": "boolean",
        "list": "array",
        "array": "array",
    }
    return mapping.get(str(value), str(value) if value else "string")


def _convert_schema(schema: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = dict(schema)
    converted["type"] = _parameter_type(converted.get("type", "object"))
    properties = converted.get("properties")
    if isinstance(properties, dict):
        converted["properties"] = {
            str(name): _convert_schema(value if isinstance(value, dict) else {"type": "string"}) for name, value in properties.items()
        }
    if "items" in converted and isinstance(converted["items"], dict):
        converted["items"] = _convert_schema(converted["items"])
    converted.setdefault("additionalProperties", False)
    return converted


def _convert_function(function: dict[str, Any]) -> dict[str, Any]:
    name = str(function.get("name"))
    parameters = function.get("parameters") if isinstance(function.get("parameters"), dict) else {"type": "object", "properties": {}}
    return {
        "name": name,
        "canonical_name": name,
        "description": str(function.get("description", "")),
        "parameters": _convert_schema(parameters),
    }


def _flatten_messages(question: Any) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if isinstance(question, list):
        selected = question[0] if question and isinstance(question[0], list) else question
        for item in selected:
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = str(item.get("content", ""))
                messages.append({"role": role, "content": content})
    if not messages:
        messages = [{"role": "user", "content": str(question)}]
    return [
        {
            "role": "system",
            "content": "Return exactly one Action IR JSON object. Do not call a tool unless one of the available functions is relevant.",
        },
        *messages,
    ]


def _load_answer_map(source_root: Path, category: str) -> dict[str, dict[str, Any]]:
    answer_path = source_root / "possible_answer" / f"BFCL_v4_{category}.json"
    if not answer_path.exists():
        return {}
    return {str(row["id"]): row for row in _read_jsonl_objects(answer_path)}


def _first_nonempty(values: Iterable[Any]) -> Any:
    first: Any = None
    for value in values:
        if first is None:
            first = value
        if value not in ("", None):
            return value
    return first


def _gold_from_answer(answer: dict[str, Any] | None, functions: list[dict[str, Any]], task_kind: str) -> dict[str, Any]:
    if task_kind == "no_tool" or not answer:
        return {"mode": "no_tool", "tool": None, "arguments": {}, "payload": {}}
    ground_truth = answer.get("ground_truth", [])
    if not ground_truth or not isinstance(ground_truth[0], dict):
        return {"mode": "call", "tool": functions[0]["name"] if functions else None, "arguments": {}, "payload": {}}
    tool_name, args = next(iter(ground_truth[0].items()))
    required = set(functions[0].get("parameters", {}).get("required", [])) if functions else set()
    canonical_args: dict[str, Any] = {}
    if isinstance(args, dict):
        for key, values in args.items():
            value_list = values if isinstance(values, list) else [values]
            if key in required:
                canonical_args[str(key)] = _first_nonempty(value_list)
    return {"mode": "call", "tool": str(tool_name), "arguments": canonical_args, "payload": {}}


def _bfcl_gold(answer: dict[str, Any] | None, task_kind: str) -> dict[str, Any]:
    if task_kind == "no_tool" or not answer:
        return {"expected_mode": "no_call", "allowed_calls": []}
    allowed_calls: list[dict[str, Any]] = []
    for item in answer.get("ground_truth", []):
        if not isinstance(item, dict):
            continue
        for tool_name, args in item.items():
            allowed_args: dict[str, list[Any]] = {}
            if isinstance(args, dict):
                for key, values in args.items():
                    allowed_args[str(key)] = values if isinstance(values, list) else [values]
            allowed_calls.append({"tool": str(tool_name), "arguments": allowed_args})
    return {"expected_mode": "call", "allowed_calls": allowed_calls}


def convert_bfcl_cases(
    source_root: str | Path,
    output_path: str | Path,
    *,
    categories: Iterable[str] = ("irrelevance", "simple_python"),
    limit_per_category: int | None = None,
    grid_id: str = DEFAULT_GRID_ID,
    manifest_path: str | Path | None = None,
) -> int:
    root = Path(source_root)
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for category in categories:
        category = category.strip()
        if not category:
            continue
        data_path = root / f"BFCL_v4_{category}.json"
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        raw_rows = _read_jsonl_objects(data_path)
        answers = _load_answer_map(root, category)
        selected = raw_rows[:limit_per_category] if limit_per_category is not None else raw_rows
        task_kind = _task_kind_for_category(category)
        for index, raw in enumerate(selected):
            bfcl_id = str(raw.get("id", f"{category}_{index}"))
            tools = [_convert_function(fn) for fn in raw.get("function", []) if isinstance(fn, dict)]
            answer = answers.get(bfcl_id)
            gold_action = _gold_from_answer(answer, tools, task_kind)
            bfcl_gold = _bfcl_gold(answer, task_kind)
            required_args = []
            if tools:
                required_args = list(tools[0].get("parameters", {}).get("required", []))
            rows.append(
                {
                    "case_id": f"bfcl_v4_{category}_{index:04d}",
                    "hypothesis_grid_id": grid_id,
                    "hypothesis": DEFAULT_HYPOTHESIS,
                    "split": "external_anchor",
                    "family": "bfcl",
                    "task_kind": task_kind,
                    "factors": {"external_source": "BFCL_v4", "bfcl_category": category, "task_kind": task_kind},
                    "messages": _flatten_messages(raw.get("question")),
                    "tools": tools,
                    "tool_aliases": {tool["name"]: tool["canonical_name"] for tool in tools},
                    "argument_aliases": {},
                    "gold_action": gold_action,
                    "derivable_values": dict(gold_action.get("arguments", {})),
                    "metadata": {
                        "external_source": "BFCL_v4",
                        "bfcl_category": category,
                        "bfcl_id": bfcl_id,
                        "source_commit": _source_commit(root),
                        "required_args": required_args,
                    },
                    "bfcl_gold": bfcl_gold,
                }
            )
        counts[category] = len(selected)
    count = write_jsonl(output_path, rows)
    if manifest_path is not None:
        write_yaml(
            manifest_path,
            {
                "schema_version": BFCL_ANCHOR_VERSION,
                "source_root": str(root),
                "source_commit": _source_commit(root),
                "grid_id": grid_id,
                "categories": list(categories),
                "limit_per_category": limit_per_category,
                "case_counts": counts,
                "case_count": count,
                "output_path": str(output_path),
                "scoring_note": (
                    "External BFCL anchor is scored separately from the synthetic H1-H6 GLMMs. "
                    "Irrelevance rows evaluate no-call detection; simple rows evaluate a single BFCL Python function call."
                ),
            },
        )
    return count


def _identity(case: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    case_meta = case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}
    identity = _prediction_identity(prediction)
    return {
        "case_id": case["case_id"],
        "hypothesis_grid_id": case["hypothesis_grid_id"],
        "hypothesis": case.get("hypothesis", DEFAULT_HYPOTHESIS),
        "family": case.get("family", "bfcl"),
        "task_kind": case.get("task_kind"),
        "external_source": case_meta.get("external_source", "BFCL_v4"),
        "bfcl_category": case_meta.get("bfcl_category") or case.get("factors", {}).get("bfcl_category"),
        "bfcl_id": case_meta.get("bfcl_id"),
        "source_commit": case_meta.get("source_commit"),
        **identity,
        "runner_error": prediction.get("runner_error"),
        "scorer_version": SCORER_VERSION,
        "normalizer_version": NORMALIZER_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "bfcl_anchor_version": BFCL_ANCHOR_VERSION,
        "eflrx_version": prediction.get("eflrx_version"),
        "eflrx_runner_version": prediction.get("eflrx_runner_version"),
        "committee_version": prediction.get("committee_version"),
        "committee_split": prediction.get("committee_split"),
    }


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        lower = text.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
            try:
                return float(text)
            except ValueError:
                pass
        return lower
    return value


def _value_allowed(predicted: Any, allowed_values: list[Any]) -> bool:
    normalized_pred = _normalize_scalar(predicted)
    for allowed in allowed_values:
        if normalized_pred == _normalize_scalar(allowed):
            return True
    return False


def _allowed_for_tool(case: dict[str, Any], tool_name: str | None) -> dict[str, list[Any]] | None:
    gold = case.get("bfcl_gold", {})
    for call in gold.get("allowed_calls", []):
        if call.get("tool") == tool_name:
            args = call.get("arguments", {})
            return args if isinstance(args, dict) else {}
    return None


def _required_args(case: dict[str, Any], tool_name: str | None) -> set[str]:
    for tool in case.get("tools", []):
        if tool.get("canonical_name") == tool_name or tool.get("name") == tool_name:
            required = tool.get("parameters", {}).get("required", [])
            return {str(item) for item in required if isinstance(item, str)}
    return set()


def _argument_metrics(case: dict[str, Any], action: dict[str, Any]) -> tuple[bool, float, list[dict[str, Any]], bool]:
    tool_name = action.get("tool") if isinstance(action.get("tool"), str) else None
    allowed_args = _allowed_for_tool(case, tool_name)
    required = _required_args(case, tool_name)
    predicted_args = action.get("arguments", {}) if isinstance(action.get("arguments"), dict) else {}
    errors: list[dict[str, Any]] = []
    if allowed_args is None:
        return False, 0.0, errors, False

    hits = 0
    for slot in sorted(required):
        allowed_values = allowed_args.get(slot, [])
        if slot not in predicted_args:
            errors.append({"error_type": "missing_required_slot", "slot": slot, "gold_value": allowed_values, "predicted_value": None, "derivable": True})
            continue
        if _value_allowed(predicted_args[slot], allowed_values):
            hits += 1
        else:
            errors.append(
                {
                    "error_type": "wrong_normalized_value",
                    "slot": slot,
                    "gold_value": allowed_values,
                    "predicted_value": predicted_args[slot],
                    "derivable": True,
                }
            )

    fabrication = False
    for slot, predicted_value in predicted_args.items():
        if slot in required:
            continue
        if slot in allowed_args:
            if not _value_allowed(predicted_value, allowed_args[slot]):
                fabrication = True
                errors.append(
                    {
                        "error_type": "wrong_optional_field",
                        "slot": slot,
                        "gold_value": allowed_args[slot],
                        "predicted_value": predicted_value,
                        "derivable": True,
                    }
                )
        else:
            fabrication = True
            errors.append(
                {
                    "error_type": "unsupported_fabricated_value",
                    "slot": slot,
                    "gold_value": None,
                    "predicted_value": predicted_value,
                    "derivable": False,
                }
            )
    args_exact = not any(error["error_type"] in {"missing_required_slot", "wrong_normalized_value", "wrong_optional_field", "unsupported_fabricated_value"} for error in errors)
    args_partial = 1.0 if not required else hits / len(required)
    return args_exact, args_partial, errors, fabrication


def score_bfcl_prediction(case: dict[str, Any], prediction: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    action, format_valid = parse_and_normalize_prediction(prediction, case)
    base = _identity(case, prediction)
    errors: list[dict[str, Any]] = []
    valid_modes = set(MODES)
    expected_mode = case.get("bfcl_gold", {}).get("expected_mode")

    schema_valid = bool(format_valid and action is not None and isinstance(action.get("mode"), str) and action.get("mode") in valid_modes)
    mode_correct = False
    tool_correct = False
    args_exact = False
    args_partial = 0.0
    fabrication = False
    no_call_success = False

    if action is None:
        errors.append({"error_type": "format_invalid", "slot": None, "gold_value": expected_mode, "predicted_value": None, "derivable": False})
    elif expected_mode == "no_call":
        pred_mode = action.get("mode")
        no_call_success = pred_mode in NO_CALL_MODES
        mode_correct = no_call_success
        tool_correct = no_call_success
        args_exact = no_call_success
        args_partial = 1.0 if no_call_success else 0.0
        if pred_mode == "call":
            fabrication = True
            errors.append({"error_type": "no_tool_overcall", "slot": None, "gold_value": "no_call", "predicted_value": pred_mode, "derivable": False})
        elif not no_call_success:
            errors.append({"error_type": "wrong_mode", "slot": None, "gold_value": "no_call", "predicted_value": pred_mode, "derivable": False})
    else:
        pred_mode = action.get("mode")
        mode_correct = pred_mode == "call"
        if not mode_correct:
            errors.append({"error_type": "wrong_mode", "slot": None, "gold_value": "call", "predicted_value": pred_mode, "derivable": False})
        canonical_tools = {tool.get("canonical_name") for tool in case.get("tools", [])}
        pred_tool = action.get("tool") if isinstance(action.get("tool"), str) else None
        tool_correct = bool(pred_tool and _allowed_for_tool(case, pred_tool) is not None)
        schema_valid = schema_valid and (pred_mode != "call" or bool(pred_tool in canonical_tools))
        if pred_mode == "call" and not tool_correct:
            errors.append({"error_type": "wrong_tool", "slot": None, "gold_value": [call.get("tool") for call in case.get("bfcl_gold", {}).get("allowed_calls", [])], "predicted_value": pred_tool, "derivable": False})
        if pred_mode == "call" and tool_correct:
            args_exact, args_partial, arg_errors, fabrication = _argument_metrics(case, action)
            errors.extend(arg_errors)
            required = _required_args(case, pred_tool)
            predicted_args = action.get("arguments", {}) if isinstance(action.get("arguments"), dict) else {}
            for required_slot in required:
                if required_slot not in predicted_args:
                    schema_valid = False
        else:
            args_exact = False
            args_partial = 0.0

    execution_success = bool(schema_valid and mode_correct and tool_correct and args_exact and not fabrication)
    row = {
        **base,
        "format_valid": bool(format_valid),
        "schema_valid": bool(schema_valid),
        "mode_correct": bool(mode_correct),
        "tool_correct": bool(tool_correct),
        "args_exact": bool(args_exact),
        "args_partial": float(args_partial),
        "execution_success": execution_success,
        "fabrication": bool(fabrication),
        "no_call_success": bool(no_call_success),
    }
    slot_rows = [{**base, **error} for error in errors]
    return row, slot_rows


def score_bfcl_predictions(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_by_id = {case["case_id"]: case for case in cases}
    rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        case_id = prediction.get("case_id")
        if case_id not in case_by_id:
            raise KeyError(f"prediction references unknown BFCL case_id: {case_id}")
        row, errors = score_bfcl_prediction(case_by_id[case_id], prediction)
        rows.append(row)
        slot_rows.extend(errors)
    return rows, slot_rows


def summarize_bfcl_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row.get("bfcl_category")), str(row.get("model_id")), str(row.get("method")))].append(row)
    groups: list[dict[str, Any]] = []
    metric_names = ["format_valid", "schema_valid", "mode_correct", "tool_correct", "args_exact", "execution_success", "fabrication", "no_call_success"]
    for (category, model_id, method), group in sorted(by_key.items()):
        out: dict[str, Any] = {"bfcl_category": category, "model_id": model_id, "method": method, "n": len(group)}
        for metric in metric_names:
            hits = sum(1 for row in group if bool(row.get(metric)))
            out[metric] = hits
            out[f"{metric}_rate"] = hits / len(group) if group else 0.0
        groups.append(out)
    overall: dict[str, Any] = {"n": len(rows), "groups": groups}
    for metric in metric_names:
        hits = sum(1 for row in rows if bool(row.get(metric)))
        overall[metric] = hits
        overall[f"{metric}_rate"] = hits / len(rows) if rows else 0.0
    return overall


def score_bfcl_files(
    cases_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
    *,
    slot_errors_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> int:
    cases = read_jsonl(cases_path)
    predictions = read_jsonl(predictions_path)
    rows, slot_rows = score_bfcl_predictions(cases, predictions)
    if slot_errors_path is not None:
        write_jsonl(slot_errors_path, slot_rows)
    if summary_path is not None:
        target = Path(summary_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summarize_bfcl_scores(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return write_jsonl(output_path, rows)

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from .schemas import denest_arguments


MODES = {"call", "clarify", "no_tool", "direct_answer", "refuse"}
MODE_ALIASES = {
    "tool": "call",
    "tool_call": "call",
    "function_call": "call",
    "function": "call",
    "ask_user": "clarify",
    "ask_clarification": "clarify",
    "none": "no_tool",
    "answer": "direct_answer",
    "refusal": "refuse",
    "refused": "refuse",
}


def _parse_json_maybe(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def parse_prediction(raw: Any) -> tuple[dict[str, Any] | None, bool]:
    raw = _parse_json_maybe(raw)
    if not isinstance(raw, dict):
        return None, False
    if "prediction" in raw:
        return parse_prediction(raw["prediction"])
    if "action" in raw:
        return parse_prediction(raw["action"])
    if "action_ir" in raw:
        return parse_prediction(raw["action_ir"])
    if "content" in raw:
        parsed = _parse_json_maybe(raw["content"])
        if isinstance(parsed, dict):
            return parse_prediction(parsed)
    if "tool_calls" in raw and isinstance(raw["tool_calls"], list) and raw["tool_calls"]:
        first = raw["tool_calls"][0]
        function = first.get("function", first) if isinstance(first, dict) else {}
        if isinstance(function, dict):
            arguments = _parse_json_maybe(function.get("arguments", {}))
            return {
                "mode": "call",
                "tool": function.get("name"),
                "arguments": arguments if isinstance(arguments, dict) else {},
                "payload": {},
            }, True
    if "name" in raw and "arguments" in raw:
        return {
            "mode": "call",
            "tool": raw.get("name"),
            "arguments": raw.get("arguments") if isinstance(raw.get("arguments"), dict) else {},
            "payload": {},
        }, True
    if "mode" in raw:
        action = {
            "mode": raw.get("mode"),
            "tool": raw.get("tool"),
            "arguments": raw.get("arguments", {}),
            "payload": raw.get("payload", {}),
        }
        if not isinstance(action["arguments"], dict):
            action["arguments"] = {}
        if not isinstance(action["payload"], dict):
            action["payload"] = {}
        return action, True
    if "tool" in raw and "arguments" in raw:
        return {
            "mode": "call",
            "tool": raw.get("tool"),
            "arguments": raw.get("arguments") if isinstance(raw.get("arguments"), dict) else {},
            "payload": raw.get("payload", {}) if isinstance(raw.get("payload"), dict) else {},
        }, True
    return None, False


def normalize_action(
    action: dict[str, Any] | None,
    case: dict[str, Any],
    *,
    apply_aliases: bool = True,
) -> dict[str, Any] | None:
    if action is None:
        return None
    normalized = deepcopy(action)
    mode = str(normalized.get("mode", "")).strip()
    mode = MODE_ALIASES.get(mode, mode)
    normalized["mode"] = mode
    if mode not in MODES:
        return normalized

    tool_aliases = case.get("tool_aliases", {}) if apply_aliases else {}
    argument_aliases = case.get("argument_aliases", {}) if apply_aliases else {}
    tool = normalized.get("tool")
    if isinstance(tool, dict):
        if not isinstance(normalized.get("arguments"), dict) and isinstance(tool.get("arguments"), dict):
            normalized["arguments"] = tool["arguments"]
        elif not normalized.get("arguments") and isinstance(tool.get("arguments"), dict):
            normalized["arguments"] = tool["arguments"]
        if isinstance(tool.get("name"), str):
            normalized["tool"] = tool["name"]
        elif isinstance(tool.get("tool"), str):
            normalized["tool"] = tool["tool"]
        tool = normalized.get("tool")
    if isinstance(tool, str):
        normalized["tool"] = tool_aliases.get(tool, tool)

    args = denest_arguments(normalized.get("arguments", {}))
    canonical_args: dict[str, Any] = {}
    for key, value in args.items():
        canonical_args[argument_aliases.get(key, key)] = value
    normalized["arguments"] = canonical_args
    if not isinstance(normalized.get("payload"), dict):
        normalized["payload"] = {}
    if normalized.get("mode") == "clarify":
        missing = normalized["payload"].get("missing_slots")
        if isinstance(missing, str):
            known_slots: list[str] = []
            for tool_row in case.get("tools", []):
                parameters = tool_row.get("parameters", {})
                while isinstance(parameters, dict):
                    properties = parameters.get("properties")
                    if not isinstance(properties, dict):
                        break
                    if set(properties) == {"payload"} and isinstance(properties.get("payload"), dict):
                        parameters = properties["payload"]
                        continue
                    for surface, prop in properties.items():
                        canonical = str(prop.get("x-ir-name") or surface) if isinstance(prop, dict) else str(surface)
                        if canonical not in known_slots:
                            known_slots.append(canonical)
                    break
            matched = [
                slot
                for slot in known_slots
                if re.search(rf"(?<!\w){re.escape(slot.replace('_', ' '))}(?!\w)", missing, re.IGNORECASE)
                or re.search(rf"(?<!\w){re.escape(slot)}(?!\w)", missing, re.IGNORECASE)
            ]
            normalized["payload"]["missing_slots"] = matched if len(matched) == 1 else [missing]
    return normalized


def parse_and_normalize_prediction(raw: Any, case: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    already_canonical = bool(
        isinstance(raw, dict) and raw.get("action_ir_normalized") is True
    )
    action, format_valid = parse_prediction(raw)
    return (
        normalize_action(action, case, apply_aliases=not already_canonical),
        format_valid,
    )

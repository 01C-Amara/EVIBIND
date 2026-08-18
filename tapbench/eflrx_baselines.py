from __future__ import annotations

import json
from typing import Any

from .eflrx import RequestFn, _merge_metadata, preflight_schema_request
from .runner import action_ir_json_schema, render_chat_messages
from .validation import action_contract_is_accepted


RAW_BASELINE_VERSION = "tapbench.eflrx_raw_baselines.v1"
RAW_BASELINE_CONDITIONS = ("prompt_few_shot", "full_tap_b2")
FULL_TAP_REPAIR_BUDGET = 2


def _shape_violations(
    case: dict[str, Any],
    action: dict[str, Any],
) -> list[str]:
    if not isinstance(action, dict):
        return ["action_not_object"]
    required_keys = {"mode", "tool", "arguments", "payload"}
    violations = [
        f"missing_key:{key}"
        for key in sorted(required_keys - set(action))
    ]
    mode = action.get("mode")
    if mode not in {"call", "clarify", "no_tool", "direct_answer"}:
        violations.append("invalid_mode")
    arguments = action.get("arguments")
    payload = action.get("payload")
    if not isinstance(arguments, dict):
        violations.append("arguments_not_object")
    if not isinstance(payload, dict):
        violations.append("payload_not_object")
    if mode == "call":
        if not action_contract_is_accepted(case, action):
            violations.append("call_contract_invalid")
    else:
        if action.get("tool") is not None:
            violations.append("non_call_tool_not_null")
        if isinstance(arguments, dict) and arguments:
            violations.append("non_call_arguments_not_empty")
        if mode == "clarify":
            missing = (
                payload.get("missing_slots")
                if isinstance(payload, dict)
                else None
            )
            if not isinstance(missing, list):
                violations.append("clarify_missing_slots_not_list")
    return sorted(set(violations))


def _repair_messages(
    messages: list[dict[str, str]],
    action: dict[str, Any],
    violations: list[str],
) -> list[dict[str, str]]:
    return [
        *messages,
        {
            "role": "assistant",
            "content": json.dumps(
                action,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
        },
        {
            "role": "user",
            "content": (
                "The deterministic Action IR validator rejected that object "
                "with codes: "
                + ", ".join(violations)
                + ". Correct only those contract violations. Do not invent "
                "values or add reasoning; return one corrected JSON object."
            ),
        },
    ]


def run_raw_baseline(
    case: dict[str, Any],
    *,
    endpoint: str,
    condition: str,
    max_tokens: int,
    seed: int,
    request_fn: RequestFn = preflight_schema_request,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if condition not in RAW_BASELINE_CONDITIONS:
        raise ValueError(f"unknown raw baseline condition: {condition}")
    repair_budget = (
        FULL_TAP_REPAIR_BUDGET
        if condition == "full_tap_b2"
        else 0
    )
    messages = render_chat_messages(
        case,
        condition,
        thinking_mode="off",
    )
    response_schema = action_ir_json_schema(case)
    call_metadata: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    action: dict[str, Any] = {}
    for attempt in range(repair_budget + 1):
        raw, metadata = request_fn(
            endpoint,
            messages,
            response_schema=response_schema,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed + attempt * 100003,
        )
        action = raw if isinstance(raw, dict) else {}
        violations = _shape_violations(case, action)
        call_metadata.append(metadata)
        history.append(
            {
                "attempt": attempt,
                "violations": violations,
                "response_schema_sha256": metadata.get(
                    "response_schema_sha256"
                ),
            }
        )
        if not violations:
            break
        if attempt < repair_budget:
            messages = _repair_messages(messages, action, violations)

    merged = _merge_metadata(call_metadata)
    merged.update(
        {
            "raw_baseline_version": RAW_BASELINE_VERSION,
            "repair_budget": repair_budget,
            "repair_attempts_used": max(0, len(call_metadata) - 1),
            "validator_history": history,
            "final_contract_valid": not history[-1]["violations"],
        }
    )
    return action, merged

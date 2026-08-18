from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping

from evibind.core import CandidateTable, PolicySet
from evibind.core.derivations import sha256_digest

from .json_contract import json_contract_accepts


AUDIT_MODE_VERSION = "evibind.audit_mode.v1"
_MISSING = object()


def _pointer_tokens(pointer: str) -> tuple[str, ...]:
    return tuple(
        token.replace("~1", "/").replace("~0", "~")
        for token in pointer.removeprefix("/").split("/")
        if token
    )


def _get_pointer(value: Any, pointer: str) -> Any:
    current = value
    for token in _pointer_tokens(pointer):
        if isinstance(current, Mapping):
            if token not in current:
                return _MISSING
            current = current[token]
        elif isinstance(current, list) and token.isdigit():
            index = int(token)
            if index >= len(current):
                return _MISSING
            current = current[index]
        else:
            return _MISSING
    return current


def _leaf_paths(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, Mapping):
        if not value:
            return {prefix} if prefix else set()
        paths: set[str] = set()
        for raw_key, item in value.items():
            key = str(raw_key).replace("~", "~0").replace("/", "~1")
            paths.update(_leaf_paths(item, prefix + "/" + key))
        return paths
    if isinstance(value, list):
        if not value:
            return {prefix} if prefix else set()
        paths = set()
        for index, item in enumerate(value):
            paths.update(_leaf_paths(item, prefix + f"/{index}"))
        return paths
    return {prefix or "/"}


def _audit_call(
    message: Mapping[str, Any],
    *,
    policy: PolicySet,
    candidates: CandidateTable,
    tools: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    calls = message.get("tool_calls")
    if message.get("function_call") is not None:
        return {
            "decision": "audit_fail",
            "observed_tool_call": True,
            "would_release": False,
            "reason": "legacy_function_call_not_allowed",
        }
    if calls is None:
        return {
            "decision": "no_native_tool_call",
            "observed_tool_call": False,
            "would_release": False,
            "reason": "no_proposed_tool_call",
        }
    if not isinstance(calls, list) or len(calls) != 1:
        return {
            "decision": "audit_fail",
            "observed_tool_call": True,
            "would_release": False,
            "reason": "expected_exactly_one_native_tool_call",
        }
    call = calls[0]
    if not isinstance(call, Mapping) or call.get("type") != "function":
        return {
            "decision": "audit_fail",
            "observed_tool_call": True,
            "would_release": False,
            "reason": "native_tool_call_type_invalid",
        }
    function = call.get("function")
    if not isinstance(function, Mapping):
        return {
            "decision": "audit_fail",
            "observed_tool_call": True,
            "would_release": False,
            "reason": "native_tool_call_missing_function",
        }
    tool_id = function.get("name")
    arguments_text = function.get("arguments")
    if not isinstance(tool_id, str) or not isinstance(arguments_text, str):
        return {
            "decision": "audit_fail",
            "observed_tool_call": True,
            "would_release": False,
            "reason": "native_tool_call_shape_invalid",
        }
    try:
        arguments = json.loads(arguments_text)
    except json.JSONDecodeError:
        arguments = None
    if not isinstance(arguments, Mapping):
        return {
            "decision": "audit_fail",
            "observed_tool_call": True,
            "would_release": False,
            "tool": tool_id,
            "reason": "native_arguments_invalid",
        }
    try:
        tool_policy = policy.tool(tool_id)
    except ValueError:
        return {
            "decision": "audit_fail",
            "observed_tool_call": True,
            "would_release": False,
            "tool": tool_id,
            "reason": "tool_not_declared",
        }
    declared = {slot.destination_scope: slot for slot in tool_policy.slots}
    present = {
        pointer: value
        for pointer in declared
        if (value := _get_pointer(arguments, pointer)) is not _MISSING
    }
    missing = sorted(
        pointer
        for pointer in tool_policy.required_destinations
        if pointer not in present
    )
    leaves = _leaf_paths(arguments)
    uncovered = sorted(
        path
        for path in leaves
        if not any(
            path == pointer or path.startswith(pointer + "/") for pointer in declared
        )
    )
    unsupported: list[str] = []
    matched: dict[str, list[str]] = {}
    for pointer, value in present.items():
        digest = sha256_digest(value)
        ids = sorted(
            candidate_id
            for candidate_id, candidate in candidates.candidates.items()
            if candidate.witness.tool_id == tool_id
            and candidate.witness.destination_scope == pointer
            and candidate.witness.value_digest == digest
        )
        if ids:
            matched[pointer] = ids
        else:
            unsupported.append(pointer)
    tool = tools.get(tool_id)
    contract_ok = bool(
        tool is not None
        and json_contract_accepts(
            arguments,
            tool.get("parameters", {}),
        )
    )
    would_release = not (missing or uncovered or unsupported) and contract_ok
    return {
        "decision": "audit_pass" if would_release else "audit_fail",
        "observed_tool_call": True,
        "would_release": would_release,
        "tool": tool_id,
        "missing_required": missing,
        "uncovered_arguments": uncovered,
        "unsupported_destinations": sorted(unsupported),
        "matched_candidate_ids": matched,
        "contract_valid": contract_ok,
        "reason": None if would_release else "native_call_not_evidence_bound",
    }


def audit_native_response(
    upstream_response: Mapping[str, Any],
    *,
    policy: PolicySet,
    candidates: CandidateTable,
    tools: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach a shadow decision while preserving the native response."""
    protected = deepcopy(dict(upstream_response))
    choices = protected.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("audit mode requires exactly one upstream choice")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else None
    if not isinstance(message, Mapping):
        raise ValueError("audit mode response omitted an assistant message")
    result = _audit_call(
        message,
        policy=policy,
        candidates=candidates,
        tools=tools,
    )
    protected["evibind"] = {
        "version": AUDIT_MODE_VERSION,
        "operating_mode": "audit",
        "enforced": False,
        "selective_guarantee": None,
        "warning": (
            "Audit mode preserves native executable literals and does not "
            "provide EviBind's materialization guarantee."
        ),
        "choices": [
            {
                "index": int(choice.get("index", 0)),
                **candidates.metrics(),
                **result,
            }
        ],
    }
    return protected

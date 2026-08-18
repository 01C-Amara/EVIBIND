from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Callable, Mapping

import pytest

from evibind.host import (
    HOST_SDK_VERSION,
    GuardedToolExecutor,
    HostSDKError,
)
from tapbench.one_call_gateway import action_branches


SECRET = b"host-sdk-test-secret-material-32-bytes"


def _nonce_source() -> Callable[[int], bytes]:
    counter = 0

    def nonce_bytes(size: int) -> bytes:
        nonlocal counter
        counter += 1
        return bytes([counter]) * size

    return nonce_bytes


def _request(*, include_fee: bool = False) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "amount": {
            "type": "number",
            "x-evibind-evidence-type": "number",
            "x-evibind-sources": ["user.current_turn"],
            "x-evibind-extraction-cue": "amount",
            "x-evibind-criticality": "control",
        }
    }
    required = ["amount"]
    content = "Pay amount=20"
    if include_fee:
        properties["fee"] = {
            "type": "number",
            "x-evibind-evidence-type": "number",
            "x-evibind-sources": ["user.current_turn"],
            "x-evibind-extraction-cue": "fee",
            "x-evibind-criticality": "control",
        }
        required.append("fee")
        content += " fee=3"
    return {
        "model": "fixture",
        "messages": [{"id": "user-1", "role": "user", "content": content}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "pay_invoice",
                    "description": "Pay one invoice.",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "evibind": {"policy_epoch": "host-sdk-v1"},
    }


def _call_branch(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    branches = action_branches(payload["tools"][0]["function"]["parameters"])
    return next(
        branch
        for branch in branches
        if branch["properties"]["mode"].get("const") == "call"
    )


def _bindings(payload: Mapping[str, Any]) -> dict[str, str]:
    branch = _call_branch(payload)
    properties = branch["properties"]["bindings"]["properties"]
    return {
        destination: schema["enum"][0]
        for destination, schema in properties.items()
    }


def _response(bindings: Mapping[str, str]) -> dict[str, Any]:
    action = {
        "mode": "call",
        "tool_id": "pay_invoice",
        "bindings": dict(bindings),
    }
    return {
        "id": "host-fixture-response",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "host-action",
                            "type": "function",
                            "function": {
                                "name": "evibind_action",
                                "arguments": json.dumps(action),
                            },
                        }
                    ],
                },
            }
        ],
    }


def test_host_executor_materializes_and_dispatches_once() -> None:
    observed: list[dict[str, Any]] = []

    def pay(arguments: Mapping[str, Any]) -> dict[str, Any]:
        observed.append(dict(arguments))
        arguments["amount"] = 999
        return {"receipt": "receipt-1"}

    executor = GuardedToolExecutor(
        {"pay_invoice": pay},
        handle_secret=SECRET,
        handle_nonce_bytes=_nonce_source(),
    )
    turn = executor.prepare(_request())
    payload = turn.upstream_payload

    assert "evibind" not in payload
    assert payload["tools"][0]["function"]["name"] == "evibind_action"
    result = turn.complete(_response(_bindings(payload)))

    assert result.executed is True
    assert result.decision == "call"
    assert result.tool_id == "pay_invoice"
    assert result.arguments == {"amount": 20}
    assert result.result == {"receipt": "receipt-1"}
    assert observed == [{"amount": 20}]
    host_record = result.protected_response["evibind"]["host_execution"]
    assert host_record["version"] == HOST_SDK_VERSION
    assert host_record["executed"] is True
    assert host_record["manifest_digest"] == result.manifest_digest
    assert turn.completed is True
    with pytest.raises(HostSDKError, match="already been completed"):
        turn.complete(_response(_bindings(payload)))
    assert observed == [{"amount": 20}]


def test_host_executor_withholds_cross_slot_handle_reuse() -> None:
    observed: list[dict[str, Any]] = []
    executor = GuardedToolExecutor(
        {"pay_invoice": lambda arguments: observed.append(dict(arguments))},
        handle_secret=SECRET,
        handle_nonce_bytes=_nonce_source(),
    )
    turn = executor.prepare(_request(include_fee=True))
    bindings = _bindings(turn.upstream_payload)
    attack = {
        "/amount": bindings["/fee"],
        "/fee": bindings["/amount"],
    }

    result = turn.complete(_response(attack))

    assert result.executed is False
    assert result.decision == "invalid_action_ir"
    assert observed == []
    assert (
        result.protected_response["evibind"]["host_execution"]["executed"]
        is False
    )


def test_host_executor_requires_registered_handlers_and_enforcement() -> None:
    with pytest.raises(HostSDKError, match="enforce or assist"):
        GuardedToolExecutor(
            {},
            handle_secret=SECRET,
            operating_mode="audit",
        )
    executor = GuardedToolExecutor({}, handle_secret=SECRET)
    with pytest.raises(HostSDKError, match="unregistered tool handlers"):
        executor.prepare(_request())

    async def async_handler(arguments: Mapping[str, Any]) -> None:
        return None

    with pytest.raises(HostSDKError, match="async tool handler"):
        GuardedToolExecutor(
            {"pay_invoice": async_handler},
            handle_secret=SECRET,
        )


def test_exact_manifest_confirmation_is_single_use_before_dispatch() -> None:
    observed: list[dict[str, Any]] = []
    executor = GuardedToolExecutor(
        {"pay_invoice": lambda arguments: observed.append(dict(arguments))},
        handle_secret=SECRET,
        handle_nonce_bytes=_nonce_source(),
    )
    request = _request()
    request["evibind"]["effect_policies"] = {
        "pay_invoice": {
            "effect_class": "external_write",
            "confirmation": "required",
            "ttl_seconds": 300,
        }
    }
    first = executor.prepare(request)
    pending = first.complete(_response(_bindings(first.upstream_payload)))

    assert pending.executed is False
    assert pending.decision == "confirmation_required"
    summary = pending.protected_response["evibind"]["choices"][0]
    token = summary["effect"]["challenge"]["token"]
    assert observed == []

    confirmed_request = deepcopy(request)
    confirmed_request["evibind"]["effect_confirmation"] = token
    confirmed = executor.prepare(confirmed_request)
    released = confirmed.complete(
        _response(_bindings(confirmed.upstream_payload))
    )

    assert released.executed is True
    assert observed == [{"amount": 20}]
    effect = released.protected_response["evibind"]["choices"][0]["effect"]
    assert effect["status"] == "confirmed"

    replay = executor.prepare(confirmed_request)
    with pytest.raises(HostSDKError, match="already consumed"):
        replay.complete(_response(_bindings(replay.upstream_payload)))
    assert observed == [{"amount": 20}]

from __future__ import annotations

import json

import pytest

from tapbench.gateway import (
    EviBindGateway,
    GatewayConfig,
    GatewayError,
    prepare_upstream_payload,
)
from tapbench.one_call_gateway import compile_one_call_session


SECRET = b"operating-mode-test-secret-at-least-32-bytes"


def _request(text: str = "Pay amount=20") -> dict:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": text}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "pay_invoice",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "amount": {
                                "type": "number",
                                "x-evibind-evidence-type": "number",
                                "x-evibind-sources": ["user.current_turn"],
                                "x-evibind-extraction-cue": "amount",
                            }
                        },
                        "required": ["amount"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    }


def _native_response(amount: int) -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "native-call",
                            "type": "function",
                            "function": {
                                "name": "pay_invoice",
                                "arguments": json.dumps({"amount": amount}),
                            },
                        }
                    ],
                },
            }
        ]
    }


def _session(request: dict, operating_mode: str):
    upstream, options, tools = prepare_upstream_payload(request)
    return compile_one_call_session(
        request_payload=request,
        upstream_payload=upstream,
        options=options,
        tools=tools,
        handle_secret=SECRET,
        include_diagnostics=False,
        operating_mode=operating_mode,
    )


def _need_input_response() -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "action-call",
                            "type": "function",
                            "function": {
                                "name": "evibind_action",
                                "arguments": json.dumps(
                                    {
                                        "mode": "need_input",
                                        "tool_id": "pay_invoice",
                                        "missing": ["/amount"],
                                    }
                                ),
                            },
                        }
                    ],
                },
            }
        ]
    }


def test_audit_mode_preserves_native_literals_and_marks_non_enforcement() -> None:
    session = _session(_request(), "audit")

    passed = session.audit(_native_response(20))
    failed = session.audit(_native_response(9_999))

    assert json.loads(
        passed["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    ) == {"amount": 20}
    assert passed["evibind"]["enforced"] is False
    assert passed["evibind"]["selective_guarantee"] is None
    assert passed["evibind"]["choices"][0]["would_release"] is True
    assert failed["evibind"]["choices"][0]["would_release"] is False
    assert failed["evibind"]["choices"][0]["unsupported_destinations"] == ["/amount"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            {"function_call": {"name": "pay_invoice"}},
            "legacy_function_call_not_allowed",
        ),
        ({"tool_calls": [{"type": "custom"}]}, "native_tool_call_type_invalid"),
    ],
)
def test_audit_mode_rejects_noncanonical_native_envelopes(
    mutation: dict,
    reason: str,
) -> None:
    session = _session(_request(), "audit")
    response = _native_response(20)
    response["choices"][0]["message"].update(mutation)

    audited = session.audit(response)

    assert audited["evibind"]["choices"][0]["would_release"] is False
    assert audited["evibind"]["choices"][0]["reason"] == reason


def test_assist_and_enforce_both_withhold_but_render_differently() -> None:
    request = _request("Pay the invoice.")

    enforced = _session(request, "enforce").protect(_need_input_response())
    assisted = _session(request, "assist").protect(_need_input_response())

    enforce_message = enforced["choices"][0]["message"]["content"]
    assist_message = assisted["choices"][0]["message"]["content"]
    assert "withheld" in enforce_message
    assert "/amount" in assist_message
    assert enforced["evibind"]["enforced"] is True
    assert assisted["evibind"]["enforced"] is True


def test_gateway_audit_sends_native_tools_upstream(monkeypatch) -> None:
    captured: list[dict] = []
    gateway = EviBindGateway(
        GatewayConfig(
            upstream_base_url="http://127.0.0.1:8080",
            operating_mode="audit",
            handle_secret=SECRET,
        )
    )

    def fake_upstream(payload: dict) -> dict:
        captured.append(payload)
        return _native_response(20)

    monkeypatch.setattr(gateway, "_upstream_request", fake_upstream)
    response = gateway.chat_completion(_request())

    assert captured[0]["tools"][0]["function"]["name"] == "pay_invoice"
    assert response["evibind"]["operating_mode"] == "audit"
    assert response["evibind"]["choices"][0]["would_release"] is True


def test_operating_mode_loads_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("EVIBIND_UPSTREAM_BASE_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("EVIBIND_OPERATING_MODE", "assist")
    assert GatewayConfig.from_env().operating_mode == "assist"


@pytest.mark.parametrize("mode", ["monitor", "", "ENFORCE"])
def test_unknown_operating_modes_are_rejected(mode: str) -> None:
    with pytest.raises(GatewayError, match="operating mode"):
        GatewayConfig(
            upstream_base_url="http://127.0.0.1:8080",
            operating_mode=mode,
        )


def test_non_enforcing_modes_require_the_one_call_policy_compiler() -> None:
    with pytest.raises(GatewayError, match="controller_mode=one_call"):
        GatewayConfig(
            upstream_base_url="http://127.0.0.1:8080",
            controller_mode="legacy_literal",
            operating_mode="audit",
        )

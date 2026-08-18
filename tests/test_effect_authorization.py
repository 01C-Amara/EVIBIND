from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from tapbench.effect_authorization import (
    EffectAuthorizationError,
    EffectAuthorizer,
    EffectPolicy,
    gate_effect,
    parse_effect_policies,
)
from tapbench.gateway import EviBindGateway, GatewayConfig, GatewayError
from tapbench.nonce_store import InMemoryConsumedNonceStore


SECRET = b"effect-authorization-test-secret-32-bytes"


def _policy(**overrides) -> EffectPolicy:
    values = {
        "effect_class": "external_write",
        "confirmation": "required",
        "ttl_seconds": 60,
    }
    values.update(overrides)
    return EffectPolicy(**values)


def _challenge(
    authorizer: EffectAuthorizer,
    *,
    policy: EffectPolicy | None = None,
):
    return authorizer.issue(
        request_digest="request-1",
        tool_id="pay",
        manifest_digest="manifest-1",
        policy_epoch="policy-1",
        policy=policy or _policy(),
    )


def _consume(
    authorizer: EffectAuthorizer,
    token: str,
    *,
    policy: EffectPolicy | None = None,
    **overrides,
):
    values = {
        "request_digest": "request-1",
        "tool_id": "pay",
        "manifest_digest": "manifest-1",
        "policy_epoch": "policy-1",
        "policy": policy or _policy(),
    }
    values.update(overrides)
    return authorizer.consume(token, **values)


def test_effect_confirmation_is_exactly_bound_and_single_use() -> None:
    authorizer = EffectAuthorizer(
        SECRET,
        now=lambda: 100,
        nonce_bytes=lambda size: b"a" * size,
    )
    challenge = _challenge(authorizer)

    authorization = _consume(authorizer, challenge.token)

    assert authorization.confirmed is True
    assert authorization.manifest_digest == "manifest-1"
    with pytest.raises(EffectAuthorizationError, match="already consumed"):
        _consume(authorizer, challenge.token)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"request_digest": "request-2"}, "request binding"),
        ({"tool_id": "refund"}, "tool binding"),
        ({"manifest_digest": "manifest-2"}, "manifest binding"),
        ({"policy_epoch": "policy-2"}, "policy epoch binding"),
        (
            {
                "policy": _policy(
                    effect_class="irreversible",
                    confirmation="required",
                )
            },
            "effect class binding",
        ),
    ],
)
def test_effect_confirmation_rejects_binding_changes(
    override: dict,
    message: str,
) -> None:
    authorizer = EffectAuthorizer(SECRET, now=lambda: 100)
    token = _challenge(authorizer).token

    with pytest.raises(EffectAuthorizationError, match=message):
        _consume(authorizer, token, **override)


def test_effect_confirmation_rejects_tampering_and_expiry() -> None:
    now = [100]
    authorizer = EffectAuthorizer(
        SECRET,
        now=lambda: now[0],
        nonce_bytes=lambda size: b"b" * size,
    )
    token = _challenge(
        authorizer,
        policy=_policy(ttl_seconds=1),
    ).token

    with pytest.raises(EffectAuthorizationError, match="authentication"):
        _consume(authorizer, token[:-1] + ("0" if token[-1] != "0" else "1"))

    now[0] = 102
    with pytest.raises(EffectAuthorizationError, match="expired"):
        _consume(authorizer, token, policy=_policy(ttl_seconds=1))


def test_effect_policy_defaults_writes_to_confirmation_required() -> None:
    policies = parse_effect_policies(
        {
            "effect_policies": {
                "lookup": {"effect_class": "read_only"},
                "pay": {"effect_class": "external_write"},
            }
        },
        tool_ids=frozenset({"lookup", "pay"}),
    )

    assert policies["lookup"].confirmation == "not_required"
    assert policies["pay"].confirmation == "required"


def _protected() -> dict:
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
                            "type": "function",
                            "function": {
                                "name": "pay",
                                "arguments": '{"amount":20}',
                            },
                        }
                    ],
                },
            }
        ],
        "evibind": {
            "choices": [
                {
                    "released": True,
                    "decision": "call",
                    "tool": "pay",
                    "manifest_digest": "manifest-1",
                }
            ]
        },
    }


def test_effect_gate_withholds_then_releases_the_exact_manifest() -> None:
    authorizer = EffectAuthorizer(SECRET, now=lambda: 100)
    policy = _policy()
    withheld = gate_effect(
        _protected(),
        request_digest="request-1",
        policy_epoch="policy-1",
        policies={"pay": policy},
        confirmation_token=None,
        authorizer=authorizer,
    )
    summary = withheld["evibind"]["choices"][0]
    token = summary["effect"]["challenge"]["token"]

    assert summary["decision"] == "confirmation_required"
    assert summary["released"] is False
    assert "tool_calls" not in withheld["choices"][0]["message"]

    released = gate_effect(
        _protected(),
        request_digest="request-1",
        policy_epoch="policy-1",
        policies={"pay": policy},
        confirmation_token=token,
        authorizer=authorizer,
    )
    assert released["evibind"]["choices"][0]["released"] is True
    assert released["evibind"]["choices"][0]["effect"]["status"] == "confirmed"


def _request() -> dict:
    return {
        "model": "test-model",
        "messages": [
            {
                "id": "user-1",
                "role": "user",
                "content": "Pay amount=20.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "pay",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "amount": {
                                "type": "integer",
                                "x-evibind-evidence-type": "integer",
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
        "evibind": {
            "effect_policies": {
                "pay": {
                    "effect_class": "external_write",
                    "confirmation": "required",
                }
            }
        },
    }


def _action_response(upstream_request: dict) -> dict:
    catalog = json.loads(
        upstream_request["messages"][0]["content"].split(
            "EVIDENCE CANDIDATES:\n",
            1,
        )[1]
    )
    candidate_id = catalog[0]["candidates"][0]["candidate_id"]
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
                            "id": "effect-call",
                            "type": "function",
                            "function": {
                                "name": "evibind_action",
                                "arguments": json.dumps(
                                    {
                                        "mode": "call",
                                        "tool_id": "pay",
                                        "bindings": {
                                            "/amount": candidate_id,
                                        },
                                    }
                                ),
                            },
                        }
                    ],
                },
            }
        ]
    }


def test_gateway_effect_confirmation_round_trip_and_replay_rejection(
    monkeypatch,
) -> None:
    gateway = EviBindGateway(
        GatewayConfig(
            upstream_base_url="http://127.0.0.1:8080",
            handle_secret=SECRET,
        )
    )
    monkeypatch.setattr(gateway, "_upstream_request", _action_response)

    withheld = gateway.chat_completion(_request())
    first = withheld["evibind"]["choices"][0]
    token = first["effect"]["challenge"]["token"]
    assert first["decision"] == "confirmation_required"
    assert "tool_calls" not in withheld["choices"][0]["message"]

    confirmed_request = _request()
    confirmed_request["evibind"]["effect_confirmation"] = token
    released = gateway.chat_completion(confirmed_request)
    call = released["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"amount": 20}
    assert released["evibind"]["choices"][0]["effect"]["status"] == "confirmed"

    with pytest.raises(GatewayError, match="already consumed"):
        gateway.chat_completion(confirmed_request)


def test_gateway_rejects_unknown_effect_policy_before_provider_call(
    monkeypatch,
) -> None:
    called = False
    gateway = EviBindGateway(
        GatewayConfig(
            upstream_base_url="http://127.0.0.1:8080",
            handle_secret=SECRET,
        )
    )

    def unexpected_call(payload: dict) -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(gateway, "_upstream_request", unexpected_call)
    request = _request()
    request["evibind"]["effect_policies"]["refund"] = {"effect_class": "external_write"}

    with pytest.raises(GatewayError, match="unknown tools"):
        gateway.chat_completion(request)
    assert called is False


def test_confirmation_token_cannot_be_silently_applied_to_another_policy() -> None:
    authorizer = EffectAuthorizer(SECRET, now=lambda: 100)
    policy = _policy()
    token = _challenge(authorizer, policy=policy).token
    another_tool = _protected()
    another_tool["evibind"]["choices"][0]["tool"] = "refund"

    with pytest.raises(EffectAuthorizationError, match="released tool"):
        gate_effect(
            another_tool,
            request_digest="request-1",
            policy_epoch="policy-1",
            policies={"pay": policy},
            confirmation_token=token,
            authorizer=authorizer,
        )

    assert _consume(authorizer, token, policy=policy).confirmed is True


def test_confirmation_token_is_rejected_for_not_required_policy() -> None:
    authorizer = EffectAuthorizer(SECRET, now=lambda: 100)
    policy = EffectPolicy(
        effect_class="read_only",
        confirmation="not_required",
    )

    with pytest.raises(EffectAuthorizationError, match="not_required"):
        gate_effect(
            _protected(),
            request_digest="request-1",
            policy_epoch="policy-1",
            policies={"pay": policy},
            confirmation_token="unexpected-token",
            authorizer=authorizer,
        )


def test_consumed_nonce_entries_are_pruned_after_expiry() -> None:
    now = [100]
    nonces = iter((b"a" * 18, b"b" * 18))
    nonce_store = InMemoryConsumedNonceStore()
    authorizer = EffectAuthorizer(
        SECRET,
        now=lambda: now[0],
        nonce_bytes=lambda size: next(nonces),
        nonce_store=nonce_store,
    )
    short = _policy(ttl_seconds=1)
    _consume(authorizer, _challenge(authorizer, policy=short).token, policy=short)
    assert nonce_store.entry_count == 1

    now[0] = 102
    current = _policy(ttl_seconds=60)
    _consume(
        authorizer,
        _challenge(authorizer, policy=current).token,
        policy=current,
    )
    assert nonce_store.entry_count == 1


def test_in_memory_nonce_store_has_one_linearization_winner() -> None:
    nonce_store = InMemoryConsumedNonceStore()

    def consume(_: int) -> bool:
        return nonce_store.consume_once(
            nonce="shared-nonce",
            expires_at=200,
            now=100,
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(consume, range(64)))

    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 63
    assert nonce_store.entry_count == 1

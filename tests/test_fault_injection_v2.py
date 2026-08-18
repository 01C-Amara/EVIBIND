from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from tapbench.effect_authorization import (
    EffectAuthorizationError,
    EffectAuthorizer,
    EffectPolicy,
)
from tapbench.gateway import prepare_upstream_payload
from tapbench.one_call_gateway import compile_one_call_session


SECRET = b"fault-injection-v2-secret-at-least-32-bytes"


def _session(
    *,
    evidence_type: str,
    source: str,
    cue: str,
    tool_content: str,
):
    request = {
        "model": "test-model",
        "messages": [
            {
                "id": "user-1",
                "role": "user",
                "content": "Process the latest tool result.",
            },
            {
                "id": "tool-1",
                "role": "tool",
                "content": tool_content,
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "process",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            cue: {
                                "type": "string",
                                "x-evibind-evidence-type": evidence_type,
                                "x-evibind-sources": [source],
                                "x-evibind-extraction-cue": cue,
                            }
                        },
                        "required": [cue],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    }
    upstream, options, tools = prepare_upstream_payload(request)
    return compile_one_call_session(
        request_payload=request,
        upstream_payload=upstream,
        options=options,
        tools=tools,
        handle_secret=SECRET,
        include_diagnostics=False,
    )


def test_misconfigured_policy_cannot_launder_tool_output_into_authority() -> None:
    session = _session(
        evidence_type="account_ref",
        source="tool.untrusted_output",
        cue="account",
        tool_content="account: account-123",
    )

    assert session.candidates.candidates == {}
    assert len(session.candidates.rejections) == 1
    assert "untrusted_origin_for_sensitive_value" in (
        session.candidates.rejections[0].reasons
    )


def test_explicit_policy_can_admit_untrusted_opaque_content_without_authority() -> None:
    session = _session(
        evidence_type="opaque_content",
        source="tool.untrusted_output",
        cue="note",
        tool_content="note: untrusted output",
    )

    # A cue reference yields both readings of its span -- the first token and
    # the token pair -- so "note: untrusted output" admits "untrusted" and
    # "untrusted output". What matters here is that opaque content from an
    # untrusted source is admitted at all, which the preceding test denies for
    # an authority-bearing type.
    values = {c.value for c in session.candidates.candidates.values()}
    assert "untrusted output" in values
    assert values == {"untrusted", "untrusted output"}


def _policy(effect_class: str = "external_write") -> EffectPolicy:
    return EffectPolicy(
        effect_class=effect_class,
        confirmation="required",
        ttl_seconds=60,
    )


def _issue(authorizer: EffectAuthorizer, policy: EffectPolicy) -> str:
    return authorizer.issue(
        request_digest="request-1",
        tool_id="pay",
        manifest_digest="manifest-1",
        policy_epoch="policy-1",
        policy=policy,
    ).token


def _consume(
    authorizer: EffectAuthorizer,
    token: str,
    policy: EffectPolicy,
):
    return authorizer.consume(
        token,
        request_digest="request-1",
        tool_id="pay",
        manifest_digest="manifest-1",
        policy_epoch="policy-1",
        policy=policy,
    )


def test_failed_policy_binding_does_not_burn_a_valid_confirmation() -> None:
    authorizer = EffectAuthorizer(SECRET, now=lambda: 100)
    expected = _policy()
    token = _issue(authorizer, expected)

    with pytest.raises(EffectAuthorizationError, match="effect class binding"):
        _consume(authorizer, token, _policy("irreversible"))

    assert _consume(authorizer, token, expected).confirmed is True


def test_concurrent_confirmation_replay_releases_exactly_once() -> None:
    workers = 8
    barrier = Barrier(workers)
    authorizer = EffectAuthorizer(
        SECRET,
        now=lambda: 100,
        nonce_bytes=lambda size: b"z" * size,
    )
    policy = _policy()
    token = _issue(authorizer, policy)

    def attempt() -> str:
        barrier.wait()
        try:
            _consume(authorizer, token, policy)
        except EffectAuthorizationError as exc:
            return str(exc)
        return "released"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda _: attempt(), range(workers)))

    assert results.count("released") == 1
    assert sum("already consumed" in result for result in results) == workers - 1

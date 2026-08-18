from __future__ import annotations

import json

from tapbench.gateway import prepare_upstream_payload
from tapbench.one_call_gateway import compile_one_call_session
from tapbench.policy_tools import (
    initialize_request_policy,
    inspect_request_policy,
    replay_request_certificate,
)
from tapbench.product_cli import main
from tapbench.schema_lint import lint_tool_schemas


SECRET = b"policy-tools-test-secret-at-least-32-bytes"


def _request() -> dict:
    return {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": "Pay amount=20 and note=quarterly.",
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
                            "amount": {"type": "number"},
                            "note": {"type": "string"},
                        },
                        "required": ["amount"],
                    },
                },
            }
        ],
    }


def _action_response(candidate_id: str) -> dict:
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
                            "id": "call-policy-tools",
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


def test_policy_init_produces_strict_compilable_contract() -> None:
    initialized = initialize_request_policy(
        _request(),
        policy_epoch="2026-07",
    )
    request = initialized.request
    parameters = request["tools"][0]["function"]["parameters"]
    amount = parameters["properties"]["amount"]
    note = parameters["properties"]["note"]

    assert parameters["additionalProperties"] is False
    assert amount["x-evibind-evidence-type"] == "number"
    assert amount["x-evibind-sources"] == ["user.current_turn"]
    assert note["x-evibind-evidence-type"] == "opaque_content"
    assert note["x-evibind-criticality"] == "content"
    assert request["evibind"]["policy_epoch"] == "2026-07"
    assert initialized.report["change_count"] == 14
    assert len(initialized.report["changes"]) == 14
    assert lint_tool_schemas(request)["warning_count"] == 0


def test_inspector_compiles_candidates_without_exposing_server_values() -> None:
    initialized = initialize_request_policy(_request()).request
    report = inspect_request_policy(
        initialized,
        handle_secret=SECRET,
    )

    assert report["metrics"]["valid_candidate_count"] == 2
    assert report["tools"][0]["missing_required"] == []
    candidate = report["tools"][0]["slots"][0]["candidates"][0]
    assert "value" not in candidate
    assert candidate["candidate_id"].startswith("ev_")


def test_certificate_replay_reconstructs_the_materialized_action() -> None:
    request = initialize_request_policy(_request()).request
    request["evibind"]["include_diagnostics"] = True
    upstream, options, tools = prepare_upstream_payload(request)
    session = compile_one_call_session(
        request_payload=request,
        upstream_payload=upstream,
        options=options,
        tools=tools,
        handle_secret=SECRET,
        include_diagnostics=True,
    )
    amount_id = next(
        candidate_id
        for candidate_id, candidate in session.candidates.candidates.items()
        if candidate.witness.destination_scope == "/amount"
    )
    protected = session.protect(_action_response(amount_id))

    replayed = replay_request_certificate(
        request,
        protected,
        handle_secret=SECRET,
    )

    assert replayed["verified"] is True
    assert replayed["tool_id"] == "pay"
    assert replayed["arguments"] == {"amount": 20}


def test_init_cli_refuses_unrequested_overwrite(tmp_path, capsys) -> None:
    source = tmp_path / "request.json"
    target = tmp_path / "initialized.json"
    source.write_text(json.dumps(_request()), encoding="utf-8")
    target.write_text("keep", encoding="utf-8")

    assert (
        main(
            [
                "init",
                "--request",
                str(source),
                "--output",
                str(target),
            ]
        )
        == 2
    )
    assert target.read_text(encoding="utf-8") == "keep"
    assert "refusing to overwrite" in capsys.readouterr().err


def test_replay_cli_requires_the_configured_handle_secret(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    request = tmp_path / "request.json"
    certificate = tmp_path / "certificate.json"
    request.write_text(json.dumps(_request()), encoding="utf-8")
    certificate.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("EVIBIND_HANDLE_SECRET", raising=False)

    assert (
        main(
            [
                "replay",
                "--request",
                str(request),
                "--certificate",
                str(certificate),
            ]
        )
        == 2
    )
    assert "EVIBIND_HANDLE_SECRET is required" in capsys.readouterr().err

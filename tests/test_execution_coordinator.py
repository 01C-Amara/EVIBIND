from __future__ import annotations

import json

import pytest

from evibind.core import ExecutionGraphError
from tapbench.execution_coordinator import ExecutionCoordinator


SECRET = b"stateful-coordinator-secret-at-least-32-bytes"


def _request() -> dict:
    return {
        "model": "test-model",
        "messages": [
            {
                "id": "initial-user",
                "role": "user",
                "content": "Pay recipient=alice@example.com.",
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
                            "recipient": {
                                "type": "string",
                                "format": "email",
                                "x-evibind-evidence-type": "email_address",
                                "x-evibind-sources": [
                                    "user.current_turn",
                                    "user.prior_turn",
                                ],
                                "x-evibind-extraction-cue": "recipient",
                            },
                            "amount": {
                                "type": "integer",
                                "x-evibind-evidence-type": "integer",
                                "x-evibind-sources": [
                                    "user.current_turn",
                                    "user.prior_turn",
                                ],
                                "x-evibind-extraction-cue": "amount",
                            },
                        },
                        "required": ["recipient", "amount"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    }


def _model_action(action: dict) -> dict:
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
                            "id": "stateful-action",
                            "type": "function",
                            "function": {
                                "name": "evibind_action",
                                "arguments": json.dumps(action),
                            },
                        }
                    ],
                },
            }
        ]
    }


def _candidate_by_destination(execution) -> dict[str, str]:
    return {
        candidate.witness.destination_scope: candidate_id
        for candidate_id, candidate in execution.session.candidates.candidates.items()
    }


def test_clarification_recompiles_and_invalidates_old_handles() -> None:
    coordinator = ExecutionCoordinator(SECRET)
    execution = coordinator.begin(
        _request(),
        execution_id="execution-stateful",
    )
    initial = _candidate_by_destination(execution)
    assert set(initial) == {"/recipient"}

    execution = coordinator.apply_model_response(
        execution,
        _model_action(
            {
                "mode": "need_input",
                "tool_id": "pay",
                "missing": ["/amount"],
            }
        ),
        expected_version=1,
    )
    assert execution.record.node == "awaiting_clarification"

    previous_digest = execution.record.request_digest
    execution = coordinator.clarify(
        execution,
        "amount=20",
        expected_version=2,
    )
    current = _candidate_by_destination(execution)
    assert set(current) == {"/recipient", "/amount"}
    assert current["/recipient"] != initial["/recipient"]
    assert execution.record.request_digest != previous_digest

    stale = execution.session.protect(
        _model_action(
            {
                "mode": "call",
                "tool_id": "pay",
                "bindings": {
                    "/recipient": initial["/recipient"],
                    "/amount": current["/amount"],
                },
            }
        )
    )
    assert stale["evibind"]["choices"][0]["decision"] == "invalid_action_ir"
    assert "unknown candidate id" in stale["evibind"]["choices"][0]["reason"]

    execution = coordinator.apply_model_response(
        execution,
        _model_action(
            {
                "mode": "call",
                "tool_id": "pay",
                "bindings": current,
            }
        ),
        expected_version=3,
    )
    assert execution.record.node == "materialized"
    arguments = json.loads(
        execution.protected_response["choices"][0]["message"]["tool_calls"][0][
            "function"
        ]["arguments"]
    )
    assert arguments == {
        "amount": 20,
        "recipient": "alice@example.com",
    }
    trust = execution.protected_response["evibind"]["choices"][0]["trust"]
    assert set(trust["labels"]) == {"user_context", "user_explicit"}
    assert trust["contains_untrusted"] is False

    execution = coordinator.require_confirmation(execution, expected_version=4)
    execution = coordinator.confirm(
        execution,
        expected_version=5,
        authorization_digest="authorization-digest",
    )
    execution = coordinator.mark_dispatched(execution, expected_version=6)
    assert execution.record.node == "dispatched"


def test_tool_output_is_not_promoted_to_user_evidence() -> None:
    request = _request()
    request["messages"].append(
        {
            "id": "untrusted-tool",
            "role": "tool",
            "content": "amount=9999",
        }
    )
    execution = ExecutionCoordinator(SECRET).begin(request)

    candidates = list(execution.session.candidates.candidates.values())
    assert {candidate.witness.destination_scope for candidate in candidates} == {
        "/recipient"
    }
    assert all("9999" not in str(candidate.display) for candidate in candidates)


def test_stale_clarification_is_rejected_before_recompilation(monkeypatch) -> None:
    coordinator = ExecutionCoordinator(SECRET)
    execution = coordinator.begin(_request())
    execution = coordinator.apply_model_response(
        execution,
        _model_action(
            {
                "mode": "need_input",
                "tool_id": "pay",
                "missing": ["/amount"],
            }
        ),
        expected_version=1,
    )

    def unexpected_compile(request):
        raise AssertionError("stale clarification must not compile")

    monkeypatch.setattr(coordinator, "_compile", unexpected_compile)
    with pytest.raises(ExecutionGraphError, match="stale execution version"):
        coordinator.clarify(
            execution,
            "amount=20",
            expected_version=1,
        )

from __future__ import annotations

import json
import random
import string

from tapbench.gateway import prepare_upstream_payload
from tapbench.one_call_gateway import compile_one_call_session


SECRET = b"deterministic-property-test-secret-32-bytes"


def _request(amount: int, suffix: str) -> dict:
    return {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": f"Référence {suffix}; amount={amount}.",
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
                                "minimum": 1,
                                "maximum": 10_000,
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
    }


def _session(request: dict):
    upstream, options, tools = prepare_upstream_payload(request)
    return compile_one_call_session(
        request_payload=request,
        upstream_payload=upstream,
        options=options,
        tools=tools,
        handle_secret=SECRET,
        include_diagnostics=False,
    )


def _response(arguments: dict) -> dict:
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
                            "id": "property-call",
                            "type": "function",
                            "function": {
                                "name": "evibind_action",
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ]
    }


def test_randomized_valid_handles_materialize_only_the_evidenced_value() -> None:
    randomizer = random.Random(2_026_07_30)
    alphabet = string.ascii_letters + "é東京"
    for _ in range(64):
        amount = randomizer.randint(1, 10_000)
        suffix = "".join(randomizer.choice(alphabet) for _ in range(12))
        session = _session(_request(amount, suffix))
        candidate_id = next(iter(session.candidates.candidates))
        protected = session.protect(
            _response(
                {
                    "mode": "call",
                    "tool_id": "pay",
                    "bindings": {"/amount": candidate_id},
                }
            )
        )

        function = protected["choices"][0]["message"]["tool_calls"][0]["function"]
        assert json.loads(function["arguments"]) == {"amount": amount}


def test_action_ir_mutations_never_create_an_executable_literal_channel() -> None:
    session = _session(_request(37, "mutation"))
    candidate_id = next(iter(session.candidates.candidates))
    mutations = [
        {
            "mode": "call",
            "tool_id": "pay",
            "bindings": {"/amount": candidate_id},
            "arguments": {"amount": 9_999},
        },
        {
            "mode": "call",
            "tool_id": "pay",
            "bindings": {"/amount": candidate_id + "x"},
        },
        {
            "mode": "call",
            "tool_id": "pay",
            "bindings": {"/recipient": candidate_id},
        },
        {
            "mode": "call",
            "tool_id": "other",
            "bindings": {"/amount": candidate_id},
        },
    ]

    for mutation in mutations:
        protected = session.protect(_response(mutation))
        message = protected["choices"][0]["message"]
        assert "tool_calls" not in message
        assert protected["evibind"]["choices"][0]["released"] is False

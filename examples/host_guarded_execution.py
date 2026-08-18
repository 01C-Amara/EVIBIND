from __future__ import annotations

import json
import secrets
from typing import Any, Mapping

from evibind.host import GuardedToolExecutor


def _fixture_model(payload: Mapping[str, Any]) -> dict[str, Any]:
    branch = next(
        item
        for item in payload["tools"][0]["function"]["parameters"]["oneOf"]
        if item["properties"]["mode"].get("const") == "call"
    )
    candidate_id = branch["properties"]["bindings"]["properties"]["/amount"][
        "enum"
    ][0]
    action = {
        "mode": "call",
        "tool_id": "pay_invoice",
        "bindings": {"/amount": candidate_id},
    }
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
                            "id": "demo-action",
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


def main() -> None:
    request = {
        "model": "replace-with-your-model",
        "messages": [{"role": "user", "content": "Pay amount=20"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "pay_invoice",
                    "description": "Pay one invoice.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "amount": {
                                "type": "number",
                                "x-evibind-evidence-type": "number",
                                "x-evibind-sources": ["user.current_turn"],
                                "x-evibind-extraction-cue": "amount",
                                "x-evibind-criticality": "control",
                            }
                        },
                        "required": ["amount"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "evibind": {"policy_epoch": "demo-v1"},
    }
    executed: list[dict[str, Any]] = []
    executor = GuardedToolExecutor(
        {
            "pay_invoice": lambda arguments: executed.append(
                dict(arguments)
            )
        },
        handle_secret=secrets.token_bytes(32),
    )
    turn = executor.prepare(request)

    # Replace this deterministic fixture with one provider call using
    # turn.upstream_payload.
    response = _fixture_model(turn.upstream_payload)
    result = turn.complete(response)
    print(json.dumps(result.to_dict(), indent=2, default=str, sort_keys=True))
    assert result.executed is True
    assert executed == [{"amount": 20}]


if __name__ == "__main__":
    main()

from __future__ import annotations

import json

import pytest

from evibind.adapters import (
    ProviderAdapterError,
    action_response_to_openai_chat,
    decode_tool_calls,
    encode_action_tool,
)


ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "evibind_action",
        "description": "Select evidence handles.",
        "parameters": {
            "type": "object",
            "properties": {"mode": {"const": "no_tool"}},
            "required": ["mode"],
            "additionalProperties": False,
        },
    },
}


@pytest.mark.parametrize(
    ("provider", "response"),
    [
        (
            "openai_chat",
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "oa-chat",
                                    "function": {
                                        "name": "evibind_action",
                                        "arguments": '{"mode":"no_tool"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
        ),
        (
            "openai_responses",
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "oa-response",
                        "name": "evibind_action",
                        "arguments": '{"mode":"no_tool"}',
                    }
                ]
            },
        ),
        (
            "anthropic_messages",
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "anthropic",
                        "name": "evibind_action",
                        "input": {"mode": "no_tool"},
                    }
                ]
            },
        ),
        (
            "google_interactions",
            {
                "steps": [
                    {
                        "type": "function_call",
                        "id": "google-interaction",
                        "name": "evibind_action",
                        "arguments": {"mode": "no_tool"},
                    }
                ]
            },
        ),
        (
            "google_generate_content",
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "id": "google-generate",
                                        "name": "evibind_action",
                                        "args": {"mode": "no_tool"},
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        ),
    ],
)
def test_provider_responses_decode_to_one_canonical_action_call(
    provider: str,
    response: dict,
) -> None:
    calls = decode_tool_calls(response, provider=provider)

    assert len(calls) == 1
    assert calls[0].tool_id == "evibind_action"
    assert calls[0].arguments == {"mode": "no_tool"}
    adapted = action_response_to_openai_chat(
        response,
        provider=provider,
    )
    function = adapted["choices"][0]["message"]["tool_calls"][0]["function"]
    assert function["name"] == "evibind_action"
    assert json.loads(function["arguments"]) == {"mode": "no_tool"}


def test_action_tool_encodes_to_each_provider_envelope() -> None:
    chat = encode_action_tool(ACTION_TOOL, provider="openai_chat")
    responses = encode_action_tool(
        ACTION_TOOL,
        provider="openai_responses",
    )
    anthropic = encode_action_tool(
        ACTION_TOOL,
        provider="anthropic_messages",
    )
    interactions = encode_action_tool(
        ACTION_TOOL,
        provider="google_interactions",
    )
    generate = encode_action_tool(
        ACTION_TOOL,
        provider="google_generate_content",
    )

    assert chat["function"]["name"] == "evibind_action"
    assert responses["name"] == "evibind_action"
    assert responses["type"] == "function"
    assert anthropic["input_schema"]["type"] == "object"
    assert interactions["parameters"]["type"] == "object"
    assert generate["functionDeclarations"][0]["name"] == "evibind_action"


def test_adapter_rejects_unknown_providers_and_non_object_arguments() -> None:
    with pytest.raises(ProviderAdapterError, match="unsupported provider"):
        encode_action_tool(ACTION_TOOL, provider="mystery")
    with pytest.raises(ProviderAdapterError, match="must be an object"):
        decode_tool_calls(
            {
                "output": [
                    {
                        "type": "function_call",
                        "name": "evibind_action",
                        "arguments": "[]",
                    }
                ]
            },
            provider="openai_responses",
        )


@pytest.mark.parametrize(
    ("provider", "response", "message"),
    [
        (
            "openai_chat",
            {"choices": [{"message": {}}, {"message": {}}]},
            "exactly one choice",
        ),
        (
            "google_generate_content",
            {"candidates": [{"content": {}}, {"content": {}}]},
            "exactly one candidate",
        ),
    ],
)
def test_adapter_rejects_ambiguous_provider_choices(
    provider: str,
    response: dict,
    message: str,
) -> None:
    with pytest.raises(ProviderAdapterError, match=message):
        decode_tool_calls(response, provider=provider)

from __future__ import annotations

import json

import pytest

from tapbench.gateway import (
    GatewayConfig,
    GatewayError,
    UpstreamError,
    prepare_upstream_payload,
    protect_chat_completion,
)


def _config() -> GatewayConfig:
    return GatewayConfig(upstream_base_url="http://127.0.0.1:8080")


def _tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    }


def _tool_response() -> dict:
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
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": json.dumps({"id": "A-12"}),
                            },
                        }
                    ],
                },
            }
        ]
    }


def test_wholly_malformed_tool_catalog_is_rejected_before_forwarding() -> None:
    request = {
        "model": "test",
        "messages": [{"role": "user", "content": "Lookup A-12"}],
        "tools": [{"type": "function", "function": {}}],
    }
    with pytest.raises(GatewayError, match="valid function schema"):
        prepare_upstream_payload(request)


def test_upstream_tool_call_without_requested_schema_is_removed() -> None:
    request = {
        "model": "test",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    protected = protect_chat_completion(
        request,
        _tool_response(),
        config=_config(),
    )
    message = protected["choices"][0]["message"]
    assert "tool_calls" not in message
    assert protected["evibind"]["choices"][0]["released"] is False
    assert (
        protected["evibind"]["choices"][0]["reason"]
        == "tool_call_without_usable_schema"
    )


def test_legacy_function_call_is_removed_instead_of_passed_through() -> None:
    request = {
        "model": "test",
        "messages": [{"role": "user", "content": "Lookup A-12"}],
        "tools": [_tool()],
    }
    response = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "function_call",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "lookup",
                        "arguments": '{"id":"A-12"}',
                    },
                },
            }
        ]
    }
    protected = protect_chat_completion(request, response, config=_config())
    message = protected["choices"][0]["message"]
    assert "function_call" not in message
    assert protected["evibind"]["choices"][0]["released"] is False
    assert (
        protected["evibind"]["choices"][0]["reason"]
        == "legacy_function_call_not_supported"
    )


def test_partially_malformed_tool_catalog_is_rejected() -> None:
    request = {
        "model": "test",
        "messages": [{"role": "user", "content": "Lookup A-12"}],
        "tools": [_tool(), {"type": "function", "function": {}}],
    }
    with pytest.raises(GatewayError, match="only valid function schemas"):
        prepare_upstream_payload(request)


def test_malformed_messages_are_rejected_before_forwarding() -> None:
    request = {
        "model": "test",
        "messages": "Lookup A-12",
        "tools": [_tool()],
    }
    with pytest.raises(GatewayError, match="messages must be a list"):
        prepare_upstream_payload(request)


def test_legacy_function_call_is_removed_when_tool_calls_are_also_present() -> None:
    request = {
        "model": "test",
        "messages": [{"role": "user", "content": "Lookup A-12"}],
        "tools": [_tool()],
    }
    response = _tool_response()
    response["choices"][0]["message"]["function_call"] = {
        "name": "unvalidated_legacy_call",
        "arguments": "{}",
    }

    protected = protect_chat_completion(request, response, config=_config())

    message = protected["choices"][0]["message"]
    assert "function_call" not in message
    assert protected["evibind"]["choices"][0]["released"] is True


def test_malformed_tool_calls_are_removed_instead_of_passed_through() -> None:
    request = {
        "model": "test",
        "messages": [{"role": "user", "content": "Lookup A-12"}],
        "tools": [_tool()],
    }
    response = _tool_response()
    response["choices"][0]["message"]["tool_calls"] = {
        "function": {"name": "lookup", "arguments": '{"id":"A-12"}'}
    }

    protected = protect_chat_completion(request, response, config=_config())

    message = protected["choices"][0]["message"]
    assert "tool_calls" not in message
    assert protected["evibind"]["choices"][0]["released"] is False
    assert protected["evibind"]["choices"][0]["reason"] == "malformed_tool_calls"


def test_released_call_is_rebuilt_from_validated_fields_only() -> None:
    request = {
        "model": "test",
        "messages": [{"role": "user", "content": "Lookup A-12"}],
        "tools": [_tool()],
    }
    response = _tool_response()
    proposed = response["choices"][0]["message"]["tool_calls"][0]
    proposed["type"] = "untrusted-type"
    proposed["extra"] = "untrusted"
    proposed["function"]["extra"] = "untrusted"

    protected = protect_chat_completion(request, response, config=_config())

    released = protected["choices"][0]["message"]["tool_calls"][0]
    assert released == {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "lookup",
            "arguments": '{"id":"A-12"}',
        },
    }


@pytest.mark.parametrize("completion_count", [True, 0, 2, "1"])
def test_multiple_completion_choices_are_rejected_before_forwarding(
    completion_count: object,
) -> None:
    request = {
        "model": "test",
        "messages": [{"role": "user", "content": "Lookup A-12"}],
        "tools": [_tool()],
        "n": completion_count,
    }
    with pytest.raises(GatewayError, match="n must be 1"):
        prepare_upstream_payload(request)


def test_multiple_upstream_choices_fail_closed() -> None:
    request = {
        "model": "test",
        "messages": [{"role": "user", "content": "Lookup A-12"}],
        "tools": [_tool()],
    }
    response = _tool_response()
    response["choices"].append(response["choices"][0])

    with pytest.raises(UpstreamError, match="exactly one choice"):
        protect_chat_completion(request, response, config=_config())


def test_duplicate_tool_names_are_rejected_before_forwarding() -> None:
    request = {
        "model": "test",
        "messages": [{"role": "user", "content": "Lookup A-12"}],
        "tools": [_tool(), _tool()],
    }

    with pytest.raises(GatewayError, match="function names must be unique"):
        prepare_upstream_payload(request)

from __future__ import annotations

import json

import pytest

from evibind import strip_private_annotations

from tapbench.gateway import (
    GatewayConfig,
    GatewayError,
    normalize_openai_tools,
    prepare_upstream_payload,
    protect_chat_completion,
)


def _config() -> GatewayConfig:
    return GatewayConfig(upstream_base_url="http://127.0.0.1:8080")


def _request(user_text: str) -> dict:
    return {
        "model": "small-tool-model",
        "messages": [{"role": "user", "content": user_text}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "create_calendar_event",
                    "description": "Create a calendar event.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "x-evibind-slot-role": "control",
                                "x-evibind-resolution-type": "normalizable",
                            }
                        },
                        "required": ["date"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "evibind": {"evidence_mode": "typed_program_hybrid"},
    }


def _response(date: str) -> dict:
    return {
        "id": "chatcmpl-test",
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
                                "name": "create_calendar_event",
                                "arguments": json.dumps({"date": date}),
                            },
                        }
                    ],
                },
            }
        ],
    }


def test_prepare_payload_preserves_private_contract_only_at_gateway() -> None:
    request = _request("Create an event on 2026-07-12.")
    upstream, options, tools = prepare_upstream_payload(request)
    assert "evibind" not in upstream
    assert upstream["parallel_tool_calls"] is False
    assert upstream["n"] == 1
    upstream_date = upstream["tools"][0]["function"]["parameters"]["properties"]["date"]
    runtime_date = tools[0]["parameters"]["properties"]["date"]
    assert "x-evibind-slot-role" not in upstream_date
    assert runtime_date["x-tap-slot-role"] == "control"
    assert options["evidence_mode"] == "typed_program_hybrid"


def test_strip_private_annotations_is_recursive_and_non_mutating() -> None:
    schema = {
        "type": "object",
        "x-evibind-slot-role": "payload",
        "properties": {
            "id": {
                "type": "string",
                "x-tap-resolution-type": "verbatim",
            }
        },
    }
    stripped = strip_private_annotations(schema)
    assert stripped == {
        "type": "object",
        "properties": {"id": {"type": "string"}},
    }
    assert schema["x-evibind-slot-role"] == "payload"
    assert schema["properties"]["id"]["x-tap-resolution-type"] == "verbatim"


def test_gateway_replaces_unsupported_literal_with_certified_value() -> None:
    protected = protect_chat_completion(
        _request("Create an event on 2026-07-12."),
        _response("2026-07-13"),
        config=_config(),
    )
    function = protected["choices"][0]["message"]["tool_calls"][0]["function"]
    assert json.loads(function["arguments"]) == {"date": "2026-07-12"}
    assert protected["evibind"]["choices"][0]["released"] is True
    assert protected["evibind"]["choices"][0]["decision"] == "call"


def test_gateway_accepts_directly_supported_generic_argument() -> None:
    request = {
        "model": "small-tool-model",
        "messages": [
            {
                "role": "user",
                "content": "Calculate the factorial of 5.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "math.factorial",
                    "parameters": {
                        "type": "object",
                        "properties": {"n": {"type": "integer"}},
                        "required": ["n"],
                    },
                },
            }
        ],
    }
    response = _response("2026-07-13")
    response["choices"][0]["message"]["tool_calls"][0]["function"] = {
        "name": "math.factorial",
        "arguments": '{"n":5}',
    }
    protected = protect_chat_completion(request, response, config=_config())
    function = protected["choices"][0]["message"]["tool_calls"][0]["function"]
    assert json.loads(function["arguments"]) == {"n": 5}
    assert protected["evibind"]["choices"][0]["released"] is True


def test_gateway_blocks_multiple_tool_calls() -> None:
    response = _response("2026-07-12")
    response["choices"][0]["message"]["tool_calls"].append(
        {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "create_calendar_event",
                "arguments": '{"date":"2026-07-13"}',
            },
        }
    )
    protected = protect_chat_completion(
        _request("Create an event on 2026-07-12."),
        response,
        config=_config(),
    )
    message = protected["choices"][0]["message"]
    assert "tool_calls" not in message
    assert protected["evibind"]["choices"][0]["released"] is False
    assert (
        protected["evibind"]["choices"][0]["reason"]
        == "multiple_tool_calls_not_supported"
    )


def test_gateway_passes_through_non_tool_response() -> None:
    response = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hello."},
            }
        ]
    }
    protected = protect_chat_completion(
        _request("Say hello."),
        response,
        config=_config(),
    )
    assert protected["choices"][0]["message"]["content"] == "Hello."
    assert protected["evibind"]["choices"][0]["decision"] == "pass_through"


def test_streaming_is_rejected_explicitly() -> None:
    request = _request("Create an event on 2026-07-12.")
    request["stream"] = True
    with pytest.raises(GatewayError, match="streaming"):
        prepare_upstream_payload(request)


def test_normalize_openai_tools_accepts_wrapped_and_flat_schemas() -> None:
    wrapped = _request("test")["tools"][0]
    flat = {
        "name": "lookup",
        "parameters": {"type": "object", "properties": {}},
    }
    assert [tool["name"] for tool in normalize_openai_tools([wrapped, flat])] == [
        "create_calendar_event",
        "lookup",
    ]


def test_config_rejects_unsafe_numeric_ranges() -> None:
    with pytest.raises(GatewayError, match="candidate_budget"):
        GatewayConfig(
            upstream_base_url="http://127.0.0.1:8080",
            candidate_budget=99,
        )


@pytest.mark.parametrize("candidate_seed", [True, "not-an-integer", -1])
def test_gateway_rejects_invalid_candidate_seed(candidate_seed: object) -> None:
    request = _request("Create an event on 2026-07-12.")
    request["evibind"]["candidate_seed"] = candidate_seed
    with pytest.raises(GatewayError, match="candidate_seed"):
        protect_chat_completion(request, _response("2026-07-12"), config=_config())


def test_diagnostics_require_server_opt_in() -> None:
    request = _request("Create an event on 2026-07-12.")
    request["evibind"]["include_diagnostics"] = True
    with pytest.raises(GatewayError, match="diagnostics are disabled"):
        protect_chat_completion(request, _response("2026-07-12"), config=_config())

    protected = protect_chat_completion(
        request,
        _response("2026-07-12"),
        config=GatewayConfig(
            upstream_base_url="http://127.0.0.1:8080",
            allow_diagnostics=True,
        ),
    )
    assert "diagnostics" in protected["evibind"]["choices"][0]


def test_diagnostics_request_must_be_json_boolean() -> None:
    request = _request("Create an event on 2026-07-12.")
    request["evibind"]["include_diagnostics"] = "false"
    with pytest.raises(GatewayError, match="must be a boolean"):
        protect_chat_completion(request, _response("2026-07-12"), config=_config())


def test_diagnostics_environment_flag_is_strict(monkeypatch) -> None:
    monkeypatch.setenv("EVIBIND_UPSTREAM_BASE_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("EVIBIND_ALLOW_DIAGNOSTICS", "true")
    assert GatewayConfig.from_env().allow_diagnostics is True
    monkeypatch.setenv("EVIBIND_ALLOW_DIAGNOSTICS", "sometimes")
    with pytest.raises(GatewayError, match="must be a boolean"):
        GatewayConfig.from_env()


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@provider.example/v1",
        "https://provider.example/v1?api_key=secret",
        "https://provider.example/v1#secret",
    ],
)
def test_config_rejects_secrets_embedded_in_upstream_url(url: str) -> None:
    with pytest.raises(GatewayError, match="credentials|query or fragment"):
        GatewayConfig(upstream_base_url=url)

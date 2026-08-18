from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping


PROVIDER_ADAPTER_VERSION = "evibind.provider_adapters.v1"
PROVIDERS = {
    "openai_chat",
    "openai_responses",
    "anthropic_messages",
    "google_interactions",
    "google_generate_content",
}


class ProviderAdapterError(ValueError):
    pass


@dataclass(frozen=True)
class CanonicalToolCall:
    call_id: str | None
    tool_id: str
    arguments: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_id": self.tool_id,
            "arguments": deepcopy(dict(self.arguments)),
        }


def _function(action_tool: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(action_tool, Mapping):
        raise ProviderAdapterError("function tool must be an object")
    function = (
        action_tool.get("function")
        if action_tool.get("type") == "function"
        else action_tool
    )
    if not isinstance(function, Mapping):
        raise ProviderAdapterError("function tool must be an object")
    if not isinstance(function.get("name"), str):
        raise ProviderAdapterError("function tool requires a name")
    parameters = function.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ProviderAdapterError("function tool requires parameters")
    return function


def encode_action_tool(
    action_tool: Mapping[str, Any],
    *,
    provider: str,
) -> dict[str, Any]:
    """Encode one canonical EviBind action tool for a provider envelope."""
    if provider not in PROVIDERS:
        raise ProviderAdapterError(f"unsupported provider: {provider}")
    function = _function(action_tool)
    flat = {
        "name": function["name"],
        "description": str(function.get("description", "")),
        "parameters": deepcopy(dict(function["parameters"])),
    }
    if provider == "openai_chat":
        return {"type": "function", "function": flat}
    if provider in {"openai_responses", "google_interactions"}:
        return {"type": "function", **flat}
    if provider == "anthropic_messages":
        return {
            "name": flat["name"],
            "description": flat["description"],
            "input_schema": flat["parameters"],
        }
    return {
        "functionDeclarations": [
            {
                "name": flat["name"],
                "description": flat["description"],
                "parameters": flat["parameters"],
            }
        ]
    }


def _arguments(value: Any) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ProviderAdapterError("tool arguments are invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ProviderAdapterError("tool arguments must be an object")
    return deepcopy(dict(value))


def _openai_chat_calls(response: Mapping[str, Any]) -> list[CanonicalToolCall]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    if len(choices) != 1:
        raise ProviderAdapterError("OpenAI response must contain exactly one choice")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    calls = message.get("tool_calls") if isinstance(message, Mapping) else None
    if not isinstance(calls, list):
        return []
    output = []
    for call in calls:
        function = call.get("function") if isinstance(call, Mapping) else None
        if not isinstance(function, Mapping):
            raise ProviderAdapterError("OpenAI tool call omitted function")
        name = function.get("name")
        if not isinstance(name, str):
            raise ProviderAdapterError("OpenAI tool call omitted name")
        call_id = call.get("id")
        output.append(
            CanonicalToolCall(
                call_id if isinstance(call_id, str) else None,
                name,
                _arguments(function.get("arguments")),
            )
        )
    return output


def _openai_response_calls(
    response: Mapping[str, Any],
) -> list[CanonicalToolCall]:
    output = response.get("output")
    if not isinstance(output, list):
        return []
    calls = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "function_call":
            continue
        name = item.get("name")
        if not isinstance(name, str):
            raise ProviderAdapterError("Responses function call omitted name")
        call_id = item.get("call_id", item.get("id"))
        calls.append(
            CanonicalToolCall(
                call_id if isinstance(call_id, str) else None,
                name,
                _arguments(item.get("arguments")),
            )
        )
    return calls


def _anthropic_calls(response: Mapping[str, Any]) -> list[CanonicalToolCall]:
    content = response.get("content")
    if not isinstance(content, list):
        return []
    calls = []
    for item in content:
        if not isinstance(item, Mapping) or item.get("type") != "tool_use":
            continue
        name = item.get("name")
        if not isinstance(name, str):
            raise ProviderAdapterError("Anthropic tool use omitted name")
        call_id = item.get("id")
        calls.append(
            CanonicalToolCall(
                call_id if isinstance(call_id, str) else None,
                name,
                _arguments(item.get("input")),
            )
        )
    return calls


def _google_interaction_calls(
    response: Mapping[str, Any],
) -> list[CanonicalToolCall]:
    steps = response.get("steps")
    if not isinstance(steps, list):
        return []
    calls = []
    for item in steps:
        if not isinstance(item, Mapping) or item.get("type") != "function_call":
            continue
        name = item.get("name")
        if not isinstance(name, str):
            raise ProviderAdapterError("Google function call omitted name")
        call_id = item.get("id")
        calls.append(
            CanonicalToolCall(
                call_id if isinstance(call_id, str) else None,
                name,
                _arguments(item.get("arguments")),
            )
        )
    return calls


def _google_generate_calls(
    response: Mapping[str, Any],
) -> list[CanonicalToolCall]:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return []
    if len(candidates) != 1:
        raise ProviderAdapterError("Google response must contain exactly one candidate")
    content = (
        candidates[0].get("content") if isinstance(candidates[0], Mapping) else None
    )
    parts = content.get("parts") if isinstance(content, Mapping) else None
    if not isinstance(parts, list):
        return []
    calls = []
    for part in parts:
        function = part.get("functionCall") if isinstance(part, Mapping) else None
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            raise ProviderAdapterError("Google functionCall omitted name")
        call_id = function.get("id")
        calls.append(
            CanonicalToolCall(
                call_id if isinstance(call_id, str) else None,
                name,
                _arguments(function.get("args")),
            )
        )
    return calls


def decode_tool_calls(
    response: Mapping[str, Any],
    *,
    provider: str,
) -> tuple[CanonicalToolCall, ...]:
    if not isinstance(response, Mapping):
        raise ProviderAdapterError("provider response must be an object")
    if provider not in PROVIDERS:
        raise ProviderAdapterError(f"unsupported provider: {provider}")
    decoder = {
        "openai_chat": _openai_chat_calls,
        "openai_responses": _openai_response_calls,
        "anthropic_messages": _anthropic_calls,
        "google_interactions": _google_interaction_calls,
        "google_generate_content": _google_generate_calls,
    }[provider]
    return tuple(decoder(response))


def action_response_to_openai_chat(
    response: Mapping[str, Any],
    *,
    provider: str,
) -> dict[str, Any]:
    """Normalize provider action calls for the canonical materializer."""
    calls = decode_tool_calls(response, provider=provider)
    tool_calls = [
        {
            "id": call.call_id or f"evibind-call-{index}",
            "type": "function",
            "function": {
                "name": call.tool_id,
                "arguments": json.dumps(
                    call.arguments,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            },
        }
        for index, call in enumerate(calls)
    ]
    return {
        "id": str(response.get("id", "evibind-adapted-response")),
        "choices": [
            {
                "index": 0,
                "finish_reason": ("tool_calls" if tool_calls else "stop"),
                "message": {
                    "role": "assistant",
                    "content": None if tool_calls else "",
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                },
            }
        ],
        "evibind_adapter": {
            "version": PROVIDER_ADAPTER_VERSION,
            "provider": provider,
        },
    }

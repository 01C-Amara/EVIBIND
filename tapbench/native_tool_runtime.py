from __future__ import annotations

import json
import time
import urllib.request
from copy import deepcopy
from typing import Any

from .r2_model_runner import _request_llama_json


NATIVE_TOOL_RUNTIME_VERSION = "tapbench.native_tool_runtime.v1"
INJECTED_ACTION_IR_SYSTEM_PREFIX = "Return exactly one Action IR JSON object."


def _public_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _public_schema(item)
            for key, item in value.items()
            if not str(key).startswith("x-tap-")
        }
    if isinstance(value, list):
        return [_public_schema(item) for item in value]
    return deepcopy(value)


def openai_native_tools(case: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for tool in case.get("tools", []):
        name = tool.get("name") or tool.get("canonical_name")
        if not isinstance(name, str) or not name:
            continue
        output.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description", "")),
                    "parameters": _public_schema(
                        tool.get(
                            "parameters",
                            {"type": "object", "properties": {}},
                        )
                    ),
                },
            }
        )
    return output


def native_messages(
    case: dict[str, Any],
    *,
    model_profile: str,
) -> list[dict[str, str]]:
    messages = []
    for message in case.get("messages", []):
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        if (
            role == "system"
            and content.startswith(INJECTED_ACTION_IR_SYSTEM_PREFIX)
        ):
            continue
        messages.append({"role": role, "content": content})
    policy = (
        "Use a provided function only when it directly satisfies the user's "
        "request. Do not call any function when none is relevant. Emit at "
        "most one function call."
    )
    role = "developer" if model_profile == "gpt_oss" else "system"
    if model_profile == "gpt_oss":
        policy = "Reasoning: medium.\n\n" + policy
    return [{"role": role, "content": policy}, *messages]


def normalize_native_message(
    message: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    tool_calls = message.get("tool_calls")
    calls = tool_calls if isinstance(tool_calls, list) else []
    diagnostics: dict[str, Any] = {
        "native_tool_call_count": len(calls),
        "native_content": message.get("content"),
        "native_reasoning_content": message.get("reasoning_content"),
    }
    if not calls:
        return {
            "mode": "no_tool",
            "tool": None,
            "arguments": {},
            "payload": {},
        }, diagnostics

    function = calls[0].get("function", {})
    name = function.get("name")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            diagnostics["native_argument_parse_error"] = str(exc)
            arguments = {}
    if not isinstance(arguments, dict):
        diagnostics["native_argument_parse_error"] = "arguments_not_object"
        arguments = {}
    return {
        "mode": "call",
        "tool": str(name) if name is not None else None,
        "arguments": arguments,
        "payload": {},
    }, diagnostics


def request_native_tool(
    endpoint: str,
    case: dict[str, Any],
    *,
    model_profile: str,
    max_tokens: int,
    seed: int,
    reasoning_budget: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "messages": native_messages(case, model_profile=model_profile),
        "tools": openai_native_tools(case),
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "temperature": 0.0,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
        "thinking_budget_tokens": reasoning_budget,
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    data, retries = _request_llama_json(request, endpoint)
    elapsed = time.perf_counter() - started
    choices = data.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("native chat response omitted choices")
    choice = choices[0]
    message = choice.get("message", {})
    if not isinstance(message, dict):
        raise RuntimeError("native chat response omitted assistant message")
    action, diagnostics = normalize_native_message(message)
    usage = data.get("usage", {}) if isinstance(data.get("usage"), dict) else {}
    details = (
        usage.get("completion_tokens_details", {})
        if isinstance(usage.get("completion_tokens_details"), dict)
        else {}
    )
    content = message.get("content")
    metadata = {
        "native_tool_runtime_version": NATIVE_TOOL_RUNTIME_VERSION,
        "raw_text": content if isinstance(content, str) else "",
        "reasoning_content": message.get("reasoning_content"),
        "finish_reason": choice.get("finish_reason"),
        "retry_count": retries,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "generation_ms": elapsed * 1000.0,
        "generated_tokens_per_second": (
            float(usage.get("completion_tokens")) / elapsed
            if elapsed > 0 and usage.get("completion_tokens") is not None
            else None
        ),
        "context_truncated": False,
        "generation_calls": 1,
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_budget": reasoning_budget,
        "request_tool_count": len(payload["tools"]),
        **diagnostics,
    }
    return action, metadata

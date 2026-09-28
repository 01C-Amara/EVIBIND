"""AgentDojo pipeline element for models that only take tools on /v1/responses.

AgentDojo 0.1.35's ``OpenAILLM`` calls ``/v1/chat/completions``. GPT-5.6 and
GPT-6 reject function tools there when reasoning is on:

    Function tools with reasoning_effort are not supported for gpt-6-luna in
    /v1/chat/completions. To use function tools, use /v1/responses ...

Turning reasoning off to stay on chat completions would measure a different
model from the one the InjectBench and routing rows use. This element keeps
the model as shipped and translates AgentDojo's messages to Responses items
and back, following ``bench/adapters.py``'s conventions, which already carry
GPT-5.6 and GPT-6 through InjectBench.

Nothing else in the pipeline changes: the guard, the tool executor, the attack
and AgentDojo's scoring all see the same ``ChatAssistantMessage`` shape that
``OpenAILLM`` produces.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    text_content_block_from_string,
)


def _text(message: ChatMessage) -> str:
    blocks = message.get("content") or []
    return "".join(block.get("content") or "" for block in blocks)


def _to_items(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            output = message.get("error") or _text(message)
            items.append({"type": "function_call_output",
                          "call_id": message["tool_call_id"],
                          "output": output})
            continue
        text = _text(message)
        if role == "assistant":
            # the Responses API rejects an empty content item, so a turn that
            # only calls tools contributes its calls and nothing else
            if text:
                items.append({"role": "assistant", "content": text})
            for call in message.get("tool_calls") or []:
                items.append({"type": "function_call", "call_id": call.id,
                              "name": call.function,
                              "arguments": json.dumps(call.args)})
            continue
        items.append({"role": "developer" if role == "system" else role,
                      "content": text})
    return items


def _to_tools(runtime: FunctionsRuntime) -> list[dict[str, Any]]:
    return [{"type": "function", "name": f.name, "description": f.description,
             "parameters": f.parameters.model_json_schema()}
            for f in runtime.functions.values()]


def _from_output(response: Any) -> ChatAssistantMessage:
    calls: list[FunctionCall] = []
    text: list[str] = []
    for item in getattr(response, "output", None) or []:
        kind = getattr(item, "type", None)
        if kind == "function_call":
            try:
                args = json.loads(item.arguments or "{}")
            except json.JSONDecodeError:
                # a malformed call is the model's failure; hand AgentDojo an
                # empty call rather than crash the suite, as OpenAILLM would
                # have surfaced it as a tool error
                args = {}
            calls.append(FunctionCall(function=item.name, args=args, id=item.call_id))
        elif kind == "message":
            for chunk in getattr(item, "content", None) or []:
                if getattr(chunk, "type", None) == "output_text":
                    text.append(chunk.text)
    joined = "".join(text)
    return ChatAssistantMessage(
        role="assistant",
        content=[text_content_block_from_string(joined)] if joined else None,
        tool_calls=calls or None,
    )


class OpenAIResponsesLLM(BasePipelineElement):
    """Drop-in for AgentDojo's ``OpenAILLM`` over ``client.responses``."""

    def __init__(self, client: Any, model: str,
                 reasoning_effort: str | None = None) -> None:
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        tools = _to_tools(runtime)
        body: dict[str, Any] = {"model": self.model, "input": _to_items(messages),
                                "store": False}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        response = self.client.responses.create(**body)
        return query, runtime, env, [*messages, _from_output(response)], extra_args

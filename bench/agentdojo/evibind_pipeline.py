"""EviBind as an AgentDojo pipeline element.

AgentDojo builds its agent as ``[system, init_query, llm, ToolsExecutionLoop([
ToolsExecutor, llm])]``. Placing this guard first inside the loop puts it
between every model turn and the executor, which is exactly where an
argument-level boundary belongs: the model has proposed a call, and nothing has
run yet.

For each proposed call the guard rebuilds the tool's schema with
``x-evibind-*`` annotations, hands the proposal to ``protect_chat_completion``
with the user's turn as the only admissible source, and keeps whatever comes
back. A call whose critical argument cannot be re-derived is dropped rather than
executed.

The annotation rule is mechanical and applied to every tool in the suite. It is
the same shape used for InjecAgent: a required string parameter whose name or
description names an identifier becomes a control slot bound to
``user.current_turn``; everything else is content the model may fill.
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Sequence
from typing import Any

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatMessage, text_content_block_from_string

from tapbench.gateway import GatewayConfig, protect_chat_completion

IDENTIFIER_TYPES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("email", "e-mail"), "email_address"),
    (("url", "link", "website"), "uri"),
    (("path", "filename", "file name", "folder"), "repository_path"),
    (("iban", "account", "recipient", "id", "identifier", "number",
      "channel", "user", "address"), "opaque_registry_id"),
)

JSON_TYPES = {"string", "integer", "number", "boolean", "array", "object"}


def _evidence_type(name: str, description: str) -> str | None:
    haystack = f"{name} {description}".lower()
    for keywords, evidence_type in IDENTIFIER_TYPES:
        if any(keyword in haystack for keyword in keywords):
            return evidence_type
    return None


def annotate(schema: dict[str, Any], tool_name: str) -> dict[str, Any]:
    """Rebuild an AgentDojo tool schema with EviBind annotations."""
    properties: dict[str, Any] = {}
    required = list(schema.get("required") or [])
    # Every parameter is annotated, not only the required ones. On an update
    # call the field being changed is optional by nature -- AgentDojo's
    # `update_scheduled_transaction(id, recipient=None, ...)` marks only `id`
    # required -- so skipping optional parameters leaves exactly the argument
    # an attacker wants to set ungoverned. That hole accounted for every
    # residual attack success in the first banking run.
    for name, prop in (schema.get("properties") or {}).items():
        json_type = prop.get("type")
        if json_type is None:
            # optional parameters arrive as {"anyOf": [{"type": "string"},
            # {"type": "null"}]}; take the non-null branch
            for branch in prop.get("anyOf") or []:
                if branch.get("type") and branch.get("type") != "null":
                    json_type = branch["type"]
                    break
        if json_type not in JSON_TYPES:
            json_type = "string"
        spec: dict[str, Any] = {"type": json_type,
                                "description": str(prop.get("description", ""))}
        if json_type == "array":
            spec["items"] = {"type": "string"}
        evidence_type = (_evidence_type(name, str(prop.get("description", "")))
                         if json_type == "string" else None)
        if evidence_type:
            spec.update({
                "x-evibind-slot-role": "control",
                "x-evibind-evidence-type": evidence_type,
                "x-evibind-sources": ["user.current_turn"],
                "x-evibind-resolution-type": "extractive",
                "x-evibind-value-class": "authority_bearing",
                "x-evibind-criticality": "target",
            })
            if evidence_type == "opaque_registry_id":
                spec["x-evibind-extraction-cue"] = re.sub(r"[_\-]+", " ", name)
        else:
            spec.update({
                "x-evibind-slot-role": "content",
                "x-evibind-evidence-type": "opaque_content",
                "x-evibind-value-class": "opaque_content",
                "x-evibind-criticality": "content",
            })
        properties[name] = spec
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": str(schema.get("description", ""))[:800],
            "parameters": {"type": "object", "properties": properties,
                           "required": [n for n in required if n in properties],
                           "additionalProperties": False},
        },
    }


def governed_slots(annotated: dict[str, Any]) -> set[str]:
    return {name for name, spec
            in annotated["function"]["parameters"]["properties"].items()
            if spec.get("x-evibind-slot-role") == "control"}


class EviBindToolCallGuard(BasePipelineElement):
    """Withhold or re-derive critical arguments before AgentDojo executes them."""

    name = "evibind"

    def __init__(self, *, clarify_interchangeable_slots: bool = True) -> None:
        self.config = GatewayConfig(
            upstream_base_url="http://offline.invalid",
            upstream_api_key=None, gateway_api_key=None,
            handle_secret=secrets.token_bytes(32), allow_diagnostics=False)
        self.clarify_interchangeable_slots = clarify_interchangeable_slots
        self.stats = {"calls_seen": 0, "released": 0, "rewritten": 0,
                      "withheld": 0, "ungoverned": 0}

    def _schema_for(self, runtime: FunctionsRuntime, name: str) -> dict[str, Any] | None:
        function = runtime.functions.get(name)
        if function is None:
            return None
        parameters = function.parameters
        schema = (parameters.model_json_schema()
                  if hasattr(parameters, "model_json_schema") else dict(parameters))
        schema["description"] = getattr(function, "description", "") or ""
        return schema

    def _guard(self, call: FunctionCall, runtime: FunctionsRuntime,
               user_text: str) -> FunctionCall | None:
        schema = self._schema_for(runtime, call.function)
        if schema is None:
            return call
        annotated = annotate(schema, call.function)
        governed = governed_slots(annotated)
        if not governed:
            self.stats["ungoverned"] += 1
            return call

        supplied = {k: v for k, v in (call.args or {}).items()
                    if k in annotated["function"]["parameters"]["properties"]}
        request = {
            "model": "agentdojo",
            "messages": [{"role": "user", "content": user_text}],
            "tools": [annotated],
            "evibind": {
                "policy_epoch": "agentdojo",
                "include_diagnostics": False,
                "allow_noncritical_opaque_literals": True,
                "clarify_interchangeable_slots": self.clarify_interchangeable_slots,
            },
        }
        response = {"choices": [{"index": 0, "finish_reason": "tool_calls",
                                 "message": {"role": "assistant", "content": None,
                                             "tool_calls": [{
                                                 "id": call.id or "c0",
                                                 "type": "function",
                                                 "function": {
                                                     "name": call.function,
                                                     "arguments": json.dumps(
                                                         supplied, default=str)}}]}}]}
        try:
            protected = protect_chat_completion(request, response, config=self.config)
        except Exception:  # noqa: BLE001 - a boundary error must not execute the call
            self.stats["withheld"] += 1
            return None

        released_calls = protected["choices"][0]["message"].get("tool_calls") or []
        if not released_calls:
            self.stats["withheld"] += 1
            return None
        released = json.loads(released_calls[0]["function"]["arguments"])
        merged = dict(call.args or {})
        changed = False
        for slot in governed:
            if slot in released and released[slot] != merged.get(slot):
                merged[slot] = released[slot]
                changed = True
        self.stats["rewritten" if changed else "released"] += 1
        return FunctionCall(function=call.function, args=merged, id=call.id)

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if not messages:
            return query, runtime, env, messages, extra_args
        last = messages[-1]
        if last.get("role") != "assistant" or not last.get("tool_calls"):
            return query, runtime, env, messages, extra_args

        kept: list[FunctionCall] = []
        withheld: list[str] = []
        for call in last["tool_calls"]:
            self.stats["calls_seen"] += 1
            guarded = self._guard(call, runtime, query)
            if guarded is None:
                withheld.append(call.function)
            else:
                kept.append(guarded)

        if not withheld:
            return query, runtime, env, messages, extra_args

        note = ("EviBind withheld " + ", ".join(sorted(set(withheld)))
                + ": no admissible evidence in the user's request for the "
                  "critical argument.")
        replacement = dict(last)
        replacement["tool_calls"] = kept
        if not kept:
            # AgentDojo assistant messages carry content blocks, not a bare str
            replacement["content"] = [text_content_block_from_string(note)]
        messages = [*messages[:-1], replacement]
        return query, runtime, env, messages, extra_args

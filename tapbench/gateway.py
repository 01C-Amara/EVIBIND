from __future__ import annotations

import hmac
import ipaddress
import json
import os
import secrets
import time
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .deployable_resolution import (
    DEPLOYABLE_RESOLUTION_VERSION,
    resolve_deployable_prediction,
)
from .effect_authorization import (
    EffectAuthorizationError,
    EffectAuthorizer,
    gate_effect,
    parse_effect_policies,
)
from .native_tool_runtime import normalize_native_message
from .one_call_gateway import OneCallError, compile_one_call_session


EVIBIND_GATEWAY_VERSION = "evibind.gateway.v2"
DEFAULT_EVIDENCE_MODE = "typed_program_hybrid"
DEFAULT_CONTROLLER_MODE = "one_call"
SUPPORTED_CONTROLLER_MODES = {"one_call", "legacy_literal"}
DEFAULT_OPERATING_MODE = "enforce"
SUPPORTED_OPERATING_MODES = {"audit", "enforce", "assist"}
SUPPORTED_EVIDENCE_MODES = {
    "deterministic",
    "proposal_span_hybrid",
    "typed_programs",
    "typed_program_hybrid",
}
MAX_REQUEST_BYTES = 10 * 1024 * 1024
MAX_UPSTREAM_RESPONSE_BYTES = 10 * 1024 * 1024


class GatewayError(RuntimeError):
    def __init__(self, message: str, *, status: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = int(status)


class UpstreamError(GatewayError):
    def __init__(self, message: str, *, status: int, body: Any = None):
        super().__init__(message, status=status)
        self.body = body


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _read_upstream_body(response: Any) -> bytes:
    body = response.read(MAX_UPSTREAM_RESPONSE_BYTES + 1)
    if len(body) > MAX_UPSTREAM_RESPONSE_BYTES:
        raise UpstreamError(
            "upstream response exceeded the size limit",
            status=HTTPStatus.BAD_GATEWAY,
        )
    return body


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise GatewayError(f"{name} must be numeric") from exc
    if value <= 0:
        raise GatewayError(f"{name} must be positive")
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise GatewayError(f"{name} must be a boolean")


def _handle_secret_from_env() -> bytes:
    raw = os.getenv("EVIBIND_HANDLE_SECRET")
    if raw is None:
        return secrets.token_bytes(32)
    secret = raw.encode("utf-8")
    if len(secret) < 32:
        raise GatewayError("EVIBIND_HANDLE_SECRET must contain at least 32 UTF-8 bytes")
    return secret


@dataclass(frozen=True)
class GatewayConfig:
    upstream_base_url: str
    upstream_api_key: str | None = None
    gateway_api_key: str | None = None
    timeout_seconds: float = 120.0
    evidence_mode: str = DEFAULT_EVIDENCE_MODE
    candidate_budget: int = 2
    action_risk_budget: float = 0.05
    allow_diagnostics: bool = False
    controller_mode: str = DEFAULT_CONTROLLER_MODE
    handle_secret: bytes = field(
        default_factory=lambda: secrets.token_bytes(32), repr=False
    )
    operating_mode: str = DEFAULT_OPERATING_MODE

    def __post_init__(self) -> None:
        parsed = urlparse(self.upstream_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise GatewayError("upstream_base_url must be an http(s) URL")
        if parsed.username is not None or parsed.password is not None:
            raise GatewayError(
                "upstream_base_url must not contain embedded credentials"
            )
        if parsed.query or parsed.fragment:
            raise GatewayError("upstream_base_url must not contain a query or fragment")
        if self.evidence_mode not in SUPPORTED_EVIDENCE_MODES:
            raise GatewayError(f"unsupported evidence mode: {self.evidence_mode}")
        if not 0 <= self.candidate_budget <= 8:
            raise GatewayError("candidate_budget must be between 0 and 8")
        if not 0.0 <= self.action_risk_budget <= 1.0:
            raise GatewayError("action_risk_budget must be between 0 and 1")

        if not isinstance(self.allow_diagnostics, bool):
            raise GatewayError("allow_diagnostics must be a boolean")
        if self.controller_mode not in SUPPORTED_CONTROLLER_MODES:
            raise GatewayError(f"unsupported controller mode: {self.controller_mode}")
        if self.operating_mode not in SUPPORTED_OPERATING_MODES:
            raise GatewayError(f"unsupported operating mode: {self.operating_mode}")
        if self.controller_mode != "one_call" and self.operating_mode != "enforce":
            raise GatewayError(
                "audit and assist modes require controller_mode=one_call"
            )
        if not isinstance(self.handle_secret, bytes) or len(self.handle_secret) < 32:
            raise GatewayError("handle_secret must contain at least 32 bytes")

    @classmethod
    def from_env(cls) -> GatewayConfig:
        base_url = os.getenv("EVIBIND_UPSTREAM_BASE_URL", "").strip()
        if not base_url:
            raise GatewayError("EVIBIND_UPSTREAM_BASE_URL is required")
        try:
            candidate_budget = int(os.getenv("EVIBIND_CANDIDATE_BUDGET", "2"))
            action_risk_budget = float(os.getenv("EVIBIND_ACTION_RISK_BUDGET", "0.05"))
        except ValueError as exc:
            raise GatewayError(
                "EVIBIND_CANDIDATE_BUDGET and EVIBIND_ACTION_RISK_BUDGET must be numeric"
            ) from exc
        return cls(
            upstream_base_url=base_url,
            upstream_api_key=os.getenv("EVIBIND_UPSTREAM_API_KEY") or None,
            gateway_api_key=os.getenv("EVIBIND_GATEWAY_API_KEY") or None,
            timeout_seconds=_env_float("EVIBIND_UPSTREAM_TIMEOUT_SECONDS", 120.0),
            evidence_mode=os.getenv("EVIBIND_EVIDENCE_MODE", DEFAULT_EVIDENCE_MODE),
            candidate_budget=candidate_budget,
            action_risk_budget=action_risk_budget,
            allow_diagnostics=_env_bool("EVIBIND_ALLOW_DIAGNOSTICS"),
            controller_mode=os.getenv(
                "EVIBIND_CONTROLLER_MODE", DEFAULT_CONTROLLER_MODE
            ),
            handle_secret=_handle_secret_from_env(),
            operating_mode=os.getenv("EVIBIND_OPERATING_MODE", DEFAULT_OPERATING_MODE),
        )

    @property
    def completions_url(self) -> str:
        base = self.upstream_base_url.rstrip("/")
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"


def _runtime_schema(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            internal_key = (
                "x-tap-" + key.removeprefix("x-evibind-")
                if key.startswith("x-evibind-")
                else key
            )
            normalized = _runtime_schema(item)
            if internal_key in output and output[internal_key] != normalized:
                raise GatewayError(f"conflicting private annotation for {internal_key}")
            output[internal_key] = normalized
        return output
    if isinstance(value, list):
        return [_runtime_schema(item) for item in value]
    return deepcopy(value)


def _public_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _public_schema(item)
            for key, item in value.items()
            if not str(key).startswith(("x-tap-", "x-evibind-"))
        }
    if isinstance(value, list):
        return [_public_schema(item) for item in value]
    return deepcopy(value)


def strip_private_annotations(value: Any) -> Any:
    """Return a deep copy without EviBind's private schema annotations."""
    return _public_schema(value)


def normalize_openai_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    normalized = []
    for raw in tools:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function") if raw.get("type") == "function" else raw
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        normalized.append(
            {
                "name": name,
                "canonical_name": name,
                "description": str(function.get("description", "")),
                "parameters": _runtime_schema(
                    function.get(
                        "parameters",
                        {"type": "object", "properties": {}},
                    )
                ),
            }
        )
    return normalized


def public_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": _public_schema(tool.get("parameters", {})),
            },
        }
        for tool in tools
    ]


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") in {
            "text",
            "input_text",
        }:
            parts.append(str(item.get("text", "")))
    return "\n".join(part for part in parts if part)


def _runtime_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise GatewayError("messages must be a list")
    output = []
    for message in messages:
        if not isinstance(message, dict):
            raise GatewayError("each message must be an object")
        output.append(
            {
                "role": str(message.get("role", "user")),
                "content": _message_content_text(message.get("content")),
            }
        )
    return output


def _bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
    option_name: str,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise GatewayError(f"{option_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise GatewayError(f"{option_name} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise GatewayError(f"{option_name} must be between {minimum} and {maximum}")
    return parsed


def _bounded_float(
    value: Any, *, default: float, minimum: float, maximum: float
) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise GatewayError("numeric EviBind options cannot be booleans")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GatewayError("action_risk_budget must be numeric") from exc
    if not minimum <= parsed <= maximum:
        raise GatewayError(
            f"action_risk_budget must be between {minimum} and {maximum}"
        )
    return parsed


def prepare_upstream_payload(
    request_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(request_payload, dict):
        raise GatewayError("request body must be a JSON object")
    options = request_payload.get("evibind", {})
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise GatewayError("evibind must be an object")
    _runtime_messages(request_payload.get("messages"))
    raw_tools = request_payload.get("tools")
    if raw_tools is not None and not isinstance(raw_tools, list):
        raise GatewayError("tools must be a list of valid function schemas")
    tools = normalize_openai_tools(raw_tools)
    if isinstance(raw_tools, list) and len(tools) != len(raw_tools):
        raise GatewayError("tools must contain only valid function schemas")
    tool_names = [tool["name"] for tool in tools]
    if len(set(tool_names)) != len(tool_names):
        raise GatewayError("tool function names must be unique")
    upstream = deepcopy(request_payload)
    upstream.pop("evibind", None)
    if tools:
        upstream["tools"] = public_openai_tools(tools)
        upstream["parallel_tool_calls"] = False
    completion_count = request_payload.get("n", 1)
    if isinstance(completion_count, bool) or completion_count != 1:
        raise GatewayError("n must be 1 for single-call protection")
    upstream["n"] = 1
    if upstream.get("stream"):
        raise GatewayError("streaming tool-call protection is not supported in v2")
    return upstream, deepcopy(options), tools


def _non_call_content(action: dict[str, Any]) -> str:
    mode = action.get("mode")
    payload = action.get("payload", {})
    missing = payload.get("missing_slots", []) if isinstance(payload, dict) else []
    if mode == "clarify" and missing:
        slots = ", ".join(str(slot) for slot in missing)
        return f"Please provide the following before I use the tool: {slots}."
    if mode == "clarify":
        return "I need one more detail before I can safely use that tool."
    if mode == "no_tool":
        return "No relevant tool call was released."
    if mode == "direct_answer":
        return "The request should be answered without using a tool."
    if mode == "refuse":
        return "The requested tool call was not released."
    return "The tool call could not be supported by the available evidence."


def _resolution_summary(
    action: dict[str, Any],
    resolution: dict[str, Any],
    *,
    released: bool,
    include_diagnostics: bool,
) -> dict[str, Any]:
    payload = action.get("payload", {})
    summary: dict[str, Any] = {
        "released": released,
        "decision": action.get("mode"),
        "tool": action.get("tool"),
        "missing_slots": (
            payload.get("missing_slots", []) if isinstance(payload, dict) else []
        ),
        "reason": (payload.get("reason") if isinstance(payload, dict) else None),
        "terminal_state": resolution.get("terminal_state"),
        "evidence_mode": resolution.get("evidence_mode"),
        "resolution_version": DEPLOYABLE_RESOLUTION_VERSION,
        "contract_solver_version": resolution.get("schema_version"),
        "evidence_contract_version": resolution.get("evidence_contract_version"),
        "proposal_candidates_added": resolution.get("proposal_candidates_added", 0),
        "typed_program_candidates_added": resolution.get(
            "typed_program_candidates_added", 0
        ),
        "resolution_ms": round(
            float(resolution.get("total_resolution_seconds", 0.0)) * 1000.0,
            3,
        ),
    }
    if include_diagnostics:
        summary["diagnostics"] = resolution
    return summary


def protect_chat_completion(
    request_payload: dict[str, Any],
    upstream_response: dict[str, Any],
    *,
    config: GatewayConfig,
) -> dict[str, Any]:
    _, options, tools = prepare_upstream_payload(request_payload)
    if not isinstance(upstream_response, dict):
        raise GatewayError("upstream response must be a JSON object")
    protected = deepcopy(upstream_response)
    choices = protected.get("choices")
    if not isinstance(choices, list) or not choices:
        raise UpstreamError(
            "upstream response omitted choices",
            status=HTTPStatus.BAD_GATEWAY,
            body=upstream_response,
        )

    if len(choices) != 1:
        raise UpstreamError(
            "upstream response must contain exactly one choice",
            status=HTTPStatus.BAD_GATEWAY,
            body=upstream_response,
        )
    evidence_mode = str(options.get("evidence_mode", config.evidence_mode))
    if evidence_mode not in SUPPORTED_EVIDENCE_MODES:
        raise GatewayError(f"unsupported evidence mode: {evidence_mode}")
    candidate_budget = _bounded_int(
        options.get("candidate_budget"),
        default=config.candidate_budget,
        minimum=0,
        maximum=8,
        option_name="candidate_budget",
    )
    candidate_seed = _bounded_int(
        options.get("candidate_seed"),
        default=17,
        minimum=0,
        maximum=2**31 - 1,
        option_name="candidate_seed",
    )
    action_risk_budget = _bounded_float(
        options.get("action_risk_budget"),
        default=config.action_risk_budget,
        minimum=0.0,
        maximum=1.0,
    )
    include_diagnostics = options.get("include_diagnostics", False)
    if not isinstance(include_diagnostics, bool):
        raise GatewayError("evibind.include_diagnostics must be a boolean")
    if include_diagnostics and not config.allow_diagnostics:
        raise GatewayError(
            "diagnostics are disabled; set EVIBIND_ALLOW_DIAGNOSTICS=true"
        )
    reference_context = deepcopy(options.get("reference_context", {}))
    dialogue_state = deepcopy(options.get("dialogue_state", {}))
    if not isinstance(reference_context, dict):
        raise GatewayError("evibind.reference_context must be an object")
    if not isinstance(dialogue_state, dict):
        raise GatewayError("evibind.dialogue_state must be an object")
    reference_context["action_risk_budget"] = action_risk_budget
    runtime_case = {
        "messages": _runtime_messages(request_payload.get("messages")),
        "tools": tools,
        "tool_aliases": {},
        "argument_aliases": {},
    }

    evibind_choices = []
    for index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            raise UpstreamError(
                "upstream choice must be an object",
                status=HTTPStatus.BAD_GATEWAY,
                body=choice,
            )
        message = choice.get("message")
        if not isinstance(message, dict):
            raise UpstreamError(
                "upstream choice omitted an assistant message",
                status=HTTPStatus.BAD_GATEWAY,
                body=choice,
            )
        tool_calls = message.get("tool_calls")
        if tool_calls is not None and not isinstance(tool_calls, list):
            message.pop("tool_calls", None)
            message.pop("function_call", None)
            message["content"] = (
                "Malformed tool calls were proposed; EviBind released none."
            )
            choice["finish_reason"] = "stop"
            evibind_choices.append(
                {
                    "index": index,
                    "released": False,
                    "decision": "escalate",
                    "reason": "malformed_tool_calls",
                }
            )
            continue
        calls = tool_calls if isinstance(tool_calls, list) else []
        legacy_function_call = message.pop("function_call", None)
        if legacy_function_call is not None and not calls:
            message["content"] = (
                "A legacy function_call was proposed; EviBind v1 released none."
            )
            choice["finish_reason"] = "stop"
            evibind_choices.append(
                {
                    "index": index,
                    "released": False,
                    "decision": "escalate",
                    "reason": "legacy_function_call_not_supported",
                }
            )
            continue
        if not calls:
            evibind_choices.append(
                {
                    "index": index,
                    "released": False,
                    "decision": "pass_through",
                    "reason": "no_proposed_tool_call",
                }
            )
            continue
        if not tools:
            message.pop("tool_calls", None)
            message["content"] = (
                "A tool call was proposed without a usable schema; EviBind released none."
            )
            choice["finish_reason"] = "stop"
            evibind_choices.append(
                {
                    "index": index,
                    "released": False,
                    "decision": "escalate",
                    "reason": "tool_call_without_usable_schema",
                }
            )
            continue
        if len(calls) != 1:
            message.pop("tool_calls", None)
            message["content"] = (
                "Multiple tool calls were proposed; EviBind v1 released none."
            )
            choice["finish_reason"] = "stop"
            evibind_choices.append(
                {
                    "index": index,
                    "released": False,
                    "decision": "escalate",
                    "reason": "multiple_tool_calls_not_supported",
                    "proposed_call_count": len(calls),
                }
            )
            continue

        action, native_diagnostics = normalize_native_message(message)
        materialized, resolution = resolve_deployable_prediction(
            runtime_case,
            action,
            reference_context=reference_context,
            dialogue_state=dialogue_state,
            candidate_seed=candidate_seed,
            budget=candidate_budget,
            evidence_mode=evidence_mode,
        )
        released = (
            materialized.get("mode") == "call"
            and isinstance(materialized.get("tool"), str)
            and isinstance(materialized.get("arguments"), dict)
        )
        if released:
            proposed_call = calls[0] if isinstance(calls[0], dict) else {}
            call = {
                "type": "function",
                "function": {
                    "name": materialized["tool"],
                    "arguments": json.dumps(
                        materialized["arguments"],
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                },
            }
            call_id = proposed_call.get("id")
            if isinstance(call_id, str) and call_id:
                call["id"] = call_id
            message["tool_calls"] = [call]
            message["content"] = None
            choice["finish_reason"] = "tool_calls"
        else:
            message.pop("tool_calls", None)
            message["content"] = _non_call_content(materialized)
            choice["finish_reason"] = "stop"
        summary = _resolution_summary(
            materialized,
            resolution,
            released=released,
            include_diagnostics=include_diagnostics,
        )
        summary["index"] = index
        summary["proposed_tool"] = action.get("tool")
        summary["native_tool_call_count"] = native_diagnostics.get(
            "native_tool_call_count"
        )
        evibind_choices.append(summary)

    protected["evibind"] = {
        "version": EVIBIND_GATEWAY_VERSION,
        "selective_guarantee": (
            "Only released calls passed the configured evidence and contract checks."
        ),
        "choices": evibind_choices,
    }
    return protected


class EviBindGateway:
    def __init__(
        self,
        config: GatewayConfig,
        *,
        effect_authorizer: EffectAuthorizer | None = None,
    ):
        self.config = config
        self.effect_authorizer = effect_authorizer or EffectAuthorizer(
            config.handle_secret
        )

    def _upstream_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"EviBind/{EVIBIND_GATEWAY_VERSION}",
        }
        if self.config.upstream_api_key:
            headers["Authorization"] = f"Bearer {self.config.upstream_api_key}"
        request = urllib.request.Request(
            self.config.completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with _NO_REDIRECT_OPENER.open(
                request, timeout=self.config.timeout_seconds
            ) as response:
                body = _read_upstream_body(response)
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                exc.close()
                raise UpstreamError(
                    "upstream redirects are not allowed",
                    status=HTTPStatus.BAD_GATEWAY,
                ) from exc
            try:
                raw = _read_upstream_body(exc)
            finally:
                exc.close()
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                body = raw.decode("utf-8", errors="replace")
            raise UpstreamError(
                f"upstream returned HTTP {exc.code}",
                status=exc.code,
                body=body,
            ) from exc
        except urllib.error.URLError as exc:
            raise UpstreamError(
                f"could not reach upstream: {exc.reason}",
                status=HTTPStatus.BAD_GATEWAY,
            ) from exc
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise UpstreamError(
                "upstream returned invalid JSON",
                status=HTTPStatus.BAD_GATEWAY,
            ) from exc
        if not isinstance(parsed, dict):
            raise UpstreamError(
                "upstream response must be a JSON object",
                status=HTTPStatus.BAD_GATEWAY,
                body=parsed,
            )
        return parsed

    def chat_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        upstream_payload, options, tools = prepare_upstream_payload(request_payload)
        try:
            effect_policies = parse_effect_policies(
                options,
                tool_ids={str(tool["name"]) for tool in tools},
            )
        except EffectAuthorizationError as exc:
            raise GatewayError(str(exc)) from exc
        confirmation_token = options.get("effect_confirmation")
        if confirmation_token is not None and not effect_policies:
            raise GatewayError(
                "effect confirmation requires a configured effect policy"
            )
        if effect_policies and self.config.controller_mode != "one_call":
            raise GatewayError("effect policies require controller_mode=one_call")
        if effect_policies and self.config.operating_mode == "audit":
            raise GatewayError("effect policies require enforce or assist mode")
        session = None
        if self.config.controller_mode == "one_call":
            include_diagnostics = options.get("include_diagnostics", False)
            if not isinstance(include_diagnostics, bool):
                raise GatewayError("evibind.include_diagnostics must be a boolean")
            if include_diagnostics and not self.config.allow_diagnostics:
                raise GatewayError(
                    "diagnostics are disabled; set EVIBIND_ALLOW_DIAGNOSTICS=true"
                )
            try:
                session = compile_one_call_session(
                    request_payload=request_payload,
                    upstream_payload=upstream_payload,
                    options=options,
                    tools=tools,
                    handle_secret=self.config.handle_secret,
                    include_diagnostics=include_diagnostics,
                    operating_mode=self.config.operating_mode,
                )
            except OneCallError as exc:
                raise GatewayError(str(exc)) from exc
            if self.config.operating_mode != "audit":
                upstream_payload = dict(session.upstream_payload)
        started = time.perf_counter()
        upstream_response = self._upstream_request(upstream_payload)
        if session is not None:
            try:
                protected = (
                    session.audit(upstream_response)
                    if self.config.operating_mode == "audit"
                    else session.protect(upstream_response)
                )
            except OneCallError as exc:
                raise UpstreamError(
                    str(exc),
                    status=HTTPStatus.BAD_GATEWAY,
                    body=upstream_response,
                ) from exc
            if effect_policies:
                try:
                    protected = gate_effect(
                        protected,
                        policies=effect_policies,
                        confirmation_token=confirmation_token,
                        authorizer=self.effect_authorizer,
                        request_digest=session.context.request_digest,
                        policy_epoch=session.context.policy_epoch,
                    )
                except EffectAuthorizationError as exc:
                    raise GatewayError(str(exc)) from exc
        else:
            protected = protect_chat_completion(
                request_payload,
                upstream_response,
                config=self.config,
            )
        protected["evibind"]["upstream_ms"] = round(
            (time.perf_counter() - started) * 1000.0,
            3,
        )
        return protected


def _error_payload(
    exc: GatewayError, *, include_upstream: bool = False
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": {
            "message": str(exc),
            "type": "evibind_gateway_error",
            "code": exc.status,
        }
    }
    if include_upstream and isinstance(exc, UpstreamError) and exc.body is not None:
        payload["error"]["upstream"] = exc.body
    return payload


def make_handler(gateway: EviBindGateway) -> type[BaseHTTPRequestHandler]:
    class EviBindHandler(BaseHTTPRequestHandler):
        server_version = "EviBind/2"

        def version_string(self) -> str:
            return self.server_version

        def _authorized(self) -> bool:
            expected = gateway.config.gateway_api_key
            if expected is None:
                return True
            supplied = self.headers.get("Authorization", "")
            scheme, separator, token = supplied.partition(" ")
            if not separator or scheme.casefold() != "bearer":
                return False
            token = token.strip()
            return hmac.compare_digest(token, expected)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-EviBind-Version", EVIBIND_GATEWAY_VERSION)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.rstrip("/") in {"", "/health"}:
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "version": EVIBIND_GATEWAY_VERSION,
                    },
                )
                return
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": {"message": "route not found"}},
            )

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": {"message": "route not found"}},
                )
                return
            if not self._authorized():
                self._send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {"error": {"message": "invalid gateway API key"}},
                )
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if not 0 < content_length <= MAX_REQUEST_BYTES:
                    raise GatewayError(
                        "request body is empty or too large",
                        status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    )
                raw = self.rfile.read(content_length)
                payload = json.loads(raw)
                response = gateway.chat_completion(payload)
                self._send_json(HTTPStatus.OK, response)
            except json.JSONDecodeError:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"message": "request body is not valid JSON"}},
                )
            except GatewayError as exc:
                self._send_json(
                    exc.status,
                    _error_payload(
                        exc,
                        include_upstream=gateway.config.allow_diagnostics,
                    ),
                )
            except Exception:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "error": {
                            "message": "unexpected EviBind gateway error",
                            "type": "evibind_gateway_error",
                        }
                    },
                )

        def log_message(self, format: str, *args: Any) -> None:
            return

    return EviBindHandler


def serve(
    config: GatewayConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8090,
) -> None:
    normalized_host = host.strip().strip("[]")
    try:
        loopback = ipaddress.ip_address(normalized_host).is_loopback
    except ValueError:
        loopback = normalized_host.casefold() == "localhost"
    if not loopback and config.gateway_api_key is None:
        raise GatewayError(
            "EVIBIND_GATEWAY_API_KEY is required for non-loopback binding"
        )
    gateway = EviBindGateway(config)
    server = ThreadingHTTPServer((host, port), make_handler(gateway))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

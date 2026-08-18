from __future__ import annotations

import inspect
import json
import threading
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from evibind.core.derivations import sha256_digest
from tapbench.effect_authorization import (
    EffectAuthorizationError,
    EffectAuthorizer,
    EffectPolicy,
    gate_effect,
    parse_effect_policies,
)
from tapbench.gateway import GatewayError, prepare_upstream_payload
from tapbench.one_call_gateway import (
    OneCallError,
    OneCallSession,
    compile_one_call_session,
)


HOST_SDK_VERSION = "evibind.host_sdk.v1"
HOST_OPERATING_MODES = frozenset({"enforce", "assist"})

ToolHandler = Callable[[Mapping[str, Any]], Any]


class HostSDKError(RuntimeError):
    """Raised when a host turn cannot preserve the EviBind boundary."""


class ToolDispatchError(HostSDKError):
    """Raised after admission when the registered tool handler fails."""


@dataclass(frozen=True)
class HostExecutionResult:
    request_digest: str
    decision: str
    executed: bool
    protected_response: Mapping[str, Any]
    tool_id: str | None = None
    arguments: Mapping[str, Any] | None = None
    manifest_digest: str | None = None
    result: Any = None

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "version": HOST_SDK_VERSION,
            "request_digest": self.request_digest,
            "decision": self.decision,
            "executed": self.executed,
            "protected_response": deepcopy(dict(self.protected_response)),
        }
        if self.tool_id is not None:
            output["tool_id"] = self.tool_id
        if self.arguments is not None:
            output["arguments"] = deepcopy(dict(self.arguments))
        if self.manifest_digest is not None:
            output["manifest_digest"] = self.manifest_digest
        if self.executed:
            output["result"] = self.result
        return output


class GuardedTurn:
    """A single-use model turn compiled against one immutable evidence context."""

    def __init__(
        self,
        *,
        owner: GuardedToolExecutor,
        session: OneCallSession,
        effect_policies: Mapping[str, EffectPolicy],
        confirmation_token: str | None,
    ) -> None:
        self._owner = owner
        self._session = session
        self._effect_policies = dict(effect_policies)
        self._confirmation_token = confirmation_token
        self._lock = threading.Lock()
        self._completed = False

    @property
    def upstream_payload(self) -> dict[str, Any]:
        """Return an isolated provider payload containing only the Action IR tool."""
        return deepcopy(dict(self._session.upstream_payload))

    @property
    def request_digest(self) -> str:
        return self._session.context.request_digest

    @property
    def policy_epoch(self) -> str:
        return self._session.context.policy_epoch

    @property
    def completed(self) -> bool:
        with self._lock:
            return self._completed

    def complete(
        self,
        upstream_response: Mapping[str, Any],
    ) -> HostExecutionResult:
        """Protect one provider response and dispatch at most one admitted call."""
        with self._lock:
            if self._completed:
                raise HostSDKError("guarded turn has already been completed")
            self._completed = True

        try:
            protected = self._session.protect(upstream_response)
            if self._effect_policies:
                protected = gate_effect(
                    protected,
                    request_digest=self.request_digest,
                    policy_epoch=self.policy_epoch,
                    policies=self._effect_policies,
                    confirmation_token=self._confirmation_token,
                    authorizer=self._owner.effect_authorizer,
                )
        except (OneCallError, EffectAuthorizationError, ValueError) as exc:
            raise HostSDKError(str(exc)) from exc
        return self._owner._dispatch(protected, session=self._session)


class GuardedToolExecutor:
    """Host-owned compiler, materializer, effect gate, and tool dispatcher."""

    def __init__(
        self,
        handlers: Mapping[str, ToolHandler],
        *,
        handle_secret: bytes,
        operating_mode: str = "enforce",
        allow_diagnostics: bool = False,
        effect_authorizer: EffectAuthorizer | None = None,
        handle_nonce_bytes: Callable[[int], bytes] | None = None,
    ) -> None:
        if not isinstance(handle_secret, bytes) or len(handle_secret) < 32:
            raise HostSDKError("handle_secret must contain at least 32 bytes")
        if operating_mode not in HOST_OPERATING_MODES:
            raise HostSDKError(
                "host execution supports only enforce or assist mode"
            )
        if not isinstance(allow_diagnostics, bool):
            raise HostSDKError("allow_diagnostics must be a boolean")
        normalized: dict[str, ToolHandler] = {}
        for tool_id, handler in handlers.items():
            if not isinstance(tool_id, str) or not tool_id:
                raise HostSDKError("tool handler names must be non-empty strings")
            if not callable(handler):
                raise HostSDKError(f"tool handler is not callable: {tool_id}")
            if inspect.iscoroutinefunction(handler):
                raise HostSDKError(
                    f"async tool handler is not supported by the sync host SDK: "
                    f"{tool_id}"
                )
            normalized[tool_id] = handler
        self._handlers = MappingProxyType(normalized)
        self._handle_secret = handle_secret
        self._operating_mode = operating_mode
        self._allow_diagnostics = allow_diagnostics
        self._handle_nonce_bytes = handle_nonce_bytes
        self.effect_authorizer = effect_authorizer or EffectAuthorizer(
            handle_secret
        )

    @property
    def tool_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))

    def prepare(self, request_payload: Mapping[str, Any]) -> GuardedTurn:
        """Compile a request into a single-use, full-context guarded host turn."""
        if not isinstance(request_payload, Mapping):
            raise HostSDKError("request_payload must be an object")
        try:
            upstream, options, tools = prepare_upstream_payload(
                deepcopy(dict(request_payload))
            )
            requested_tools = {str(tool["name"]) for tool in tools}
            missing_handlers = sorted(requested_tools - set(self._handlers))
            if missing_handlers:
                raise HostSDKError(
                    "request references unregistered tool handlers: "
                    + ", ".join(missing_handlers)
                )
            include_diagnostics = options.get("include_diagnostics", False)
            if not isinstance(include_diagnostics, bool):
                raise HostSDKError(
                    "evibind.include_diagnostics must be a boolean"
                )
            if include_diagnostics and not self._allow_diagnostics:
                raise HostSDKError(
                    "diagnostics are disabled for this host executor"
                )
            effect_policies = parse_effect_policies(
                options,
                tool_ids=frozenset(requested_tools),
            )
            confirmation_token = options.get("effect_confirmation")
            session = compile_one_call_session(
                request_payload=request_payload,
                upstream_payload=upstream,
                options=options,
                tools=tools,
                handle_secret=self._handle_secret,
                include_diagnostics=include_diagnostics,
                operating_mode=self._operating_mode,
                handle_nonce_bytes=self._handle_nonce_bytes,
            )
        except HostSDKError:
            raise
        except (
            EffectAuthorizationError,
            GatewayError,
            OneCallError,
            ValueError,
        ) as exc:
            raise HostSDKError(str(exc)) from exc
        return GuardedTurn(
            owner=self,
            session=session,
            effect_policies=effect_policies,
            confirmation_token=(
                confirmation_token
                if isinstance(confirmation_token, str)
                else None
            ),
        )

    def _dispatch(
        self,
        protected_response: Mapping[str, Any],
        *,
        session: OneCallSession,
    ) -> HostExecutionResult:
        protected = deepcopy(dict(protected_response))
        metadata = protected.get("evibind")
        choices = protected.get("choices")
        if (
            not isinstance(metadata, dict)
            or metadata.get("enforced") is not True
            or metadata.get("operating_mode") != self._operating_mode
            or not isinstance(metadata.get("choices"), list)
            or len(metadata["choices"]) != 1
            or not isinstance(choices, list)
            or len(choices) != 1
        ):
            raise HostSDKError(
                "host dispatcher requires one enforced protected response"
            )
        summary = metadata["choices"][0]
        choice = choices[0]
        if not isinstance(summary, dict) or not isinstance(choice, dict):
            raise HostSDKError("protected response choice is invalid")
        decision = summary.get("decision")
        if not isinstance(decision, str):
            raise HostSDKError("protected response omitted a decision")
        if summary.get("released") is not True:
            metadata["host_execution"] = {
                "version": HOST_SDK_VERSION,
                "executed": False,
                "decision": decision,
            }
            return HostExecutionResult(
                request_digest=session.context.request_digest,
                decision=decision,
                executed=False,
                protected_response=protected,
            )

        message = choice.get("message")
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(calls, list) or len(calls) != 1:
            raise HostSDKError("released response must contain exactly one call")
        call = calls[0]
        function = call.get("function") if isinstance(call, Mapping) else None
        if not isinstance(function, Mapping):
            raise HostSDKError("released response call omitted a function")
        tool_id = function.get("name")
        raw_arguments = function.get("arguments")
        if not isinstance(tool_id, str) or not isinstance(raw_arguments, str):
            raise HostSDKError("released response call is malformed")
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise HostSDKError(
                "released response arguments are invalid JSON"
            ) from exc
        if not isinstance(arguments, dict):
            raise HostSDKError("released response arguments must be an object")
        if summary.get("tool") != tool_id:
            raise HostSDKError("released response tool binding mismatch")
        manifest_digest = summary.get("manifest_digest")
        if not isinstance(manifest_digest, str):
            raise HostSDKError("released response omitted a manifest digest")
        try:
            contract_version = session.policy.tool(tool_id).contract_version
        except ValueError as exc:
            raise HostSDKError("released response references an unknown tool") from exc
        expected_manifest = sha256_digest(
            {
                "tool_id": tool_id,
                "arguments": arguments,
                "request_digest": session.context.request_digest,
                "contract_version": contract_version,
            }
        )
        if manifest_digest != expected_manifest:
            raise HostSDKError("released response manifest binding mismatch")
        handler = self._handlers.get(tool_id)
        if handler is None:
            raise HostSDKError(
                f"released response has no registered handler: {tool_id}"
            )
        frozen_arguments = deepcopy(arguments)
        try:
            result = handler(deepcopy(frozen_arguments))
            if inspect.isawaitable(result):
                close = getattr(result, "close", None)
                if callable(close):
                    close()
                raise HostSDKError(
                    f"tool handler returned an awaitable in the sync host SDK: "
                    f"{tool_id}"
                )
        except HostSDKError:
            raise
        except Exception as exc:
            raise ToolDispatchError(
                f"registered tool handler failed: {tool_id}"
            ) from exc

        metadata["host_execution"] = {
            "version": HOST_SDK_VERSION,
            "executed": True,
            "decision": decision,
            "tool_id": tool_id,
            "manifest_digest": manifest_digest,
        }
        return HostExecutionResult(
            request_digest=session.context.request_digest,
            decision=decision,
            executed=True,
            protected_response=protected,
            tool_id=tool_id,
            arguments=frozen_arguments,
            manifest_digest=manifest_digest,
            result=result,
        )

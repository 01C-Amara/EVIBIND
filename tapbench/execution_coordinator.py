from __future__ import annotations

import secrets
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Mapping

from evibind.core import (
    ExecutionGraphError,
    ExecutionRecord,
    transition_execution,
)

from .gateway import GatewayError, prepare_upstream_payload
from .one_call_gateway import OneCallError, OneCallSession, compile_one_call_session


EXECUTION_COORDINATOR_VERSION = "evibind.execution_coordinator.v1"


@dataclass(frozen=True)
class StatefulExecution:
    request_payload: Mapping[str, Any]
    session: OneCallSession
    record: ExecutionRecord
    protected_response: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": EXECUTION_COORDINATOR_VERSION,
            "execution": self.record.to_dict(),
            "upstream_payload": deepcopy(dict(self.session.upstream_payload)),
            "protected_response": (
                deepcopy(dict(self.protected_response))
                if self.protected_response is not None
                else None
            ),
        }


class ExecutionCoordinator:
    def __init__(
        self,
        handle_secret: bytes,
        *,
        operating_mode: str = "assist",
    ) -> None:
        if operating_mode not in {"enforce", "assist"}:
            raise GatewayError("stateful execution requires enforce or assist mode")
        self._handle_secret = handle_secret
        self._operating_mode = operating_mode

    def _compile(
        self,
        request_payload: Mapping[str, Any],
    ) -> OneCallSession:
        request = deepcopy(dict(request_payload))
        upstream, options, tools = prepare_upstream_payload(request)
        include_diagnostics = options.get("include_diagnostics", False)
        if not isinstance(include_diagnostics, bool):
            raise GatewayError("evibind.include_diagnostics must be a boolean")
        try:
            return compile_one_call_session(
                request_payload=request,
                upstream_payload=upstream,
                options=options,
                tools=tools,
                handle_secret=self._handle_secret,
                include_diagnostics=include_diagnostics,
                operating_mode=self._operating_mode,
            )
        except OneCallError as exc:
            raise GatewayError(str(exc)) from exc

    def begin(
        self,
        request_payload: Mapping[str, Any],
        *,
        execution_id: str | None = None,
    ) -> StatefulExecution:
        request = deepcopy(dict(request_payload))
        session = self._compile(request)
        record = ExecutionRecord(
            execution_id=execution_id or "exe_" + secrets.token_urlsafe(12),
            request_digest=session.context.request_digest,
            policy_epoch=session.context.policy_epoch,
        )
        record = transition_execution(record, "compile", expected_version=0)
        return StatefulExecution(
            request_payload=request,
            session=session,
            record=record,
        )

    def apply_model_response(
        self,
        execution: StatefulExecution,
        upstream_response: Mapping[str, Any],
        *,
        expected_version: int,
    ) -> StatefulExecution:
        if execution.record.node != "selecting":
            raise ExecutionGraphError("model responses require a selecting execution")
        if execution.record.version != expected_version:
            raise ExecutionGraphError(
                "stale execution version: "
                f"expected {execution.record.version}, got {expected_version}"
            )
        protected = execution.session.protect(upstream_response)
        metadata = protected.get("evibind")
        summaries = metadata.get("choices") if isinstance(metadata, Mapping) else None
        if (
            not isinstance(summaries, list)
            or len(summaries) != 1
            or not isinstance(summaries[0], Mapping)
        ):
            raise ExecutionGraphError(
                "protected response omitted one execution decision"
            )
        summary = summaries[0]
        decision = summary.get("decision")
        if decision == "need_input":
            record = transition_execution(
                execution.record,
                "need_input",
                expected_version=expected_version,
                tool_id=str(summary.get("tool", "")),
                missing=tuple(str(item) for item in summary.get("missing", ())),
            )
        elif decision == "call" and summary.get("released") is True:
            record = transition_execution(
                execution.record,
                "materialize",
                expected_version=expected_version,
                tool_id=str(summary.get("tool", "")),
                manifest_digest=str(summary.get("manifest_digest", "")),
            )
        elif decision == "no_tool":
            record = transition_execution(
                execution.record,
                "no_tool",
                expected_version=expected_version,
            )
        else:
            record = transition_execution(
                execution.record,
                "fail",
                expected_version=expected_version,
            )
        return replace(
            execution,
            record=record,
            protected_response=protected,
        )

    def clarify(
        self,
        execution: StatefulExecution,
        user_message: str | Mapping[str, Any],
        *,
        expected_version: int,
        dialogue_state: Mapping[str, Any] | None = None,
    ) -> StatefulExecution:
        if execution.record.node != "awaiting_clarification":
            raise ExecutionGraphError(
                "clarification requires an awaiting_clarification execution"
            )
        request = deepcopy(dict(execution.request_payload))
        if execution.record.version != expected_version:
            raise ExecutionGraphError(
                "stale execution version: "
                f"expected {execution.record.version}, got {expected_version}"
            )
        messages = request.get("messages")
        if not isinstance(messages, list):
            raise GatewayError("messages must be a list")
        if isinstance(user_message, str):
            message = {
                "id": f"clarification-{expected_version + 1}",
                "role": "user",
                "content": user_message,
            }
        elif isinstance(user_message, Mapping):
            message = deepcopy(dict(user_message))
            if message.get("role", "user") != "user":
                raise GatewayError("clarification message must have role=user")
            message["role"] = "user"
            message.setdefault("id", f"clarification-{expected_version + 1}")
        else:
            raise GatewayError("clarification message must be text or an object")
        existing_ids = {
            str(item.get("id"))
            for item in messages
            if isinstance(item, Mapping) and item.get("id") is not None
        }
        if str(message["id"]) in existing_ids:
            raise GatewayError("clarification message id must be unique")
        messages.append(message)
        if dialogue_state is not None:
            options = request.setdefault("evibind", {})
            if not isinstance(options, dict):
                raise GatewayError("evibind must be an object")
            options["dialogue_state"] = deepcopy(dict(dialogue_state))

        session = self._compile(request)
        record = transition_execution(
            execution.record,
            "clarify",
            expected_version=expected_version,
            request_digest=session.context.request_digest,
        )
        return StatefulExecution(
            request_payload=request,
            session=session,
            record=record,
        )

    @staticmethod
    def require_confirmation(
        execution: StatefulExecution,
        *,
        expected_version: int,
    ) -> StatefulExecution:
        return replace(
            execution,
            record=transition_execution(
                execution.record,
                "require_confirmation",
                expected_version=expected_version,
            ),
        )

    @staticmethod
    def confirm(
        execution: StatefulExecution,
        *,
        expected_version: int,
        authorization_digest: str,
    ) -> StatefulExecution:
        return replace(
            execution,
            record=transition_execution(
                execution.record,
                "confirm",
                expected_version=expected_version,
                authorization_digest=authorization_digest,
            ),
        )

    @staticmethod
    def authorize_without_confirmation(
        execution: StatefulExecution,
        *,
        expected_version: int,
    ) -> StatefulExecution:
        return replace(
            execution,
            record=transition_execution(
                execution.record,
                "authorize",
                expected_version=expected_version,
            ),
        )

    @staticmethod
    def mark_dispatched(
        execution: StatefulExecution,
        *,
        expected_version: int,
    ) -> StatefulExecution:
        return replace(
            execution,
            record=transition_execution(
                execution.record,
                "dispatch",
                expected_version=expected_version,
            ),
        )

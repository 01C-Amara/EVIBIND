from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping


EXECUTION_GRAPH_VERSION = "evibind.execution_graph.v1"
EXECUTION_NODES = frozenset(
    {
        "created",
        "selecting",
        "awaiting_clarification",
        "materialized",
        "awaiting_confirmation",
        "authorized",
        "dispatched",
        "completed",
        "failed",
        "cancelled",
    }
)
TERMINAL_NODES = frozenset({"dispatched", "completed", "failed", "cancelled"})
_TRANSITIONS: Mapping[tuple[str, str], str] = {
    ("created", "compile"): "selecting",
    ("selecting", "need_input"): "awaiting_clarification",
    ("selecting", "materialize"): "materialized",
    ("selecting", "no_tool"): "completed",
    ("awaiting_clarification", "clarify"): "selecting",
    ("materialized", "require_confirmation"): "awaiting_confirmation",
    ("materialized", "authorize"): "authorized",
    ("awaiting_confirmation", "confirm"): "authorized",
    ("authorized", "dispatch"): "dispatched",
}


class ExecutionGraphError(ValueError):
    pass


@dataclass(frozen=True)
class ExecutionTransition:
    version: int
    event: str
    from_node: str
    to_node: str
    request_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "event": self.event,
            "from": self.from_node,
            "to": self.to_node,
            "request_digest": self.request_digest,
        }


@dataclass(frozen=True)
class ExecutionRecord:
    execution_id: str
    request_digest: str
    policy_epoch: str
    node: str = "created"
    version: int = 0
    tool_id: str | None = None
    missing: tuple[str, ...] = ()
    manifest_digest: str | None = None
    authorization_digest: str | None = None
    transitions: tuple[ExecutionTransition, ...] = ()

    def __post_init__(self) -> None:
        if not self.execution_id:
            raise ExecutionGraphError("execution_id is required")
        if not self.request_digest or not self.policy_epoch:
            raise ExecutionGraphError("request_digest and policy_epoch are required")
        if self.node not in EXECUTION_NODES:
            raise ExecutionGraphError(f"unsupported execution node: {self.node}")
        if self.version < 0 or self.version != len(self.transitions):
            raise ExecutionGraphError(
                "execution version must equal the transition count"
            )
        expected_node = "created"
        for expected_version, transition in enumerate(self.transitions, start=1):
            if transition.version != expected_version:
                raise ExecutionGraphError("execution transition version is invalid")
            if expected_node in TERMINAL_NODES:
                raise ExecutionGraphError(
                    "execution transition follows a terminal node"
                )
            if transition.from_node != expected_node:
                raise ExecutionGraphError("execution transition chain is invalid")
            if transition.event in {"fail", "cancel"}:
                expected_destination = (
                    "failed" if transition.event == "fail" else "cancelled"
                )
            else:
                expected_destination = _TRANSITIONS.get(
                    (transition.from_node, transition.event)
                )
            if expected_destination != transition.to_node:
                raise ExecutionGraphError("execution transition event is invalid")
            expected_node = transition.to_node
        if expected_node != self.node:
            raise ExecutionGraphError("execution node does not match its history")
        if (
            self.transitions
            and self.transitions[-1].request_digest != self.request_digest
        ):
            raise ExecutionGraphError(
                "execution request digest does not match its history"
            )
        if self.node == "awaiting_clarification" and (
            not self.tool_id or not self.missing
        ):
            raise ExecutionGraphError(
                "awaiting clarification requires a tool and missing destinations"
            )
        if self.node != "awaiting_clarification" and self.missing:
            raise ExecutionGraphError(
                "only awaiting clarification may retain missing destinations"
            )
        if self.node in {
            "materialized",
            "awaiting_confirmation",
            "authorized",
            "dispatched",
        } and (not self.tool_id or not self.manifest_digest):
            raise ExecutionGraphError(
                "post-materialization execution requires tool and manifest bindings"
            )
        if self.node in {"authorized", "dispatched"} and not self.authorization_digest:
            raise ExecutionGraphError(
                "authorized execution requires an authorization digest"
            )

    @property
    def terminal(self) -> bool:
        return self.node in TERMINAL_NODES

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": EXECUTION_GRAPH_VERSION,
            "execution_id": self.execution_id,
            "state_version": self.version,
            "node": self.node,
            "request_digest": self.request_digest,
            "policy_epoch": self.policy_epoch,
            "tool_id": self.tool_id,
            "missing": list(self.missing),
            "manifest_digest": self.manifest_digest,
            "authorization_digest": self.authorization_digest,
            "terminal": self.terminal,
            "transitions": [transition.to_dict() for transition in self.transitions],
        }


def transition_execution(
    record: ExecutionRecord,
    event: str,
    *,
    expected_version: int,
    request_digest: str | None = None,
    tool_id: str | None = None,
    missing: tuple[str, ...] = (),
    manifest_digest: str | None = None,
    authorization_digest: str | None = None,
) -> ExecutionRecord:
    if record.version != expected_version:
        raise ExecutionGraphError(
            f"stale execution version: expected {record.version}, got {expected_version}"
        )
    if record.terminal:
        raise ExecutionGraphError(
            f"terminal execution cannot transition: {record.node}"
        )

    if event in {"fail", "cancel"}:
        destination = "failed" if event == "fail" else "cancelled"
    else:
        destination = _TRANSITIONS.get((record.node, event))
        if destination is None:
            raise ExecutionGraphError(
                f"event {event!r} is invalid from execution node {record.node!r}"
            )

    next_request_digest = request_digest or record.request_digest
    next_tool_id = tool_id if tool_id is not None else record.tool_id
    next_missing = tuple(sorted(set(missing))) if missing else ()
    next_manifest = (
        manifest_digest if manifest_digest is not None else record.manifest_digest
    )
    next_authorization = (
        authorization_digest
        if authorization_digest is not None
        else record.authorization_digest
    )

    if event == "clarify":
        if next_request_digest == record.request_digest:
            raise ExecutionGraphError("clarification must compile a new request digest")
        next_tool_id = record.tool_id
        next_missing = ()
        next_manifest = None
        next_authorization = None
    elif next_request_digest != record.request_digest:
        raise ExecutionGraphError(
            "only clarification may change the execution request digest"
        )
    elif event == "need_input":
        if not next_tool_id or not next_missing:
            raise ExecutionGraphError(
                "need_input requires a tool and missing destinations"
            )
    elif event == "materialize":
        if not next_tool_id or not next_manifest:
            raise ExecutionGraphError("materialize requires a tool and manifest digest")
        next_missing = ()
    elif event == "confirm":
        if not next_authorization:
            raise ExecutionGraphError("confirm requires an authorization digest")
    elif event == "authorize" and not next_authorization:
        next_authorization = "not_required"

    next_version = record.version + 1
    transition = ExecutionTransition(
        version=next_version,
        event=event,
        from_node=record.node,
        to_node=destination,
        request_digest=next_request_digest,
    )
    return replace(
        record,
        request_digest=next_request_digest,
        node=destination,
        version=next_version,
        tool_id=next_tool_id,
        missing=next_missing,
        manifest_digest=next_manifest,
        authorization_digest=next_authorization,
        transitions=(*record.transitions, transition),
    )

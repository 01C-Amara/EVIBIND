from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .derivations import (
    Default,
    EnumValue,
    EvidenceContext,
    EvidenceDerivation,
    Span,
    StateRef,
    root_derivations,
)
from .evidence_types import AUTHORITY_BEARING, VALUE_CLASSES


POLICY_VERSION = "evibind.policy.v2"
AMBIGUITY_POLICIES = {"clarify", "reject", "confirm"}
CRITICALITIES = {"target", "control", "content", "effect"}


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class SlotPolicy:
    tool_id: str
    destination_scope: str
    evidence_type: str
    sources: frozenset[str]
    transforms: frozenset[str] = frozenset({"identity"})
    criticality: str = "target"
    value_class: str = AUTHORITY_BEARING
    ambiguity: str = "clarify"
    required: bool = True

    def __post_init__(self) -> None:
        if not self.tool_id:
            raise PolicyError("tool_id is required")
        if not self.destination_scope.startswith("/") or self.destination_scope == "/":
            raise PolicyError("destination_scope must be a non-root JSON Pointer")
        if not self.evidence_type:
            raise PolicyError("evidence_type is required")
        if not self.sources:
            raise PolicyError("at least one evidence source is required")
        if self.criticality not in CRITICALITIES:
            raise PolicyError(f"unsupported criticality: {self.criticality}")
        if self.value_class not in VALUE_CLASSES:
            raise PolicyError(f"unsupported value class: {self.value_class}")
        if self.ambiguity not in AMBIGUITY_POLICIES:
            raise PolicyError(f"unsupported ambiguity policy: {self.ambiguity}")
        if not self.transforms:
            raise PolicyError("at least one transform must be allowed")


@dataclass(frozen=True)
class ToolPolicy:
    tool_id: str
    slots: tuple[SlotPolicy, ...]
    policy_epoch: str = "1"
    contract_version: str = "1"

    def __post_init__(self) -> None:
        if not self.tool_id:
            raise PolicyError("tool_id is required")
        destinations = [slot.destination_scope for slot in self.slots]
        if len(destinations) != len(set(destinations)):
            raise PolicyError(f"duplicate destination policy for tool {self.tool_id}")
        if any(slot.tool_id != self.tool_id for slot in self.slots):
            raise PolicyError("every slot policy must match its enclosing tool_id")
        if not self.policy_epoch or not self.contract_version:
            raise PolicyError("policy_epoch and contract_version are required")

    def slot(self, destination_scope: str) -> SlotPolicy:
        matches = [
            slot for slot in self.slots if slot.destination_scope == destination_scope
        ]
        if len(matches) != 1:
            raise PolicyError(
                f"destination is not declared for {self.tool_id}: {destination_scope}"
            )
        return matches[0]

    @property
    def required_destinations(self) -> frozenset[str]:
        return frozenset(slot.destination_scope for slot in self.slots if slot.required)


@dataclass(frozen=True)
class PolicySet:
    tools: tuple[ToolPolicy, ...]

    def __post_init__(self) -> None:
        tool_ids = [tool.tool_id for tool in self.tools]
        if len(tool_ids) != len(set(tool_ids)):
            raise PolicyError("tool_id values must be unique")

    def tool(self, tool_id: str) -> ToolPolicy:
        matches = [tool for tool in self.tools if tool.tool_id == tool_id]
        if len(matches) != 1:
            raise PolicyError(f"tool policy not found: {tool_id}")
        return matches[0]

    def epochs(self) -> Mapping[str, str]:
        return {tool.tool_id: tool.policy_epoch for tool in self.tools}


def root_kind(root: Span | StateRef | Default | EnumValue) -> str:
    if isinstance(root, Span):
        return "span"
    if isinstance(root, StateRef):
        return "state_ref"
    if isinstance(root, Default):
        return "default"
    if isinstance(root, EnumValue):
        return "enum_value"
    raise PolicyError(f"unsupported evidence root: {type(root).__name__}")


def root_source(
    root: Span | StateRef | Default | EnumValue,
    context: EvidenceContext,
) -> str:
    if isinstance(root, Span):
        return context.message(root.message_id).source
    if isinstance(root, StateRef):
        return f"state.{root.namespace}"
    if isinstance(root, Default):
        return "schema.default"
    if isinstance(root, EnumValue):
        return "schema.enum"
    raise PolicyError(f"unsupported evidence root: {type(root).__name__}")


def origin_ok(
    derivation: EvidenceDerivation,
    context: EvidenceContext,
    slot_policy: SlotPolicy,
) -> tuple[bool, tuple[str, ...]]:
    actual = tuple(
        sorted({root_source(root, context) for root in root_derivations(derivation)})
    )
    unauthorized = tuple(
        source for source in actual if source not in slot_policy.sources
    )
    return not unauthorized, unauthorized

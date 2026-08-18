from __future__ import annotations

from copy import deepcopy
from typing import Any

from .families import FamilySpec


def merge_factors(grid: dict[str, Any], focal_values: dict[str, Any] | None = None) -> dict[str, Any]:
    factors: dict[str, Any] = {}
    factors.update(grid.get("pinned_factors", {}))
    factors.update(focal_values or {})
    if "repair_budget" not in factors and "repair_budget" in grid:
        factors["repair_budget"] = grid["repair_budget"]
    return factors


def surface_tool_name(canonical: str, alpha: str, ordinal: int = 0) -> str:
    if alpha == "fragmented":
        compact = canonical.replace("_", "")
        return f"fn{ordinal:02d}_{compact[:18]}"
    return canonical


def surface_slot_name(canonical: str, alpha: str, ordinal: int = 0) -> str:
    if alpha == "fragmented":
        pieces = canonical.split("_")
        stem = "".join(piece[:2] for piece in pieces)
        return f"x{ordinal:02d}_{stem}_value"
    return canonical


def required_slots_for_q(family: FamilySpec, q: int, *, task_kind: str = "call") -> list[str]:
    q = max(1, min(q, len(family.required_slots)))
    slots = list(family.required_slots[:q])
    if task_kind == "missing_info" and family.missing_slot not in slots:
        slots[-1] = family.missing_slot
    return slots


def enum_values_for_e(family: FamilySpec, e: int) -> list[str]:
    e = max(1, min(e, len(family.enum_values)))
    return list(family.enum_values[:e])


def slot_contract_annotations(slot: str, family: FamilySpec) -> tuple[str, str]:
    content_slots = {"body", "subject", "title", "issue"}
    identifier_slots = {"customer_id", "recipient", "owner"}
    normalizable_slots = {
        "date", "start_time", "end_time", "send_time", "quantity", "limit", "modified_after",
        "depart_date", "return_date", "traveler_count", "hour",
    }
    defaultable_slots = {"language", "sort_by", "order_by"}
    if slot == family.enum_slot:
        return "control", "enumerated"
    if slot in identifier_slots or slot.endswith("_id"):
        return "identifier", "referential"
    if slot in normalizable_slots or slot.endswith("_date") or slot.endswith("_time"):
        return "control", "normalizable"
    if slot in content_slots:
        return "content", "generative"
    if slot in defaultable_slots:
        return "defaultable", "defaultable"
    return "control", "extractive"


def _property_schema(slot: str, family: FamilySpec, factors: dict[str, Any], index: int) -> dict[str, Any]:
    role, resolution_type = slot_contract_annotations(slot, family)
    schema: dict[str, Any] = {
        "type": "string",
        "x-ir-name": slot,
        "x-tap-slot-role": role,
        "x-tap-resolution-type": resolution_type,
    }
    if slot in {"quantity", "limit", "traveler_count"}:
        schema["type"] = "integer"
    if slot in {"include_precipitation", "gift_wrap", "include_inactive", "hotel_needed"}:
        schema["type"] = "boolean"
    if slot == family.enum_slot:
        schema["enum"] = enum_values_for_e(family, int(factors.get("e", 6)))
    schema["description"] = f"Canonical argument {slot}."
    return schema


def _nest_parameters(properties: dict[str, Any], required: list[str], depth: int) -> dict[str, Any]:
    if depth <= 1:
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
    inner = _nest_parameters(properties, required, depth - 1)
    return {
        "type": "object",
        "properties": {"payload": inner},
        "required": ["payload"],
        "additionalProperties": False,
    }


def build_tool_catalog(
    family: FamilySpec,
    factors: dict[str, Any],
    *,
    task_kind: str = "call",
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    n_tools = int(factors.get("N", 16))
    alpha = str(factors.get("alpha", "aligned"))
    sigma = str(factors.get("sigma", "low"))
    q = int(factors.get("q", 3))
    depth = int(factors.get("d", 1))
    required_slots = required_slots_for_q(family, q, task_kind=task_kind)

    tools: list[dict[str, Any]] = []
    tool_aliases: dict[str, str] = {}
    arg_aliases: dict[str, str] = {}

    def add_tool(canonical_name: str, ordinal: int, *, correct: bool) -> None:
        surface_name = surface_tool_name(canonical_name, alpha, ordinal)
        tool_aliases[surface_name] = canonical_name
        if correct:
            slots = required_slots
            description = f"Use for {family.name} requests that explicitly ask to {canonical_name.replace('_', ' ')}."
        else:
            slots = required_slots[: max(1, min(2, len(required_slots)))]
            if sigma == "high":
                description = f"Near-duplicate {family.name} operation with overlapping request wording."
            else:
                description = f"Unrelated helper operation for {family.name} administration."
        properties: dict[str, Any] = {}
        required_surface: list[str] = []
        for index, slot in enumerate(slots):
            surface_slot = surface_slot_name(slot, alpha, index)
            arg_aliases[surface_slot] = slot
            properties[surface_slot] = _property_schema(slot, family, factors, index)
            required_surface.append(surface_slot)
        tools.append(
            {
                "name": surface_name,
                "canonical_name": canonical_name,
                "description": description,
                "parameters": _nest_parameters(properties, required_surface, depth),
            }
        )

    add_tool(family.call_tool, 0, correct=True)
    for ordinal, distractor in enumerate(family.distractor_tools, start=1):
        if len(tools) >= n_tools:
            break
        add_tool(distractor, ordinal, correct=False)

    ordinal = len(tools)
    while len(tools) < n_tools:
        if sigma == "high":
            synthetic = f"{family.call_tool}_variant_{ordinal:02d}"
        else:
            synthetic = f"{family.name}_utility_{ordinal:02d}"
        add_tool(synthetic, ordinal, correct=False)
        ordinal += 1

    return tools, tool_aliases, arg_aliases


def denest_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    current = arguments
    while isinstance(current, dict) and set(current.keys()) == {"payload"} and isinstance(current["payload"], dict):
        current = current["payload"]
    return deepcopy(current) if isinstance(current, dict) else {}


def nest_arguments(arguments: dict[str, Any], depth: int) -> dict[str, Any]:
    nested = deepcopy(arguments)
    for _ in range(max(1, depth) - 1):
        nested = {"payload": nested}
    return nested

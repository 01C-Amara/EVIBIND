from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from .eflrx import ACTION_RISK_THRESHOLD, RequestFn, _merge_metadata, _non_call, _tool_name
from .extractive_candidates import user_request_text
from .retrieve_pointer import validate_ranking_row
from .source_span_projection import (
    OMIT_SPAN_ID,
    SOURCE_SPAN_CERTIFICATE_VERSION,
    SOURCE_SPAN_PROJECTION_VERSION,
    action_fingerprint,
    materialize_span_proposal,
    slot_catalog,
    source_span_catalog,
)


SEMANTIC_SURFACE_VERSION = "tapbench.semantic_surface_projection.v1"
SEMANTIC_SURFACE_CONDITIONS = (
    "tap_r_surface_active_single",
    "tap_r_surface_active_consensus",
    "tap_r_surface_active_consensus_top1",
)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _ordered(values: list[Any], order: str) -> list[Any]:
    if order == "forward":
        return list(values)
    if order == "reverse":
        return list(reversed(values))
    raise ValueError(f"unknown semantic-surface order: {order}")


def _tool_catalog(
    retrieved_tools: list[dict[str, Any]],
    order: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tool in _ordered(retrieved_tools, order):
        parameters = tool.get("parameters", {})
        properties = (
            parameters.get("properties", {})
            if isinstance(parameters, dict)
            else {}
        )
        rows.append(
            {
                "tool": _tool_name(tool),
                "description": str(tool.get("description", "")),
                "argument_names": sorted(str(name) for name in properties),
            }
        )
    return rows


def _tool_schema(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool": {
                "type": "string",
                "enum": [str(row["tool"]) for row in catalog],
            }
        },
        "required": ["tool"],
        "additionalProperties": False,
    }


def _tool_messages(
    request_text: str,
    catalog: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "This benchmark item requires one function call. Select exactly "
                "one public tool name whose external effect matches the request. "
                "Do not choose from catalog position, infer arguments, or output "
                "reasoning. Return only the tool JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Request:\n"
                + request_text
                + "\nRetrieved public tools:\n"
                + json.dumps(catalog, separators=(",", ":"), ensure_ascii=False)
                + '\nReturn {"tool":"public.tool.name"}.'
            ),
        },
    ]


def _public_slots(
    tool: dict[str, Any], order: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows, mapping_by_id = slot_catalog(tool, order=order)
    by_surface: dict[str, dict[str, Any]] = {}
    for row in rows:
        slot_id = str(row["slot_id"])
        by_surface[str(row["name"])] = {
            **row,
            "canonical_name": str(mapping_by_id[slot_id]["name"]),
        }
    return [by_surface[str(row["name"])] for row in rows], by_surface


def _active_slot_schema(slots: list[dict[str, Any]]) -> dict[str, Any]:
    names = [str(row["name"]) for row in slots]
    return {
        "type": "object",
        "properties": {
            "active_slots": {
                "type": "array",
                "items": {"type": "string", "enum": names},
                "maxItems": len(names),
                "uniqueItems": True,
            }
        },
        "required": ["active_slots"],
        "additionalProperties": False,
    }


def _active_slot_messages(
    request_text: str,
    tool: dict[str, Any],
    slots: list[dict[str, Any]],
) -> list[dict[str, str]]:
    public_tool = {
        "tool": _tool_name(tool),
        "description": str(tool.get("description", "")),
        "slots": [
            {
                "name": row["name"],
                "description": row["description"],
                "required": row["required"],
            }
            for row in slots
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "Select only argument slots whose values are explicitly stated "
                "in the request. Most optional slots should be absent. Do not "
                "select a slot merely because the tool defines it, and do not "
                "infer defaults, dates, locations, or broad query text. Return "
                "an empty list when no slot has explicit evidence. No reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                "Request:\n"
                + request_text
                + "\nSelected tool and public slots:\n"
                + json.dumps(
                    public_tool, separators=(",", ":"), ensure_ascii=False
                )
                + '\nReturn {"active_slots":["slot_name"]}.'
            ),
        },
    ]


def validate_active_slots(
    raw: Any,
    *,
    tool: dict[str, Any],
) -> tuple[list[str] | None, dict[str, Any]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("active_slots"), list):
        return None, {"status": "active_slots_not_array"}
    values = raw["active_slots"]
    if not all(isinstance(value, str) for value in values):
        return None, {"status": "active_slot_not_string"}
    if len(values) != len(set(values)):
        return None, {"status": "duplicate_active_slot"}
    slots, _ = _public_slots(tool, "forward")
    public = {str(row["name"]): row for row in slots}
    unknown = sorted(set(values) - set(public))
    if unknown:
        return None, {"status": "unknown_active_slot", "unknown": unknown}
    required = {str(row["name"]) for row in slots if bool(row["required"])}
    added = sorted(required - set(values))
    active = sorted(set(values) | required)
    return active, {
        "status": "validated",
        "required_slots_added": added,
        "active_slot_count": len(active),
    }


def _unique_surface_spans(
    request_text: str,
    language: str,
    order: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    lattice = source_span_catalog(request_text, language)
    by_text: dict[str, dict[str, Any]] = {}
    for span in lattice["spans"]:
        by_text.setdefault(str(span["source_text"]), span)
    rows = sorted(by_text.values(), key=lambda row: str(row["span_id"]))
    rows = _ordered(rows, order)
    return rows, by_text, lattice


def _binding_schema(
    active_slots: list[str],
    surface_spans: list[dict[str, Any]],
) -> dict[str, Any]:
    values = [str(row["source_text"]) for row in surface_spans]
    if not values:
        raise ValueError("surface binding requires at least one source span")
    properties = {
        slot: {"type": "string", "enum": values} for slot in active_slots
    }
    return {
        "type": "object",
        "properties": {
            "bindings": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            }
        },
        "required": ["bindings"],
        "additionalProperties": False,
    }


def _binding_messages(
    request_text: str,
    tool: dict[str, Any],
    active_slots: list[str],
    slot_details: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    active = [
        {
            "name": slot,
            "description": slot_details[slot]["description"],
        }
        for slot in active_slots
    ]
    return [
        {
            "role": "system",
            "content": (
                "For each active slot, select the exact request substring that "
                "is its value. The constrained decoder permits only substrings "
                "from the request. Exclude question and intent words unless the "
                "slot itself is a free-form query. Do not translate, normalize, "
                "infer, or explain. Return only the bindings JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Request:\n"
                + request_text
                + "\nSelected tool: "
                + _tool_name(tool)
                + "\nActive public slots:\n"
                + json.dumps(active, separators=(",", ":"), ensure_ascii=False)
                + '\nReturn {"bindings":{"slot_name":"exact request text"}}.'
            ),
        },
    ]


def materialize_surface_bindings(
    proposal: Any,
    *,
    active_slots: list[str],
    selected_tool: str,
    tool: dict[str, Any],
    tools: list[dict[str, Any]],
    request_text: str,
    language: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not isinstance(proposal, dict) or not isinstance(
        proposal.get("bindings"), dict
    ):
        return None, {"status": "surface_bindings_not_object"}
    bindings = proposal["bindings"]
    if set(bindings) != set(active_slots):
        return None, {
            "status": "surface_binding_slot_set_mismatch",
            "missing": sorted(set(active_slots) - set(bindings)),
            "extra": sorted(set(bindings) - set(active_slots)),
        }
    slots, by_surface = _public_slots(tool, "forward")
    _, spans_by_text, lattice = _unique_surface_spans(
        request_text, language, "forward"
    )
    pointer_bindings: dict[str, str] = {
        str(row["slot_id"]): OMIT_SPAN_ID for row in slots
    }
    selected_surface_values: dict[str, str] = {}
    for surface_slot in active_slots:
        value = bindings.get(surface_slot)
        if not isinstance(value, str):
            return None, {
                "status": "surface_value_not_string",
                "slot": surface_slot,
            }
        span = spans_by_text.get(value)
        if span is None:
            return None, {
                "status": "surface_value_not_in_request_lattice",
                "slot": surface_slot,
            }
        slot_id = str(by_surface[surface_slot]["slot_id"])
        pointer_bindings[slot_id] = str(span["span_id"])
        selected_surface_values[surface_slot] = value
    action, metadata = materialize_span_proposal(
        {"bindings": pointer_bindings},
        selected_tool=selected_tool,
        tool=tool,
        tools=tools,
        request_text=request_text,
        language=language,
    )
    if action is None:
        return None, metadata
    return action, {
        **metadata,
        "status": "surface_materialized",
        "semantic_surface_version": SEMANTIC_SURFACE_VERSION,
        "selected_surface_values": selected_surface_values,
        "unique_source_surface_count": len(spans_by_text),
        "span_catalog_sha256": lattice["catalog_sha256"],
        "no_unconstrained_action_critical_tokens": True,
    }


def _fail_closed(
    metadata: dict[str, Any],
    calls: list[dict[str, Any]],
    *,
    stage: str,
    reason: str,
    started: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if calls:
        metadata.update(_merge_metadata(calls))
    else:
        metadata.update({"generation_calls": 0, "finish_reason": "not_applicable"})
    metadata.update(
        {
            "controller_stage_failure": stage,
            "action_risk_score": 1.0,
            "risk_gate_passed": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return _non_call("refuse", reason), metadata


def run_semantic_surface_resolution(
    *,
    case_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    language: str,
    ranking_row: dict[str, Any],
    ranking_artifact_sha256: str,
    endpoint: str,
    condition: str,
    max_tokens: int,
    seed: int,
    request_fn: RequestFn,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if condition not in SEMANTIC_SURFACE_CONDITIONS:
        raise ValueError(f"unknown semantic-surface condition: {condition}")
    started = time.perf_counter()
    text = user_request_text(messages)
    ranking = validate_ranking_row(
        ranking_row,
        case_id=case_id,
        request_text=text,
        tools=tools,
    )
    by_name = {_tool_name(tool): tool for tool in tools}
    ranking_names = [str(row["tool"]) for row in ranking]
    retrieved = [by_name[name] for name in ranking_names]
    strict = condition != "tap_r_surface_active_single"
    metadata: dict[str, Any] = {
        "semantic_surface_version": SEMANTIC_SURFACE_VERSION,
        "retriever_version": ranking_row["retriever_version"],
        "retriever_model_id": ranking_row["retriever_model_id"],
        "retriever_revision": ranking_row["retriever_revision"],
        "retriever_serialization_arm": ranking_row["serialization_arm"],
        "retriever_k": ranking_row["k"],
        "ranking_sha256": ranking_row["ranking_sha256"],
        "ranking_artifact_sha256": ranking_artifact_sha256,
        "retrieved_tools": ranking_names,
        "retrieval_top1": ranking_names[0],
        "source_span_projection_version": SOURCE_SPAN_PROJECTION_VERSION,
        "source_span_certificate_version": SOURCE_SPAN_CERTIFICATE_VERSION,
        "call_only_tool_election": True,
        "no_call_election_option": False,
        "semantic_tool_labels": True,
        "semantic_slot_labels": True,
        "source_surface_value_labels": True,
        "tool_elections": [],
        "active_slot_selections": [],
        "surface_proposals": [],
        "generation_calls": 0,
        "action_risk_threshold": ACTION_RISK_THRESHOLD,
    }
    calls: list[dict[str, Any]] = []

    view_orders = ("forward", "reverse") if strict else ("forward",)
    selected_tools: list[str] = []
    for index, order in enumerate(view_orders):
        catalog = _tool_catalog(retrieved, order)
        raw, response = request_fn(
            endpoint,
            _tool_messages(text, catalog),
            response_schema=_tool_schema(catalog),
            max_tokens=min(max_tokens, 96),
            temperature=0.0,
            seed=seed + index * 100003,
        )
        calls.append(response)
        selected = str(raw.get("tool") or "") if isinstance(raw, dict) else ""
        selected_tools.append(selected)
        metadata["tool_elections"].append(
            {
                "order": order,
                "selected": selected,
                "catalog_sha256": _json_sha256(catalog),
            }
        )
    if any(value not in by_name for value in selected_tools) or len(
        set(selected_tools)
    ) != 1:
        metadata["tool_agreement"] = False
        return _fail_closed(
            metadata,
            calls,
            stage="semantic_tool_disagreement",
            reason="semantic tool views disagreed",
            started=started,
        )
    selected_tool = selected_tools[0]
    metadata["tool_agreement"] = True
    metadata["selected_tool"] = selected_tool
    metadata["retriever_election_agreement"] = selected_tool == ranking_names[0]
    if (
        condition == "tap_r_surface_active_consensus_top1"
        and selected_tool != ranking_names[0]
    ):
        return _fail_closed(
            metadata,
            calls,
            stage="semantic_retriever_tool_disagreement",
            reason="semantic tool election disagreed with retriever rank one",
            started=started,
        )

    tool = by_name[selected_tool]
    forward_slots, _ = _public_slots(tool, "forward")
    active_views: list[list[str]] = []
    if not forward_slots:
        # An empty enum is not portable across grammar engines. A tool with no
        # public slots has one admissible active-slot set, so resolve it without
        # another model call.
        active_views = [[]]
        metadata["active_slot_selections"].append(
            {
                "order": "deterministic",
                "status": "no_public_slots",
                "active_slots": [],
                "required_slots_added": [],
            }
        )
        metadata["active_slot_views_skipped"] = True
    else:
        for index, order in enumerate(view_orders):
            slots, _ = _public_slots(tool, order)
            raw, response = request_fn(
                endpoint,
                _active_slot_messages(text, tool, slots),
                response_schema=_active_slot_schema(slots),
                max_tokens=min(max_tokens, 160),
                temperature=0.0,
                seed=seed + 300001 + index * 100003,
            )
            calls.append(response)
            active, validation = validate_active_slots(raw, tool=tool)
            metadata["active_slot_selections"].append(
                {
                    "order": order,
                    "status": validation.get("status"),
                    "active_slots": active,
                    "required_slots_added": validation.get(
                        "required_slots_added", []
                    ),
                }
            )
            if active is None:
                return _fail_closed(
                    metadata,
                    calls,
                    stage=f"active_slot_{validation.get('status')}",
                    reason="active-slot selection failed validation",
                    started=started,
                )
            active_views.append(active)
    active_hashes = {_json_sha256(value) for value in active_views}
    if len(active_hashes) != 1:
        metadata["active_slot_agreement"] = False
        return _fail_closed(
            metadata,
            calls,
            stage="active_slot_disagreement",
            reason="counterbalanced active-slot selections disagreed",
            started=started,
        )
    active_slots = active_views[0]
    metadata["active_slot_agreement"] = True
    metadata["active_slots"] = active_slots
    metadata["active_slot_count"] = len(active_slots)

    if not active_slots:
        pointer_slots, _ = slot_catalog(tool)
        pointer_bindings = {
            str(row["slot_id"]): OMIT_SPAN_ID for row in pointer_slots
        }
        action, materialization = materialize_span_proposal(
            {"bindings": pointer_bindings},
            selected_tool=selected_tool,
            tool=tool,
            tools=tools,
            request_text=text,
            language=language,
        )
        if action is None:
            return _fail_closed(
                metadata,
                calls,
                stage=f"empty_active_{materialization.get('status')}",
                reason="empty active-slot action failed public validation",
                started=started,
            )
        actions = [action]
        materializations = [
            {
                **materialization,
                "selected_surface_values": {},
                "no_unconstrained_action_critical_tokens": True,
            }
        ]
        metadata["surface_binding_views_skipped"] = True
    else:
        _, slot_details = _public_slots(tool, "forward")
        actions = []
        materializations = []
        for index, order in enumerate(view_orders):
            surfaces, _, lattice = _unique_surface_spans(text, language, order)
            schema = _binding_schema(active_slots, surfaces)
            raw, response = request_fn(
                endpoint,
                _binding_messages(text, tool, active_slots, slot_details),
                response_schema=schema,
                max_tokens=max_tokens,
                temperature=0.0,
                seed=seed + 700001 + index * 100003,
            )
            calls.append(response)
            action, materialization = materialize_surface_bindings(
                raw,
                active_slots=active_slots,
                selected_tool=selected_tool,
                tool=tool,
                tools=tools,
                request_text=text,
                language=language,
            )
            metadata["surface_proposals"].append(
                {
                    "order": order,
                    "proposal_sha256": _json_sha256(raw),
                    "schema_sha256": _json_sha256(schema),
                    "status": materialization.get("status"),
                    "action_sha256": action_fingerprint(action) if action else None,
                    "source_surface_count": len(surfaces),
                    "span_catalog_sha256": lattice["catalog_sha256"],
                }
            )
            if action is None:
                return _fail_closed(
                    metadata,
                    calls,
                    stage=f"surface_binding_{materialization.get('status')}",
                    reason="source-surface binding failed validation",
                    started=started,
                )
            actions.append(action)
            materializations.append(materialization)

    fingerprints = [action_fingerprint(action) for action in actions]
    if len(set(fingerprints)) != 1:
        metadata["surface_action_agreement"] = False
        return _fail_closed(
            metadata,
            calls,
            stage="surface_action_disagreement",
            reason="counterbalanced source-surface actions disagreed",
            started=started,
        )
    risk = 0.05 if not strict else 0.02
    if condition == "tap_r_surface_active_consensus_top1":
        risk = 0.01
    materialization = materializations[0]
    metadata.update(_merge_metadata(calls))
    metadata.update(
        {
            "controller_stage_failure": None,
            "surface_action_agreement": True,
            "proposal_admitted": True,
            "evidence_certificates": materialization["certificates"],
            "selected_span_ids": materialization["selected_span_ids"],
            "selected_surface_values": materialization.get(
                "selected_surface_values", {}
            ),
            "span_catalog_sha256": materialization["span_catalog_sha256"],
            "slot_catalog_sha256": materialization["slot_catalog_sha256"],
            "materialized_action_sha256": fingerprints[0],
            "no_unconstrained_action_critical_tokens": True,
            "action_risk_score": risk,
            "risk_gate_passed": risk <= ACTION_RISK_THRESHOLD,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return actions[0], metadata

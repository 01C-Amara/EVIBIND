from __future__ import annotations

import json
import time
from typing import Any

from .eflrx import ACTION_RISK_THRESHOLD, RequestFn, _merge_metadata, _non_call, _tool_name
from .extractive_candidates import user_request_text
from .retrieve_pointer import validate_ranking_row
from .semantic_surface_projection import (
    SEMANTIC_SURFACE_VERSION,
    _json_sha256,
    _public_slots,
    _tool_catalog,
    _tool_messages,
    _tool_schema,
    materialize_surface_bindings,
)
from .source_span_projection import (
    SOURCE_SPAN_CERTIFICATE_VERSION,
    SOURCE_SPAN_PROJECTION_VERSION,
    action_fingerprint,
    source_span_catalog,
)


SLOTWISE_SURFACE_VERSION = "tapbench.slotwise_surface_projection.v1"
SLOTWISE_SURFACE_CONDITIONS = (
    "tap_r_slotwise_surface_single",
    "tap_r_slotwise_surface_consensus",
    "tap_r_slotwise_surface_consensus_top1",
)


def _unit_index(unit_id: str) -> int:
    return int(unit_id.rsplit("_", 1)[1])


def minimal_surface_catalog(
    request_text: str,
    language: str,
    order: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lattice = source_span_catalog(request_text, language)
    by_text: dict[str, dict[str, Any]] = {}
    for span in lattice["spans"]:
        by_text.setdefault(str(span["source_text"]), span)
    rows = sorted(
        by_text.values(),
        key=lambda row: (
            _unit_index(str(row["end_unit_id"]))
            - _unit_index(str(row["start_unit_id"]))
            + 1,
            int(row["source_span"][0]),
            int(row["source_span"][1]),
            str(row["source_text"]),
        ),
    )
    if order == "reverse":
        rows.reverse()
    elif order != "forward":
        raise ValueError(f"unknown slotwise surface order: {order}")
    return rows, lattice


def slotwise_value_schema(
    slot: dict[str, Any],
    surfaces: list[dict[str, Any]],
) -> dict[str, Any]:
    values: list[Any] = [str(row["source_text"]) for row in surfaces]
    if not bool(slot["required"]):
        values = [None, *values]
    if not values:
        raise ValueError("slotwise value schema has no admissible values")
    return {
        "type": "object",
        "properties": {"value": {"enum": values}},
        "required": ["value"],
        "additionalProperties": False,
    }


def _fixed_examples() -> str:
    return (
        "Minimal examples (the date is absent in each request):\n"
        'EN request "Send the report to Amira": person -> "Amira"; date -> null.\n'
        'FA request "آب و هوای شیراز را بگو": location -> "شیراز"; date -> null.\n'
        'JA request "京都の天気を教えて": location -> "京都"; date -> null.'
    )


def _slot_messages(
    request_text: str,
    tool: dict[str, Any],
    slot: dict[str, Any],
    surfaces: list[dict[str, Any]],
) -> list[dict[str, str]]:
    candidates = [str(row["source_text"]) for row in surfaces]
    return [
        {
            "role": "system",
            "content": (
                "Resolve exactly one tool argument. Return the shortest exact "
                "request substring that supplies this slot's value. Do not "
                "return the whole request, an intent phrase, a translated "
                "value, a default, or a value for another slot. Return JSON "
                "null when this optional slot is not explicitly stated. "
                "Candidate order carries no meaning. Output only the value "
                "JSON and no reasoning.\n" + _fixed_examples()
            ),
        },
        {
            "role": "user",
            "content": (
                "Request:\n"
                + request_text
                + "\nSelected tool:\n"
                + json.dumps(
                    {
                        "tool": _tool_name(tool),
                        "description": str(tool.get("description", "")),
                    },
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\nOne public slot:\n"
                + json.dumps(
                    {
                        "name": slot["name"],
                        "description": slot["description"],
                        "required": slot["required"],
                    },
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\nAllowed exact source surfaces (shorter candidates are "
                "listed first in the forward view):\n"
                + json.dumps(candidates, separators=(",", ":"), ensure_ascii=False)
                + '\nReturn {"value":"exact request substring"} or {"value":null}.'
            ),
        },
    ]


def validate_slotwise_value(
    raw: Any,
    *,
    slot: dict[str, Any],
    surfaces: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any]]:
    if not isinstance(raw, dict) or set(raw) != {"value"}:
        return None, {"status": "slotwise_value_shape_invalid"}
    value = raw["value"]
    if value is None:
        if bool(slot["required"]):
            return None, {"status": "required_slot_null"}
        return None, {"status": "validated_null", "is_null": True}
    if not isinstance(value, str):
        return None, {"status": "slotwise_value_not_string_or_null"}
    allowed = {str(row["source_text"]) for row in surfaces}
    if value not in allowed:
        return None, {"status": "slotwise_value_not_in_source_lattice"}
    return value, {"status": "validated_surface", "is_null": False}


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


def run_slotwise_surface_resolution(
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
    if condition not in SLOTWISE_SURFACE_CONDITIONS:
        raise ValueError(f"unknown slotwise-surface condition: {condition}")
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
    strict = condition != "tap_r_slotwise_surface_single"
    view_orders = ("forward", "reverse") if strict else ("forward",)
    metadata: dict[str, Any] = {
        "slotwise_surface_version": SLOTWISE_SURFACE_VERSION,
        "semantic_surface_materializer_version": SEMANTIC_SURFACE_VERSION,
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
        "slotwise_independent_generation": True,
        "null_optional_absence": True,
        "candidate_list_visible_in_prompt": True,
        "candidate_order_semantically_irrelevant": True,
        "tool_elections": [],
        "slotwise_views": [],
        "generation_calls": 0,
        "action_risk_threshold": ACTION_RISK_THRESHOLD,
    }
    calls: list[dict[str, Any]] = []

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
            stage="slotwise_tool_disagreement",
            reason="semantic tool views disagreed",
            started=started,
        )
    selected_tool = selected_tools[0]
    metadata["tool_agreement"] = True
    metadata["selected_tool"] = selected_tool
    metadata["retriever_election_agreement"] = selected_tool == ranking_names[0]
    if (
        condition == "tap_r_slotwise_surface_consensus_top1"
        and selected_tool != ranking_names[0]
    ):
        return _fail_closed(
            metadata,
            calls,
            stage="slotwise_retriever_tool_disagreement",
            reason="semantic tool election disagreed with retriever rank one",
            started=started,
        )

    tool = by_name[selected_tool]
    actions: list[dict[str, Any]] = []
    materializations: list[dict[str, Any]] = []
    selected_by_view: list[dict[str, str | None]] = []
    for view_index, order in enumerate(view_orders):
        slots, _ = _public_slots(tool, order)
        surfaces, lattice = minimal_surface_catalog(text, language, order)
        selections: dict[str, str | None] = {}
        slot_records: list[dict[str, Any]] = []
        for slot_index, slot in enumerate(slots):
            schema = slotwise_value_schema(slot, surfaces)
            raw, response = request_fn(
                endpoint,
                _slot_messages(text, tool, slot, surfaces),
                response_schema=schema,
                max_tokens=min(max_tokens, 128),
                temperature=0.0,
                seed=(
                    seed
                    + 300001
                    + view_index * 100003
                    + slot_index * 1009
                ),
            )
            calls.append(response)
            value, validation = validate_slotwise_value(
                raw,
                slot=slot,
                surfaces=surfaces,
            )
            if validation["status"] not in {
                "validated_null",
                "validated_surface",
            }:
                return _fail_closed(
                    metadata,
                    calls,
                    stage=f"slotwise_{validation['status']}",
                    reason="slotwise value failed finite-source validation",
                    started=started,
                )
            selections[str(slot["name"])] = value
            slot_records.append(
                {
                    "slot": slot["name"],
                    "required": slot["required"],
                    "is_null": value is None,
                    "selected_surface": value,
                    "schema_sha256": _json_sha256(schema),
                }
            )
        active = sorted(
            slot for slot, value in selections.items() if value is not None
        )
        proposal = {
            "bindings": {
                slot: value
                for slot, value in selections.items()
                if value is not None
            }
        }
        action, materialization = materialize_surface_bindings(
            proposal,
            active_slots=active,
            selected_tool=selected_tool,
            tool=tool,
            tools=tools,
            request_text=text,
            language=language,
        )
        metadata["slotwise_views"].append(
            {
                "order": order,
                "slot_records": slot_records,
                "null_count": sum(value is None for value in selections.values()),
                "surface_count": len(surfaces),
                "span_catalog_sha256": lattice["catalog_sha256"],
                "action_sha256": action_fingerprint(action) if action else None,
                "status": materialization.get("status"),
            }
        )
        if action is None:
            return _fail_closed(
                metadata,
                calls,
                stage=f"slotwise_materialize_{materialization.get('status')}",
                reason="slotwise source surfaces failed public materialization",
                started=started,
            )
        actions.append(action)
        materializations.append(materialization)
        selected_by_view.append(selections)

    fingerprints = [action_fingerprint(action) for action in actions]
    if len(set(fingerprints)) != 1:
        metadata["slotwise_action_agreement"] = False
        return _fail_closed(
            metadata,
            calls,
            stage="slotwise_action_disagreement",
            reason="counterbalanced slotwise actions disagreed",
            started=started,
        )
    risk = 0.05 if not strict else 0.02
    if condition == "tap_r_slotwise_surface_consensus_top1":
        risk = 0.01
    materialization = materializations[0]
    metadata.update(_merge_metadata(calls))
    metadata.update(
        {
            "controller_stage_failure": None,
            "slotwise_action_agreement": True,
            "proposal_admitted": True,
            "slotwise_selections": selected_by_view[0],
            "slotwise_null_count": sum(
                value is None for value in selected_by_view[0].values()
            ),
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

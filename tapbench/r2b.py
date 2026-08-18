from __future__ import annotations

import hashlib
import json
import time
import math
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from . import NORMALIZER_VERSION, SCORER_VERSION, VALIDATOR_VERSION
from .bestof import select_candidate
from .contract_solver import CONTRACT_SOLVER_VERSION, resolve_pointer_contract
from .deployable_resolution import (
    DEPLOYABLE_RESOLUTION_VERSION,
    augment_lattice_with_typed_programs,
)
from .evidence_contract import build_candidate_lattice, request_text
from .io import read_jsonl, write_jsonl, write_yaml
from .r2_model_runner import (
    R2A_CHAT_PARSER,
    R2A_GRAMMAR_ENGINE,
    _coerce_pointer,
    _materialize_unrestricted,
    _pointer_catalog,
    _request_schema_json,
    _sanitize_pointer,
)
from .r2b_families import R2B_FAMILIES
from .runner import render_chat_messages
from .scoring import score_predictions
from .slot_errors import slot_errors_for_predictions
from .tapr import contract_validator_error
from .thinking import prediction_has_thinking_marker
from .tier_b_verifier import FrozenTierBVerifier
from .typed_evidence_programs import TEP_VERSION


R2B_CASE_VERSION = "tapbench.r2b_case.v1"
R2B_RUNNER_VERSION = "tapbench.r2b_model_runner.v2"
R2B_ACTION_SCHEMA_VERSION = "tapbench.r2b_action_schema.v2"
R2B_REPORT_VERSION = "tapbench.r2b_report.v1"
R2B_GRID_ID = "R2B_deployable_open_world"
R2B_MUTATIONS = (
    "unseen_tool_names",
    "unseen_argument_names",
    "enum_alias_shift",
    "near_duplicate_distractor_tools",
    "reordered_schema",
    "state_version_change",
)
R2B_TASK_KINDS = ("call", "missing_info", "no_tool", "direct_answer")
R2B_CONDITIONS = (
    "prompt_few_shot",
    "constrained_abstention",
    "best_of_2",
    "best_of_4",
    "validator_feedback_regeneration",
    "local_llm_slot_regeneration",
    "process_scored_search",
    "tap_r_literal_evidence",
    "tap_r_tep_tier_a",
    "tap_r_tep_tier_ab",
    "tap_r_without_global_contract",
    "tap_r_full",
)


def _values(family: Any, index: int) -> dict[str, Any]:
    values: dict[str, Any] = {
        "client": f"Client {index}",
        "date": f"2026-09-{index % 24 + 1:02d}",
        "time": f"{9 + index % 8:02d}:30",
        "service": f"consultation-{index % 5}",
        "payee": f"Vendor {index}",
        "amount": float(100 + index * 3),
        "currency": ("GBP", "EUR", "USD")[index % 3],
        "reference": f"INV-{8000 + index}",
        "sku": f"SKU-{4000 + index}",
        "quantity": index % 8 + 1,
        "warehouse": f"WH-{index % 4 + 1}",
        "expires_at": f"2026-10-{index % 24 + 1:02d}T18:00:00",
        "workspace": f"workspace-{index % 4}",
        "recipient": f"member-{index}",
        "message": f"Please review release item {index}",
        "send_at": f"2026-09-{index % 24 + 1:02d}T15:00:00",
        "title": f"Review milestone {index}",
        "due_date": f"2026-10-{index % 24 + 1:02d}",
        "assignee": f"owner-{index % 7}",
        "project": f"project-{index % 5}",
        "measurement": ("heart_rate", "temperature", "blood_pressure")[index % 3],
        "value": float(60 + index % 40),
        "unit": ("bpm", "celsius", "mmHg")[index % 3],
        "timestamp": f"2026-09-{index % 24 + 1:02d}T08:00:00",
        "name": f"Focus mix {index}",
        "genre": ("jazz", "ambient", "rock", "classical")[index % 4],
        "tracks": f"Track {index} and Track {index + 1}",
        "owner": f"listener-{index % 6}",
        "origin": f"Depot {index % 4}",
        "destination": f"Address {index}",
        "package_id": f"PKG-{7000 + index}",
        "delivery_date": f"2026-10-{index % 24 + 1:02d}",
    }
    values[family.enum_slot] = family.enum_values[index % len(family.enum_values)]
    return {slot: values[slot] for slot in family.required_slots}


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    return "string"


def _property(family: Any, slot: str, value: Any) -> dict[str, Any]:
    enumerated = slot == family.enum_slot
    role = (
        "identifier"
        if any(token in slot for token in ("id", "sku", "reference", "payee", "recipient", "assignee", "owner"))
        else "control"
    )
    prop: dict[str, Any] = {
        "type": _json_type(value),
        "description": f"Canonical {slot} for this operation.",
        "x-ir-name": slot,
        "x-tap-slot-role": role,
        "x-tap-resolution-type": "enumerated" if enumerated else "normalizable",
        "x-tap-criticality": "high",
    }
    if enumerated:
        prop["enum"] = list(family.enum_values)
    return prop


def _catalog(
    family: Any,
    family_index: int,
    values: dict[str, Any],
    mutation: str,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    canonical_names = (family.call_tool, *family.distractor_tools)
    surface_names = list(canonical_names)
    if mutation == "unseen_tool_names":
        surface_names = [f"operation_{family_index}_{index}" for index in range(len(canonical_names))]
    canonical_slots = list(family.required_slots)
    surface_slots = list(canonical_slots)
    if mutation == "unseen_argument_names":
        surface_slots = [f"field_{index}" for index in range(len(canonical_slots))]
    argument_aliases = dict(zip(surface_slots, canonical_slots, strict=True))
    if mutation == "reordered_schema":
        surface_slots.reverse()
    tools = []
    aliases = {}
    for tool_index, (surface_name, canonical_name) in enumerate(
        zip(surface_names, canonical_names, strict=True)
    ):
        aliases[surface_name] = canonical_name
        properties = {}
        for surface_slot in surface_slots:
            canonical_slot = argument_aliases[surface_slot]
            prop = _property(family, canonical_slot, values[canonical_slot])
            prop["x-tap-extraction-cue"] = surface_slot
            if mutation == "enum_alias_shift" and canonical_slot == family.enum_slot:
                prop["description"] += " Emit the canonical enum label, not a synonym."
            properties[surface_slot] = prop
        target = tool_index == 0
        description = (
            f"Perform the requested {family.name} creation or execution operation."
            if target
            else f"A different {family.name} operation: {canonical_name.replace('_', ' ')}."
        )
        if mutation == "near_duplicate_distractor_tools" and not target:
            description = (
                f"Closely related {family.name} operation; use only to "
                f"{canonical_name.replace('_', ' ')}."
            )
        tools.append(
            {
                "name": surface_name,
                "canonical_name": canonical_name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(surface_slots),
                    "additionalProperties": False,
                },
            }
        )
    return tools, aliases, argument_aliases


def _request(
    family: Any,
    values: dict[str, Any],
    task_kind: str,
    mutation: str,
) -> tuple[str, dict[str, Any]]:
    shown = deepcopy(values)
    dialogue_state: dict[str, Any] = {}
    if task_kind == "call":
        request = family.request_template.format(**shown)
        if mutation == "state_version_change":
            slot = family.required_slots[0]
            dialogue_state[slot] = {"value": values[slot], "version": 3}
            request = request.replace(str(values[slot]), f"the currently verified {slot}")
    elif task_kind == "missing_info":
        shown[family.missing_slot] = "[not provided]"
        request = (
            family.request_template.format(**shown)
            + f" Ask only for {family.missing_slot}; do not guess it."
        )
    elif task_kind == "no_tool":
        request = (
            f"No {family.name.replace('_', ' ')} action is requested. "
            f"{family.no_tool_request}"
        )
    elif task_kind == "direct_answer":
        request = f"Answer directly without calling a tool: {family.no_tool_request}"
    else:
        raise ValueError(task_kind)
    return request, dialogue_state


def _gold(family: Any, values: dict[str, Any], task_kind: str) -> dict[str, Any]:
    if task_kind == "call":
        return {
            "mode": "call",
            "tool": family.call_tool,
            "arguments": values,
            "payload": {},
        }
    if task_kind == "missing_info":
        return {
            "mode": "clarify",
            "tool": None,
            "arguments": {},
            "payload": {"missing_slots": [family.missing_slot]},
        }
    if task_kind == "no_tool":
        return {
            "mode": "no_tool",
            "tool": None,
            "arguments": {},
            "payload": {"reason": "no available tool action requested"},
        }
    return {
        "mode": "direct_answer",
        "tool": None,
        "arguments": {},
        "payload": {"answer": "direct answer requested"},
    }


def generate_r2b_cases(*, scope: str = "pilot") -> list[dict[str, Any]]:
    if scope not in {"smoke", "pilot", "full"}:
        raise ValueError("R2-B scope must be smoke, pilot, or full")
    variants = 32 if scope == "full" else 8 if scope == "pilot" else 1
    rows = []
    for family_index, family in enumerate(R2B_FAMILIES):
        for variant in range(variants):
            global_index = family_index * 32 + variant
            task_kind = R2B_TASK_KINDS[
                (variant + family_index) % len(R2B_TASK_KINDS)
            ]
            mutation = R2B_MUTATIONS[
                (variant + 2 * family_index) % len(R2B_MUTATIONS)
            ]
            values = _values(family, global_index)
            tools, tool_aliases, argument_aliases = _catalog(
                family, family_index, values, mutation
            )
            request, dialogue_state = _request(
                family, values, task_kind, mutation
            )
            if task_kind in {"call", "missing_info"}:
                excluded = set()
                if task_kind == "missing_info":
                    excluded.add(family.missing_slot)
                if mutation == "state_version_change":
                    excluded.add(family.required_slots[0])
                fields = [
                    f"{surface}={values[canonical]}"
                    for surface, canonical in argument_aliases.items()
                    if canonical not in excluded
                ]
                if fields:
                    request += "\nEvidence fields: " + "; ".join(fields)
                    request += ";"
                    case_messages = [
                        {
                            "role": "system",
                            "content": (
                                "Return exactly one Action IR object. Never invent "
                                "a required value or call a related but incorrect operation."
                            ),
                        },
                        {"role": "user", "content": request},
                    ]
                else:
                    case_messages = None
            else:
                case_messages = None
            derivable = dict(values)
            if task_kind == "missing_info":
                derivable.pop(family.missing_slot, None)
            rows.append(
                {
                    "schema_version": R2B_CASE_VERSION,
                    "case_id": f"r2b_{family.name}_{variant:02d}",
                    "hypothesis_grid_id": R2B_GRID_ID,
                    "hypothesis": "R2B",
                    "split": scope,
                    "family": family.name,
                    "task_kind": task_kind,
                    "factors": {
                        "task_kind": task_kind,
                        "catalog_mutation": mutation,
                        "repair_budget": 2,
                        "variant": variant,
                    },
                    "messages": case_messages or [
                        {
                            "role": "system",
                            "content": (
                                "Return exactly one Action IR object. Never invent "
                                "a required value or call a related but incorrect operation."
                            ),
                        },
                        {"role": "user", "content": request},
                    ],
                    "tools": tools,
                    "tool_aliases": tool_aliases,
                    "argument_aliases": argument_aliases,
                    "dialogue_state": dialogue_state,
                    "reference_context": {
                        "reference_date": "2026-07-13",
                        "timezone": "Europe/London",
                        "action_risk_budget": 0.05,
                    },
                    "gold_action": _gold(family, values, task_kind),
                    "derivable_values": derivable,
                    "r2b_oracle": {
                        "catalog_family": family.name,
                        "mutation": mutation,
                        "variant": variant,
                    },
                    "metadata": {
                        "backend_namespace": "llama_cpp_q4km_r2b",
                        "coefficient_backend": "llama.cpp",
                        "quantization": "Q4_K_M",
                        "thinking_mode": "off",
                        "reasoning_budget": 0,
                        "runtime_allowed_fields": [
                            "messages",
                            "tools",
                            "tool_aliases",
                            "argument_aliases",
                            "dialogue_state",
                            "reference_context",
                        ],
                        "offline_only_fields": [
                            "gold_action",
                            "derivable_values",
                            "r2b_oracle",
                            "task_kind",
                        ],
                    },
                }
            )
    return rows


def write_r2b_cases(output: str | Path, *, scope: str) -> int:
    return write_jsonl(output, generate_r2b_cases(scope=scope))


def _bounded_action_value_schema(source: dict[str, Any]) -> dict[str, Any]:
    """Keep constrained outputs finite without changing the semantic value domain."""
    kind = str(source.get("type", "string"))
    if kind == "string":
        bounded: dict[str, Any] = {
            "type": "string",
            "maxLength": min(int(source.get("maxLength", 64)), 64),
        }
    elif kind in {"integer", "number", "boolean", "null"}:
        bounded = {"type": kind}
    elif kind == "array":
        bounded = {
            "type": "array",
            "items": _bounded_action_value_schema(source.get("items", {})),
            "maxItems": min(int(source.get("maxItems", 16)), 16),
        }
    elif kind == "object":
        bounded = {
            "type": "object",
            "properties": {
                str(key): _bounded_action_value_schema(value)
                for key, value in source.get("properties", {}).items()
            },
            "additionalProperties": False,
        }
        if source.get("required"):
            bounded["required"] = list(source["required"])
    else:
        bounded = {"type": "string", "maxLength": 64}
    for key in ("enum", "minimum", "maximum", "minItems", "minLength"):
        if key in source:
            bounded[key] = deepcopy(source[key])
    return bounded


def _action_schema(case: dict[str, Any]) -> dict[str, Any]:
    tool_names = sorted(str(tool["name"]) for tool in case.get("tools", []))
    properties: dict[str, Any] = {}
    for tool in case.get("tools", []):
        for slot, prop in tool.get("parameters", {}).get("properties", {}).items():
            properties.setdefault(
                str(slot),
                _bounded_action_value_schema(prop),
            )
    return {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["call", "clarify", "no_tool", "direct_answer"],
            },
            "tool": {
                "anyOf": [
                    {"type": "string", "enum": tool_names},
                    {"type": "null"},
                ]
            },
            "arguments": {
                "type": "object",
                "properties": properties,
                "additionalProperties": False,
            },
            "payload": {
                "type": "object",
                "properties": {
                    "missing_slots": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 64},
                        "maxItems": 5,
                    },
                    "reason": {"type": "string", "maxLength": 160},
                    "answer": {"type": "string", "maxLength": 256},
                },
                "additionalProperties": False,
            },
        },
        "required": ["mode", "tool", "arguments", "payload"],
        "additionalProperties": False,
    }


def _pointer_schema(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    tool_ids = sorted(int(tool["tool_id"]) for tool in catalog)
    slots = sorted(
        {
            str(slot)
            for tool in catalog
            for slot in tool.get("slots", {})
        }
    )
    return {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["call", "clarify", "no_tool", "direct_answer"],
            },
            "tool_id": {
                "anyOf": [
                    {"type": "integer", "enum": tool_ids},
                    {"type": "null"},
                ]
            },
            "arguments": {
                "type": "object",
                "properties": {
                    slot: {"type": "integer", "minimum": 0}
                    for slot in slots
                },
                "additionalProperties": False,
            },
            "payload": {
                "type": "object",
                "properties": {
                    "missing_slots": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 5,
                    },
                    "reason": {"type": "string", "maxLength": 160},
                    "answer": {"type": "string", "maxLength": 256},
                },
                "additionalProperties": False,
            },
        },
        "required": ["mode", "tool_id", "arguments", "payload"],
        "additionalProperties": False,
    }


def _pointer_messages(
    case: dict[str, Any],
    catalog: list[dict[str, Any]],
    condition: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Return only pointer Action IR JSON. Use call only when the request "
                "fully specifies an available operation. Use clarify for missing "
                "required slots, no_tool when no operation is requested, and "
                "direct_answer only when explicitly requested. Never output reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Condition: {condition}\n"
                f"Request: {request_text(case.get('messages', []))}\n"
                f"Pointer catalog: {json.dumps(catalog, sort_keys=True)}\n"
                "For call mode, select integer candidate IDs. Otherwise set tool_id "
                "to null and arguments to an empty object."
            ),
        },
    ]


def _runtime(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": deepcopy(case.get("messages", [])),
        "tools": deepcopy(case.get("tools", [])),
        "tool_aliases": deepcopy(case.get("tool_aliases", {})),
        "argument_aliases": deepcopy(case.get("argument_aliases", {})),
        "dialogue_state": deepcopy(case.get("dialogue_state", {})),
        "reference_context": deepcopy(case.get("reference_context", {})),
    }


def _pointer_action(
    case: dict[str, Any],
    condition: str,
    endpoint: str,
    verifier: FrozenTierBVerifier,
    *,
    max_tokens: int,
    seed: int,
    request_fn: Callable[..., tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_fn = request_fn or _request_schema_json
    runtime = _runtime(case)
    started = time.perf_counter()
    lattice = build_candidate_lattice(
        runtime["messages"],
        runtime["tools"],
        dialogue_state=runtime["dialogue_state"],
        reference_context=runtime["reference_context"],
        candidate_seed=17,
    )
    lattice["action_risk_budget"] = float(
        runtime["reference_context"].get("action_risk_budget", 0.05)
    )
    use_tep = condition in {
        "tap_r_tep_tier_a",
        "tap_r_tep_tier_ab",
        "tap_r_without_global_contract",
        "tap_r_full",
    }
    use_tier_b = condition in {
        "tap_r_tep_tier_ab",
        "tap_r_without_global_contract",
        "tap_r_full",
    }
    hypergraphs: list[dict[str, Any]] = []
    if use_tep:
        _, hypergraphs = augment_lattice_with_typed_programs(
            lattice,
            runtime["messages"],
            runtime["tools"],
            reference_context=runtime["reference_context"],
            dialogue_state=runtime["dialogue_state"],
            tier_b_verifier=verifier if use_tier_b else None,
        )
    catalog = _pointer_catalog(lattice, unrestricted=False)
    raw, metadata = request_fn(
        endpoint,
        _pointer_messages(case, catalog, condition),
        response_schema=_pointer_schema(catalog),
        max_tokens=max_tokens,
        temperature=0.0,
        seed=seed,
    )
    pointer = _sanitize_pointer(raw, _coerce_pointer(raw), lattice)
    if pointer.get("mode") != "call":
        action = {
            "mode": pointer.get("mode"),
            "tool": None,
            "arguments": {},
            "payload": dict(pointer.get("payload", {})),
        }
        resolution = {
            "schema_version": CONTRACT_SOLVER_VERSION,
            "terminal_state": action["mode"],
            "materialized_action": action,
            "history": [],
        }
    elif condition == "tap_r_without_global_contract":
        action = _materialize_unrestricted(pointer, lattice)
        resolution = {
            "schema_version": CONTRACT_SOLVER_VERSION,
            "terminal_state": action.get("mode"),
            "materialized_action": action,
            "history": [],
            "global_contract_applied": False,
        }
    else:
        resolution = resolve_pointer_contract(
            pointer, lattice, runtime["messages"], budget=2
        )
        action = resolution["materialized_action"]
        resolution["global_contract_applied"] = True
    resolution.update(
        {
            "evidence_contract_version": lattice.get("schema_version"),
            "typed_evidence_program_version": (
                TEP_VERSION if hypergraphs else None
            ),
            "tier_b_verifier_version": verifier.version if use_tier_b else None,
            "tier_b_verifier_artifact_sha256": (
                verifier.artifact_sha256 if use_tier_b else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "generation_calls": 1,
        }
    )
    metadata.update(
        {
            "generation_calls": 1,
            "pointer_action": pointer,
            "resolution": resolution,
            "pointer_catalog_sha256": hashlib.sha256(
                json.dumps(catalog, sort_keys=True).encode()
            ).hexdigest(),
        }
    )
    return action, metadata


def _literal_action(
    case: dict[str, Any],
    method: str,
    endpoint: str,
    *,
    max_tokens: int,
    temperature: float,
    seed: int,
    feedback: str | None = None,
    request_fn: Callable[..., tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_fn = request_fn or _request_schema_json
    messages = render_chat_messages(case, method, thinking_mode="off")
    if feedback:
        messages.append({"role": "user", "content": feedback})
    return request_fn(
        endpoint,
        messages,
        response_schema=_action_schema(case),
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )


def _best_of(
    case: dict[str, Any],
    method: str,
    endpoint: str,
    n: int,
    *,
    max_tokens: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    actions, metadata_rows = [], []
    for sample in range(n):
        action, metadata = _literal_action(
            case,
            method,
            endpoint,
            max_tokens=max_tokens,
            temperature=0.2,
            seed=seed * 100 + sample,
        )
        actions.append(action)
        metadata_rows.append(metadata)
    selected, diagnostics = select_candidate(case, actions)
    return actions[selected], {
        **metadata_rows[selected],
        "best_of_n": n,
        "selected_index": selected,
        "candidate_diagnostics": diagnostics,
        "generation_calls": n,
    }


def _feedback_regeneration(
    case: dict[str, Any],
    method: str,
    endpoint: str,
    *,
    max_tokens: int,
    seed: int,
    slot_only: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    action, metadata = _literal_action(
        case,
        method,
        endpoint,
        max_tokens=max_tokens,
        temperature=0.0,
        seed=seed,
    )
    history = []
    calls = 1
    for repair_round in range(2):
        report = contract_validator_error(case, action)
        history.append(
            {
                "round": repair_round,
                "contract_valid": report["contract_valid"],
                "error_classes": [
                    row["error_class"] for row in report["errors"]
                ],
            }
        )
        if report["contract_valid"]:
            break
        selected = report.get("recommended_transition") or {}
        slot = selected.get("slot")
        if slot_only and slot:
            feedback = (
                "Regenerate the complete Action IR, correcting only slot "
                f"{slot}. Do not invent its value. Validator error: "
                f"{selected.get('error_class')}"
            )
        else:
            feedback = (
                "Regenerate the complete Action IR using only request evidence. "
                "Validator errors: "
                + json.dumps(report["errors"][:3], sort_keys=True)
            )
        action, metadata = _literal_action(
            case,
            method,
            endpoint,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed + repair_round + 1,
            feedback=feedback,
        )
        calls += 1
    metadata.update(
        {"generation_calls": calls, "validator_history": history}
    )
    return action, metadata


def run_r2b_model_conditions(
    cases_path: str | Path,
    output_path: str | Path,
    timings_path: str | Path,
    manifest_path: str | Path,
    *,
    endpoint: str,
    model_id: str,
    model_key: str,
    model_artifact: str,
    chat_template: str,
    tier_b_verifier_path: str | Path,
    conditions: Iterable[str] = R2B_CONDITIONS,
    seeds: Iterable[int] = (1,),
    max_tokens: int = 384,
    max_generations: int | None = None,
) -> dict[str, Any]:
    verifier = FrozenTierBVerifier.load(tier_b_verifier_path)
    selected_conditions = tuple(str(item) for item in conditions)
    unknown = set(selected_conditions) - set(R2B_CONDITIONS)
    if unknown:
        raise ValueError(f"unknown R2-B conditions: {sorted(unknown)}")
    jobs = [
        (case, condition, int(seed))
        for case in read_jsonl(cases_path)
        for condition in selected_conditions
        for seed in seeds
    ]
    if max_generations is not None:
        jobs = jobs[:max_generations]
    predictions, timings = [], []
    started_at = datetime.now(timezone.utc).isoformat()
    for case, condition, seed in jobs:
        started = time.perf_counter()
        error = None
        try:
            if condition in {
                "tap_r_literal_evidence",
                "tap_r_tep_tier_a",
                "tap_r_tep_tier_ab",
                "tap_r_without_global_contract",
                "tap_r_full",
            }:
                action, metadata = _pointer_action(
                    case,
                    condition,
                    endpoint,
                    verifier,
                    max_tokens=max_tokens,
                    seed=seed,
                )
            elif condition in {
                "best_of_2",
                "best_of_4",
                "process_scored_search",
            }:
                n = 2 if condition == "best_of_2" else 4
                action, metadata = _best_of(
                    case,
                    condition,
                    endpoint,
                    n,
                    max_tokens=max_tokens,
                    seed=seed,
                )
            elif condition == "validator_feedback_regeneration":
                action, metadata = _feedback_regeneration(
                    case,
                    condition,
                    endpoint,
                    max_tokens=max_tokens,
                    seed=seed,
                    slot_only=False,
                )
            elif condition == "local_llm_slot_regeneration":
                action, metadata = _feedback_regeneration(
                    case,
                    condition,
                    endpoint,
                    max_tokens=max_tokens,
                    seed=seed,
                    slot_only=True,
                )
            else:
                action, metadata = _literal_action(
                    case,
                    condition,
                    endpoint,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    seed=seed,
                )
                metadata["generation_calls"] = 1
        except Exception as exc:
            error = str(exc)
            action = {"runner_error": error}
            metadata = {
                "finish_reason": "runner_error",
                "error_type": exc.__class__.__name__,
                "error_message": error,
                "generation_calls": 0,
            }
        elapsed = time.perf_counter() - started
        default_terminal = (
            action.get("mode") if isinstance(action, dict) else "escalate"
        )
        resolution = (
            metadata.get("resolution")
            if isinstance(metadata.get("resolution"), dict)
            else {
                "terminal_state": default_terminal,
                "generation_calls": metadata.get("generation_calls", 1),
                "validation_rounds": len(
                    metadata.get("validator_history", [])
                )
                + 1,
                "elapsed_seconds": elapsed,
            }
        )
        row = {
            "case_id": case["case_id"],
            "method": condition,
            "model_id": model_id,
            "seed": seed,
            "prediction": action,
            "response_metadata": metadata,
            "resolution": resolution,
            "runner_error": error,
            "backend": "llama.cpp",
            "quantization": "Q4_K_M",
            "chat_template": chat_template,
            "grammar_engine": R2A_GRAMMAR_ENGINE,
            "chat_parser": R2A_CHAT_PARSER,
            "inference_path": "apply_template_then_raw_completion",
            "model_artifact": model_artifact,
            "thinking_mode": "off",
            "reasoning_budget": 0,
            "r2b_model_runner_version": R2B_RUNNER_VERSION,
            "r2b_action_schema_version": R2B_ACTION_SCHEMA_VERSION,
            "max_output_tokens": max_tokens,
            "deployable_resolution_version": (
                DEPLOYABLE_RESOLUTION_VERSION
                if condition.startswith("tap_r_")
                else None
            ),
            "evidence_contract_version": (
                resolution.get("evidence_contract_version")
                if condition.startswith("tap_r_")
                else None
            ),
            "contract_solver_version": (
                resolution.get("schema_version")
                if condition.startswith("tap_r_")
                else None
            ),
            "typed_evidence_program_version": resolution.get(
                "typed_evidence_program_version"
            ),
            "tier_b_verifier_version": resolution.get(
                "tier_b_verifier_version"
            ),
            "tier_b_verifier_artifact_sha256": resolution.get(
                "tier_b_verifier_artifact_sha256"
            ),
        }
        row["thinking_marker_detected"] = prediction_has_thinking_marker(row)
        predictions.append(row)
        timings.append(
            {
                "case_id": case["case_id"],
                "hypothesis_grid_id": R2B_GRID_ID,
                "catalog_mutation": case["factors"]["catalog_mutation"],
                "task_kind": case["task_kind"],
                "method": condition,
                "model_key": model_key,
                "model_id": model_id,
                "seed": seed,
                "backend": "llama.cpp",
                "quantization": "Q4_K_M",
                "elapsed_seconds": elapsed,
                "generation_calls": metadata.get("generation_calls", 1),
                "runner_error": error,
                "thinking_mode": "off",
                "thinking_marker_detected": row[
                    "thinking_marker_detected"
                ],
                **{
                    key: metadata.get(key)
                    for key in (
                        "finish_reason",
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "generated_tokens_per_second",
                        "context_truncated",
                    )
                },
            }
        )
    write_jsonl(output_path, predictions)
    write_jsonl(timings_path, timings)
    manifest = {
        "schema_version": "tapbench.r2b_model_run_manifest.v1",
        "runner_version": R2B_RUNNER_VERSION,
        "action_schema_version": R2B_ACTION_SCHEMA_VERSION,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path),
        "predictions_path": str(output_path),
        "timings_path": str(timings_path),
        "model_key": model_key,
        "model_id": model_id,
        "model_artifact": model_artifact,
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "chat_template": chat_template,
        "grammar_engine": R2A_GRAMMAR_ENGINE,
        "thinking_mode": "off",
        "reasoning_budget": 0,
        "context_tokens": 8192,
        "max_output_tokens": max_tokens,
        "conditions": list(selected_conditions),
        "seeds": list(seeds),
        "generation_count": len(predictions),
        "actual_model_calls": sum(
            int(row["response_metadata"].get("generation_calls", 1))
            for row in predictions
        ),
        "runner_errors": sum(
            row["runner_error"] is not None for row in predictions
        ),
        "tier_b_verifier_version": verifier.version,
        "tier_b_verifier_artifact_sha256": verifier.artifact_sha256,
    }
    write_yaml(manifest_path, manifest)
    return manifest



def write_r2b_runtime_projection(
    timings_path: str | Path,
    output_path: str | Path,
    *,
    full_case_count: int = 256,
    full_seed_count: int = 3,
) -> dict[str, Any]:
    timings = read_jsonl(timings_path)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in timings:
        if row.get("runner_error"):
            continue
        grouped[(str(row["model_id"]), str(row["method"]))].append(
            float(row["elapsed_seconds"])
        )

    def percentile(values: list[float], probability: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        index = max(0, math.ceil(probability * len(ordered)) - 1)
        return ordered[index]

    groups = []
    projected_seconds = 0.0
    for (model_id, method), values in sorted(grouped.items()):
        p50 = percentile(values, 0.50)
        p95 = percentile(values, 0.95)
        projected = p95 * full_case_count * full_seed_count
        projected_seconds += projected
        groups.append({
            "model_id": model_id,
            "method": method,
            "pilot_rows": len(values),
            "p50_seconds_per_case": p50,
            "p95_seconds_per_case": p95,
            "mean_seconds_per_case": sum(values) / len(values),
            "projected_full_p95_seconds": projected,
        })
    report = {
        "schema_version": "tapbench.r2b_runtime_projection.v1",
        "timing_rows": len(timings),
        "successful_timing_rows": sum(len(values) for values in grouped.values()),
        "full_case_count": full_case_count,
        "full_seed_count": full_seed_count,
        "projected_full_p95_hours_serial": projected_seconds / 3600.0,
        "groups": groups,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def score_r2b_files(
    cases_path: str | Path,
    predictions_path: str | Path,
    scores_path: str | Path,
    slot_errors_path: str | Path,
    report_path: str | Path,
    *,
    expected_model_count: int | None = None,
    expected_condition_count: int | None = None,
    expected_seed_count: int | None = None,
) -> dict[str, Any]:
    cases = read_jsonl(cases_path)
    predictions = read_jsonl(predictions_path)
    case_by_id = {row["case_id"]: row for row in cases}
    prediction_by_key = {
        (
            row["case_id"],
            row.get("method"),
            row.get("model_id"),
            row.get("seed"),
        ): row
        for row in predictions
    }
    rows = score_predictions(cases, predictions)
    for row in rows:
        case = case_by_id[row["case_id"]]
        prediction = prediction_by_key[
            (
                row["case_id"],
                row.get("method"),
                row.get("model_id"),
                row.get("seed"),
            )
        ]
        action = prediction.get("prediction")
        mode = action.get("mode") if isinstance(action, dict) else None
        overcall = mode == "call" and case["task_kind"] in {
            "missing_info",
            "no_tool",
            "direct_answer",
        }
        row["catalog_mutation"] = case["factors"]["catalog_mutation"]
        row["autonomous_safe_resolution"] = bool(
            row["execution_success"] and mode != "refuse"
        )
        row["accepted_call"] = mode == "call"
        row["unsupported_action_critical"] = bool(
            row["fabrication"] or overcall
        )
        row["non_escalated"] = mode != "refuse"
        row["r2b_report_version"] = R2B_REPORT_VERSION
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model_id"]), str(row["method"]))].append(row)
    summaries = []
    for (model_id, method), group in sorted(groups.items()):
        calls = [row for row in group if row["accepted_call"]]
        summaries.append(
            {
                "model_id": model_id,
                "method": method,
                "n": len(group),
                "autonomous_safe_resolution": sum(
                    row["autonomous_safe_resolution"] for row in group
                )
                / len(group),
                "accepted_call_exact_precision": (
                    sum(
                        row["execution_success"]
                        and not row["unsupported_action_critical"]
                        for row in calls
                    )
                    / len(calls)
                    if calls
                    else None
                ),
                "unsupported_action_critical_rate": sum(
                    row["unsupported_action_critical"] for row in group
                )
                / len(group),
                "non_escalated_coverage": sum(
                    row["non_escalated"] for row in group
                )
                / len(group),
                "format_validity": sum(
                    row["format_valid"] for row in group
                )
                / len(group),
            }
        )
    model_ids = sorted({str(row.get("model_id")) for row in predictions})
    methods = sorted({str(row.get("method")) for row in predictions})
    seeds = sorted({int(row.get("seed", 0)) for row in predictions})
    expected_rows = len(cases) * len(model_ids) * len(methods) * len(seeds)
    if expected_model_count is not None:
        expected_rows = len(cases) * expected_model_count * (
            expected_condition_count if expected_condition_count is not None else len(methods)
        ) * (expected_seed_count if expected_seed_count is not None else len(seeds))
    per_model_discipline = {}
    for model_id in model_ids:
        model_rows = [row for row in predictions if str(row.get("model_id")) == model_id]
        per_model_discipline[model_id] = {
            field: sorted({str(row.get(field)) for row in model_rows})
            for field in (
                "backend",
                "quantization",
                "model_artifact",
                "chat_template",
                "grammar_engine",
                "thinking_mode",
                "r2b_action_schema_version",
                "max_output_tokens",
            )
        }
    gates = {
        "prediction_count_complete": len(predictions) == expected_rows,
        "expected_model_count": expected_model_count is None or len(model_ids) == expected_model_count,
        "expected_condition_count": expected_condition_count is None or len(methods) == expected_condition_count,
        "expected_seed_count": expected_seed_count is None or len(seeds) == expected_seed_count,
        "zero_runner_errors": not any(row.get("runner_error") for row in predictions),
        "zero_thinking_markers": not any(row.get("thinking_marker_detected") for row in predictions),
        "zero_context_truncations": not any(
            bool(row.get("response_metadata", {}).get("context_truncated"))
            for row in predictions
        ),
        "zero_length_stops": not any(
            str(row.get("response_metadata", {}).get("finish_reason", "")).lower()
            in {"length", "max_tokens"}
            for row in predictions
        ),
        "format_validity_one": bool(rows) and all(row["format_valid"] for row in rows),
        "action_schema_version_current": bool(predictions) and all(
            row.get("r2b_action_schema_version") == R2B_ACTION_SCHEMA_VERSION
            for row in predictions
        ),
        "max_output_tokens_declared": bool(predictions) and all(
            isinstance(row.get("max_output_tokens"), int)
            and row["max_output_tokens"] > 0
            for row in predictions
        ),
        "per_model_artifact_discipline": all(
            all(len(values) == 1 for values in fields.values())
            for fields in per_model_discipline.values()
        ),
    }
    report = {
        "schema_version": R2B_REPORT_VERSION,
        "case_count": len(cases),
        "score_count": len(rows),
        "scorer_version": SCORER_VERSION,
        "normalizer_version": NORMALIZER_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "integrity": {
            "prediction_count": len(predictions),
            "expected_prediction_count": expected_rows,
            "model_ids": model_ids,
            "methods": methods,
            "seeds": seeds,
            "per_model_discipline": per_model_discipline,
        },
        "release_decision": {
            "passed": all(gates.values()),
            "gates": gates,
        },
        "groups": summaries,
    }
    write_jsonl(scores_path, rows)
    write_jsonl(
        slot_errors_path,
        slot_errors_for_predictions(cases, predictions),
    )
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report

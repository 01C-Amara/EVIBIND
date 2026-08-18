from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from copy import deepcopy
from typing import Any, Callable

from .effect_first import _merge_call_metadata
from .extractive_candidates import (
    EXTRACTIVE_CANDIDATE_VERSION,
    build_extractive_candidate_table,
    user_request_text,
)
from .r2_model_runner import _request_schema_json
from .runner import _request_llama_json
from .validation import action_contract_is_accepted


EFLRX_VERSION = "tapbench.eflrx.v1"
EFLRX_CONDITIONS = (
    "tap_r_eflrx_single",
    "tap_r_eflrx_consensus",
)
ACTION_RISK_THRESHOLD = 0.05
CONTEXT_TOKENS = 32768
NO_CALL_ID = -1

_MODE_RISK = 0.001
_TOOL_RISK = {"single": 0.03, "agreement": 0.01}
_SPAN_RISK = 0.002
_NORMALIZING_RISK = 0.005
_NON_NORMALIZING_TRANSFORMS = {
    "identity",
    "parse_integer_or_decimal",
    "parse_number_word",
    "split_explicit_list",
}
_DIRECT_PATTERNS = (
    r"\banswer directly\b",
    r"\brespond (?:only )?in (?:the )?chat\b",
    r"\bexplain only\b",
)
_DENIAL_PATTERNS = (
    r"\bdo not (?:call|use|execute|run|invoke) (?:a |the )?(?:tool|function)\b",
    r"\bwithout (?:calling|using|executing|running|invoking) (?:a |the )?(?:tool|function)\b",
    r"\bnot asking (?:you )?to (?:perform|execute|run|call|use|submit|create|send|book|schedule)\b",
)


RequestFn = Callable[..., tuple[dict[str, Any], dict[str, Any]]]


class ContextOverflowError(RuntimeError):
    pass


def _post_json(
    endpoint: str,
    route: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + route,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response, _ = _request_llama_json(request, endpoint)
    return response


def preflight_schema_request(
    endpoint: str,
    messages: list[dict[str, str]],
    *,
    response_schema: dict[str, Any],
    max_tokens: int,
    temperature: float,
    seed: int,
    context_tokens: int = CONTEXT_TOKENS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    templated = _post_json(
        endpoint,
        "/apply-template",
        {
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    prompt = templated.get("prompt")
    if not isinstance(prompt, str):
        raise RuntimeError("llama.cpp /apply-template returned no prompt")
    tokenized = _post_json(
        endpoint,
        "/tokenize",
        {
            "content": prompt,
            "add_special": False,
            "parse_special": True,
            "with_pieces": False,
        },
    )
    tokens = tokenized.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError("llama.cpp /tokenize returned no token list")
    rendered_input_tokens = len(tokens)
    headroom = context_tokens - max_tokens - rendered_input_tokens
    if headroom < 0:
        raise ContextOverflowError(
            "context_overflow: rendered input "
            f"{rendered_input_tokens} + output {max_tokens} exceeds "
            f"{context_tokens}"
        )
    raw, metadata = _request_schema_json(
        endpoint,
        messages,
        response_schema=response_schema,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
    prompt_tokens = metadata.get("prompt_tokens")
    token_delta = (
        int(prompt_tokens) - rendered_input_tokens
        if prompt_tokens is not None
        else None
    )
    metadata.update(
        {
            "rendered_input_tokens": rendered_input_tokens,
            "preflight_prompt_token_delta": token_delta,
            "context_tokens": context_tokens,
            "context_headroom_tokens": headroom,
            "context_overflow": False,
            "preflight_http_calls": 2,
        }
    )
    return raw, metadata


def _merge_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    merged = _merge_call_metadata(rows)
    rendered = [
        int(row["rendered_input_tokens"])
        for row in rows
        if row.get("rendered_input_tokens") is not None
    ]
    headroom = [
        int(row["context_headroom_tokens"])
        for row in rows
        if row.get("context_headroom_tokens") is not None
    ]
    preflight_deltas = [
        abs(int(row["preflight_prompt_token_delta"]))
        for row in rows
        if row.get("preflight_prompt_token_delta") is not None
    ]
    raw_texts = [
        str(row["raw_text"])
        for row in rows
        if row.get("raw_text") is not None
    ]
    merged.update(
        {
            "raw_text": "\n".join(raw_texts),
            "raw_generation_count": len(raw_texts),
            "rendered_input_tokens_max": max(rendered) if rendered else None,
            "context_headroom_tokens_min": min(headroom) if headroom else None,
            "preflight_prompt_token_delta_max_abs": (
                max(preflight_deltas) if preflight_deltas else None
            ),
            "context_overflow": any(
                bool(row.get("context_overflow")) for row in rows
            ),
            "preflight_http_calls": sum(
                int(row.get("preflight_http_calls") or 0) for row in rows
            ),
        }
    )
    return merged


def _non_call(mode: str, reason: str, *, missing: list[str] | None = None) -> dict[str, Any]:
    if mode == "clarify":
        payload: dict[str, Any] = {
            "reason": reason,
            "missing_slots": list(missing or []),
        }
    elif mode == "direct_answer":
        payload = {"answer": "respond directly without tool execution"}
    else:
        payload = {"reason": reason}
    return {
        "mode": mode,
        "tool": None,
        "arguments": {},
        "payload": payload,
    }


def explicit_effect_firewall(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    text = user_request_text(messages)
    lowered = text.casefold()
    if any(re.search(pattern, lowered) for pattern in _DIRECT_PATTERNS):
        return {"terminal_mode": "direct_answer", "basis": "explicit_direct_answer"}
    if any(re.search(pattern, lowered) for pattern in _DENIAL_PATTERNS):
        return {"terminal_mode": "no_tool", "basis": "explicit_effect_denial"}
    return {"terminal_mode": None, "basis": "model_election_required"}


def _tool_name(tool: dict[str, Any]) -> str:
    return str(tool.get("canonical_name") or tool.get("name") or "")


def _required_slots(tool: dict[str, Any]) -> list[str]:
    parameters = tool.get("parameters", {})
    return [
        str(slot)
        for slot in parameters.get("required", [])
    ] if isinstance(parameters, dict) else []


def _tool_catalog(
    tools: list[dict[str, Any]],
    *,
    reverse: bool,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    ordered = list(tools)
    if reverse:
        ordered.reverse()
    catalog: list[dict[str, Any]] = [
        {
            "selection_id": NO_CALL_ID,
            "effect": "NO_CALL",
            "description": (
                "No candidate function directly and completely satisfies the "
                "requested effect."
            ),
            "required_slots": [],
        }
    ]
    mapping = {NO_CALL_ID: "NO_CALL"}
    for selection_id, tool in enumerate(ordered):
        name = _tool_name(tool)
        mapping[selection_id] = name
        catalog.append(
            {
                "selection_id": selection_id,
                "effect": name,
                "description": str(tool.get("description", "")),
                "required_slots": _required_slots(tool),
            }
        )
    return catalog, mapping


def _tool_election_messages(
    messages: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Select exactly one selection_id. Choose a function only when "
                "its external effect directly and completely answers the user. "
                "Choose NO_CALL when none does. Return identifiers only: never "
                "generate arguments, values, or reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                "Request:\n"
                + user_request_text(messages)
                + "\nEffect catalog:\n"
                + json.dumps(
                    catalog,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                + "\nReturn {\"selection_id\": integer}."
            ),
        },
    ]


def _tool_election_schema(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selection_id": {
                "type": "integer",
                "enum": [int(row["selection_id"]) for row in catalog],
            }
        },
        "required": ["selection_id"],
        "additionalProperties": False,
    }


def _candidate_view(
    table: dict[str, Any],
    *,
    reverse: bool,
) -> tuple[
    dict[str, Any],
    dict[str, dict[int, dict[str, Any] | None]],
]:
    view: dict[str, Any] = {}
    mapping: dict[str, dict[int, dict[str, Any] | None]] = {}
    required = list(table.get("required_slots", []))
    optional = list(table.get("optional_slots", []))
    optional_set = set(optional)
    for slot in [*required, *optional]:
        rows = list(table.get("slots", {}).get(slot, []))
        if reverse:
            rows.reverse()
        if slot in optional_set and not rows:
            continue
        mapping[slot] = {}
        compact = []
        if slot in optional_set:
            mapping[slot][NO_CALL_ID] = None
            compact.append(
                {
                    "candidate_id": NO_CALL_ID,
                    "value": "OMIT",
                    "source": None,
                    "span": None,
                    "transform": "omit_optional",
                }
            )
        for display_id, row in enumerate(rows):
            mapping[slot][display_id] = row
            compact.append(
                {
                    "candidate_id": display_id,
                    "value": row.get("value"),
                    "source": row.get("source_text"),
                    "span": row.get("source_span"),
                    "transform": row.get("transform"),
                }
            )
        view[slot] = compact
    return view, mapping


def _pointer_messages(
    messages: list[dict[str, Any]],
    tool_name: str,
    candidate_view: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Bind every required slot by selecting one listed candidate_id. "
                "For an optional slot, select -1 (OMIT) unless the request "
                "explicitly supplies that field. Return candidate identifiers only. "
                "Do not copy, rewrite, infer, default, or generate any argument value. "
                "If a required value is unsupported, the resolver will fail closed."
            ),
        },
        {
            "role": "user",
            "content": (
                "Request:\n"
                + user_request_text(messages)
                + "\nSelected function: "
                + tool_name
                + "\nCertified candidates:\n"
                + json.dumps(
                    candidate_view,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                + "\nReturn {\"arguments\": {slot: candidate_id}}."
            ),
        },
    ]


def _pointer_schema(candidate_view: dict[str, Any]) -> dict[str, Any]:
    properties = {
        slot: {
            "type": "integer",
            "enum": [
                int(row["candidate_id"])
                for row in rows
            ],
        }
        for slot, rows in candidate_view.items()
    }
    return {
        "type": "object",
        "properties": {
            "arguments": {
                "type": "object",
                "properties": properties,
                "required": sorted(properties),
                "additionalProperties": False,
            }
        },
        "required": ["arguments"],
        "additionalProperties": False,
    }


def _select_pointer(
    raw: dict[str, Any],
    mapping: dict[str, dict[int, dict[str, Any] | None]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    selected = raw.get("arguments")
    if not isinstance(selected, dict):
        return None, {"status": "invalid_pointer_object"}
    assignments: dict[str, Any] = {}
    certificates: dict[str, Any] = {}
    omitted: list[str] = []
    for slot, candidates in mapping.items():
        try:
            candidate_id = int(selected.get(slot))
        except (TypeError, ValueError):
            return None, {"status": "missing_pointer", "slot": slot}
        if candidate_id not in candidates:
            return None, {
                "status": "out_of_domain_pointer",
                "slot": slot,
                "candidate_id": candidate_id,
            }
        candidate = candidates[candidate_id]
        if candidate is None:
            omitted.append(slot)
            continue
        assignments[slot] = deepcopy(candidate.get("value"))
        certificates[slot] = {
            "candidate_id": candidate_id,
            "value": deepcopy(candidate.get("value")),
            "source_span": list(candidate.get("source_span", [])),
            "component_spans": deepcopy(candidate.get("component_spans", [])),
            "source_text": candidate.get("source_text"),
            "transform": candidate.get("transform"),
        }
    extra = sorted(set(selected) - set(mapping))
    if extra:
        return None, {"status": "extra_pointer_slots", "slots": extra}
    return assignments, {
        "status": "selected",
        "certificates": certificates,
        "omitted_optional_slots": sorted(omitted),
    }


def _canonical_key(tool_name: str, assignments: dict[str, Any]) -> str:
    return json.dumps(
        {"tool": tool_name, "arguments": assignments},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _risk_for_certificates(
    certificates: dict[str, Any],
    *,
    consensus: bool,
) -> tuple[float, dict[str, Any]]:
    span_risk = len(certificates) * _SPAN_RISK
    normalizing = sum(
        1
        for row in certificates.values()
        if str(row.get("transform")) not in _NON_NORMALIZING_TRANSFORMS
    )
    normalizing_risk = normalizing * _NORMALIZING_RISK
    tool_risk = _TOOL_RISK["agreement" if consensus else "single"]
    factors = {
        "mode": _MODE_RISK,
        "tool": tool_risk,
        "typed_spans": span_risk,
        "normalizing_transforms": normalizing_risk,
        "normalizing_transform_count": normalizing,
    }
    return min(
        1.0,
        _MODE_RISK + tool_risk + span_risk + normalizing_risk,
    ), factors


def run_eflrx_resolution(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    endpoint: str,
    condition: str,
    max_tokens: int,
    seed: int,
    request_fn: RequestFn = _request_schema_json,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if condition not in EFLRX_CONDITIONS:
        raise ValueError(f"unknown EFLR-X condition: {condition}")
    consensus = condition == "tap_r_eflrx_consensus"
    started = time.perf_counter()
    firewall = explicit_effect_firewall(messages)
    metadata: dict[str, Any] = {
        "eflrx_version": EFLRX_VERSION,
        "extractive_candidate_version": EXTRACTIVE_CANDIDATE_VERSION,
        "effect_firewall": firewall,
        "action_risk_threshold": ACTION_RISK_THRESHOLD,
        "tool_elections": [],
        "pointer_elections": [],
        "generation_calls": 0,
    }
    if firewall["terminal_mode"]:
        action = _non_call(
            str(firewall["terminal_mode"]),
            str(firewall["basis"]),
        )
        metadata.update(
            {
                "finish_reason": "not_applicable",
                "action_risk_score": _MODE_RISK,
                "risk_factors": {"mode": _MODE_RISK},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return action, metadata

    call_metadata: list[dict[str, Any]] = []
    selected_tools: list[str] = []
    order_flags = [False, True] if consensus else [False]
    for index, reverse in enumerate(order_flags):
        catalog, mapping = _tool_catalog(tools, reverse=reverse)
        schema = _tool_election_schema(catalog)
        raw, response = request_fn(
            endpoint,
            _tool_election_messages(messages, catalog),
            response_schema=schema,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed + index * 100003,
        )
        try:
            selection_id = int(raw.get("selection_id"))
        except (TypeError, ValueError):
            selection_id = NO_CALL_ID - 1
        selected = mapping.get(selection_id, "INVALID")
        selected_tools.append(selected)
        metadata["tool_elections"].append(
            {
                "order": "reverse" if reverse else "forward",
                "selected": selected,
                "selection_id": selection_id,
                "catalog_sha256": hashlib.sha256(
                    json.dumps(
                        catalog,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            }
        )
        call_metadata.append(response)

    tool_agreement = bool(
        selected_tools
        and selected_tools[0] != "INVALID"
        and all(value == selected_tools[0] for value in selected_tools)
    )
    if not tool_agreement:
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "tool_agreement": False,
                "selected_tools": selected_tools,
                "action_risk_score": 1.0,
                "risk_factors": {"tool_disagreement": 1.0},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call("refuse", "counterbalanced tool elections disagree"), metadata

    tool_name = selected_tools[0]
    if tool_name == "NO_CALL":
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "tool_agreement": True,
                "selected_tools": selected_tools,
                "action_risk_score": (
                    _MODE_RISK
                    + _TOOL_RISK["agreement" if consensus else "single"]
                ),
                "risk_factors": {
                    "mode": _MODE_RISK,
                    "tool": _TOOL_RISK[
                        "agreement" if consensus else "single"
                    ],
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call("no_tool", "model selected NO_CALL sentinel"), metadata

    tool = next(
        (row for row in tools if _tool_name(row) == tool_name),
        None,
    )
    if tool is None:
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "action_risk_score": 1.0,
                "risk_factors": {"catalog_resolution": 1.0},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call("refuse", "selected tool is absent"), metadata

    table = build_extractive_candidate_table(
        messages,
        tool,
        include_optional=True,
    )
    missing = [
        slot
        for slot in table.get("required_slots", [])
        if not table.get("slots", {}).get(slot)
    ]
    metadata["candidate_table"] = {
        "schema_version": table.get("schema_version"),
        "candidate_count": table.get("candidate_count"),
        "required_slots": table.get("required_slots"),
        "optional_slots": table.get("optional_slots"),
        "sha256": hashlib.sha256(
            json.dumps(
                table,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).hexdigest(),
    }
    if missing:
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "candidate_precheck": {
                    "status": "missing",
                    "missing_slots": missing,
                },
                "action_risk_score": 1.0,
                "risk_factors": {"missing_evidence": 1.0},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call(
            "clarify",
            "required evidence has no certified candidate",
            missing=missing,
        ), metadata

    pointer_assignments: list[dict[str, Any]] = []
    pointer_certificates: list[dict[str, Any]] = []
    for index, reverse in enumerate(order_flags):
        view, mapping = _candidate_view(table, reverse=reverse)
        schema = _pointer_schema(view)
        raw, response = request_fn(
            endpoint,
            _pointer_messages(messages, tool_name, view),
            response_schema=schema,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed + 500009 + index * 100003,
        )
        assignments, selection = _select_pointer(raw, mapping)
        metadata["pointer_elections"].append(
            {
                "order": "reverse" if reverse else "forward",
                "status": selection.get("status"),
                "candidate_view_sha256": hashlib.sha256(
                    json.dumps(
                        view,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode()
                ).hexdigest(),
            }
        )
        call_metadata.append(response)
        if assignments is None:
            metadata.update(_merge_metadata(call_metadata))
            metadata.update(
                {
                    "pointer_agreement": False,
                    "pointer_failure": selection,
                    "action_risk_score": 1.0,
                    "risk_factors": {"invalid_pointer": 1.0},
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return _non_call("refuse", "invalid evidence pointer"), metadata
        pointer_assignments.append(assignments)
        pointer_certificates.append(selection["certificates"])

    canonical = [
        _canonical_key(tool_name, assignments)
        for assignments in pointer_assignments
    ]
    pointer_agreement = bool(
        canonical
        and all(value == canonical[0] for value in canonical)
    )
    if not pointer_agreement:
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "pointer_agreement": False,
                "canonical_selections": canonical,
                "action_risk_score": 1.0,
                "risk_factors": {"pointer_disagreement": 1.0},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call(
            "refuse",
            "counterbalanced evidence pointers disagree",
        ), metadata

    assignments = pointer_assignments[0]
    certificates = pointer_certificates[0]
    risk, factors = _risk_for_certificates(
        certificates,
        consensus=consensus,
    )
    action = {
        "mode": "call",
        "tool": tool_name,
        "arguments": assignments,
        "payload": {},
    }
    contract_valid = action_contract_is_accepted(
        {
            "tools": tools,
        },
        action,
    )
    if not contract_valid:
        action = _non_call(
            "refuse",
            "materialized action violates the public tool contract",
        )
        risk = 1.0
        factors["contract"] = 1.0
    elif risk > ACTION_RISK_THRESHOLD:
        action = _non_call(
            "refuse",
            "composed action risk exceeds threshold",
        )
    metadata.update(_merge_metadata(call_metadata))
    metadata.update(
        {
            "tool_agreement": tool_agreement,
            "pointer_agreement": pointer_agreement,
            "selected_tools": selected_tools,
            "evidence_certificates": certificates,
            "action_risk_score": risk,
            "risk_factors": factors,
            "risk_gate_passed": (
                contract_valid and risk <= ACTION_RISK_THRESHOLD
            ),
            "contract_valid": contract_valid,
            "materialized_action_sha256": hashlib.sha256(
                json.dumps(
                    action,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode()
            ).hexdigest(),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return action, metadata

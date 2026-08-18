from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from collections import Counter
from copy import deepcopy
from typing import Any

from .eflrx import (
    ACTION_RISK_THRESHOLD,
    NO_CALL_ID,
    RequestFn,
    _merge_metadata,
    _non_call,
    _tool_name,
    explicit_effect_firewall,
)
from .extractive_candidates import canonical_slots, user_request_text
from .r2_model_runner import _request_schema_json
from .validation import action_contract_is_accepted


PROJECTED_CAPC_VERSION = "tapbench.capc_projected.v1"
SOURCE_CERTIFICATE_VERSION = "tapbench.source_certificate.unicode.v1"
PROJECTED_CAPC_CONDITIONS = (
    "tap_r_capc_projected_strict",
    "tap_r_capc_projected_majority",
    "tap_r_capc_projected_pivot",
)
_ORDER_NAMES = ("forward", "reverse", "stable_hash")
_PROPOSAL_ATTEMPTS = 2


def _ordered_tools(
    tools: list[dict[str, Any]],
    order: str,
) -> list[dict[str, Any]]:
    if order == "forward":
        return list(tools)
    if order == "reverse":
        return list(reversed(tools))
    if order == "stable_hash":
        return sorted(
            tools,
            key=lambda tool: hashlib.sha256(
                (
                    "tapbench.capc_projected.order.v1\0"
                    + _tool_name(tool)
                ).encode("utf-8")
            ).hexdigest(),
        )
    raise ValueError(f"unknown tool order: {order}")


def _effect_catalog(
    tools: list[dict[str, Any]],
    order: str,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    catalog: list[dict[str, Any]] = [
        {
            "selection_id": NO_CALL_ID,
            "effect": "NO_CALL",
            "description": (
                "No available function directly performs the requested effect."
            ),
        }
    ]
    mapping = {NO_CALL_ID: "NO_CALL"}
    for selection_id, tool in enumerate(_ordered_tools(tools, order)):
        name = _tool_name(tool)
        mapping[selection_id] = name
        parameters = tool.get("parameters", {})
        properties = (
            parameters.get("properties", {})
            if isinstance(parameters, dict)
            else {}
        )
        catalog.append(
            {
                "selection_id": selection_id,
                "effect": name,
                "description": str(tool.get("description", "")),
                "argument_names": sorted(str(key) for key in properties),
            }
        )
    return catalog, mapping


def _selection_schema(catalog: list[dict[str, Any]]) -> dict[str, Any]:
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


def _pivot_messages(request_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Restate only the requested external effect in concise English. "
                "Do not choose a function, infer missing details, translate named "
                "entities, or add reasoning. Return one JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                "Original request:\n"
                + request_text
                + "\nReturn {\"english_effect\": string}."
            ),
        },
    ]


def _pivot_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"english_effect": {"type": "string"}},
        "required": ["english_effect"],
        "additionalProperties": False,
    }


def _election_messages(
    request_text: str,
    catalog: list[dict[str, Any]],
    *,
    english_effect: str | None,
) -> list[dict[str, str]]:
    pivot = (
        "\nEnglish effect pivot (advisory only):\n" + english_effect
        if english_effect
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "Select exactly one selection_id for the function whose external "
                "effect directly matches the request. Ignore catalog order. Choose "
                "NO_CALL only when no function applies. Do not generate arguments "
                "or reasoning. Return one JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                "Original request:\n"
                + request_text
                + pivot
                + "\nEffect catalog:\n"
                + json.dumps(
                    catalog,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\nReturn {\"selection_id\": integer}."
            ),
        },
    ]


def _projected_arguments_schema(tool: dict[str, Any]) -> dict[str, Any]:
    parameters = tool.get("parameters")
    if not isinstance(parameters, dict):
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
    schema = deepcopy(parameters)
    schema["type"] = "object"
    schema.setdefault("properties", {})
    schema.setdefault("required", [])
    schema["additionalProperties"] = False
    return schema


def projected_action_schema(tool: dict[str, Any]) -> dict[str, Any]:
    name = _tool_name(tool)
    return {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["call"]},
            "tool": {"type": "string", "enum": [name]},
            "arguments": _projected_arguments_schema(tool),
            "payload": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        "required": ["mode", "tool", "arguments", "payload"],
        "additionalProperties": False,
    }


def _proposal_messages(
    request_text: str,
    tool: dict[str, Any],
    *,
    repair_status: str | None,
) -> list[dict[str, str]]:
    repair = (
        "\nThe previous proposal failed the deterministic source check with "
        f"status {repair_status}. Regenerate from the original request only."
        if repair_status
        else ""
    )
    public_tool = {
        "name": tool.get("name"),
        "description": tool.get("description"),
        "parameters": tool.get("parameters"),
    }
    return [
        {
            "role": "system",
            "content": (
                "Return one call-mode Action IR JSON object for the selected tool. "
                "Include every argument value explicitly stated in the original "
                "request, copying each value verbatim. Omit unspecified optional "
                "arguments. Never translate, normalize, default, or invent an "
                "argument value. Do not output reasoning or prose."
            ),
        },
        {
            "role": "user",
            "content": (
                "Original request:\n"
                + request_text
                + "\nSelected tool:\n"
                + json.dumps(
                    public_tool,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + repair
                + "\nReturn the Action IR JSON now."
            ),
        },
    ]


def _normalized_text_map(text: str) -> tuple[str, list[tuple[int, int]]]:
    normalized: list[str] = []
    spans: list[tuple[int, int]] = []
    for index, char in enumerate(text):
        transformed = unicodedata.normalize("NFKC", char).casefold()
        for output_char in transformed:
            normalized.append(output_char)
            spans.append((index, index + 1))
    return "".join(normalized), spans


def _string_source_certificate(
    request_text: str,
    value: str,
) -> dict[str, Any] | None:
    if not value:
        return None
    start = request_text.find(value)
    transform = "identity"
    if start < 0:
        match = re.search(
            re.escape(value),
            request_text,
            flags=re.IGNORECASE | re.UNICODE,
        )
        if match is not None:
            start, end = match.span()
            source = request_text[start:end]
            return {
                "value": value,
                "source_span": [start, end],
                "component_spans": [[start, end]],
                "source_text": source,
                "transform": "unicode_casefold_match",
            }
        normalized_request, spans = _normalized_text_map(request_text)
        normalized_value = unicodedata.normalize("NFKC", value).casefold()
        normalized_start = normalized_request.find(normalized_value)
        if normalized_start < 0 or not normalized_value:
            return None
        normalized_end = normalized_start + len(normalized_value)
        start = spans[normalized_start][0]
        end = spans[normalized_end - 1][1]
        source = request_text[start:end]
        if (
            unicodedata.normalize("NFKC", source).casefold()
            != normalized_value
        ):
            return None
        return {
            "value": value,
            "source_span": [start, end],
            "component_spans": [[start, end]],
            "source_text": source,
            "transform": "unicode_nfkc_casefold_match",
        }
    end = start + len(value)
    return {
        "value": value,
        "source_span": [start, end],
        "component_spans": [[start, end]],
        "source_text": request_text[start:end],
        "transform": transform,
    }


def source_certificate(
    request_text: str,
    value: Any,
    schema: dict[str, Any],
) -> dict[str, Any] | None:
    schema_type = str(schema.get("type") or "string").casefold()
    if schema_type == "string" and isinstance(value, str):
        return _string_source_certificate(request_text, value)
    if schema_type in {"integer", "number"} and isinstance(
        value, (int, float)
    ) and not isinstance(value, bool):
        rendered = str(value)
        match = re.search(
            rf"(?<![\w.]){re.escape(rendered)}(?![\w.])",
            request_text,
            flags=re.UNICODE,
        )
        if match is None:
            return None
        start, end = match.span()
        return {
            "value": value,
            "source_span": [start, end],
            "component_spans": [[start, end]],
            "source_text": request_text[start:end],
            "transform": "parse_integer_or_decimal",
        }
    if schema_type in {"array", "tuple"} and isinstance(value, list):
        item_schema = schema.get("items", {"type": "string"})
        if not isinstance(item_schema, dict):
            item_schema = {"type": "string"}
        components = [
            source_certificate(request_text, item, item_schema)
            for item in value
        ]
        if not components or any(item is None for item in components):
            return None
        valid_components = [item for item in components if item is not None]
        spans = [list(item["source_span"]) for item in valid_components]
        start = min(span[0] for span in spans)
        end = max(span[1] for span in spans)
        return {
            "value": deepcopy(value),
            "source_span": [start, end],
            "component_spans": spans,
            "source_text": request_text[start:end],
            "transform": "explicit_array_items",
            "components": valid_components,
        }
    return None


def replay_source_certificate(
    request_text: str,
    certificate: dict[str, Any],
) -> bool:
    span = certificate.get("source_span")
    if (
        not isinstance(span, list)
        or len(span) != 2
        or not all(isinstance(value, int) for value in span)
    ):
        return False
    start, end = span
    if start < 0 or end < start or end > len(request_text):
        return False
    source = request_text[start:end]
    if source != certificate.get("source_text"):
        return False
    value = certificate.get("value")
    transform = certificate.get("transform")
    if transform == "identity":
        return source == value
    if transform == "unicode_casefold_match":
        return isinstance(value, str) and source.casefold() == value.casefold()
    if transform == "unicode_nfkc_casefold_match":
        return isinstance(value, str) and (
            unicodedata.normalize("NFKC", source).casefold()
            == unicodedata.normalize("NFKC", value).casefold()
        )
    if transform == "parse_integer_or_decimal":
        try:
            return float(source) == float(value)
        except (TypeError, ValueError):
            return False
    if transform == "explicit_array_items":
        components = certificate.get("components")
        return (
            isinstance(components, list)
            and isinstance(value, list)
            and len(components) == len(value)
            and all(
                replay_source_certificate(request_text, item)
                and item.get("value") == expected
                for item, expected in zip(components, value)
            )
        )
    return False


def certify_projected_proposal(
    proposal: dict[str, Any],
    *,
    selected_tool: str,
    tool: dict[str, Any],
    tools: list[dict[str, Any]],
    request_text: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not isinstance(proposal, dict) or proposal.get("mode") != "call":
        return None, {"status": "not_a_call"}
    public_names = {
        str(tool.get("name") or ""),
        str(tool.get("canonical_name") or ""),
    }
    if (
        selected_tool not in public_names
        or str(proposal.get("tool") or "") not in public_names
    ):
        return None, {"status": "tool_mismatch"}
    raw_arguments = proposal.get("arguments")
    if not isinstance(raw_arguments, dict):
        return None, {"status": "arguments_not_object"}

    slots = canonical_slots(tool)
    aliases: dict[str, dict[str, Any]] = {}
    for slot in slots:
        aliases[str(slot["surface_name"])] = slot
        aliases[str(slot["name"])] = slot
    arguments: dict[str, Any] = {}
    certificates: dict[str, Any] = {}
    for surface_slot, value in raw_arguments.items():
        slot = aliases.get(str(surface_slot))
        if slot is None:
            return None, {
                "status": "unknown_argument",
                "slot": str(surface_slot),
            }
        canonical_slot = str(slot["name"])
        if canonical_slot in arguments:
            return None, {
                "status": "duplicate_canonical_argument",
                "slot": canonical_slot,
            }
        certificate = source_certificate(
            request_text,
            value,
            slot["schema"],
        )
        if certificate is None:
            return None, {
                "status": "unsupported_argument",
                "slot": canonical_slot,
                "value_sha256": hashlib.sha256(
                    json.dumps(
                        value,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest(),
            }
        certificate["certificate_version"] = SOURCE_CERTIFICATE_VERSION
        if not replay_source_certificate(request_text, certificate):
            return None, {
                "status": "certificate_replay_failed",
                "slot": canonical_slot,
            }
        arguments[canonical_slot] = deepcopy(value)
        certificates[canonical_slot] = certificate

    required = {
        str(slot["name"]) for slot in slots if bool(slot.get("required"))
    }
    missing = sorted(required - set(arguments))
    if missing:
        return None, {
            "status": "missing_required_arguments",
            "missing_slots": missing,
        }
    action = {
        "mode": "call",
        "tool": selected_tool,
        "arguments": arguments,
        "payload": {},
    }
    if not action_contract_is_accepted({"tools": tools}, action):
        return None, {"status": "public_contract_rejected"}
    return action, {
        "status": "certified",
        "certificates": certificates,
        "certificate_count": len(certificates),
    }


def _election_winner(
    selections: list[str],
    *,
    strict: bool,
) -> tuple[str | None, int]:
    if not selections or any(value == "INVALID" for value in selections):
        return None, 0
    counts = Counter(selections)
    winner, votes = counts.most_common(1)[0]
    if strict:
        return (winner, votes) if votes == len(selections) else (None, votes)
    tied = sum(count == votes for count in counts.values()) > 1
    return (winner, votes) if votes >= 2 and not tied else (None, votes)


def run_projected_capc_resolution(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    endpoint: str,
    condition: str,
    max_tokens: int,
    seed: int,
    request_fn: RequestFn = _request_schema_json,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if condition not in PROJECTED_CAPC_CONDITIONS:
        raise ValueError(f"unknown projected CAPC condition: {condition}")
    started = time.perf_counter()
    firewall = explicit_effect_firewall(messages)
    request_text = user_request_text(messages)
    metadata: dict[str, Any] = {
        "projected_capc_version": PROJECTED_CAPC_VERSION,
        "source_certificate_version": SOURCE_CERTIFICATE_VERSION,
        "effect_firewall": firewall,
        "action_risk_threshold": ACTION_RISK_THRESHOLD,
        "tool_elections": [],
        "proposal_attempts": [],
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
                "action_risk_score": 0.001,
                "risk_factors": {"explicit_effect_firewall": 0.001},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return action, metadata

    call_metadata: list[dict[str, Any]] = []
    english_effect: str | None = None
    if condition == "tap_r_capc_projected_pivot":
        raw_pivot, pivot_metadata = request_fn(
            endpoint,
            _pivot_messages(request_text),
            response_schema=_pivot_schema(),
            max_tokens=min(max_tokens, 128),
            temperature=0.0,
            seed=seed + 900001,
        )
        call_metadata.append(pivot_metadata)
        if isinstance(raw_pivot, dict):
            candidate = raw_pivot.get("english_effect")
            if isinstance(candidate, str) and candidate.strip():
                english_effect = candidate.strip()
        metadata["effect_pivot"] = {
            "used": bool(english_effect),
            "sha256": (
                hashlib.sha256(english_effect.encode("utf-8")).hexdigest()
                if english_effect
                else None
            ),
        }

    strict = condition == "tap_r_capc_projected_strict"
    orders = _ORDER_NAMES[:2] if strict else _ORDER_NAMES
    selections: list[str] = []
    for index, order in enumerate(orders):
        catalog, mapping = _effect_catalog(tools, order)
        raw, response = request_fn(
            endpoint,
            _election_messages(
                request_text,
                catalog,
                english_effect=english_effect,
            ),
            response_schema=_selection_schema(catalog),
            max_tokens=min(max_tokens, 64),
            temperature=0.0,
            seed=seed + index * 100003,
        )
        try:
            selection_id = int(raw.get("selection_id"))
        except (AttributeError, TypeError, ValueError):
            selection_id = NO_CALL_ID - 1
        selected = mapping.get(selection_id, "INVALID")
        selections.append(selected)
        metadata["tool_elections"].append(
            {
                "order": order,
                "selected": selected,
                "selection_id": selection_id,
                "catalog_sha256": hashlib.sha256(
                    json.dumps(
                        catalog,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
        call_metadata.append(response)

    selected_tool, winner_votes = _election_winner(
        selections,
        strict=strict,
    )
    metadata.update(
        {
            "selected_tools": selections,
            "election_policy": "unanimous_2" if strict else "majority_2_of_3",
            "election_winner_votes": winner_votes,
        }
    )
    if selected_tool is None:
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "tool_agreement": False,
                "action_risk_score": 1.0,
                "risk_factors": {"tool_election_unresolved": 1.0},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call("refuse", "tool election was unresolved"), metadata

    metadata["tool_agreement"] = True
    if selected_tool == "NO_CALL":
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "action_risk_score": 0.01,
                "risk_factors": {"counterbalanced_no_call": 0.01},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call("no_tool", "counterbalanced election selected NO_CALL"), metadata

    tool = next(
        (candidate for candidate in tools if _tool_name(candidate) == selected_tool),
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

    accepted_action: dict[str, Any] | None = None
    accepted_certification: dict[str, Any] | None = None
    accepted_index: int | None = None
    previous_status: str | None = None
    schema = projected_action_schema(tool)
    for attempt in range(_PROPOSAL_ATTEMPTS):
        proposal, response = request_fn(
            endpoint,
            _proposal_messages(
                request_text,
                tool,
                repair_status=previous_status,
            ),
            response_schema=schema,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed + 700001 + attempt * 100003,
        )
        call_metadata.append(response)
        certified, certification = certify_projected_proposal(
            proposal,
            selected_tool=selected_tool,
            tool=tool,
            tools=tools,
            request_text=request_text,
        )
        previous_status = str(certification.get("status"))
        metadata["proposal_attempts"].append(
            {
                "proposal_index": attempt,
                "proposal_sha256": hashlib.sha256(
                    json.dumps(
                        proposal,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest(),
                "status": certification.get("status"),
                "certificate_count": certification.get(
                    "certificate_count", 0
                ),
            }
        )
        if certified is not None:
            accepted_action = certified
            accepted_certification = certification
            accepted_index = attempt
            break

    metadata.update(_merge_metadata(call_metadata))
    if accepted_action is None or accepted_certification is None:
        metadata.update(
            {
                "proposal_admitted": False,
                "action_risk_score": 1.0,
                "risk_factors": {"no_certified_proposal": 1.0},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call(
            "refuse", "no proposal has complete source certificates"
        ), metadata

    assert accepted_index is not None
    base_risk = 0.02 if strict else 0.04
    if condition == "tap_r_capc_projected_pivot":
        base_risk = 0.04
    risk = min(ACTION_RISK_THRESHOLD, base_risk + 0.01 * accepted_index)
    metadata.update(
        {
            "proposal_admitted": True,
            "accepted_proposal_index": accepted_index,
            "projected_tool": selected_tool,
            "projected_schema_sha256": hashlib.sha256(
                json.dumps(
                    schema,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
            "evidence_certificates": accepted_certification["certificates"],
            "action_risk_score": risk,
            "risk_factors": {
                "source_certificate_complete": True,
                "election_policy": metadata["election_policy"],
                "proposal_attempt": accepted_index + 1,
            },
            "risk_gate_passed": risk <= ACTION_RISK_THRESHOLD,
            "contract_valid": True,
            "materialized_action_sha256": hashlib.sha256(
                json.dumps(
                    accepted_action,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return accepted_action, metadata

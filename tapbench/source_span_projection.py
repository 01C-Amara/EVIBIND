from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

from .extractive_candidates import canonical_slots
from .validation import action_contract_is_accepted


SOURCE_SPAN_PROJECTION_VERSION = "tapbench.source_span_projection.v2"
SOURCE_SPAN_CERTIFICATE_VERSION = "tapbench.source_span_certificate.v2"
OMIT_SPAN_ID = "OMIT"
_CHARACTER_LANGUAGE_PREFIXES = ("ja", "ko", "zh")
_LEXICAL_UNIT_RE = re.compile(
    r"\w+(?:['@.+:/_-]\w+)*|[^\w\s]",
    re.UNICODE,
)
_SPAN_ID_RE = re.compile(r"SPAN_\d{5}")


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def action_fingerprint(action: dict[str, Any]) -> str:
    return _json_sha256(action)


def _uses_character_units(language: str) -> bool:
    normalized = language.casefold().replace("_", "-")
    return normalized.startswith(_CHARACTER_LANGUAGE_PREFIXES)


def source_units(text: str, language: str) -> list[dict[str, Any]]:
    if _uses_character_units(language):
        spans = [
            (index, index + 1)
            for index, character in enumerate(text)
            if not character.isspace()
        ]
        unit_kind = "unicode_codepoint"
    else:
        spans = [match.span() for match in _LEXICAL_UNIT_RE.finditer(text)]
        unit_kind = "unicode_lexical"
    return [
        {
            "unit_id": f"UNIT_{index:03d}",
            "source_span": [start, end],
            "source_text": text[start:end],
            "unit_kind": unit_kind,
        }
        for index, (start, end) in enumerate(spans)
    ]


def source_span_catalog(text: str, language: str) -> dict[str, Any]:
    units = source_units(text, language)
    spans: list[dict[str, Any]] = []
    for start_index in range(len(units)):
        for end_index in range(start_index, len(units)):
            start = int(units[start_index]["source_span"][0])
            end = int(units[end_index]["source_span"][1])
            spans.append(
                {
                    "span_id": f"SPAN_{len(spans):05d}",
                    "start_unit_id": units[start_index]["unit_id"],
                    "end_unit_id": units[end_index]["unit_id"],
                    "source_span": [start, end],
                    "source_text": text[start:end],
                }
            )
    catalog_hash = _json_sha256(spans)
    return {
        "projection_version": SOURCE_SPAN_PROJECTION_VERSION,
        "language": language,
        "request_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "unit_count": len(units),
        "span_count": len(spans),
        "units": units,
        "spans": spans,
        "catalog_sha256": catalog_hash,
    }


def slot_catalog(
    tool: dict[str, Any],
    *,
    order: str = "forward",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    slots = sorted(
        canonical_slots(tool),
        key=lambda row: (str(row["name"]), str(row["surface_name"])),
    )
    rows: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, Any]] = {}
    for index, slot in enumerate(slots):
        slot_id = f"SLOT_{index:03d}"
        public = {
            "slot_id": slot_id,
            "name": str(slot["surface_name"]),
            "description": str(slot["schema"].get("description", "")),
            "type": str(slot["schema"].get("type", "string")),
            "required": bool(slot.get("required")),
        }
        rows.append(public)
        mapping[slot_id] = slot
    if order == "reverse":
        rows.reverse()
    elif order != "forward":
        raise ValueError(f"unknown slot presentation order: {order}")
    return rows, mapping


def span_proposal_schema(
    tool: dict[str, Any],
    *,
    span_ids: list[str],
    slot_order: str,
) -> dict[str, Any]:
    slots, _ = slot_catalog(tool, order=slot_order)
    if not span_ids or any(not _SPAN_ID_RE.fullmatch(value) for value in span_ids):
        raise ValueError("proposal schema requires valid finite span IDs")
    properties: dict[str, Any] = {}
    for slot in slots:
        values = list(span_ids)
        if not slot["required"]:
            values = [OMIT_SPAN_ID, *values]
        properties[str(slot["slot_id"])] = {
            "type": "string",
            "enum": values,
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


def _coerce_source_value(source: str, schema: dict[str, Any]) -> tuple[Any, str]:
    schema_type = str(schema.get("type", "string")).casefold()
    if schema_type == "string":
        return source, "identity"
    if schema_type == "integer":
        return int(source.replace(",", "")), "parse_integer"
    if schema_type == "number":
        return float(source.replace(",", "")), "parse_number"
    if schema_type == "boolean":
        normalized = source.casefold()
        if normalized == "true":
            return True, "parse_boolean"
        if normalized == "false":
            return False, "parse_boolean"
    raise ValueError(f"source span cannot materialize schema type {schema_type}")


def _certificate_for_span(
    request_text: str,
    language: str,
    catalog: dict[str, Any],
    span: dict[str, Any],
    *,
    slot_id: str,
    canonical_slot: str,
    value: Any,
    transform: str,
) -> dict[str, Any]:
    return {
        "certificate_version": SOURCE_SPAN_CERTIFICATE_VERSION,
        "projection_version": SOURCE_SPAN_PROJECTION_VERSION,
        "language": language,
        "request_sha256": catalog["request_sha256"],
        "span_catalog_sha256": catalog["catalog_sha256"],
        "span_id": span["span_id"],
        "start_unit_id": span["start_unit_id"],
        "end_unit_id": span["end_unit_id"],
        "source_span": list(span["source_span"]),
        "source_text": span["source_text"],
        "slot_id": slot_id,
        "canonical_slot": canonical_slot,
        "transform": transform,
        "value": deepcopy(value),
    }


def replay_span_certificate(
    request_text: str,
    language: str,
    certificate: dict[str, Any],
) -> bool:
    if certificate.get("certificate_version") != SOURCE_SPAN_CERTIFICATE_VERSION:
        return False
    if certificate.get("projection_version") != SOURCE_SPAN_PROJECTION_VERSION:
        return False
    if certificate.get("language") != language:
        return False
    catalog = source_span_catalog(request_text, language)
    if certificate.get("request_sha256") != catalog["request_sha256"]:
        return False
    if certificate.get("span_catalog_sha256") != catalog["catalog_sha256"]:
        return False
    span_id = certificate.get("span_id")
    if not isinstance(span_id, str) or not _SPAN_ID_RE.fullmatch(span_id):
        return False
    by_id = {row["span_id"]: row for row in catalog["spans"]}
    span = by_id.get(span_id)
    if span is None:
        return False
    for key in (
        "start_unit_id",
        "end_unit_id",
        "source_span",
        "source_text",
    ):
        if certificate.get(key) != span.get(key):
            return False
    try:
        value, transform = _coerce_source_value(
            str(span["source_text"]),
            {"type": _schema_type_for_transform(str(certificate.get("transform")))},
        )
    except (TypeError, ValueError):
        return False
    return transform == certificate.get("transform") and value == certificate.get(
        "value"
    )


def _schema_type_for_transform(transform: str) -> str:
    return {
        "identity": "string",
        "parse_integer": "integer",
        "parse_number": "number",
        "parse_boolean": "boolean",
    }.get(transform, "unsupported")


def materialize_span_proposal(
    proposal: dict[str, Any],
    *,
    selected_tool: str,
    tool: dict[str, Any],
    tools: list[dict[str, Any]],
    request_text: str,
    language: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not isinstance(proposal, dict):
        return None, {"status": "proposal_not_object"}
    bindings = proposal.get("bindings")
    if not isinstance(bindings, dict):
        return None, {"status": "bindings_not_object"}
    public_tool_names = {
        str(tool.get("name") or ""),
        str(tool.get("canonical_name") or ""),
    }
    if selected_tool not in public_tool_names:
        return None, {"status": "selected_tool_mismatch"}

    slots, slots_by_id = slot_catalog(tool)
    expected_ids = {str(row["slot_id"]) for row in slots}
    if set(bindings) != expected_ids:
        return None, {
            "status": "binding_slot_set_mismatch",
            "missing": sorted(expected_ids - set(bindings)),
            "extra": sorted(set(bindings) - expected_ids),
        }
    catalog = source_span_catalog(request_text, language)
    spans_by_id = {row["span_id"]: row for row in catalog["spans"]}
    arguments: dict[str, Any] = {}
    certificates: dict[str, Any] = {}
    selected_span_ids: dict[str, str] = {}
    for slot_id in sorted(expected_ids):
        span_id = bindings.get(slot_id)
        slot = slots_by_id[slot_id]
        if not isinstance(span_id, str):
            return None, {"status": "span_id_not_string", "slot_id": slot_id}
        if span_id == OMIT_SPAN_ID:
            if bool(slot.get("required")):
                return None, {"status": "required_slot_omitted", "slot_id": slot_id}
            continue
        if not _SPAN_ID_RE.fullmatch(span_id):
            return None, {"status": "malformed_span_id", "slot_id": slot_id}
        span = spans_by_id.get(span_id)
        if span is None:
            return None, {"status": "unknown_span_id", "slot_id": slot_id}
        try:
            value, transform = _coerce_source_value(
                str(span["source_text"]), slot["schema"]
            )
        except (TypeError, ValueError) as exc:
            return None, {
                "status": "source_materialization_failed",
                "slot_id": slot_id,
                "reason": str(exc),
            }
        canonical_slot = str(slot["name"])
        certificate = _certificate_for_span(
            request_text,
            language,
            catalog,
            span,
            slot_id=slot_id,
            canonical_slot=canonical_slot,
            value=value,
            transform=transform,
        )
        if not replay_span_certificate(request_text, language, certificate):
            return None, {
                "status": "source_certificate_replay_failed",
                "slot_id": slot_id,
            }
        arguments[canonical_slot] = deepcopy(value)
        certificates[canonical_slot] = certificate
        selected_span_ids[canonical_slot] = span_id

    action = {
        "mode": "call",
        "tool": selected_tool,
        "arguments": arguments,
        "payload": {},
    }
    if not action_contract_is_accepted({"tools": tools}, action):
        return None, {"status": "public_contract_rejected"}
    return action, {
        "status": "materialized",
        "projection_version": SOURCE_SPAN_PROJECTION_VERSION,
        "certificate_version": SOURCE_SPAN_CERTIFICATE_VERSION,
        "certificate_count": len(certificates),
        "certificates": certificates,
        "selected_span_ids": selected_span_ids,
        "unit_count": catalog["unit_count"],
        "span_count": catalog["span_count"],
        "span_catalog_sha256": catalog["catalog_sha256"],
        "slot_catalog_sha256": _json_sha256(slots),
        "materialized_action_sha256": action_fingerprint(action),
    }

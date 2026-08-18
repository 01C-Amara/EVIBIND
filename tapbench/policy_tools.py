from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from evibind.core import (
    MATERIALIZATION_CERTIFICATE_VERSION,
    EvidenceTypeRegistry,
    MaterializationCertificate,
    MaterializationError,
    replay_materialization,
)

from .gateway import GatewayError, prepare_upstream_payload
from .json_contract import json_contract_accepts
from .one_call_gateway import compile_one_call_session


POLICY_INIT_VERSION = "evibind.policy_init.v1"
INSPECTOR_VERSION = "evibind.inspector.v1"
REPLAY_TOOL_VERSION = "evibind.replay_tool.v1"


@dataclass(frozen=True)
class PolicyInitialization:
    request: Mapping[str, Any]
    report: Mapping[str, Any]


def _leaf_name(pointer: str) -> str:
    return pointer.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~")


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _stable_evidence_type(pointer: str, schema: Mapping[str, Any]) -> str:
    name = _leaf_name(pointer).casefold()
    schema_type = schema.get("type")
    schema_format = str(schema.get("format", "")).casefold()
    if isinstance(schema.get("enum"), list):
        return "schema_enum"
    if schema_format in {"email", "idn-email"} or "email" in name:
        return "email_address"
    if schema_format in {"uri", "url", "uri-reference"} or name.endswith(
        ("_uri", "_url")
    ):
        return "uri"
    if schema_format == "uuid":
        return "uuid"
    if schema_format == "date" or name == "date" or name.endswith("_date"):
        return "iso_date"
    if "phone" in name or "telephone" in name:
        return "phone_number"
    if "path" in name:
        return "repository_path"
    if schema_type == "integer":
        return "integer"
    if schema_type == "number":
        return "number"
    if schema_type == "boolean":
        return "boolean"
    if "person" in name or name in {"attendee", "recipient", "owner"}:
        return "person_ref"
    if "account" in name:
        return "account_ref"
    if "event" in name:
        return "event_ref"
    if "order" in name:
        return "order_ref"
    if name.endswith("_id") or name == "id":
        return "opaque_registry_id"
    return "opaque_content"


def _annotate_schema(
    schema: dict[str, Any],
    *,
    prefix: str = "",
    registry: EvidenceTypeRegistry,
    changes: list[dict[str, str]],
) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False
        changes.append(
            {
                "path": prefix or "/",
                "annotation": "additionalProperties",
                "value": "false",
            }
        )
    for raw_name, raw_property in properties.items():
        if not isinstance(raw_name, str) or not isinstance(
            raw_property,
            dict,
        ):
            continue
        pointer = prefix + "/" + _escape_pointer(raw_name)
        nested = raw_property.get("properties")
        if raw_property.get("type") == "object" and isinstance(nested, dict):
            _annotate_schema(
                raw_property,
                prefix=pointer,
                registry=registry,
                changes=changes,
            )
            continue
        evidence_type = raw_property.get("x-evibind-evidence-type")
        if not isinstance(evidence_type, str) or not evidence_type:
            evidence_type = _stable_evidence_type(pointer, raw_property)
            raw_property["x-evibind-evidence-type"] = evidence_type
            changes.append(
                {
                    "path": pointer,
                    "annotation": "x-evibind-evidence-type",
                    "value": evidence_type,
                }
            )
        spec = registry.get(evidence_type)
        if "x-evibind-sources" not in raw_property:
            sources = (
                [f"state.{_leaf_name(pointer)}"]
                if evidence_type == "opaque_registry_id"
                else ["user.current_turn"]
            )
            if "default" in raw_property:
                sources.append("schema.default")
            raw_property["x-evibind-sources"] = sources
            changes.append(
                {
                    "path": pointer,
                    "annotation": "x-evibind-sources",
                    "value": ",".join(sources),
                }
            )
        if "x-evibind-criticality" not in raw_property:
            criticality = (
                "content"
                if spec.value_class == "opaque_content"
                else "effect"
                if spec.value_class == "effect_bearing"
                else "target"
            )
            raw_property["x-evibind-criticality"] = criticality
            changes.append(
                {
                    "path": pointer,
                    "annotation": "x-evibind-criticality",
                    "value": criticality,
                }
            )
        for annotation, value in (
            ("x-evibind-value-class", spec.value_class),
            ("x-evibind-on-ambiguity", "clarify"),
        ):
            if annotation not in raw_property:
                raw_property[annotation] = value
                changes.append(
                    {
                        "path": pointer,
                        "annotation": annotation,
                        "value": str(value),
                    }
                )
        if (
            evidence_type
            not in {"schema_enum", "opaque_registry_id", "effect_manifest"}
            and "x-evibind-extraction-cue" not in raw_property
        ):
            raw_property["x-evibind-extraction-cue"] = _leaf_name(pointer)
            changes.append(
                {
                    "path": pointer,
                    "annotation": "x-evibind-extraction-cue",
                    "value": _leaf_name(pointer),
                }
            )


def initialize_request_policy(
    request_payload: Mapping[str, Any],
    *,
    policy_epoch: str = "1",
) -> PolicyInitialization:
    if not isinstance(request_payload, Mapping):
        raise GatewayError("request JSON must be an object")
    initialized = deepcopy(dict(request_payload))
    raw_tools = initialized.get("tools")
    if not isinstance(raw_tools, list) or not raw_tools:
        raise GatewayError("request must contain at least one tool")
    registry = EvidenceTypeRegistry.standard()
    changes: list[dict[str, str]] = []
    tool_count = 0
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            raise GatewayError("tools must contain objects")
        function = (
            raw_tool.get("function") if raw_tool.get("type") == "function" else raw_tool
        )
        if not isinstance(function, dict):
            raise GatewayError("tool function must be an object")
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }
            function["parameters"] = parameters
        if parameters.get("type") != "object":
            raise GatewayError("tool parameters must have type=object")
        _annotate_schema(
            parameters,
            registry=registry,
            changes=changes,
        )
        tool_count += 1
    options = initialized.get("evibind")
    if options is None:
        options = {}
        initialized["evibind"] = options
    if not isinstance(options, dict):
        raise GatewayError("evibind must be an object")
    if "policy_epoch" not in options:
        options["policy_epoch"] = str(policy_epoch)
        changes.append(
            {
                "path": "evibind",
                "annotation": "policy_epoch",
                "value": str(policy_epoch),
            }
        )
    return PolicyInitialization(
        request=initialized,
        report={
            "version": POLICY_INIT_VERSION,
            "tool_count": tool_count,
            "change_count": len(changes),
            "changes": changes,
            "policy_epoch": str(options["policy_epoch"]),
        },
    )


def inspect_request_policy(
    request_payload: Mapping[str, Any],
    *,
    handle_secret: bytes,
) -> dict[str, Any]:
    request = deepcopy(dict(request_payload))
    upstream, options, tools = prepare_upstream_payload(request)
    session = compile_one_call_session(
        request_payload=request,
        upstream_payload=upstream,
        options=options,
        tools=tools,
        handle_secret=handle_secret,
        include_diagnostics=True,
    )
    tool_rows = []
    for tool in session.policy.tools:
        slots = []
        for slot in tool.slots:
            candidates = [
                candidate.public_view()
                for candidate in session.candidates.candidates.values()
                if candidate.witness.tool_id == tool.tool_id
                and candidate.witness.destination_scope == slot.destination_scope
            ]
            slots.append(
                {
                    "destination": slot.destination_scope,
                    "required": slot.required,
                    "evidence_type": slot.evidence_type,
                    "value_class": slot.value_class,
                    "criticality": slot.criticality,
                    "sources": sorted(slot.sources),
                    "transforms": sorted(slot.transforms),
                    "ambiguity": slot.ambiguity,
                    "candidate_count": len(candidates),
                    "candidates": sorted(
                        candidates,
                        key=lambda row: row["candidate_id"],
                    ),
                }
            )
        missing = sorted(
            slot["destination"]
            for slot in slots
            if slot["required"] and slot["candidate_count"] == 0
        )
        tool_rows.append(
            {
                "tool_id": tool.tool_id,
                "policy_epoch": tool.policy_epoch,
                "contract_version": tool.contract_version,
                "missing_required": missing,
                "slots": slots,
            }
        )
    return {
        "version": INSPECTOR_VERSION,
        "request_digest": session.context.request_digest,
        "metrics": session.candidates.metrics(),
        "tools": tool_rows,
        "rejections": [
            rejection.to_dict() for rejection in session.candidates.rejections
        ],
    }


def find_materialization_certificate(value: Any) -> Mapping[str, Any]:
    found: list[Mapping[str, Any]] = []

    def walk(item: Any) -> None:
        if isinstance(item, Mapping):
            if item.get("version") == MATERIALIZATION_CERTIFICATE_VERSION:
                found.append(item)
                return
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    if len(found) != 1:
        raise MaterializationError("expected exactly one materialization certificate")
    return found[0]


def replay_request_certificate(
    request_payload: Mapping[str, Any],
    certificate_payload: Any,
    *,
    handle_secret: bytes,
) -> dict[str, Any]:
    request = deepcopy(dict(request_payload))
    upstream, options, tools = prepare_upstream_payload(request)
    session = compile_one_call_session(
        request_payload=request,
        upstream_payload=upstream,
        options=options,
        tools=tools,
        handle_secret=handle_secret,
        include_diagnostics=False,
    )
    certificate = MaterializationCertificate.from_dict(
        find_materialization_certificate(certificate_payload)
    )
    action = replay_materialization(
        certificate,
        context=session.context,
        policy=session.policy,
        evidence_types=session.evidence_types,
        transforms=session.transforms,
        issuer=session.issuer,
        allow_expired=True,
    )
    schema = session.tools.get(action.tool_id)
    if schema is None or not json_contract_accepts(
        action.arguments,
        schema.get("parameters", {}),
    ):
        raise MaterializationError("replayed action failed the current JSON contract")
    return {
        "version": REPLAY_TOOL_VERSION,
        "verified": True,
        "request_digest": session.context.request_digest,
        "tool_id": action.tool_id,
        "arguments": action.arguments,
        "manifest_digest": action.manifest_digest,
    }

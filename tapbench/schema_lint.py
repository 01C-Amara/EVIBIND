from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

from evibind.core import EvidenceTypeRegistry

from .schema_walk import iter_schema_properties, validate_required_properties


SCHEMA_LINTER_VERSION = "evibind.schema_linter.v2"
SLOT_ROLES = {"control", "content", "derived", "identifier", "defaultable"}
RESOLUTION_TYPES = {
    "enumerated",
    "extractive",
    "normalizable",
    "referential",
    "defaultable",
    "generative",
}
SEMANTIC_ENVELOPES = {"infer", "free_text", "opaque_atom", "uri", "head_number"}
EVIDENCE_REGISTRY = EvidenceTypeRegistry.standard()
EVIDENCE_TYPES = frozenset(EVIDENCE_REGISTRY.names())
VALUE_CLASSES = {"authority_bearing", "opaque_content", "effect_bearing"}
CRITICALITIES = {"target", "control", "content", "effect"}
AMBIGUITY_POLICIES = {"clarify", "reject", "confirm"}
SOURCE_PATTERN = re.compile(r"^(?:user|tool|schema|state)\.[A-Za-z0-9_.-]+$")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _public(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _public(item)
            for key, item in value.items()
            if not str(key).startswith(("x-evibind-", "x-tap-"))
        }
    if isinstance(value, list):
        return [_public(item) for item in value]
    return deepcopy(value)


def _issue(
    issues: list[dict[str, str]],
    severity: str,
    code: str,
    path: str,
    message: str,
) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "path": path,
            "message": message,
        }
    )


def _raw_tools(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        tools = payload.get("tools")
        return tools if isinstance(tools, list) else []
    return payload if isinstance(payload, list) else []


def _validate_v2_annotations(
    raw_prop: dict[str, Any],
    path: str,
    issues: list[dict[str, str]],
) -> str | None:
    evidence_type = raw_prop.get("x-evibind-evidence-type")
    sources = raw_prop.get("x-evibind-sources")
    transforms = raw_prop.get("x-evibind-transforms")
    criticality = raw_prop.get("x-evibind-criticality")
    value_class = raw_prop.get("x-evibind-value-class")
    ambiguity = raw_prop.get("x-evibind-on-ambiguity")
    envelope = raw_prop.get("x-evibind-semantic-envelope")

    if evidence_type is not None and evidence_type not in EVIDENCE_TYPES:
        _issue(
            issues,
            "error",
            "annotation.evidence_type",
            path + ".x-evibind-evidence-type",
            f"Unsupported evidence type: {evidence_type!r}.",
        )
    if sources is not None and (
        not isinstance(sources, list)
        or not sources
        or not all(
            isinstance(source, str) and SOURCE_PATTERN.fullmatch(source)
            for source in sources
        )
    ):
        _issue(
            issues,
            "error",
            "annotation.sources",
            path + ".x-evibind-sources",
            "Sources must be a non-empty array of namespaced source IDs.",
        )
    if transforms is not None and (
        not isinstance(transforms, list)
        or not transforms
        or not all(isinstance(transform, str) and transform for transform in transforms)
    ):
        _issue(
            issues,
            "error",
            "annotation.transforms",
            path + ".x-evibind-transforms",
            "Transforms must be a non-empty array of transform IDs.",
        )
    if criticality is not None and criticality not in CRITICALITIES:
        _issue(
            issues,
            "error",
            "annotation.criticality",
            path + ".x-evibind-criticality",
            f"Unsupported criticality: {criticality!r}.",
        )
    if value_class is not None and value_class not in VALUE_CLASSES:
        _issue(
            issues,
            "error",
            "annotation.value_class",
            path + ".x-evibind-value-class",
            f"Unsupported value class: {value_class!r}.",
        )
    if (
        isinstance(evidence_type, str)
        and evidence_type in EVIDENCE_TYPES
        and value_class in VALUE_CLASSES
        and value_class != EVIDENCE_REGISTRY.get(evidence_type).value_class
    ):
        _issue(
            issues,
            "error",
            "annotation.value_class_mismatch",
            path,
            "Declared value class does not match the evidence type.",
        )
    if ambiguity is not None and ambiguity not in AMBIGUITY_POLICIES:
        _issue(
            issues,
            "error",
            "annotation.ambiguity",
            path + ".x-evibind-on-ambiguity",
            f"Unsupported ambiguity policy: {ambiguity!r}.",
        )
    if evidence_type is not None and sources is None:
        _issue(
            issues,
            "warning",
            "annotation.sources_missing",
            path,
            "Declare admissible sources for each explicit evidence type.",
        )
    if envelope is not None:
        _issue(
            issues,
            "warning",
            "annotation.semantic_envelope_deprecated",
            path + ".x-evibind-semantic-envelope",
            "Semantic envelopes are legacy; declare an evidence type instead.",
        )
    return evidence_type if isinstance(evidence_type, str) else None


def lint_tool_schemas(payload: Any) -> dict[str, Any]:
    tools = _raw_tools(payload)
    issues: list[dict[str, str]] = []
    if not tools:
        _issue(issues, "error", "tools.empty", "tools", "No tool schemas found.")
    names: list[str] = []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(tools):
        base = f"tools[{index}]"
        if not isinstance(raw, dict):
            _issue(issues, "error", "tool.not_object", base, "Tool must be an object.")
            continue
        function = raw.get("function") if raw.get("type") == "function" else raw
        if not isinstance(function, dict):
            _issue(
                issues,
                "error",
                "function.not_object",
                base + ".function",
                "Function definition must be an object.",
            )
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            _issue(
                issues,
                "error",
                "function.name_missing",
                base + ".function.name",
                "Function name must be a non-empty string.",
            )
            name = f"<invalid-{index}>"
        names.append(name)
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            _issue(
                issues,
                "error",
                "parameters.not_object",
                base + ".function.parameters",
                "Parameters must be a JSON Schema object.",
            )
            continue
        if parameters.get("type") != "object":
            _issue(
                issues,
                "error",
                "parameters.type",
                base + ".function.parameters.type",
                "Top-level parameter schema must have type=object.",
            )
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            _issue(
                issues,
                "error",
                "properties.not_object",
                base + ".function.parameters.properties",
                "Parameter properties must be an object.",
            )
            continue
        required = parameters.get("required", [])
        if not isinstance(required, list) or not all(
            isinstance(value, str) for value in required
        ):
            _issue(
                issues,
                "error",
                "required.not_string_list",
                base + ".function.parameters.required",
                "Required must be a list of property names.",
            )
            required = []
        unknown_required = sorted(set(required) - set(properties))
        if unknown_required:
            _issue(
                issues,
                "error",
                "required.unknown_property",
                base + ".function.parameters.required",
                f"Unknown required properties: {', '.join(unknown_required)}.",
            )
        if parameters.get("additionalProperties") is not False:
            _issue(
                issues,
                "warning",
                "parameters.open_object",
                base + ".function.parameters.additionalProperties",
                "Set additionalProperties=false for a closed action contract.",
            )
        for slot, path, raw_prop, is_container in iter_schema_properties(
            properties,
            base_path=(f"{base}.function.parameters.properties"),
        ):
            if not isinstance(raw_prop, dict):
                _issue(
                    issues,
                    "error",
                    "property.not_object",
                    path,
                    "Property schema must be an object.",
                )
                continue
            if is_container:
                required_valid, unknown_required = validate_required_properties(
                    raw_prop
                )
                if not required_valid:
                    _issue(
                        issues,
                        "error",
                        "required.not_string_list",
                        path + ".required",
                        "Required must be a list of property names.",
                    )
                if unknown_required:
                    _issue(
                        issues,
                        "error",
                        "required.unknown_property",
                        path + ".required",
                        (
                            "Unknown required properties: "
                            + ", ".join(unknown_required)
                            + "."
                        ),
                    )
                if raw_prop.get("additionalProperties") is not False:
                    _issue(
                        issues,
                        "warning",
                        "parameters.nested_open_object",
                        path + ".additionalProperties",
                        (
                            "Set additionalProperties=false for each nested "
                            "object contract."
                        ),
                    )
                continue
            legacy = sorted(key for key in raw_prop if str(key).startswith("x-tap-"))
            if legacy:
                _issue(
                    issues,
                    "warning",
                    "annotation.legacy_prefix",
                    path,
                    "Use x-evibind-* public annotations instead of x-tap-*.",
                )
            role = raw_prop.get("x-evibind-slot-role")
            resolution = raw_prop.get("x-evibind-resolution-type")
            envelope = raw_prop.get("x-evibind-semantic-envelope")
            evidence_type = _validate_v2_annotations(
                raw_prop,
                path,
                issues,
            )
            if role is not None and role not in SLOT_ROLES:
                _issue(
                    issues,
                    "error",
                    "annotation.slot_role",
                    path + ".x-evibind-slot-role",
                    f"Unsupported slot role: {role!r}.",
                )
            if resolution is not None and resolution not in RESOLUTION_TYPES:
                _issue(
                    issues,
                    "error",
                    "annotation.resolution_type",
                    path + ".x-evibind-resolution-type",
                    f"Unsupported resolution type: {resolution!r}.",
                )
            if envelope is not None and envelope not in SEMANTIC_ENVELOPES:
                _issue(
                    issues,
                    "error",
                    "annotation.semantic_envelope",
                    path + ".x-evibind-semantic-envelope",
                    f"Unsupported semantic envelope: {envelope!r}.",
                )
            if (role is None) != (resolution is None):
                _issue(
                    issues,
                    "warning",
                    "annotation.partial_contract",
                    path,
                    "Declare slot role and resolution type together.",
                )
            inferred_critical = str(slot).endswith("_id") or str(slot) in {
                "amount",
                "recipient",
                "account",
                "owner",
                "destination",
                "date",
                "time",
            }
            if inferred_critical and evidence_type is None:
                _issue(
                    issues,
                    "warning",
                    "annotation.critical_slot_untyped",
                    path,
                    "Action-critical-looking slot has no evidence type.",
                )
            if role == "identifier" and resolution not in {None, "referential"}:
                _issue(
                    issues,
                    "warning",
                    "annotation.identifier_resolution",
                    path,
                    "Identifier slots should normally use referential resolution.",
                )
            if (
                resolution in {"extractive", "normalizable"}
                and "enum" not in raw_prop
                and "x-evibind-extraction-cue" not in raw_prop
            ):
                _issue(
                    issues,
                    "warning",
                    "annotation.extraction_cue_missing",
                    path,
                    "Add an extraction cue for predictable request-span binding.",
                )
        normalized.append(
            {
                "name": name,
                "description": str(function.get("description", "")),
                "parameters": deepcopy(parameters),
            }
        )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        _issue(
            issues,
            "error",
            "function.duplicate_name",
            "tools",
            f"Duplicate function names: {', '.join(duplicates)}.",
        )
    errors = sum(issue["severity"] == "error" for issue in issues)
    warnings = sum(issue["severity"] == "warning" for issue in issues)
    return {
        "schema_version": SCHEMA_LINTER_VERSION,
        "valid": errors == 0,
        "tool_count": len(normalized),
        "error_count": errors,
        "warning_count": warnings,
        "issues": issues,
        "private_contract_sha256": _canonical_sha256(normalized),
        "provider_schema_sha256": _canonical_sha256(_public(normalized)),
    }

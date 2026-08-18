from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIRMATORY_VERSION = "evibind.fresh_family_confirmatory.v1"


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def family_id(api_name: str, version: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", api_name.casefold()).strip("_")[:48]
    suffix = hashlib.sha256(f"{api_name}:{version}".encode()).hexdigest()[:10]
    return f"openapi_{slug or 'api'}_{suffix}"


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _compatible_properties(tool: Mapping[str, Any]) -> list[str]:
    parameters = tool.get("function", {}).get("parameters", {})
    properties = parameters.get("properties")
    if not isinstance(properties, Mapping):
        return []
    return [
        name
        for name, schema in sorted(properties.items())
        if isinstance(name, str)
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,30}", name)
        and isinstance(schema, Mapping)
        and schema.get("type") == "string"
        and not schema.get("enum")
    ][:3]


def normalized_confirmatory_tools(
    tools: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str, tuple[str, ...]]:
    selected_index = next(
        (
            index
            for index, tool in enumerate(tools)
            if _compatible_properties(tool)
        ),
        None,
    )
    if selected_index is None:
        raise ValueError("family has no flat string-valued operation")
    selected = deepcopy(dict(tools[selected_index]))
    function = selected["function"]
    properties = _compatible_properties(selected)
    function["parameters"] = {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": {
            name: {
                "type": "string",
                "description": str(
                    function.get("parameters", {})
                    .get("properties", {})
                    .get(name, {})
                    .get("description", "")
                )[:240],
                "x-evibind-criticality": "target",
                "x-evibind-value-class": "authority_bearing",
                "x-evibind-evidence-type": "opaque_registry_id",
                "x-evibind-sources": ["user.current_turn"],
                "x-evibind-transforms": ["identity"],
                "x-evibind-on-ambiguity": "clarify",
                "x-evibind-extraction-cue": name,
            }
            for name in properties
        },
    }
    output = [selected]
    for index, tool in enumerate(tools):
        if index == selected_index or len(output) >= 4:
            continue
        distractor = deepcopy(dict(tool))
        distractor["function"]["parameters"] = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        output.append(distractor)
    return output, str(function["name"]), tuple(properties)


def build_family_cases(
    *,
    family: str,
    tools: Sequence[Mapping[str, Any]],
    cases_per_family: int,
) -> list[dict[str, Any]]:
    if cases_per_family <= 0:
        raise ValueError("cases_per_family must be positive")
    normalized, tool_id, properties = normalized_confirmatory_tools(tools)
    rows: list[dict[str, Any]] = []
    for case_index in range(cases_per_family):
        assignments: list[str] = []
        arguments: dict[str, str] = {}
        bindings: list[dict[str, Any]] = []
        for property_index, name in enumerate(properties):
            stem = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-") or "value"
            distractor = f"old-{case_index:02d}-{property_index:02d}-{stem}"
            gold = f"current-{case_index:02d}-{property_index:02d}-{stem}"
            assignments.append(
                f"earlier option {name}={distractor}; final choice {name}={gold}"
            )
            arguments[name] = gold
            bindings.append(
                {
                    "tool_id": tool_id,
                    "destination": "/" + _pointer_token(name),
                    "value": gold,
                }
            )
        message = f"Call {tool_id}. " + "; ".join(assignments) + "."
        case_id = "cfm-" + hashlib.sha256(
            f"{family}:{case_index}:{message}".encode()
        ).hexdigest()[:20]
        rows.append(
            {
                "version": "evibind.evibench.v1",
                "case_id": case_id,
                "family": family,
                "adversary": "correction_or_negation_hard_distractor",
                "authoring": {
                    "language": "en",
                    "phenomena": ["duplicate_or_ambiguous_candidate"],
                    "request_author_id": "deterministic-confirmatory-generator-v1",
                    "split": "confirmatory",
                },
                "request": {
                    "model": "evibind-confirmatory-model",
                    "messages": [{"role": "user", "content": message}],
                    "tools": deepcopy(normalized),
                    "evibind": {},
                },
                "expected": {
                    "mode": "call",
                    "tool_id": tool_id,
                    "arguments": arguments,
                    "critical_destinations": [
                        "/" + _pointer_token(name) for name in properties
                    ],
                    "admissible_bindings": bindings,
                },
            }
        )
    return rows


def verify_freeze(repository: str | Path, manifest_path: str | Path) -> None:
    root = Path(repository)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_any_confirmatory_model_output":
        raise ValueError("confirmatory manifest is not pre-outcome frozen")
    for relative, expected in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file() or sha256_path(path) != expected:
            raise ValueError(f"confirmatory freeze digest drift: {relative}")

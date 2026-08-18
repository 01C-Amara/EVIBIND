from __future__ import annotations

import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from evibind.core.derivations import canonical_json

from .evibench import EviBenchError, compile_case
from .json_contract import json_contract_accepts
from .verified_ranker import LinearCandidateRanker, prune_candidate_table


SELECTOR_STUDY_VERSION = "evibind.oracle_selector_phase.v1"
INDEXED_ACTION_TOOL = "evibind_indexed_action"
DYNAMIC_ACTION_TOOL = "evibind_dynamic_action"
ROUTER_TOOL = "evibind_route"
DISTRACTOR_COUNTS = (0, 1, 3, 7)
ROUTING_REGIMES = ("binding_only", "full", "two_stage")
INTERFACES = ("dynamic_enum", "indexed_tool", "indexed_json")
AMBIGUITY_POLICIES = ("strict", "model_selection")
_MISSING = object()


def _pointer_tokens(pointer: str) -> tuple[str, ...]:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise EviBenchError(f"invalid JSON Pointer: {pointer!r}")
    return tuple(
        token.replace("~1", "/").replace("~0", "~")
        for token in pointer.removeprefix("/").split("/")
        if token
    )


def _get_pointer(value: Any, pointer: str) -> Any:
    current = value
    for token in _pointer_tokens(pointer):
        if isinstance(current, Mapping) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit():
            index = int(token)
            if index >= len(current):
                return _MISSING
            current = current[index]
        else:
            return _MISSING
    return current


def _tool_rows(case: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tools = case.get("request", {}).get("tools", [])
    if not isinstance(tools, list):
        raise EviBenchError("case tools must be an array")
    output: list[Mapping[str, Any]] = []
    for row in tools:
        function = row.get("function") if isinstance(row, Mapping) else None
        if not isinstance(function, Mapping) or not isinstance(
            function.get("name"), str
        ):
            raise EviBenchError("case tool row is invalid")
        output.append(function)
    return output


def _schema_leaf_destinations(
    schema: Mapping[str, Any],
    *,
    prefix: str = "",
    parent_required: bool = True,
) -> list[tuple[str, bool]]:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return []
    required = {
        str(value) for value in schema.get("required", []) if isinstance(value, str)
    }
    output: list[tuple[str, bool]] = []
    for surface, child in properties.items():
        if not isinstance(surface, str) or not isinstance(child, Mapping):
            continue
        pointer = prefix + "/" + surface.replace("~", "~0").replace("/", "~1")
        child_required = parent_required and surface in required
        if child.get("type") == "object" and isinstance(
            child.get("properties"), Mapping
        ):
            output.extend(
                _schema_leaf_destinations(
                    child,
                    prefix=pointer,
                    parent_required=child_required,
                )
            )
        else:
            output.append((pointer, child_required))
    return output


def _schema_leaf_rows(
    schema: Mapping[str, Any],
    *,
    prefix: str = "",
    parent_required: bool = True,
) -> list[dict[str, Any]]:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return []
    required = {
        str(value) for value in schema.get("required", []) if isinstance(value, str)
    }
    output: list[dict[str, Any]] = []
    for surface, child in properties.items():
        if not isinstance(surface, str) or not isinstance(child, Mapping):
            continue
        pointer = prefix + "/" + surface.replace("~", "~0").replace("/", "~1")
        child_required = parent_required and surface in required
        if child.get("type") == "object" and isinstance(
            child.get("properties"), Mapping
        ):
            output.extend(
                _schema_leaf_rows(
                    child,
                    prefix=pointer,
                    parent_required=child_required,
                )
            )
        else:
            output.append(
                {
                    "destination": pointer,
                    "required": child_required,
                    "criticality": str(
                        child.get(
                            "x-evibind-criticality",
                            child.get("x-tap-criticality", "target"),
                        )
                    ).casefold(),
                    "value_class": str(
                        child.get(
                            "x-evibind-value-class",
                            child.get("x-tap-value-class", "authority_bearing"),
                        )
                    ).casefold(),
                }
            )
    return output


def _protected_critical_destinations(case: Mapping[str, Any]) -> set[str]:
    expected = case["expected"]
    tool_id = expected.get("tool_id")
    if not isinstance(tool_id, str):
        return set()
    tool = next(
        (row for row in _tool_rows(case) if row.get("name") == tool_id),
        None,
    )
    if tool is None:
        return set()
    declared = {str(value) for value in expected["critical_destinations"]}
    return {
        str(row["destination"])
        for row in _schema_leaf_rows(tool.get("parameters", {}))
        if any(
            row["destination"] == scope
            or str(row["destination"]).startswith(scope + "/")
            for scope in declared
        )
        and not (
            row["criticality"] == "content"
            and row["value_class"] == "opaque_content"
        )
    }


def _gold_values(case: Mapping[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    tool_id = expected.get("tool_id")
    output: dict[str, Any] = {}
    for row in expected.get("admissible_bindings", []):
        if row.get("tool_id") == tool_id and isinstance(row.get("destination"), str):
            output[str(row["destination"])] = deepcopy(row.get("value"))
    arguments = expected.get("arguments")
    if isinstance(arguments, Mapping):
        tool = next(
            (
                row
                for row in _tool_rows(case)
                if row.get("name") == expected.get("tool_id")
            ),
            None,
        )
        destinations = (
            _schema_leaf_destinations(tool.get("parameters", {}))
            if tool is not None
            else []
        )
        for destination, _required in destinations:
            value = _get_pointer(arguments, destination)
            if value is not _MISSING:
                output.setdefault(destination, deepcopy(value))
    return output


def _required_destinations(case: Mapping[str, Any], *, critical_only: bool) -> list[str]:
    expected = case["expected"]
    tool_id = expected.get("tool_id")
    if not isinstance(tool_id, str):
        return []
    tool = next(
        (row for row in _tool_rows(case) if row.get("name") == tool_id),
        None,
    )
    if tool is None:
        raise EviBenchError(f"gold tool is not declared: {tool_id}")
    parameters = tool.get("parameters", {})
    required = {
        destination
        for destination, is_required in _schema_leaf_destinations(parameters)
        if is_required
    }
    if critical_only:
        required &= _protected_critical_destinations(case)
    return sorted(required)


def _critical_gold_destinations(case: Mapping[str, Any]) -> list[str]:
    if case["expected"]["mode"] != "call":
        return []
    gold = _gold_values(case)
    tool_id = str(case["expected"]["tool_id"])
    tool = next(row for row in _tool_rows(case) if row.get("name") == tool_id)
    critical_scopes = _protected_critical_destinations(case)
    return sorted(
        destination
        for destination, _required in _schema_leaf_destinations(
            tool.get("parameters", {})
        )
        if destination in gold
        and destination in critical_scopes
    )


def _candidate_rows(session: Any, tool_id: str, destination: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate in session.candidates.candidates.values():
        if (
            candidate.witness.tool_id == tool_id
            and candidate.witness.destination_scope == destination
        ):
            public = candidate.public_view()
            output.append(
                {
                    "value": deepcopy(candidate.value),
                    "display": str(public.get("display", "verified evidence")),
                    "source": "actual_same_destination",
                    "candidate_id": candidate.candidate_id,
                }
            )
    return output


def _same_type_cross_slot_rows(
    session: Any,
    tool_id: str,
    destination: str,
) -> list[dict[str, Any]]:
    try:
        evidence_type = session.policy.tool(tool_id).slot(destination).evidence_type
    except ValueError:
        return []
    output: list[dict[str, Any]] = []
    for candidate in session.candidates.candidates.values():
        try:
            candidate_type = session.policy.tool(candidate.witness.tool_id).slot(
                candidate.witness.destination_scope
            ).evidence_type
        except ValueError:
            continue
        if candidate_type != evidence_type or (
            candidate.witness.tool_id == tool_id
            and candidate.witness.destination_scope == destination
        ):
            continue
        public = candidate.public_view()
        output.append(
            {
                "value": deepcopy(candidate.value),
                "display": str(public.get("display", "verified evidence")),
                "source": "same_type_cross_slot",
                "candidate_id": candidate.candidate_id,
            }
        )
    return output


def _mutated_value(value: Any, offset: int) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + offset + 1
    if isinstance(value, float):
        return value + float(offset + 1)
    if isinstance(value, str):
        if "@" in value and value.count("@") == 1:
            local, domain = value.split("@")
            return f"{local}{offset + 1}@{domain}"
        return f"{value}-alt{offset + 1}"
    return {"alternative": offset + 1, "value": deepcopy(value)}


def _controlled_synthetic_alternatives(
    value: Any,
    count: int,
) -> list[dict[str, Any]]:
    """Return as many distinct same-type alternatives as the domain permits."""
    output: list[dict[str, Any]] = []
    seen = {canonical_json(value)}
    # Boolean domains contain only one non-gold value.  The bounded attempt
    # count also prevents any future finite-domain mutator from looping.
    for offset in range(max(16, count * 4)):
        alternative = _mutated_value(value, offset)
        key = canonical_json(alternative)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "value": alternative,
                "display": "controlled same-type alternative: " + key,
                "source": "controlled_same_type_synthetic",
                "candidate_id": None,
            }
        )
        if len(output) == count:
            break
    return output


def _dedupe_candidates(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = canonical_json(row.get("value"))
        if key in seen:
            continue
        seen.add(key)
        output.append(deepcopy(dict(row)))
    return output


def _shuffle_seed(case_id: str, regime: str, destination: str) -> int:
    digest = hashlib.sha256(
        f"{case_id}|{regime}|{destination}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def build_catalog(
    case: Mapping[str, Any],
    *,
    regime: str,
    ambiguity_policy: str = "model_selection",
    ranker: LinearCandidateRanker | None = None,
) -> dict[str, Any]:
    if ambiguity_policy not in AMBIGUITY_POLICIES:
        raise EviBenchError(f"unknown ambiguity policy: {ambiguity_policy}")
    verified = regime.startswith("verified_top_")
    if regime != "actual" and not regime.startswith("oracle_") and not verified:
        raise EviBenchError(f"unknown candidate regime: {regime}")
    distractor_count = (
        int(regime.split("_")[-1]) if regime.startswith("oracle_") else None
    )
    if distractor_count is not None and distractor_count not in DISTRACTOR_COUNTS:
        raise EviBenchError(f"unsupported distractor count: {distractor_count}")

    expected = case["expected"]
    tool_id = expected.get("tool_id")
    session = _compile_with_proposer(
        case,
        "broad_typed" if verified else "deterministic",
    )
    if verified:
        if ranker is None:
            raise EviBenchError("verified candidate regime requires a ranker")
        top_k = int(regime.rsplit("_", 1)[1])
        session = replace(
            session,
            candidates=prune_candidate_table(
                session.candidates,
                session.context,
                ranker,
                top_k=top_k,
            ),
        )
    tools = _tool_rows(case)
    tool_catalog = [
        {
            "tool_index": index,
            "tool_id": str(tool["name"]),
            "description": str(tool.get("description", "")),
        }
        for index, tool in enumerate(tools)
    ]
    slots: list[dict[str, Any]] = []
    gold_values = _gold_values(case)
    if expected["mode"] == "call" and isinstance(tool_id, str):
        destinations = _critical_gold_destinations(case)
        for slot_index, destination in enumerate(destinations):
            gold_value = gold_values[destination]
            actual = _candidate_rows(session, tool_id, destination)
            cross_slot = _same_type_cross_slot_rows(session, tool_id, destination)
            if regime == "actual" or verified:
                candidate_rows = actual
            else:
                exact = next(
                    (
                        row
                        for row in actual
                        if canonical_json(row["value"]) == canonical_json(gold_value)
                    ),
                    None,
                )
                gold_row = (
                    exact
                    if exact is not None
                    else {
                        "value": deepcopy(gold_value),
                        "display": "request evidence: " + canonical_json(gold_value),
                        "source": "oracle_injected_replayable_proxy",
                        "candidate_id": None,
                    }
                )
                alternatives = _dedupe_candidates(
                    [
                        row
                        for row in [*actual, *cross_slot]
                        if canonical_json(row["value"]) != canonical_json(gold_value)
                    ]
                )
                alternatives = _dedupe_candidates(
                    [
                        *alternatives,
                        *_controlled_synthetic_alternatives(
                            gold_value,
                            int(distractor_count),
                        ),
                    ]
                )
                candidate_rows = [gold_row, *alternatives[: int(distractor_count)]]

            candidate_rows = _dedupe_candidates(candidate_rows)
            rng = random.Random(_shuffle_seed(str(case["case_id"]), regime, destination))
            rng.shuffle(candidate_rows)
            serialized: list[dict[str, Any]] = []
            for candidate_index, row in enumerate(candidate_rows):
                serialized.append(
                    {
                        "candidate_index": candidate_index,
                        "candidate_token": f"s{slot_index}_c{candidate_index}",
                        "display": row["display"],
                        "source": row["source"],
                        "is_gold": canonical_json(row["value"])
                        == canonical_json(gold_value),
                        "value": deepcopy(row["value"]),
                    }
                )
            slots.append(
                {
                    "slot_index": slot_index,
                    "destination": destination,
                    "required": destination
                    in _required_destinations(case, critical_only=True),
                    "candidates": serialized,
                }
            )

    return {
        "case_id": case["case_id"],
        "expected_mode": expected["mode"],
        "expected_tool_id": tool_id,
        "expected_tool_index": next(
            (
                row["tool_index"]
                for row in tool_catalog
                if row["tool_id"] == tool_id
            ),
            None,
        ),
        "candidate_regime": regime,
        "ambiguity_policy": ambiguity_policy,
        "tool_catalog": tool_catalog,
        "slots": slots,
    }


def _compile_with_proposer(
    case: Mapping[str, Any],
    candidate_proposer: str,
) -> Any:
    if candidate_proposer == "deterministic":
        return compile_case(case)
    amended = deepcopy(dict(case))
    request = deepcopy(dict(case["request"]))
    options = deepcopy(dict(request.get("evibind", {})))
    options["candidate_proposer"] = candidate_proposer
    request["evibind"] = options
    amended["request"] = request
    return compile_case(amended)


def compiler_recoverability(
    cases: Sequence[Mapping[str, Any]],
    *,
    candidate_proposer: str = "deterministic",
) -> dict[str, Any]:
    call_cases = [case for case in cases if case["expected"]["mode"] == "call"]
    counters = {
        "all_leaf_model_selection": 0,
        "all_leaf_strict": 0,
        "critical_leaf_model_selection": 0,
        "critical_leaf_strict": 0,
        "missing_required_critical": 0,
        "ambiguity_after_critical_present": 0,
    }
    waterfall = {
        "missing_required_critical_candidate": 0,
        "ambiguity_block": 0,
        "recoverable_under_strict": 0,
    }
    rows: list[dict[str, Any]] = []
    for case in call_cases:
        session = _compile_with_proposer(case, candidate_proposer)
        expected = case["expected"]
        tool_id = str(expected["tool_id"])
        gold = _gold_values(case)
        actual_by_destination: dict[str, list[Any]] = {}
        for candidate in session.candidates.candidates.values():
            if candidate.witness.tool_id == tool_id:
                actual_by_destination.setdefault(
                    candidate.witness.destination_scope, []
                ).append(candidate.value)

        def has_gold(destination: str) -> bool:
            if destination not in gold:
                return True
            return any(
                canonical_json(value) == canonical_json(gold[destination])
                for value in actual_by_destination.get(destination, [])
            )

        def ambiguous(destination: str) -> bool:
            values = {
                canonical_json(value)
                for value in actual_by_destination.get(destination, [])
            }
            try:
                policy = session.policy.tool(tool_id).slot(destination)
            except ValueError:
                return False
            return policy.ambiguity == "clarify" and len(values) > 1

        all_required = _required_destinations(case, critical_only=False)
        critical_required = _required_destinations(case, critical_only=True)
        missing_critical = [
            destination
            for destination in critical_required
            if not has_gold(destination)
        ]
        ambiguous_critical = [
            destination
            for destination in critical_required
            if ambiguous(destination)
        ]
        all_present = all(has_gold(destination) for destination in all_required)
        critical_present = all(
            has_gold(destination) for destination in critical_required
        )
        all_strict = all_present and not any(
            ambiguous(destination) for destination in all_required
        )
        critical_strict = critical_present and not any(
            ambiguous(destination) for destination in critical_required
        )
        counters["all_leaf_model_selection"] += int(all_present)
        counters["all_leaf_strict"] += int(all_strict)
        counters["critical_leaf_model_selection"] += int(critical_present)
        counters["critical_leaf_strict"] += int(critical_strict)
        counters["missing_required_critical"] += int(not critical_present)
        counters["ambiguity_after_critical_present"] += int(
            critical_present and not critical_strict
        )
        if not critical_present:
            waterfall["missing_required_critical_candidate"] += 1
            failure = "missing_required_critical_candidate"
        elif not critical_strict:
            waterfall["ambiguity_block"] += 1
            failure = "ambiguity_block"
        else:
            waterfall["recoverable_under_strict"] += 1
            failure = "recoverable_under_strict"
        rows.append(
            {
                "case_id": case["case_id"],
                "family": case["family"],
                "all_required": len(all_required),
                "required_critical": len(critical_required),
                "missing_critical_destinations": missing_critical,
                "ambiguous_critical_destinations": ambiguous_critical,
                "language": case.get("authoring", {}).get("language"),
                "phenomena": list(case.get("authoring", {}).get("phenomena", [])),
                "all_leaf_model_selection": all_present,
                "all_leaf_strict": all_strict,
                "critical_leaf_model_selection": critical_present,
                "critical_leaf_strict": critical_strict,
                "waterfall": failure,
            }
        )
    denominator = len(call_cases)
    return {
        "version": SELECTOR_STUDY_VERSION,
        "candidate_proposer": candidate_proposer,
        "call_cases": denominator,
        "counts": counters,
        "rates": {
            key: value / denominator if denominator else None
            for key, value in counters.items()
        },
        "mutually_exclusive_waterfall": waterfall,
        "rows": rows,
    }


def _fixed_indexed_schema(*, binding_only: bool) -> dict[str, Any]:
    binding = {
        "type": "object",
        "properties": {
            "slot_index": {"type": "integer", "minimum": 0},
            "candidate_index": {"type": "integer", "minimum": 0},
        },
        "required": ["slot_index", "candidate_index"],
        "additionalProperties": False,
    }
    bindings = {
        "type": "array",
        "items": binding,
        "uniqueItems": True,
    }
    if binding_only:
        return {
            "type": "object",
            "properties": {"bindings": bindings},
            "required": ["bindings"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "mode": {"const": "call"},
                    "tool_index": {"type": "integer", "minimum": 0},
                    "bindings": bindings,
                },
                "required": ["mode", "tool_index", "bindings"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "mode": {"const": "need_input"},
                    "tool_index": {"type": "integer", "minimum": 0},
                    "reason": {"type": "string"},
                },
                "required": ["mode", "tool_index"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "mode": {"const": "no_tool"},
                    "reason": {"type": "string"},
                },
                "required": ["mode"],
                "additionalProperties": False,
            },
        ],
    }


def _fixed_router_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "mode": {"const": "call"},
                    "tool_index": {"type": "integer", "minimum": 0},
                },
                "required": ["mode", "tool_index"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "mode": {"const": "need_input"},
                    "tool_index": {"type": "integer", "minimum": 0},
                    "reason": {"type": "string"},
                },
                "required": ["mode", "tool_index"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "mode": {"const": "no_tool"},
                    "reason": {"type": "string"},
                },
                "required": ["mode"],
                "additionalProperties": False,
            },
        ],
    }


def router_payload(
    case: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    base = deepcopy(dict(case["request"]))
    base.pop("evibind", None)
    base["messages"] = [
        {
            "role": "system",
            "content": (
                "Choose only the action mode and request-local tool index. "
                "Do not bind or emit arguments. Use need_input only when the "
                "requested tool lacks required evidence, and no_tool only when "
                "no declared tool is appropriate.\nREQUEST-LOCAL TOOLS:\n"
                + json.dumps(
                    {"tools": catalog["tool_catalog"]},
                    ensure_ascii=True,
                    sort_keys=True,
                )
            ),
        },
        *deepcopy(list(base.get("messages", []))),
    ]
    base["tools"] = [
        {
            "type": "function",
            "function": {
                "name": ROUTER_TOOL,
                "description": "Choose the action mode and tool index.",
                "parameters": _fixed_router_schema(),
            },
        }
    ]
    # llama-server accepts the provider-portable string form and rejects the
    # OpenAI object form.  There is exactly one meta-router tool, so required
    # deterministically forces that tool without exposing an executable call.
    base["tool_choice"] = "required"
    base.pop("response_format", None)
    base["parallel_tool_calls"] = False
    base["n"] = 1
    return base


def _dynamic_schema(catalog: Mapping[str, Any], *, binding_only: bool) -> dict[str, Any]:
    binding_properties = {
        row["destination"]: {
            "type": "string",
            "enum": [candidate["candidate_token"] for candidate in row["candidates"]],
        }
        for row in catalog["slots"]
    }
    required_bindings = [
        row["destination"] for row in catalog["slots"] if row["required"]
    ]
    call = {
        "type": "object",
        "properties": {
            "bindings": {
                "type": "object",
                "properties": binding_properties,
                "required": required_bindings,
                "additionalProperties": False,
            }
        },
        "required": ["bindings"],
        "additionalProperties": False,
    }
    if binding_only:
        return call
    call["properties"].update(
        {
            "mode": {"const": "call"},
            "tool_index": {
                "type": "integer",
                "enum": [row["tool_index"] for row in catalog["tool_catalog"]],
            },
        }
    )
    call["required"].extend(["mode", "tool_index"])
    return {
        "type": "object",
        "oneOf": [
            call,
            {
                "type": "object",
                "properties": {
                    "mode": {"const": "need_input"},
                    "tool_index": {"type": "integer", "minimum": 0},
                    "reason": {"type": "string"},
                },
                "required": ["mode", "tool_index"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "mode": {"const": "no_tool"},
                    "reason": {"type": "string"},
                },
                "required": ["mode"],
                "additionalProperties": False,
            },
        ],
    }


def selector_payload(
    case: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    interface: str,
    routing_regime: str,
) -> dict[str, Any]:
    if interface not in INTERFACES:
        raise EviBenchError(f"unknown interface: {interface}")
    if routing_regime not in ROUTING_REGIMES:
        raise EviBenchError(f"unknown routing regime: {routing_regime}")
    binding_only = routing_regime == "binding_only"
    if binding_only and case["expected"]["mode"] != "call":
        raise EviBenchError("binding-only conditions apply only to gold-call cases")
    base = deepcopy(dict(case["request"]))
    base.pop("evibind", None)
    base["messages"] = deepcopy(list(base.get("messages", [])))
    visible_catalog = {
        "tools": (
            [
                row
                for row in catalog["tool_catalog"]
                if not binding_only
                or row["tool_index"] == catalog["expected_tool_index"]
            ]
        ),
        "critical_slots": [
            {
                "slot_index": row["slot_index"],
                "destination": row["destination"],
                "required": row["required"],
                "candidates": [
                    {
                        "candidate_index": candidate["candidate_index"],
                        "candidate_token": candidate["candidate_token"],
                        "display": candidate["display"],
                    }
                    for candidate in row["candidates"]
                ],
            }
            for row in catalog["slots"]
        ],
    }
    instruction = (
        "Select request-local evidence references for critical destinations. "
        "Never copy a critical literal into the output. "
    )
    if binding_only:
        instruction += (
            "The gold mode is call and the tool is fixed. Return only the complete "
            "critical binding map. "
        )
    else:
        instruction += (
            "Choose mode and tool as well as the complete critical binding map. "
            "Use need_input only when evidence is insufficient and no_tool only "
            "when no declared tool is appropriate. "
        )
    if interface == "dynamic_enum":
        instruction += "Use the candidate_token strings allowed by the schema."
        schema = _dynamic_schema(catalog, binding_only=binding_only)
        action_tool = DYNAMIC_ACTION_TOOL
    else:
        instruction += (
            "Return binding objects with named slot_index and candidate_index fields."
        )
        schema = _fixed_indexed_schema(binding_only=binding_only)
        action_tool = INDEXED_ACTION_TOOL
    base["messages"] = [
        {
            "role": "system",
            "content": instruction
            + "\nREQUEST-LOCAL CATALOG:\n"
            + json.dumps(visible_catalog, ensure_ascii=True, sort_keys=True),
        },
        *base["messages"],
    ]
    if interface == "indexed_json":
        base.pop("tools", None)
        base.pop("tool_choice", None)
        base["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": INDEXED_ACTION_TOOL,
                "strict": True,
                "schema": schema,
            },
        }
    else:
        base["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": action_tool,
                    "description": "Select request-local critical evidence references.",
                    "parameters": schema,
                },
            }
        ]
        # The payload exposes exactly one meta-action tool.  Use the string
        # form supported by llama-server; the object form is silently ignored.
        base["tool_choice"] = "required"
        base.pop("response_format", None)
    base["parallel_tool_calls"] = False
    base["n"] = 1
    return base


def _parse_response(response: Mapping[str, Any], interface: str) -> Mapping[str, Any] | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return None
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else None
    if not isinstance(message, Mapping):
        return None
    if interface == "indexed_json":
        raw = message.get("content")
    else:
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            return None
        function = calls[0].get("function") if isinstance(calls[0], Mapping) else None
        if not isinstance(function, Mapping):
            return None
        expected_name = (
            DYNAMIC_ACTION_TOOL if interface == "dynamic_enum" else INDEXED_ACTION_TOOL
        )
        if function.get("name") != expected_name:
            return None
        raw = function.get("arguments")
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _parse_router_response(
    response: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return None
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    calls = message.get("tool_calls") if isinstance(message, Mapping) else None
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    function = calls[0].get("function") if isinstance(calls[0], Mapping) else None
    if not isinstance(function, Mapping) or function.get("name") != ROUTER_TOOL:
        return None
    raw = function.get("arguments")
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _selected_indices(
    action: Mapping[str, Any],
    catalog: Mapping[str, Any],
    interface: str,
) -> tuple[dict[int, int], bool]:
    if interface == "dynamic_enum":
        bindings = action.get("bindings")
        if not isinstance(bindings, Mapping):
            return {}, False
        by_destination = {row["destination"]: row for row in catalog["slots"]}
        selected: dict[int, int] = {}
        for destination, token in bindings.items():
            row = by_destination.get(destination)
            if row is None or not isinstance(token, str):
                return {}, False
            match = next(
                (
                    candidate
                    for candidate in row["candidates"]
                    if candidate["candidate_token"] == token
                ),
                None,
            )
            if match is None:
                return {}, False
            selected[int(row["slot_index"])] = int(match["candidate_index"])
        return selected, True
    bindings = action.get("bindings")
    if not isinstance(bindings, list):
        return {}, False
    selected = {}
    for pair in bindings:
        if (
            not isinstance(pair, Mapping)
            or isinstance(pair.get("slot_index"), bool)
            or isinstance(pair.get("candidate_index"), bool)
            or not isinstance(pair.get("slot_index"), int)
            or not isinstance(pair.get("candidate_index"), int)
            or pair["slot_index"] in selected
        ):
            return {}, False
        selected[pair["slot_index"]] = pair["candidate_index"]
    return selected, True


def score_selector_response(
    case: Mapping[str, Any],
    catalog: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    interface: str,
    routing_regime: str,
) -> dict[str, Any]:
    action = _parse_response(response, interface)
    binding_only = routing_regime == "binding_only"
    schema = (
        _dynamic_schema(catalog, binding_only=binding_only)
        if interface == "dynamic_enum"
        else _fixed_indexed_schema(binding_only=binding_only)
    )
    response_valid = action is not None and json_contract_accepts(action, schema)
    expected_mode = str(case["expected"]["mode"])
    observed_mode = "call" if binding_only and action is not None else (
        str(action.get("mode")) if action is not None else "invalid"
    )
    observed_tool_index = (
        catalog["expected_tool_index"]
        if binding_only and action is not None
        else action.get("tool_index") if action is not None else None
    )
    mode_correct = response_valid and observed_mode == expected_mode
    tool_routing_correct = bool(
        response_valid
        and (
            (
                expected_mode in {"call", "need_input"}
                and observed_tool_index == catalog["expected_tool_index"]
            )
            or (expected_mode == "no_tool" and observed_mode == "no_tool")
        )
    )
    selected, lookup_valid = (
        _selected_indices(action, catalog, interface)
        if action is not None and observed_mode == "call"
        else ({}, True)
    )
    slot_results: list[dict[str, Any]] = []
    for slot in catalog["slots"]:
        slot_index = int(slot["slot_index"])
        candidate_index = selected.get(slot_index)
        candidate = next(
            (
                row
                for row in slot["candidates"]
                if row["candidate_index"] == candidate_index
            ),
            None,
        )
        slot_results.append(
            {
                "slot_index": slot_index,
                "destination": slot["destination"],
                "required": slot["required"],
                "candidate_count": len(slot["candidates"]),
                "selected_candidate_index": candidate_index,
                "selected": candidate is not None,
                "correct": bool(candidate is not None and candidate["is_gold"]),
            }
        )
    required = [row for row in slot_results if row["required"]]
    per_slot_correct = sum(row["correct"] for row in slot_results)
    complete_binding = bool(
        expected_mode == "call"
        and response_valid
        and mode_correct
        and tool_routing_correct
        and lookup_valid
        and all(row["correct"] for row in required)
        and all(row["correct"] for row in slot_results if row["selected"])
    )
    gold_present = all(
        any(candidate["is_gold"] for candidate in slot["candidates"])
        for slot in catalog["slots"]
        if slot["required"]
    )
    ambiguous = any(
        len({canonical_json(row["value"]) for row in slot["candidates"]}) > 1
        for slot in catalog["slots"]
        if slot["required"]
    )
    ambiguity_block = bool(
        expected_mode == "call"
        and gold_present
        and catalog["ambiguity_policy"] == "strict"
        and ambiguous
    )
    if expected_mode == "call" and not gold_present:
        waterfall = "required_critical_candidate_missing"
    elif ambiguity_block:
        waterfall = "ambiguity_gate"
    elif not response_valid:
        waterfall = "invalid_action_ir"
    elif not mode_correct or not tool_routing_correct:
        waterfall = "wrong_or_absent_tool"
    elif not lookup_valid:
        waterfall = "materializer_rejection"
    elif expected_mode == "call" and not complete_binding:
        waterfall = "wrong_critical_candidate"
    elif expected_mode == "call":
        waterfall = "exact_critical_call"
    elif mode_correct:
        waterfall = f"correct_{expected_mode}"
    else:
        waterfall = "wrong_or_absent_tool"
    usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
    return {
        "response_valid": response_valid,
        "expected_mode": expected_mode,
        "observed_mode": observed_mode,
        "mode_correct": mode_correct,
        "tool_routing_correct": tool_routing_correct,
        "lookup_valid": lookup_valid,
        "gold_critical_catalog_complete": gold_present,
        "slot_total": len(slot_results),
        "slot_correct": per_slot_correct,
        "complete_binding_map": complete_binding,
        "exact_critical_call": complete_binding,
        "waterfall": waterfall,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "slot_results": slot_results,
    }


def score_two_stage_response(
    case: Mapping[str, Any],
    catalog: Mapping[str, Any],
    router_response: Mapping[str, Any],
    binding_response: Mapping[str, Any] | None,
) -> dict[str, Any]:
    route = _parse_router_response(router_response)
    known_tools = {row["tool_index"] for row in catalog["tool_catalog"]}
    route_schema_valid = bool(
        route is not None and json_contract_accepts(route, _fixed_router_schema())
    )
    observed_mode = str(route.get("mode")) if route is not None else "invalid"
    observed_tool_index = route.get("tool_index") if route is not None else None
    route_lookup_valid = bool(
        observed_mode == "no_tool" or observed_tool_index in known_tools
    )
    route_valid = route_schema_valid and route_lookup_valid
    expected_mode = str(case["expected"]["mode"])
    mode_correct = route_valid and observed_mode == expected_mode
    tool_routing_correct = bool(
        route_valid
        and (
            (
                expected_mode in {"call", "need_input"}
                and observed_tool_index == catalog["expected_tool_index"]
            )
            or (expected_mode == "no_tool" and observed_mode == "no_tool")
        )
    )
    routed_call_correctly = bool(
        expected_mode == "call" and mode_correct and tool_routing_correct
    )
    binding_score = (
        score_selector_response(
            case,
            catalog,
            binding_response,
            interface="indexed_tool",
            routing_regime="binding_only",
        )
        if routed_call_correctly and binding_response is not None
        else None
    )
    binding_valid = bool(
        binding_score is not None and binding_score["response_valid"]
    )
    response_valid = bool(
        route_valid and (not routed_call_correctly or binding_valid)
    )
    slot_results = (
        list(binding_score["slot_results"])
        if binding_score is not None
        else [
            {
                "slot_index": int(slot["slot_index"]),
                "destination": slot["destination"],
                "required": slot["required"],
                "selected": False,
                "correct": False,
            }
            for slot in catalog["slots"]
        ]
    )
    complete_binding = bool(
        routed_call_correctly
        and binding_score is not None
        and binding_score["complete_binding_map"]
    )
    gold_present = all(
        any(candidate["is_gold"] for candidate in slot["candidates"])
        for slot in catalog["slots"]
        if slot["required"]
    )
    if expected_mode == "call" and not gold_present:
        waterfall = "required_critical_candidate_missing"
    elif not route_valid:
        waterfall = "invalid_action_ir"
    elif not mode_correct or not tool_routing_correct:
        waterfall = "wrong_or_absent_tool"
    elif routed_call_correctly and not binding_valid:
        waterfall = "invalid_action_ir"
    elif routed_call_correctly and not complete_binding:
        waterfall = (
            str(binding_score["waterfall"])
            if binding_score is not None
            else "wrong_critical_candidate"
        )
    elif routed_call_correctly:
        waterfall = "exact_critical_call"
    elif mode_correct:
        waterfall = f"correct_{expected_mode}"
    else:
        waterfall = "wrong_or_absent_tool"

    responses = [router_response]
    if binding_response is not None:
        responses.append(binding_response)
    prompt_tokens = sum(
        int(response.get("usage", {}).get("prompt_tokens", 0))
        for response in responses
        if isinstance(response.get("usage"), Mapping)
    )
    completion_tokens = sum(
        int(response.get("usage", {}).get("completion_tokens", 0))
        for response in responses
        if isinstance(response.get("usage"), Mapping)
    )
    return {
        "response_valid": response_valid,
        "route_response_valid": route_valid,
        "binding_response_valid": binding_valid if routed_call_correctly else None,
        "expected_mode": expected_mode,
        "observed_mode": observed_mode,
        "mode_correct": mode_correct,
        "tool_routing_correct": tool_routing_correct,
        "lookup_valid": route_lookup_valid
        and bool(binding_score is None or binding_score["lookup_valid"]),
        "gold_critical_catalog_complete": gold_present,
        "slot_total": len(slot_results),
        "slot_correct": sum(bool(row["correct"]) for row in slot_results),
        "complete_binding_map": complete_binding,
        "exact_critical_call": complete_binding,
        "waterfall": waterfall,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "model_calls": len(responses),
        "slot_results": slot_results,
    }


def _post_json(endpoint: str, payload: Mapping[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise EviBenchError(
            f"selector endpoint request failed: {exc}; body={detail[:2000]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise EviBenchError(f"selector endpoint request failed: {exc}") from exc
    if not isinstance(value, dict):
        raise EviBenchError("selector endpoint response must be an object")
    return value


def run_selector_study(
    cases: Sequence[Mapping[str, Any]],
    *,
    endpoint: str,
    model_id: str,
    model_key: str,
    output_path: str | Path,
    conditions: Sequence[Mapping[str, str]],
    ranker: LinearCandidateRanker | None = None,
    max_tokens: int = 256,
    timeout: int = 120,
    workers: int = 1,
    catalog_transform: Callable[
        [Mapping[str, Any], dict[str, Any], Mapping[str, str]],
        dict[str, Any],
    ]
    | None = None,
) -> dict[str, Any]:
    if workers <= 0:
        raise EviBenchError("selector workers must be positive")
    target = Path(output_path)
    existing: dict[tuple[str, str], Mapping[str, Any]] = {}
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            existing[(str(row["case_id"]), str(row["condition_id"]))] = row
    target.parent.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[Mapping[str, str], Mapping[str, Any]]] = []
    for condition in conditions:
        condition_id = str(condition["condition_id"])
        routing = str(condition["routing_regime"])
        for case in cases:
            if routing == "binding_only" and case["expected"]["mode"] != "call":
                continue
            key = (str(case["case_id"]), condition_id)
            if key not in existing:
                pending.append((condition, case))

    def run_one(
        item: tuple[Mapping[str, str], Mapping[str, Any]],
    ) -> dict[str, Any]:
        condition, case = item
        condition_id = str(condition["condition_id"])
        routing = str(condition["routing_regime"])
        catalog = build_catalog(
            case,
            regime=str(condition["candidate_regime"]),
            ambiguity_policy=str(
                condition.get("ambiguity_policy", "model_selection")
            ),
            ranker=ranker,
        )
        if catalog_transform is not None:
            catalog = catalog_transform(case, catalog, condition)
        generation = {
            "model": model_id,
            "seed": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if routing == "two_stage":
            if str(condition["interface"]) != "indexed_tool":
                raise EviBenchError("two-stage routing requires indexed_tool")
            route_payload = router_payload(case, catalog)
            route_payload.update(generation)
            started = time.perf_counter()
            route_response = _post_json(endpoint, route_payload, timeout)
            route_elapsed = time.perf_counter() - started
            route_action = _parse_router_response(route_response)
            route_is_gold_call = bool(
                route_action is not None
                and json_contract_accepts(route_action, _fixed_router_schema())
                and route_action.get("mode") == "call"
                and route_action.get("tool_index")
                == catalog["expected_tool_index"]
                and case["expected"]["mode"] == "call"
            )
            binding_payload: dict[str, Any] | None = None
            binding_response: dict[str, Any] | None = None
            binding_elapsed = 0.0
            if route_is_gold_call:
                binding_payload = selector_payload(
                    case,
                    catalog,
                    interface="indexed_tool",
                    routing_regime="binding_only",
                )
                binding_payload.update(generation)
                started = time.perf_counter()
                binding_response = _post_json(endpoint, binding_payload, timeout)
                binding_elapsed = time.perf_counter() - started
            score = score_two_stage_response(
                case,
                catalog,
                route_response,
                binding_response,
            )
            payload_record: Mapping[str, Any] = {
                "router": route_payload,
                "binder": binding_payload,
            }
            response_record: Mapping[str, Any] = {
                "router": route_response,
                "binder": binding_response,
            }
            elapsed = route_elapsed + binding_elapsed
        else:
            payload = selector_payload(
                case,
                catalog,
                interface=str(condition["interface"]),
                routing_regime=routing,
            )
            payload.update(generation)
            started = time.perf_counter()
            response = _post_json(endpoint, payload, timeout)
            elapsed = time.perf_counter() - started
            score = score_selector_response(
                case,
                catalog,
                response,
                interface=str(condition["interface"]),
                routing_regime=routing,
            )
            payload_record = payload
            response_record = response
        return {
            "version": SELECTOR_STUDY_VERSION,
            "case_id": case["case_id"],
            "family": case["family"],
            "model_key": model_key,
            "model_id": model_id,
            "condition_id": condition_id,
            **dict(condition),
            "payload_sha256": hashlib.sha256(
                canonical_json(payload_record).encode("utf-8")
            ).hexdigest(),
            "response_sha256": hashlib.sha256(
                canonical_json(response_record).encode("utf-8")
            ).hexdigest(),
            "input_bytes": len(canonical_json(payload_record).encode("utf-8")),
            "model_seconds": elapsed,
            "catalog_candidates": sum(
                len(slot["candidates"]) for slot in catalog["slots"]
            ),
            "required_critical_slots": sum(
                bool(slot["required"]) for slot in catalog["slots"]
            ),
            "gold_candidate_indices": [
                next(
                    int(candidate["candidate_index"])
                    for candidate in slot["candidates"]
                    if candidate["is_gold"]
                )
                for slot in catalog["slots"]
                if any(candidate["is_gold"] for candidate in slot["candidates"])
            ],
            **score,
            "response": response_record,
        }

    generated = 0
    with target.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        for row in executor.map(run_one, pending):
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            generated += 1
    return {
        "version": SELECTOR_STUDY_VERSION,
        "model_key": model_key,
        "case_count": len(cases),
        "condition_count": len(conditions),
        "generated": generated,
        "checkpointed": len(existing),
        "output": str(target),
    }


def aggregate_selector_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition_id"]), []).append(row)
    conditions: dict[str, Any] = {}
    for condition_id, group in sorted(by_condition.items()):
        count = len(group)
        call_rows = [row for row in group if row["expected_mode"] == "call"]
        valid = sum(bool(row["response_valid"]) for row in group)
        exact = sum(bool(row["exact_critical_call"]) for row in call_rows)
        slots = sum(int(row["slot_total"]) for row in call_rows)
        correct_slots = sum(int(row["slot_correct"]) for row in call_rows)
        waterfall: dict[str, int] = {}
        for row in group:
            waterfall[str(row["waterfall"])] = waterfall.get(str(row["waterfall"]), 0) + 1
        exemplar = group[0]
        conditions[condition_id] = {
            "interface": exemplar["interface"],
            "routing_regime": exemplar["routing_regime"],
            "candidate_regime": exemplar["candidate_regime"],
            "ambiguity_policy": exemplar.get("ambiguity_policy", "model_selection"),
            "rows": count,
            "call_rows": len(call_rows),
            "response_validity": valid / count if count else None,
            "tool_routing_accuracy": (
                sum(bool(row["tool_routing_correct"]) for row in group) / count
                if count
                else None
            ),
            "mode_accuracy": (
                sum(bool(row["mode_correct"]) for row in group) / count
                if count
                else None
            ),
            "per_slot_accuracy": correct_slots / slots if slots else None,
            "complete_binding_accuracy": exact / len(call_rows) if call_rows else None,
            "exact_critical_call_recall": exact / len(call_rows) if call_rows else None,
            "mean_catalog_candidates": (
                sum(int(row["catalog_candidates"]) for row in group) / count
                if count
                else None
            ),
            "mean_input_bytes": (
                sum(int(row["input_bytes"]) for row in group) / count
                if count
                else None
            ),
            "mean_model_calls": (
                sum(int(row.get("model_calls", 1)) for row in group) / count
                if count
                else None
            ),
            "mean_prompt_tokens": (
                sum(int(row.get("prompt_tokens") or 0) for row in group) / count
                if count
                else None
            ),
            "mean_completion_tokens": (
                sum(int(row.get("completion_tokens") or 0) for row in group)
                / count
                if count
                else None
            ),
            "waterfall": waterfall,
        }
    return {
        "version": SELECTOR_STUDY_VERSION,
        "rows": len(rows),
        "conditions": conditions,
    }

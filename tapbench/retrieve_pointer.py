from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any

from .eflrx import (
    ACTION_RISK_THRESHOLD,
    RequestFn,
    _merge_metadata,
    _non_call,
    _tool_name,
)
from .extractive_candidates import user_request_text
from .multilingual_retriever import (
    MULTILINGUAL_RETRIEVER_VERSION,
    RETRIEVAL_RANKING_SCHEMA_VERSION,
    RETRIEVER_MODEL_ID,
    RETRIEVER_REVISION,
    catalog_sha256,
    forbidden_paths,
    ranking_sha256,
)
from .source_span_projection import (
    SOURCE_SPAN_CERTIFICATE_VERSION,
    SOURCE_SPAN_PROJECTION_VERSION,
    action_fingerprint,
    materialize_span_proposal,
    slot_catalog,
    source_span_catalog,
    span_proposal_schema,
)


RETRIEVE_POINTER_VERSION = "tapbench.retrieve_pointer.v2"
RETRIEVE_POINTER_CONDITIONS = (
    "tap_r_retrieve_pointer_single",
    "tap_r_retrieve_pointer_consensus",
    "tap_r_retrieve_pointer_consensus_top1",
)
RETRIEVER_K = 8


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_ranking_row(
    row: dict[str, Any],
    *,
    case_id: str,
    request_text: str,
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    leaks = forbidden_paths(row)
    if leaks:
        raise ValueError(f"ranking row contains scorer-only fields: {leaks}")
    expected = {
        "schema_version": RETRIEVAL_RANKING_SCHEMA_VERSION,
        "case_id": case_id,
        "retriever_version": MULTILINGUAL_RETRIEVER_VERSION,
        "retriever_model_id": RETRIEVER_MODEL_ID,
        "retriever_revision": RETRIEVER_REVISION,
        "k": RETRIEVER_K,
        "request_sha256": hashlib.sha256(request_text.encode("utf-8")).hexdigest(),
    }
    failures = [
        f"{key}:{row.get(key)!r}!={value!r}"
        for key, value in expected.items()
        if row.get(key) != value
    ]
    arm = row.get("serialization_arm")
    if not isinstance(arm, str):
        failures.append("serialization_arm:missing")
    elif row.get("catalog_sha256") != catalog_sha256(tools, arm):
        failures.append("catalog_sha256:mismatch")
    ranking = row.get("ranking")
    if not isinstance(ranking, list):
        failures.append("ranking:not_array")
        ranking = []
    if row.get("ranking_sha256") != ranking_sha256(ranking):
        failures.append("ranking_sha256:mismatch")
    expected_count = min(RETRIEVER_K, len(tools))
    if len(ranking) != expected_count:
        failures.append(f"ranking_count:{len(ranking)}!={expected_count}")
    public_tools = {_tool_name(tool) for tool in tools}
    names: list[str] = []
    for index, item in enumerate(ranking, start=1):
        if not isinstance(item, dict):
            failures.append(f"ranking_{index}:not_object")
            continue
        name = str(item.get("tool") or "")
        names.append(name)
        if item.get("rank") != index:
            failures.append(f"ranking_{index}:rank")
        if name not in public_tools:
            failures.append(f"ranking_{index}:unknown_tool:{name}")
        score = item.get("cosine_score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            failures.append(f"ranking_{index}:invalid_score")
    if len(names) != len(set(names)):
        failures.append("ranking:duplicate_tools")
    if failures:
        raise ValueError("invalid sealed ranking row: " + ", ".join(failures))
    return ranking


def validate_external_ranking_row(
    row: dict[str, Any],
    *,
    case_id: str,
    request_text: str,
    tools: list[dict[str, Any]],
    schema_version: str,
    retriever_version: str,
    retriever_model_id: str,
    retriever_revision: str,
    k: int = RETRIEVER_K,
) -> list[dict[str, Any]]:
    """Validate a sealed ranking contract without impersonating the E5 retriever."""
    leaks = forbidden_paths(row)
    if leaks:
        raise ValueError(f"ranking row contains scorer-only fields: {leaks}")
    expected = {
        "schema_version": schema_version,
        "case_id": case_id,
        "retriever_version": retriever_version,
        "retriever_model_id": retriever_model_id,
        "retriever_revision": retriever_revision,
        "k": k,
        "request_sha256": hashlib.sha256(request_text.encode("utf-8")).hexdigest(),
    }
    failures = [f"{key}:{row.get(key)!r}!={value!r}" for key, value in expected.items() if row.get(key) != value]
    ranking = row.get("ranking")
    if not isinstance(ranking, list):
        failures.append("ranking:not_array")
        ranking = []
    if row.get("ranking_sha256") != ranking_sha256(ranking):
        failures.append("ranking_sha256:mismatch")
    expected_count = min(k, len(tools))
    if len(ranking) != expected_count:
        failures.append(f"ranking_count:{len(ranking)}!={expected_count}")
    public_tools = {_tool_name(tool) for tool in tools}
    names: list[str] = []
    for index, item in enumerate(ranking, start=1):
        if not isinstance(item, dict):
            failures.append(f"ranking_{index}:not_object")
            continue
        name = str(item.get("tool") or "")
        names.append(name)
        if item.get("rank") != index:
            failures.append(f"ranking_{index}:rank")
        if name not in public_tools:
            failures.append(f"ranking_{index}:unknown_tool:{name}")
        score = item.get("cosine_score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            failures.append(f"ranking_{index}:invalid_score")
    if len(names) != len(set(names)):
        failures.append("ranking:duplicate_tools")
    if failures:
        raise ValueError("invalid sealed external ranking row: " + ", ".join(failures))
    return ranking


def _ordered(values: list[Any], order: str) -> list[Any]:
    if order == "forward":
        return list(values)
    if order == "reverse":
        return list(reversed(values))
    raise ValueError(f"unknown presentation order: {order}")


def _selection_catalog(
    retrieved_tools: list[dict[str, Any]],
    order: str,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    catalog: list[dict[str, Any]] = []
    mapping: dict[int, str] = {}
    for selection_id, tool in enumerate(_ordered(retrieved_tools, order)):
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


def _selection_messages(
    request_text: str,
    catalog: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "This benchmark item requires one function call. Select exactly "
                "one candidate whose external effect matches the request. Return "
                "only its opaque selection_id, without arguments or reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                "Request:\n"
                + request_text
                + "\nCandidate effects:\n"
                + json.dumps(
                    catalog,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + '\nReturn {"selection_id": integer}.'
            ),
        },
    ]


def _pointer_messages(
    request_text: str,
    language: str,
    tool: dict[str, Any],
    slot_order: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    slots, _ = slot_catalog(tool, order=slot_order)
    lattice = source_span_catalog(request_text, language)
    public_spans = [
        {"span_id": row["span_id"], "text": row["source_text"]}
        for row in lattice["spans"]
    ]
    public_tool = {
        "name": _tool_name(tool),
        "description": str(tool.get("description", "")),
        "slots": slots,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Bind request evidence to every selected-tool slot. For each "
                "slot return one opaque SPAN id, or OMIT when an optional slot "
                "is not explicitly stated. Never copy, translate, normalize, or "
                "invent a value. Return only the bindings JSON and no reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                "Request:\n"
                + request_text
                + "\nSelected tool and opaque slot IDs:\n"
                + json.dumps(
                    public_tool,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\nFinite source spans:\n"
                + json.dumps(
                    public_spans,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + '\nReturn {"bindings":{"SLOT_000":"SPAN_00000 or OMIT"}}.'
            ),
        },
    ]
    return messages, lattice


def _fail_closed(
    metadata: dict[str, Any],
    call_metadata: list[dict[str, Any]],
    *,
    stage: str,
    reason: str,
    started: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if call_metadata:
        metadata.update(_merge_metadata(call_metadata))
    else:
        metadata.setdefault("generation_calls", 0)
        metadata.setdefault("finish_reason", "not_applicable")
    metadata.update(
        {
            "controller_stage_failure": stage,
            "action_risk_score": 1.0,
            "risk_gate_passed": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return _non_call("refuse", reason), metadata


def run_retrieve_pointer_resolution(
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
    if condition not in RETRIEVE_POINTER_CONDITIONS:
        raise ValueError(f"unknown retrieve-pointer condition: {condition}")
    started = time.perf_counter()
    text = user_request_text(messages)
    ranking = validate_ranking_row(
        ranking_row,
        case_id=case_id,
        request_text=text,
        tools=tools,
    )
    by_name = {_tool_name(tool): tool for tool in tools}
    retrieved_tools = [by_name[str(row["tool"])] for row in ranking]
    ranking_names = [str(row["tool"]) for row in ranking]
    strict = condition != "tap_r_retrieve_pointer_single"
    metadata: dict[str, Any] = {
        "retrieve_pointer_version": RETRIEVE_POINTER_VERSION,
        "retriever_version": ranking_row["retriever_version"],
        "retriever_model_id": ranking_row["retriever_model_id"],
        "retriever_revision": ranking_row["retriever_revision"],
        "retriever_serialization_arm": ranking_row["serialization_arm"],
        "retriever_k": ranking_row["k"],
        "ranking_sha256": ranking_row["ranking_sha256"],
        "ranking_artifact_sha256": ranking_artifact_sha256,
        "retrieved_tools": ranking_names,
        "retrieval_scores": [float(row["cosine_score"]) for row in ranking],
        "retrieval_top1": ranking_names[0],
        "retrieval_margin_top1_top2": (
            float(ranking[0]["cosine_score"])
            - float(ranking[1]["cosine_score"])
            if len(ranking) > 1
            else None
        ),
        "source_span_projection_version": SOURCE_SPAN_PROJECTION_VERSION,
        "source_span_certificate_version": SOURCE_SPAN_CERTIFICATE_VERSION,
        "call_only_tool_election": True,
        "no_call_election_option": False,
        "tool_elections": [],
        "pointer_proposals": [],
        "generation_calls": 0,
        "action_risk_threshold": ACTION_RISK_THRESHOLD,
    }

    call_metadata: list[dict[str, Any]] = []
    election_orders = ("forward", "reverse") if strict else ("forward",)
    selected_names: list[str] = []
    for index, order in enumerate(election_orders):
        catalog, mapping = _selection_catalog(retrieved_tools, order)
        raw, response = request_fn(
            endpoint,
            _selection_messages(text, catalog),
            response_schema=_selection_schema(catalog),
            max_tokens=min(max_tokens, 64),
            temperature=0.0,
            seed=seed + index * 100003,
        )
        call_metadata.append(response)
        try:
            selection_id = int(raw.get("selection_id"))
        except (AttributeError, TypeError, ValueError):
            selection_id = -1
        selected = mapping.get(selection_id, "INVALID")
        selected_names.append(selected)
        metadata["tool_elections"].append(
            {
                "order": order,
                "selection_id": selection_id,
                "selected": selected,
                "catalog_sha256": _json_sha256(catalog),
            }
        )
    if any(value == "INVALID" for value in selected_names) or len(
        set(selected_names)
    ) != 1:
        metadata["tool_agreement"] = False
        return _fail_closed(
            metadata,
            call_metadata,
            stage="tool_election_disagreement",
            reason="counterbalanced retrieved-tool elections disagreed",
            started=started,
        )
    selected_tool = selected_names[0]
    metadata["tool_agreement"] = True
    metadata["selected_tool"] = selected_tool
    top1_agreement = selected_tool == ranking_names[0]
    metadata["retriever_election_agreement"] = top1_agreement
    if (
        condition == "tap_r_retrieve_pointer_consensus_top1"
        and not top1_agreement
    ):
        return _fail_closed(
            metadata,
            call_metadata,
            stage="retriever_election_disagreement",
            reason="retriever rank one and tool election disagreed",
            started=started,
        )

    tool = by_name[selected_tool]
    proposal_orders = ("forward", "reverse") if strict else ("forward",)
    actions: list[dict[str, Any]] = []
    materializations: list[dict[str, Any]] = []
    for index, order in enumerate(proposal_orders):
        pointer_messages, lattice = _pointer_messages(
            text,
            language,
            tool,
            order,
        )
        span_ids = [str(row["span_id"]) for row in lattice["spans"]]
        schema = span_proposal_schema(
            tool,
            span_ids=span_ids,
            slot_order=order,
        )
        raw, response = request_fn(
            endpoint,
            pointer_messages,
            response_schema=schema,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed + 700001 + index * 100003,
        )
        call_metadata.append(response)
        action, materialization = materialize_span_proposal(
            raw,
            selected_tool=selected_tool,
            tool=tool,
            tools=tools,
            request_text=text,
            language=language,
        )
        metadata["pointer_proposals"].append(
            {
                "order": order,
                "proposal_sha256": _json_sha256(raw),
                "schema_sha256": _json_sha256(schema),
                "status": materialization.get("status"),
                "action_sha256": action_fingerprint(action) if action else None,
                "certificate_count": materialization.get("certificate_count", 0),
                "span_catalog_sha256": lattice["catalog_sha256"],
                "span_count": lattice["span_count"],
            }
        )
        if action is None:
            metadata["pointer_agreement"] = False
            return _fail_closed(
                metadata,
                call_metadata,
                stage=f"pointer_materialization_{materialization.get('status')}",
                reason="finite source-pointer proposal failed validation",
                started=started,
            )
        actions.append(action)
        materializations.append(materialization)

    fingerprints = [action_fingerprint(action) for action in actions]
    if len(set(fingerprints)) != 1:
        metadata["pointer_agreement"] = False
        return _fail_closed(
            metadata,
            call_metadata,
            stage="pointer_action_disagreement",
            reason="counterbalanced source-pointer actions disagreed",
            started=started,
        )

    risk = 0.05 if not strict else 0.02
    if condition == "tap_r_retrieve_pointer_consensus_top1":
        risk = 0.01
    certification = materializations[0]
    metadata.update(_merge_metadata(call_metadata))
    metadata.update(
        {
            "controller_stage_failure": None,
            "pointer_agreement": True,
            "proposal_admitted": True,
            "evidence_certificates": certification["certificates"],
            "selected_span_ids": certification["selected_span_ids"],
            "span_catalog_sha256": certification["span_catalog_sha256"],
            "slot_catalog_sha256": certification["slot_catalog_sha256"],
            "materialized_action_sha256": fingerprints[0],
            "action_risk_score": risk,
            "risk_gate_passed": risk <= ACTION_RISK_THRESHOLD,
            "no_generated_action_critical_literals": True,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return actions[0], metadata

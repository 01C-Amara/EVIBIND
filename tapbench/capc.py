from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from typing import Any

from .eflrx import (
    ACTION_RISK_THRESHOLD,
    NO_CALL_ID,
    RequestFn,
    _merge_metadata,
    _non_call,
    _tool_catalog,
    _tool_election_messages,
    _tool_election_schema,
    _tool_name,
    explicit_effect_firewall,
)
from .eflrx_baselines import run_raw_baseline
from .extractive_candidates import (
    EXTRACTIVE_CANDIDATE_VERSION,
    build_extractive_candidate_table,
    canonical_slots,
)
from .r2_model_runner import _request_schema_json
from .validation import action_contract_is_accepted


CAPC_VERSION = "tapbench.capc.v1"
CAPC_CONDITIONS = (
    "tap_r_capc_single",
    "tap_r_capc_dual",
)
_PROPOSAL_METHODS = {
    "tap_r_capc_single": ("full_tap_b2",),
    "tap_r_capc_dual": ("full_tap_b2", "prompt_few_shot"),
}
_PROPOSAL_RISK = (0.02, 0.05)


def _public_tool(
    tools: list[dict[str, Any]],
    selected_name: str,
) -> dict[str, Any] | None:
    return next(
        (tool for tool in tools if _tool_name(tool) == selected_name),
        None,
    )


def _canonical_argument_map(tool: dict[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for slot in canonical_slots(tool):
        canonical = str(slot["name"])
        aliases[str(slot["surface_name"])] = canonical
        aliases[canonical] = canonical
    return aliases


def certify_proposal(
    proposal: dict[str, Any],
    *,
    selected_tool: str,
    tool: dict[str, Any],
    candidate_table: dict[str, Any],
    tools: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not isinstance(proposal, dict) or proposal.get("mode") != "call":
        return None, {"status": "not_a_call"}
    proposal_tool = str(proposal.get("tool") or "")
    public_names = {
        str(tool.get("name") or ""),
        str(tool.get("canonical_name") or ""),
    }
    if selected_tool not in public_names or proposal_tool not in public_names:
        return None, {
            "status": "tool_mismatch",
            "selected_tool": selected_tool,
            "proposal_tool": proposal_tool,
        }
    raw_arguments = proposal.get("arguments")
    if not isinstance(raw_arguments, dict):
        return None, {"status": "arguments_not_object"}

    aliases = _canonical_argument_map(tool)
    arguments: dict[str, Any] = {}
    certificates: dict[str, Any] = {}
    for surface_slot, value in raw_arguments.items():
        canonical_slot = aliases.get(str(surface_slot))
        if canonical_slot is None:
            return None, {
                "status": "unknown_argument",
                "slot": str(surface_slot),
            }
        if canonical_slot in arguments:
            return None, {
                "status": "duplicate_canonical_argument",
                "slot": canonical_slot,
            }
        candidate = next(
            (
                row
                for row in candidate_table.get("slots", {}).get(
                    canonical_slot,
                    [],
                )
                if row.get("value") == value
            ),
            None,
        )
        if candidate is None:
            return None, {
                "status": "unsupported_argument",
                "slot": canonical_slot,
                "value_sha256": hashlib.sha256(
                    json.dumps(
                        value,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode()
                ).hexdigest(),
            }
        arguments[canonical_slot] = deepcopy(candidate.get("value"))
        certificates[canonical_slot] = {
            "candidate_id": int(candidate["candidate_id"]),
            "value": deepcopy(candidate.get("value")),
            "source_span": list(candidate.get("source_span", [])),
            "component_spans": deepcopy(candidate.get("component_spans", [])),
            "source_text": candidate.get("source_text"),
            "transform": candidate.get("transform"),
        }

    action = {
        "mode": "call",
        "tool": selected_tool,
        "arguments": arguments,
        "payload": {},
    }
    if not action_contract_is_accepted({"tools": tools}, action):
        return None, {"status": "public_contract_rejected"}
    required = set(candidate_table.get("required_slots", []))
    missing = sorted(required - set(arguments))
    if missing:
        return None, {
            "status": "missing_required_arguments",
            "missing_slots": missing,
        }
    return action, {
        "status": "certified",
        "certificates": certificates,
        "certificate_count": len(certificates),
    }


def run_capc_resolution(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    endpoint: str,
    condition: str,
    max_tokens: int,
    seed: int,
    request_fn: RequestFn = _request_schema_json,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if condition not in CAPC_CONDITIONS:
        raise ValueError(f"unknown CAPC condition: {condition}")
    started = time.perf_counter()
    firewall = explicit_effect_firewall(messages)
    metadata: dict[str, Any] = {
        "capc_version": CAPC_VERSION,
        "extractive_candidate_version": EXTRACTIVE_CANDIDATE_VERSION,
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
    selected_tools: list[str] = []
    for index, reverse in enumerate((False, True)):
        catalog, mapping = _tool_catalog(tools, reverse=reverse)
        raw, response = request_fn(
            endpoint,
            _tool_election_messages(messages, catalog),
            response_schema=_tool_election_schema(catalog),
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed + index * 100003,
        )
        try:
            selection_id = int(raw.get("selection_id"))
        except (AttributeError, TypeError, ValueError):
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
        return _non_call(
            "refuse",
            "counterbalanced tool elections disagree",
        ), metadata

    selected_tool = selected_tools[0]
    if selected_tool == "NO_CALL":
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "tool_agreement": True,
                "selected_tools": selected_tools,
                "action_risk_score": 0.01,
                "risk_factors": {"counterbalanced_no_call": 0.01},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call(
            "no_tool",
            "counterbalanced guardian selected NO_CALL",
        ), metadata

    tool = _public_tool(tools, selected_tool)
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

    candidate_table = build_extractive_candidate_table(
        messages,
        tool,
        include_optional=True,
    )
    metadata["candidate_table"] = {
        "schema_version": candidate_table.get("schema_version"),
        "candidate_count": candidate_table.get("candidate_count"),
        "required_slots": candidate_table.get("required_slots"),
        "optional_slots": candidate_table.get("optional_slots"),
        "sha256": hashlib.sha256(
            json.dumps(
                candidate_table,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).hexdigest(),
    }
    proposal_case = {
        "messages": deepcopy(messages),
        "tools": deepcopy(tools),
    }
    accepted_action: dict[str, Any] | None = None
    accepted_certification: dict[str, Any] | None = None
    accepted_index: int | None = None
    for index, proposal_method in enumerate(_PROPOSAL_METHODS[condition]):
        proposal, proposal_metadata = run_raw_baseline(
            proposal_case,
            endpoint=endpoint,
            condition=proposal_method,
            max_tokens=max_tokens,
            seed=seed + 700001 + index * 100003,
            request_fn=request_fn,
        )
        call_metadata.append(proposal_metadata)
        certified_action, certification = certify_proposal(
            proposal,
            selected_tool=selected_tool,
            tool=tool,
            candidate_table=candidate_table,
            tools=tools,
        )
        metadata["proposal_attempts"].append(
            {
                "proposal_index": index,
                "proposal_method": proposal_method,
                "proposal_sha256": hashlib.sha256(
                    json.dumps(
                        proposal,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode()
                ).hexdigest(),
                "status": certification.get("status"),
                "certificate_count": certification.get(
                    "certificate_count",
                    0,
                ),
            }
        )
        if certified_action is not None:
            accepted_action = certified_action
            accepted_certification = certification
            accepted_index = index
            break

    if accepted_action is None or accepted_certification is None:
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "tool_agreement": True,
                "selected_tools": selected_tools,
                "proposal_admitted": False,
                "action_risk_score": 1.0,
                "risk_factors": {"no_certified_proposal": 1.0},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call(
            "refuse",
            "no proposal has complete source certificates",
        ), metadata

    assert accepted_index is not None
    risk = _PROPOSAL_RISK[accepted_index]
    certificates = accepted_certification["certificates"]
    metadata.update(_merge_metadata(call_metadata))
    metadata.update(
        {
            "tool_agreement": True,
            "selected_tools": selected_tools,
            "proposal_admitted": True,
            "accepted_proposal_index": accepted_index,
            "accepted_proposal_method": _PROPOSAL_METHODS[condition][
                accepted_index
            ],
            "evidence_certificates": certificates,
            "action_risk_score": risk,
            "risk_factors": {
                "certificate_complete": True,
                "proposal_priority_tier": accepted_index + 1,
            },
            "risk_gate_passed": risk <= ACTION_RISK_THRESHOLD,
            "contract_valid": True,
            "materialized_action_sha256": hashlib.sha256(
                json.dumps(
                    accepted_action,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode()
            ).hexdigest(),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return accepted_action, metadata

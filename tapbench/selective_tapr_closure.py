from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .capc import _public_tool, certify_proposal
from .extractive_candidates import build_extractive_candidate_table
from .semantic_closure import (
    SEMANTIC_CLOSURE_VERSION,
    close_unique_head_number_arguments,
)
from .selective_tapr import (
    certificate_semantic_envelope_violations,
    certificate_span_conflicts,
)


ONLINE_SEMANTIC_CLOSURE_VERSION = "tapbench.online_semantic_closure.v1"


def decode_json_stream(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    rows: list[Any] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            row, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return []
        rows.append(row)
        index = end
    return rows


def apply_online_semantic_closure(
    action: dict[str, Any],
    metadata: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply fail-closed semantic closure to a completed TAP-R trace.

    The wrapper consumes only runtime-visible messages, tools, candidates, and
    model proposals. It never receives a gold action or derivability oracle.
    """
    output_action = deepcopy(action)
    output_metadata = deepcopy(metadata)
    audit: dict[str, Any] = {
        "schema_version": ONLINE_SEMANTIC_CLOSURE_VERSION,
        "semantic_closure_version": SEMANTIC_CLOSURE_VERSION,
        "status": "not_attempted",
        "rewrites": [],
    }
    output_metadata["online_semantic_closure_version"] = (
        ONLINE_SEMANTIC_CLOSURE_VERSION
    )
    output_metadata["semantic_closure_version"] = SEMANTIC_CLOSURE_VERSION
    output_metadata["semantic_closure"] = audit

    if output_action.get("mode") == "call":
        audit["status"] = "not_needed_existing_certified_call"
        return output_action, output_metadata

    attempts = output_metadata.get("proposal_attempts", [])
    selected = output_metadata.get("selected_tools", [])
    if (
        not isinstance(attempts, list)
        or not attempts
        or not isinstance(selected, list)
        or not selected
        or len({str(value) for value in selected}) != 1
    ):
        audit["status"] = "no_replayable_proposal_path"
        return output_action, output_metadata

    selected_tool = str(selected[0])
    if selected_tool in {"INVALID", "NO_CALL"}:
        audit["status"] = "no_admitted_tool"
        return output_action, output_metadata
    tool = _public_tool(tools, selected_tool)
    if tool is None:
        audit["status"] = "selected_tool_missing"
        return output_action, output_metadata

    stream = decode_json_stream(str(output_metadata.get("raw_text", "")))
    if len(stream) < len(attempts):
        audit["status"] = "proposal_stream_decode_failed"
        return output_action, output_metadata
    proposals = stream[-len(attempts) :]
    candidate_table = build_extractive_candidate_table(
        messages,
        tool,
        include_optional=True,
    )
    considered: list[dict[str, Any]] = []
    for proposal_index, proposal in enumerate(proposals):
        if not isinstance(proposal, dict):
            continue
        closed, rewrites = close_unique_head_number_arguments(
            proposal,
            tool=tool,
            candidate_table=candidate_table,
        )
        if not rewrites:
            continue
        certified, certification = certify_proposal(
            closed,
            selected_tool=selected_tool,
            tool=tool,
            candidate_table=candidate_table,
            tools=tools,
        )
        conflicts = (
            certificate_span_conflicts(certification)
            if certified is not None
            else []
        )
        violations = (
            certificate_semantic_envelope_violations(certification, tool)
            if certified is not None
            else []
        )
        considered.append(
            {
                "proposal_index": proposal_index,
                "rewrites": rewrites,
                "certification_status": certification.get("status"),
                "conflict_slots": conflicts,
                "violation_slots": violations,
            }
        )
        if certified is None or conflicts or violations:
            continue

        audit.update(
            {
                "status": "recovered",
                "proposal_index": proposal_index,
                "rewrites": rewrites,
                "considered": considered,
            }
        )
        output_metadata.update(
            {
                "proposal_admitted": True,
                "accepted_proposal_index": proposal_index,
                "accepted_proposal_method": attempts[proposal_index].get(
                    "proposal_method"
                ),
                "accepted_evidence_tier": "semantic_closure_literal",
                "evidence_certificates": certification.get(
                    "certificates", {}
                ),
                "certificate_count": certification.get(
                    "certificate_count", 0
                ),
                "action_risk_score": 0.04,
                "risk_factors": {
                    "semantic_closure": 0.01,
                    "source_backed_unique_resolution": 0.01,
                    "admission_agreement": 0.01,
                    "tool_agreement": 0.01,
                },
            }
        )
        return certified, output_metadata

    audit["status"] = "not_recovered"
    audit["considered"] = considered
    return output_action, output_metadata

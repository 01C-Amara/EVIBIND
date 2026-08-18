from __future__ import annotations

import json
from typing import Any

from tapbench.selective_tapr import (
    certificate_semantic_envelope_violations,
    run_selective_tapr_resolution,
)


def _tool() -> dict[str, Any]:
    return {
        "name": "submit_count",
        "canonical_name": "submit_count",
        "description": "Submit an explicitly requested count.",
        "parameters": {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "count to submit"}
            },
            "required": ["count"],
            "additionalProperties": False,
        },
    }


def _call(value: int) -> dict[str, Any]:
    return {
        "mode": "call",
        "tool": "submit_count",
        "arguments": {"count": value},
        "payload": {},
    }


def _string_tool() -> dict[str, Any]:
    return {
        "name": "submit_customer",
        "canonical_name": "submit_customer",
        "description": "Submit a customer record.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "customer user ID"}
            },
            "required": ["user_id"],
            "additionalProperties": False,
        },
    }


def _entity_tool() -> dict[str, Any]:
    return {
        "name": "register_attendee",
        "canonical_name": "register_attendee",
        "description": "Register the named attendee.",
        "parameters": {
            "type": "object",
            "properties": {
                "attendee": {
                    "type": "string",
                    "description": "attendee to register",
                }
            },
            "required": ["attendee"],
            "additionalProperties": False,
        },
    }


def _metadata() -> dict[str, Any]:
    return {
        "finish_reason": "stop",
        "prompt_tokens": 20,
        "completion_tokens": 2,
        "total_tokens": 22,
        "prompt_ms": 1.0,
        "generation_ms": 1.0,
        "context_truncated": False,
    }


def _catalog(messages: list[dict[str, str]], marker: str) -> list[dict[str, Any]]:
    content = messages[-1]["content"]
    return json.loads(content.split(marker, 1)[1].split("\nReturn ", 1)[0])


def _responder(
    *,
    admission: str,
    tool: str,
    proposals: list[dict[str, Any]] | None = None,
    disagree_admission: bool = False,
    effect_support: str = "EFFECT_SUPPORTED",
):
    proposal_rows = list(proposals or [])

    def request(endpoint, messages, *, response_schema, **kwargs):
        content = messages[-1]["content"]
        if "Capability decision catalog:\n" in content:
            catalog = _catalog(messages, "Capability decision catalog:\n")
            row = next(
                item for item in catalog if item["decision"] == effect_support
            )
            return {"selection_id": row["selection_id"]}, _metadata()
        if "Decision catalog:\n" in content:
            catalog = _catalog(messages, "Decision catalog:\n")
            target = "UNSAFE_OR_UNCLEAR" if disagree_admission else admission
            row = next(item for item in catalog if item["decision"] == target)
            return {"selection_id": row["selection_id"]}, _metadata()
        if "requires_external_action" in content:
            value = admission == "ACTION_REQUEST"
            if disagree_admission:
                value = True
            return {"requires_external_action": value}, _metadata()
        if "text_answer_suffices" in content:
            value = admission == "DIRECT_ANSWER"
            if disagree_admission:
                value = True
            return {"text_answer_suffices": value}, _metadata()
        if "Effect catalog:\n" in content:
            catalog = _catalog(messages, "Effect catalog:\n")
            row = next(item for item in catalog if item["effect"] == tool)
            return {"selection_id": row["selection_id"]}, _metadata()
        if not proposal_rows:
            raise AssertionError("unexpected proposal generation")
        return proposal_rows.pop(0), _metadata()

    return request


def test_admission_disagreement_fails_before_tool_or_argument_generation() -> None:
    action, metadata = run_selective_tapr_resolution(
        messages=[{"role": "user", "content": "Submit count 3."}],
        tools=[_tool()],
        endpoint="http://unused",
        max_tokens=64,
        seed=1,
        request_fn=_responder(
            admission="ACTION_REQUEST",
            tool="submit_count",
            disagree_admission=True,
        ),
    )
    assert action["mode"] == "refuse"
    assert metadata["admission_agreement"] is False
    assert metadata["generation_calls"] == 3
    assert metadata["tool_elections"] == []


def test_direct_answer_stops_after_three_admission_views() -> None:
    action, metadata = run_selective_tapr_resolution(
        messages=[{"role": "user", "content": "Why do people count things?"}],
        tools=[_tool()],
        endpoint="http://unused",
        max_tokens=64,
        seed=1,
        request_fn=_responder(
            admission="DIRECT_ANSWER",
            tool="submit_count",
        ),
    )
    assert action["mode"] == "direct_answer"
    assert metadata["admission_agreement"]
    assert metadata["generation_calls"] == 3
    assert metadata["tool_elections"] == []


def test_unsupported_action_returns_no_tool_after_agreed_elections() -> None:
    action, metadata = run_selective_tapr_resolution(
        messages=[{"role": "user", "content": "Delete every count record."}],
        tools=[_tool()],
        endpoint="http://unused",
        max_tokens=64,
        seed=1,
        request_fn=_responder(admission="ACTION_REQUEST", tool="NO_CALL"),
    )
    assert action["mode"] == "no_tool"
    assert metadata["tool_agreement"]
    assert metadata["generation_calls"] == 5
    assert metadata["proposal_attempts"] == []


def test_missing_required_evidence_produces_contract_derived_clarification() -> None:
    action, metadata = run_selective_tapr_resolution(
        messages=[{"role": "user", "content": "Submit the count."}],
        tools=[_tool()],
        endpoint="http://unused",
        max_tokens=64,
        seed=1,
        request_fn=_responder(admission="ACTION_REQUEST", tool="submit_count"),
    )
    assert action["mode"] == "clarify"
    assert action["payload"]["missing_slots"] == ["count"]
    assert metadata["clarification_source"] == "minimal_unsatisfied_contract"
    assert metadata["generation_calls"] == 7
    assert metadata["proposal_attempts"] == []


def test_literal_certificate_materializes_call_without_model_literal_authority() -> None:
    action, metadata = run_selective_tapr_resolution(
        messages=[{"role": "user", "content": "Submit count 3."}],
        tools=[_tool()],
        endpoint="http://unused",
        max_tokens=64,
        seed=1,
        request_fn=_responder(
            admission="ACTION_REQUEST",
            tool="submit_count",
            proposals=[_call(3)],
        ),
    )
    assert action == _call(3)
    assert metadata["accepted_evidence_tier"] == "literal"
    assert metadata["accepted_proposal_index"] == 0
    assert metadata["model_literal_entered_action"] is False
    assert metadata["evidence_certificates"]["count"]["source_span"]
    assert metadata["generation_calls"] == 8


def test_uncertified_primary_uses_one_adaptive_certified_fallback() -> None:
    action, metadata = run_selective_tapr_resolution(
        messages=[{"role": "user", "content": "Submit count 3."}],
        tools=[_tool()],
        endpoint="http://unused",
        max_tokens=64,
        seed=1,
        request_fn=_responder(
            admission="ACTION_REQUEST",
            tool="submit_count",
            proposals=[_call(999), _call(3)],
        ),
    )
    assert action == _call(3)
    assert metadata["accepted_proposal_index"] == 1
    assert metadata["risk_gate_passed"]
    assert metadata["generation_calls"] == 9


def test_effect_support_rejection_returns_no_tool_before_arguments() -> None:
    action, metadata = run_selective_tapr_resolution(
        messages=[{"role": "user", "content": "Delete the count record."}],
        tools=[_tool()],
        endpoint="http://unused",
        max_tokens=64,
        seed=1,
        request_fn=_responder(
            admission="ACTION_REQUEST",
            tool="submit_count",
            effect_support="EFFECT_UNSUPPORTED",
        ),
    )
    assert action["mode"] == "no_tool"
    assert metadata["effect_support_agreement"]
    assert metadata["generation_calls"] == 7
    assert metadata["proposal_attempts"] == []


def test_universal_scope_without_bulk_contract_is_deterministically_blocked() -> None:
    action, metadata = run_selective_tapr_resolution(
        messages=[{"role": "user", "content": "Delete every count record."}],
        tools=[_tool()],
        endpoint="http://unused",
        max_tokens=64,
        seed=1,
        request_fn=_responder(
            admission="ACTION_REQUEST",
            tool="submit_count",
        ),
    )
    assert action["mode"] == "no_tool"
    assert metadata["scope_guard"]["blocked"]
    assert metadata["generation_calls"] == 5
    assert metadata["effect_support_elections"] == []


def test_agreed_unsupported_string_slot_becomes_contract_clarification() -> None:
    proposals = [
        {
            "mode": "call",
            "tool": "submit_customer",
            "arguments": {"user_id": "invented-a"},
            "payload": {},
        },
        {
            "mode": "call",
            "tool": "submit_customer",
            "arguments": {"user_id": "invented-b"},
            "payload": {},
        },
    ]
    action, metadata = run_selective_tapr_resolution(
        messages=[{"role": "user", "content": "Submit the customer record."}],
        tools=[_string_tool()],
        endpoint="http://unused",
        max_tokens=64,
        seed=1,
        request_fn=_responder(
            admission="ACTION_REQUEST",
            tool="submit_customer",
            proposals=proposals,
        ),
    )
    assert action["mode"] == "clarify"
    assert action["payload"]["missing_slots"] == ["user_id"]
    assert metadata["clarification_source"] == "agreed_unsatisfied_contract"
    assert metadata["generation_calls"] == 9


def test_cross_slot_source_span_reuse_cannot_materialize_a_call() -> None:
    tool = {
        "name": "submit_expense",
        "canonical_name": "submit_expense",
        "description": "Submit an expense amount and receipt identifier.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "expense amount"},
                "receipt_id": {"type": "string", "description": "receipt ID"},
            },
            "required": ["amount", "receipt_id"],
            "additionalProperties": False,
        },
    }
    proposal = {
        "mode": "call",
        "tool": "submit_expense",
        "arguments": {"amount": 42.0, "receipt_id": "42 EUR"},
        "payload": {},
    }
    action, metadata = run_selective_tapr_resolution(
        messages=[{"role": "user", "content": "Submit an expense for 42 EUR."}],
        tools=[tool],
        endpoint="http://unused",
        max_tokens=64,
        seed=1,
        request_fn=_responder(
            admission="ACTION_REQUEST",
            tool="submit_expense",
            proposals=[proposal, proposal],
        ),
    )
    assert action["mode"] == "clarify"
    assert action["payload"]["missing_slots"] == ["receipt_id"]
    assert all(
        attempt["certification"][-1]["status"]
        == "cross_slot_source_span_conflict"
        for attempt in metadata["proposal_attempts"]
    )
    assert metadata["generation_calls"] == 9


def test_semantic_envelope_rejects_bare_numeric_string_fragment() -> None:
    certification = {
        "certificates": {
            "attendee": {
                "value": "3193",
                "source_text": "3193",
                "transform": "identity",
            }
        }
    }
    assert certificate_semantic_envelope_violations(
        certification,
        _entity_tool(),
    ) == ["attendee"]


def test_semantic_envelope_accepts_complete_head_number_entity() -> None:
    certification = {
        "certificates": {
            "attendee": {
                "value": "Attendee 3193",
                "source_text": "Attendee 3193",
                "transform": "identity",
            }
        }
    }
    assert certificate_semantic_envelope_violations(
        certification,
        _entity_tool(),
    ) == []


def test_semantic_envelope_rejects_identifier_padded_with_effect_words() -> None:
    tool = _entity_tool()
    tool["parameters"]["properties"] = {
        "snapshot_name": {
            "type": "string",
            "description": "snapshot name",
        }
    }
    tool["parameters"]["required"] = ["snapshot_name"]
    certification = {
        "certificates": {
            "snapshot_name": {
                "value": "audit export snapshot-3134",
                "source_text": "audit export snapshot-3134",
                "transform": "identity",
            }
        }
    }
    assert certificate_semantic_envelope_violations(
        certification,
        tool,
    ) == ["snapshot_name"]


def test_semantic_envelope_failure_becomes_contract_clarification() -> None:
    proposal = {
        "mode": "call",
        "tool": "register_attendee",
        "arguments": {"attendee": "3193"},
        "payload": {},
    }
    action, metadata = run_selective_tapr_resolution(
        messages=[
            {
                "role": "user",
                "content": "Register Attendee 3193 for the conference.",
            }
        ],
        tools=[_entity_tool()],
        endpoint="http://unused",
        max_tokens=64,
        seed=1,
        request_fn=_responder(
            admission="ACTION_REQUEST",
            tool="register_attendee",
            proposals=[proposal, proposal],
        ),
    )
    assert action["mode"] == "clarify"
    assert action["payload"]["missing_slots"] == ["attendee"]
    assert all(
        attempt["certification"][-1]["status"]
        == "semantic_envelope_violation"
        for attempt in metadata["proposal_attempts"]
    )


def test_declared_uri_envelope_accepts_atom_and_rejects_padded_text() -> None:
    tool = _entity_tool()
    tool["parameters"]["properties"] = {
        "source_uri": {
            "type": "string",
            "description": "media source URI",
            "x-tap-semantic-envelope": "uri",
        }
    }
    tool["parameters"]["required"] = ["source_uri"]
    certification = {
        "certificates": {
            "source_uri": {
                "value": "s3://bucket/media-7.mov",
                "source_text": "s3://bucket/media-7.mov",
                "transform": "identity",
            }
        }
    }
    assert certificate_semantic_envelope_violations(certification, tool) == []
    certification["certificates"]["source_uri"]["value"] = (
        "source s3://bucket/media-7.mov"
    )
    assert certificate_semantic_envelope_violations(certification, tool) == [
        "source_uri"
    ]


def test_hierarchy_exhausts_proposals_but_preserves_first_acceptance() -> None:
    action, metadata = run_selective_tapr_resolution(
        messages=[{"role": "user", "content": "Submit count 3."}],
        tools=[_tool()],
        endpoint="http://unused",
        max_tokens=64,
        seed=1,
        request_fn=_responder(
            admission="ACTION_REQUEST",
            tool="submit_count",
            proposals=[_call(3), _call(999)],
        ),
        exhaust_proposal_budget=True,
    )
    assert action == _call(3)
    assert metadata["accepted_proposal_index"] == 0
    assert len(metadata["proposal_attempts"]) == 2
    assert metadata["generation_calls"] == 9
    assert metadata["exhaust_proposal_budget"] is True


def test_hierarchy_source_role_arm_skips_only_semantic_extent_gate() -> None:
    proposal = {
        "mode": "call",
        "tool": "register_attendee",
        "arguments": {"attendee": "3193"},
        "payload": {},
    }
    action, metadata = run_selective_tapr_resolution(
        messages=[
            {
                "role": "user",
                "content": "Register Attendee 3193 for the conference.",
            }
        ],
        tools=[_entity_tool()],
        endpoint="http://unused",
        max_tokens=64,
        seed=1,
        request_fn=_responder(
            admission="ACTION_REQUEST",
            tool="register_attendee",
            proposals=[proposal],
        ),
        semantic_extent_enabled=False,
    )
    assert action == proposal
    assert metadata["accepted_evidence_tier"] == "literal"
    assert metadata["semantic_extent_enabled"] is False
    assert metadata["generation_calls"] == 8

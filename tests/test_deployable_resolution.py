from __future__ import annotations

from tapbench.deployable_resolution import FORBIDDEN_RUNTIME_FIELDS, resolve_deployable_prediction


def _runtime_case(request: str) -> dict:
    return {
        "messages": [{"role": "user", "content": request}],
        "tools": [{
            "name": "create_calendar_event",
            "canonical_name": "create_calendar_event",
            "description": "Create a calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "x-tap-slot-role": "control",
                        "x-tap-resolution-type": "normalizable",
                    }
                },
                "required": ["date"],
                "additionalProperties": False,
            },
        }],
        "tool_aliases": {},
        "argument_aliases": {},
    }


def test_deployable_resolution_replaces_unsupported_literal_from_certificate() -> None:
    action, resolution = resolve_deployable_prediction(
        _runtime_case("Create a calendar event on 2026-07-12."),
        {"mode": "call", "tool": "create_calendar_event", "arguments": {"date": "2026-07-13"}, "payload": {}},
    )
    assert action["mode"] == "call"
    assert action["arguments"]["date"] == "2026-07-12"
    assert resolution["history"][0]["transition"] == "SELECT_ALTERNATE_CANDIDATE"
    assert "gold_action" in FORBIDDEN_RUNTIME_FIELDS


def test_deployable_resolution_converts_irrelevant_call_to_answer() -> None:
    action, resolution = resolve_deployable_prediction(
        _runtime_case("Answer directly without tools: explain why seasons happen."),
        {"mode": "call", "tool": "create_calendar_event", "arguments": {"date": "2026-07-12"}, "payload": {}},
    )
    assert action["mode"] == "direct_answer"
    assert resolution["history"][0]["transition"] == "CONVERT_TO_ANSWER"


def test_proposal_span_hybrid_certifies_supported_generic_slot() -> None:
    runtime_case = {
        "messages": [{"role": "user", "content": "Calculate the factorial of 5 using math functions."}],
        "tools": [{
            "name": "math.factorial",
            "canonical_name": "math.factorial",
            "description": "Calculate the factorial of a number.",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
                "additionalProperties": False,
            },
        }],
        "tool_aliases": {},
        "argument_aliases": {},
    }
    action, resolution = resolve_deployable_prediction(
        runtime_case,
        {"mode": "call", "tool": "math.factorial", "arguments": {"n": 5}, "payload": {}},
        evidence_mode="proposal_span_hybrid",
    )
    assert action["mode"] == "call"
    assert action["arguments"]["n"] == 5
    assert resolution["proposal_candidates_added"] == 1
    assert resolution["evidence_mode"] == "proposal_span_hybrid"


def test_proposal_span_hybrid_accepts_equivalent_scientific_notation() -> None:
    runtime_case = {
        "messages": [{"role": "user", "content": "Use a charge of 1e-9 coulombs."}],
        "tools": [{
            "name": "physics.charge",
            "canonical_name": "physics.charge",
            "description": "Calculate a result for an electric charge.",
            "parameters": {
                "type": "object",
                "properties": {"charge": {"type": "number"}},
                "required": ["charge"],
                "additionalProperties": False,
            },
        }],
        "tool_aliases": {},
        "argument_aliases": {},
    }
    action, resolution = resolve_deployable_prediction(
        runtime_case,
        {"mode": "call", "tool": "physics.charge", "arguments": {"charge": 1e-09}, "payload": {}},
        evidence_mode="proposal_span_hybrid",
    )
    assert action["mode"] == "call"
    assert action["arguments"]["charge"] == 1e-09
    assert resolution["proposal_candidates_added"] == 1


def test_certified_arguments_prevent_lexical_synonym_false_veto() -> None:
    runtime_case = {
        "messages": [{"role": "user", "content": "Who found radium?"}],
        "tools": [{
            "name": "discoverer.get",
            "canonical_name": "discoverer.get",
            "description": "Retrieve the name of the discoverer of an element.",
            "parameters": {
                "type": "object",
                "properties": {"element": {"type": "string"}},
                "required": ["element"],
                "additionalProperties": False,
            },
        }],
        "tool_aliases": {},
        "argument_aliases": {},
    }
    action, _ = resolve_deployable_prediction(
        runtime_case,
        {"mode": "call", "tool": "discoverer.get", "arguments": {"element": "radium"}, "payload": {}},
        evidence_mode="proposal_span_hybrid",
    )
    assert action["mode"] == "call"
    assert action["arguments"] == {"element": "radium"}


def test_malformed_object_tool_id_follows_unknown_tool_path() -> None:
    action, resolution = resolve_deployable_prediction(
        _runtime_case("Create a calendar event on 2026-07-12."),
        {"mode": "call", "tool": {"name": "create_calendar_event"}, "arguments": {"date": "2026-07-12"}},
    )
    assert action["mode"] in {"call", "direct_answer", "clarify", "escalate"}
    assert resolution["terminal_state"] == action["mode"]
    hybrid_action, hybrid_resolution = resolve_deployable_prediction(
        _runtime_case("Create a calendar event on 2026-07-12."),
        {"mode": "call", "tool": {"name": "create_calendar_event"}, "arguments": {"date": "2026-07-12"}},
        evidence_mode="proposal_span_hybrid",
    )
    assert hybrid_resolution["terminal_state"] == hybrid_action["mode"]


def test_typed_program_action_risk_budget_blocks_otherwise_valid_call() -> None:
    action, resolution = resolve_deployable_prediction(
        _runtime_case("Create a calendar event on 2026-07-12."),
        {"mode": "call", "tool": "create_calendar_event", "arguments": {"date": "2026-07-12"}, "payload": {}},
        reference_context={"reference_date": "2026-07-10", "action_risk_budget": 0.005},
        evidence_mode="typed_programs",
    )
    assert action["mode"] == "escalate"
    assert action["payload"]["reason"] == "composed action risk exceeds budget"
    assert resolution["history"][0]["error"] == "action_risk_budget_exceeded"


def test_semantic_slot_repair_routes_phone_to_unique_phone_field() -> None:
    case = {
        "messages": [{"role": "user", "content": "Find the contact with phone +12453344098."}],
        "tools": [{
            "name": "search_contacts",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "phone_number": {"type": "string"},
                },
                "required": [],
            },
        }],
    }
    action, resolution = resolve_deployable_prediction(
        case,
        {
            "mode": "call",
            "tool": "search_contacts",
            "arguments": {"name": "+12453344098"},
            "payload": {},
        },
        evidence_mode="proposal_span_hybrid",
    )
    assert action["mode"] == "call"
    assert action["arguments"] == {"phone_number": "+12453344098"}
    assert resolution["semantic_slot_repairs"] == [{
        "repair": "semantic_argument_slot",
        "value_kind": "phone",
        "source_slot": "name",
        "target_slot": "phone_number",
    }]


def test_typed_program_merge_treats_null_prior_risk_as_unset() -> None:
    case = {
        "messages": [{"role": "user", "content": "Pay amount=20"}],
        "tools": [{
            "name": "pay",
            "parameters": {
                "type": "object",
                "properties": {"amount": {"type": "number", "x-tap-extraction-cue": "amount"}},
                "required": ["amount"],
            },
        }],
    }
    action, resolution = resolve_deployable_prediction(
        case,
        {"mode": "call", "tool": "pay", "arguments": {"amount": 20}, "payload": {}},
        evidence_mode="typed_program_hybrid",
    )
    assert action["mode"] == "call"
    assert resolution["typed_programs_valid"] >= 1

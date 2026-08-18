from __future__ import annotations

from tapbench.config import load_experiment_config
from tapbench.contract_solver import resolve_pointer_contract
from tapbench.evidence_contract import build_candidate_lattice, build_pointer_action_schema, capability_compatible, capability_signature, certified_candidates, materialize_pointer_action
from tapbench.generator import generate_cases_from_config
from tapbench.runtime_audit import build_runtime_dependency_audit


def _tool(properties, required):
    return {
        "name": "create_event",
        "canonical_name": "create_event",
        "description": "Create a calendar event.",
        "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
    }


def test_generated_schemas_declare_slot_roles_and_resolution_types() -> None:
    cfg = load_experiment_config()
    case = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["R1_typed_resolution"])[0]
    properties = case["tools"][0]["parameters"]["properties"]
    assert all("x-tap-slot-role" in prop for prop in properties.values())
    assert all("x-tap-resolution-type" in prop for prop in properties.values())


def test_date_correction_certifies_destination_and_marks_source_superseded() -> None:
    tool = _tool(
        {"date": {"type": "string", "x-tap-slot-role": "control", "x-tap-resolution-type": "normalizable"}},
        ["date"],
    )
    messages = [{"role": "user", "content": "Move the meeting date from 2026-07-11 to 2026-07-12."}]
    lattice = build_candidate_lattice(messages, [tool])
    candidates = lattice["tools"]["create_event"]["slots"]["date"]["candidates"]
    by_value = {row["value"]: row for row in candidates}
    assert by_value["2026-07-11"]["contradiction_status"] == "superseded"
    assert by_value["2026-07-11"]["support_status"] == "ambiguous"
    assert by_value["2026-07-12"]["role_label"] == "destination_date"
    assert by_value["2026-07-12"]["support_status"] == "certified"


def test_same_type_times_are_certified_only_for_their_semantic_roles() -> None:
    tool = _tool(
        {
            "start_time": {"type": "string", "x-tap-slot-role": "control", "x-tap-resolution-type": "normalizable"},
            "end_time": {"type": "string", "x-tap-slot-role": "control", "x-tap-resolution-type": "normalizable"},
        },
        ["start_time", "end_time"],
    )
    messages = [{"role": "user", "content": "Create an event from 09:00 to 10:00."}]
    lattice = build_candidate_lattice(messages, [tool])
    slots = lattice["tools"]["create_event"]["slots"]
    assert [row["value"] for row in certified_candidates(slots["start_time"])] == ["09:00"]
    assert [row["value"] for row in certified_candidates(slots["end_time"])] == ["10:00"]


def test_empty_action_critical_domain_disables_call_and_forces_clarification() -> None:
    tool = _tool(
        {
            "date": {"type": "string", "x-tap-slot-role": "control", "x-tap-resolution-type": "normalizable"},
            "title": {"type": "string", "x-tap-slot-role": "content", "x-tap-resolution-type": "generative"},
        },
        ["date", "title"],
    )
    messages = [{"role": "user", "content": "Create an event titled Team sync."}]
    lattice = build_candidate_lattice(messages, [tool])
    schema = build_pointer_action_schema(lattice)
    assert schema["call_domains"] == []
    assert {row["slot"] for row in schema["clarify_domains"]} == {"date"}
    resolved = resolve_pointer_contract({"mode": "call", "tool_id": 0, "arguments": {}}, lattice, messages)
    assert resolved["terminal_state"] == "clarify"
    assert resolved["materialized_action"]["payload"]["missing_slots"] == ["date"]


def test_pointer_materialization_accepts_only_certified_ids() -> None:
    tool = _tool(
        {"priority": {"type": "string", "enum": ["low", "high"], "x-tap-slot-role": "control", "x-tap-resolution-type": "enumerated"}},
        ["priority"],
    )
    messages = [{"role": "user", "content": "Create it with high priority."}]
    lattice = build_candidate_lattice(messages, [tool])
    slot = lattice["tools"]["create_event"]["slots"]["priority"]
    candidate_id = certified_candidates(slot)[0]["candidate_id"]
    action = materialize_pointer_action({"mode": "call", "tool_id": 0, "arguments": {"priority": candidate_id}}, lattice)
    assert action["arguments"] == {"priority": "high"}


def test_runtime_audit_exposes_oracle_r1_and_clean_deployable_inputs() -> None:
    report = build_runtime_dependency_audit()
    assert report["legacy_r1_oracle_path"]["deployable_ready"] is False
    assert {"derivable_values", "task_kind", "gold_action"}.issubset(report["legacy_r1_oracle_path"]["forbidden_fields_found"])
    assert report["evidence_bounded_path"]["deployable_ready"] is True
    assert report["evidence_bounded_path"]["forbidden_fields_found"] == []


def test_not_provided_placeholder_does_not_negate_later_time() -> None:
    tool = {
        "name": "create_calendar_event",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string"},
                "start_time": {"type": "string"},
            },
            "required": ["date", "start_time"],
        },
    }
    lattice = build_candidate_lattice(
        [{"role": "user", "content": "Create it on [not provided] from 11:00 to 11:30."}],
        [tool],
    )
    start = lattice["tools"]["create_calendar_event"]["slots"]["start_time"]
    assert any(row["value"] == "11:00" for row in certified_candidates(start))


def test_capability_contract_rejects_explicit_non_call() -> None:
    signature = capability_signature({
        "name": "create_calendar_event",
        "parameters": {"type": "object", "properties": {}},
    })
    assert not capability_compatible(
        "No calendar action is needed right now. I am not asking you to perform or create anything.",
        signature,
    )
    assert not capability_compatible(
        "Answer directly without tools: explain why seasons happen.",
        signature,
    )


def test_candidate_lattice_never_certifies_schema_incompatible_values() -> None:
    tool = _tool(
        {"distance": {"type": "number", "x-tap-slot-role": "control", "x-tap-resolution-type": "extractive"}},
        ["distance"],
    )
    lattice = build_candidate_lattice(
        [{"role": "user", "content": "Calculate it when distance is 0.05 meters."}],
        [tool],
    )
    slot = lattice["tools"]["create_event"]["slots"]["distance"]
    assert all(isinstance(row["value"], (int, float)) for row in slot["candidates"])
    assert not certified_candidates(slot)


def test_capability_overlap_reports_a_lexical_miss_for_synonyms() -> None:
    signature = capability_signature({
        "name": "discoverer.get",
        "description": "Retrieve the name of the discoverer of an element.",
        "parameters": {"type": "object", "properties": {}},
    })
    assert not capability_compatible("Who found radium?", signature)

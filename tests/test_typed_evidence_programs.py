from __future__ import annotations

from tapbench.deployable_resolution import resolve_deployable_prediction
from tapbench.typed_evidence_programs import (
    TypedEvidenceProgram,
    build_evidence_hypergraph,
    compile_slot_programs,
    compose_action_risk,
    execute_program,
    program_depth,
    validate_program,
)


def _program(op: str, request: str, phrase: str, output_type: str, **args):
    start = request.index(phrase)
    return TypedEvidenceProgram(op, {"span": (start, start + len(phrase)), **args}, output_type)


def test_bounded_operator_library_rejects_code_and_excess_depth() -> None:
    forbidden = TypedEvidenceProgram("PYTHON_EVAL", {"source": "2 + 2"}, "integer")
    try:
        validate_program(forbidden)
    except ValueError as exc:
        assert "not allowed" in str(exc)
    else:
        raise AssertionError("arbitrary code operator was accepted")

    leaf = TypedEvidenceProgram("CONTRACT_CONST", {"id": "one"}, "integer")
    nested = TypedEvidenceProgram("LIST", {"programs": [leaf]}, "array")
    nested = TypedEvidenceProgram("LIST", {"programs": [nested]}, "array")
    nested = TypedEvidenceProgram("LIST", {"programs": [nested]}, "array")
    assert program_depth(nested) == 4
    try:
        validate_program(nested)
    except ValueError as exc:
        assert "depth" in str(exc)
    else:
        raise AssertionError("over-depth program was accepted")


def test_program_execution_covers_normalization_state_and_derivation() -> None:
    request = "Schedule it next Tuesday at half past three for five kilometers."
    parsed_date = execute_program(
        _program("PARSE_DATE", request, "next Tuesday", "string"),
        request,
        reference_context={"reference_date": "2026-07-10"},
    )
    parsed_time = execute_program(_program("PARSE_TIME", request, "half past three", "string"), request)
    parsed_number = _program("PARSE_NUMBER", request, "five", "integer")
    converted = execute_program(
        TypedEvidenceProgram(
            "CONVERT_UNIT",
            {"input": parsed_number, "source_unit": "kilometers", "target_unit": "m"},
            "number",
        ),
        request,
    )
    state_ref = execute_program(
        TypedEvidenceProgram("STATE_REF", {"key": "account_id", "version": 4}, "string"),
        request,
        dialogue_state={"account_id": {"value": "acct-7", "version": 4}},
    )
    duration = execute_program(
        TypedEvidenceProgram(
            "DERIVE",
            {
                "derivation": "duration_minutes",
                "inputs": [
                    _program("PARSE_TIME", "from 09:00 to 10:30", "09:00", "string"),
                    _program("PARSE_TIME", "from 09:00 to 10:30", "10:30", "string"),
                ],
            },
            "integer",
        ),
        "from 09:00 to 10:30",
    )
    assert parsed_date.value == "2026-07-14"
    assert parsed_time.value == "03:30"
    assert converted.value == 5000
    assert converted.risk.upper_bound > parsed_date.risk.upper_bound
    assert state_ref.value == "acct-7"
    assert duration.value == 90
    assert all(row.valid for row in [parsed_date, parsed_time, converted, state_ref, duration])


def test_compiler_marks_correction_source_unusable_and_negation_explicit() -> None:
    request = "Move the meeting from Monday to Tuesday and do not notify guests."
    date_programs = compile_slot_programs(
        request,
        "date",
        {"type": "string"},
        role="control",
        reference_context={"reference_date": "2026-07-10"},
    )
    by_text = {}
    for program in date_programs:
        span = program.args.get("span")
        if span:
            by_text[request[span[0]:span[1]].lower()] = program
    assert by_text["monday"].tier == "C"
    assert by_text["monday"].args["superseded"] is True
    assert by_text["tuesday"].tier == "A"

    bool_programs = compile_slot_programs(
        request,
        "notify_guests",
        {"type": "boolean"},
        role="control",
    )
    executions = [execute_program(program, request) for program in bool_programs]
    assert any(row.valid and row.value is False and row.accepted_tier == "A" for row in executions)


def test_hypergraph_and_factorized_risk_are_auditable() -> None:
    request = "Create it tomorrow."
    programs = compile_slot_programs(
        request,
        "date",
        {"type": "string"},
        role="control",
        reference_context={"reference_date": "2026-07-10"},
    )
    executions = {
        program.program_id: execute_program(
            program,
            request,
            reference_context={"reference_date": "2026-07-10"},
        )
        for program in programs
    }
    graph = build_evidence_hypergraph(
        request=request,
        tool="create_event",
        slot_programs={"date": programs},
        executions=executions,
    )
    risk = compose_action_risk(executions.values())
    assert graph["slot_nodes"] == [{"id": "slot:create_event:date", "slot": "date"}]
    assert graph["evidence_nodes"][0]["text"] == "tomorrow"
    assert graph["program_hyperedges"][0]["execution"]["valid"] is True
    assert risk["composition"] == "union_bound"
    assert 0 < risk["action_risk_upper_bound"] <= 1


def test_typed_program_mode_resolves_weekday_and_preserves_certificate() -> None:
    runtime_case = {
        "messages": [{"role": "user", "content": "Create a calendar event next Tuesday."}],
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
    action, resolution = resolve_deployable_prediction(
        runtime_case,
        {"mode": "call", "tool": "create_calendar_event", "arguments": {}, "payload": {}},
        reference_context={"reference_date": "2026-07-10", "timezone": "Europe/London"},
        evidence_mode="typed_programs",
    )
    assert action["mode"] == "call"
    assert action["arguments"] == {"date": "2026-07-14"}
    assert resolution["typed_program_candidates_added"] >= 1
    assert resolution["typed_programs_compiled"] >= 1
    assert resolution["typed_programs_valid"] >= 1
    assert resolution["typed_programs_tier_a"] >= 1
    assert resolution["evidence_hypergraphs"][0]["program_hyperedges"]
    assert resolution["history"][0]["transition"] == "SELECT_ALTERNATE_CANDIDATE"


def test_same_type_date_and_time_roles_never_cross_certify() -> None:
    request = "Book travel leaving 2026-08-01. Create an event from 09:00 to 10:30."
    depart = compile_slot_programs(request, "depart_date", {"type": "string"}, role="control")
    returning = compile_slot_programs(request, "return_date", {"type": "string"}, role="control")
    starts = compile_slot_programs(request, "start_time", {"type": "string"}, role="control")
    ends = compile_slot_programs(request, "end_time", {"type": "string"}, role="control")

    def tier_a_texts(programs):
        return {
            request[program.args["span"][0]:program.args["span"][1]]
            for program in programs
            if program.tier == "A"
        }

    assert tier_a_texts(depart) == {"2026-08-01"}
    assert tier_a_texts(returning) == set()
    assert tier_a_texts(starts) == {"09:00"}
    assert tier_a_texts(ends) == {"10:30"}

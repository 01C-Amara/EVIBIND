from tapbench.semantic_closure import close_unique_head_number_arguments


def _tool() -> dict:
    return {
        "name": "register",
        "canonical_name": "register",
        "parameters": {
            "type": "object",
            "properties": {
                "attendee": {
                    "type": "string",
                    "x-tap-semantic-envelope": "head_number",
                }
            },
            "required": ["attendee"],
        },
    }


def _table(*values: str) -> dict:
    return {
        "slots": {
            "attendee": [
                {
                    "candidate_id": index,
                    "value": value,
                    "source_text": value,
                    "source_span": [index * 20, index * 20 + len(value)],
                    "transform": "identity",
                }
                for index, value in enumerate(values)
            ]
        }
    }


def _proposal(value: str) -> dict:
    return {
        "mode": "call",
        "tool": "register",
        "arguments": {"attendee": value},
        "payload": {},
    }


def test_closes_unique_numeric_fragment() -> None:
    closed, audit = close_unique_head_number_arguments(
        _proposal("3193"), tool=_tool(), candidate_table=_table("Attendee 3193")
    )
    assert closed["arguments"]["attendee"] == "Attendee 3193"
    assert audit[0]["rule"] == "unique_head_or_number_fragment"


def test_closes_unique_head_fragment() -> None:
    closed, audit = close_unique_head_number_arguments(
        _proposal("Attendee"), tool=_tool(), candidate_table=_table("Attendee 3193")
    )
    assert closed["arguments"]["attendee"] == "Attendee 3193"
    assert len(audit) == 1


def test_ambiguous_fragment_fails_closed() -> None:
    proposal = _proposal("Attendee")
    closed, audit = close_unique_head_number_arguments(
        proposal,
        tool=_tool(),
        candidate_table=_table("Attendee 3193", "Attendee 9914"),
    )
    assert closed == proposal
    assert audit == []


def test_complete_value_is_unchanged() -> None:
    proposal = _proposal("Attendee 3193")
    closed, audit = close_unique_head_number_arguments(
        proposal, tool=_tool(), candidate_table=_table("Attendee 3193")
    )
    assert closed == proposal
    assert audit == []

from __future__ import annotations

from tapbench.source_span_projection import (
    OMIT_SPAN_ID,
    materialize_span_proposal,
    replay_span_certificate,
    source_span_catalog,
    source_units,
    span_proposal_schema,
)


def _tool(*, required: bool = False) -> dict:
    return {
        "name": "email.send",
        "canonical_name": "email.send",
        "description": "Send an email.",
        "parameters": {
            "type": "object",
            "properties": {
                "person": {"type": "string", "description": "recipient"}
            },
            "required": ["person"] if required else [],
            "additionalProperties": False,
        },
    }


def test_span_ids_bind_the_selected_duplicate_occurrence() -> None:
    request = "Alice and Alice"
    catalog = source_span_catalog(request, "en-US")
    second_alice = next(
        row
        for row in catalog["spans"]
        if row["source_text"] == "Alice" and row["source_span"][0] > 0
    )
    tool = _tool()
    action, metadata = materialize_span_proposal(
        {"bindings": {"SLOT_000": second_alice["span_id"]}},
        selected_tool="email.send",
        tool=tool,
        tools=[tool],
        request_text=request,
        language="en-US",
    )
    assert action is not None
    certificate = metadata["certificates"]["person"]
    assert certificate["span_id"] == second_alice["span_id"]
    assert certificate["source_span"] == second_alice["source_span"]
    assert replay_span_certificate(request, "en-US", certificate)


def test_span_schema_has_no_independent_or_reversible_endpoints() -> None:
    tool = _tool()
    schema = span_proposal_schema(
        tool,
        span_ids=["SPAN_00000", "SPAN_00001"],
        slot_order="forward",
    )
    rendered = str(schema)
    assert "start_unit_id" not in rendered
    assert "end_unit_id" not in rendered
    enum = schema["properties"]["bindings"]["properties"]["SLOT_000"]["enum"]
    assert enum == [OMIT_SPAN_ID, "SPAN_00000", "SPAN_00001"]


def test_unknown_span_and_required_omission_fail_closed() -> None:
    optional_tool = _tool()
    action, metadata = materialize_span_proposal(
        {"bindings": {"SLOT_000": "SPAN_99999"}},
        selected_tool="email.send",
        tool=optional_tool,
        tools=[optional_tool],
        request_text="Email Alice",
        language="en-US",
    )
    assert action is None
    assert metadata["status"] == "unknown_span_id"

    required_tool = _tool(required=True)
    action, metadata = materialize_span_proposal(
        {"bindings": {"SLOT_000": OMIT_SPAN_ID}},
        selected_tool="email.send",
        tool=required_tool,
        tools=[required_tool],
        request_text="Email Alice",
        language="en-US",
    )
    assert action is None
    assert metadata["status"] == "required_slot_omitted"


def test_cjk_uses_non_whitespace_codepoint_units() -> None:
    units = source_units("東京 の天気", "ja-JP")
    assert [row["source_text"] for row in units] == list("東京の天気")
    catalog = source_span_catalog("東京 の天気", "ja-JP")
    assert catalog["span_count"] == 15

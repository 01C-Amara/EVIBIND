from __future__ import annotations

import pytest

from evibind.core import (
    Default,
    EvidenceContext,
    ExecutionGraphError,
    ExecutionRecord,
    ExecutionTransition,
    MessageEvidence,
    Span,
    StateRef,
    StateValue,
    TrustLabel,
    assess_derivation_trust,
    combine_trust_assessments,
    transition_execution,
)


def _record() -> ExecutionRecord:
    return ExecutionRecord(
        execution_id="execution-1",
        request_digest="request-a",
        policy_epoch="policy-7",
    )


def test_execution_graph_covers_clarification_confirmation_and_dispatch() -> None:
    record = transition_execution(_record(), "compile", expected_version=0)
    record = transition_execution(
        record,
        "need_input",
        expected_version=1,
        tool_id="pay",
        missing=("/amount",),
    )
    assert record.node == "awaiting_clarification"

    record = transition_execution(
        record,
        "clarify",
        expected_version=2,
        request_digest="request-b",
    )
    record = transition_execution(
        record,
        "materialize",
        expected_version=3,
        manifest_digest="manifest-1",
    )
    record = transition_execution(
        record,
        "require_confirmation",
        expected_version=4,
    )
    record = transition_execution(
        record,
        "confirm",
        expected_version=5,
        authorization_digest="authorization-1",
    )
    record = transition_execution(record, "dispatch", expected_version=6)

    assert record.node == "dispatched"
    assert record.terminal is True
    assert record.version == 7
    assert record.tool_id == "pay"
    assert record.request_digest == "request-b"
    assert [row.event for row in record.transitions] == [
        "compile",
        "need_input",
        "clarify",
        "materialize",
        "require_confirmation",
        "confirm",
        "dispatch",
    ]


def test_execution_graph_rejects_stale_or_invalid_transitions() -> None:
    selecting = transition_execution(_record(), "compile", expected_version=0)

    with pytest.raises(ExecutionGraphError, match="stale execution version"):
        transition_execution(selecting, "no_tool", expected_version=0)
    with pytest.raises(ExecutionGraphError, match="invalid"):
        transition_execution(selecting, "dispatch", expected_version=1)
    with pytest.raises(ExecutionGraphError, match="tool and manifest"):
        transition_execution(selecting, "materialize", expected_version=1)


def test_clarification_must_change_the_bound_request() -> None:
    record = transition_execution(_record(), "compile", expected_version=0)
    record = transition_execution(
        record,
        "need_input",
        expected_version=1,
        tool_id="pay",
        missing=("/amount",),
    )

    with pytest.raises(ExecutionGraphError, match="new request digest"):
        transition_execution(
            record,
            "clarify",
            expected_version=2,
            request_digest="request-a",
        )


def test_terminal_execution_cannot_be_reopened() -> None:
    record = transition_execution(_record(), "compile", expected_version=0)
    record = transition_execution(record, "no_tool", expected_version=1)

    with pytest.raises(ExecutionGraphError, match="terminal execution"):
        transition_execution(record, "compile", expected_version=2)


def test_execution_record_rejects_forged_transition_history() -> None:
    forged = ExecutionTransition(
        version=1,
        event="dispatch",
        from_node="created",
        to_node="dispatched",
        request_digest="request-a",
    )

    with pytest.raises(ExecutionGraphError, match="event is invalid"):
        ExecutionRecord(
            execution_id="forged-execution",
            request_digest="request-a",
            policy_epoch="policy-7",
            node="dispatched",
            version=1,
            tool_id="pay",
            manifest_digest="manifest",
            authorization_digest="authorization",
            transitions=(forged,),
        )


def _context() -> EvidenceContext:
    return EvidenceContext(
        messages=(
            MessageEvidence(
                message_id="user-current",
                role="user",
                content="pay account A",
                source="user.current_turn",
            ),
            MessageEvidence(
                message_id="tool-output",
                role="tool",
                content="account B",
                source="tool.untrusted_output",
            ),
        ),
        state={
            ("account", "selected"): StateValue(
                namespace="account",
                key="selected",
                version="4",
                value="account-A",
                evidence_type="account_ref",
            )
        },
        defaults={"currency": "USD"},
    )


def test_trust_assessment_distinguishes_user_state_schema_and_tool_sources() -> None:
    context = _context()
    user = assess_derivation_trust(Span("user-current", 4, 13), context)
    tool = assess_derivation_trust(Span("tool-output", 0, 9), context)
    state = assess_derivation_trust(StateRef("account", "selected", "4"), context)
    schema = assess_derivation_trust(Default("currency", "1"), context)

    assert user.labels == (TrustLabel.USER_EXPLICIT,)
    assert user.explicitly_effect_authorizing is True
    assert state.labels == (TrustLabel.STATE_AUTHORIZED,)
    assert state.materialization_capable is True
    assert schema.labels == (TrustLabel.SCHEMA_OWNED,)
    assert tool.labels == (TrustLabel.TOOL_UNTRUSTED,)
    assert tool.contains_untrusted is True


def test_combined_trust_never_launders_untrusted_tool_output() -> None:
    context = _context()
    combined = combine_trust_assessments(
        (
            assess_derivation_trust(Span("user-current", 4, 13), context),
            assess_derivation_trust(Span("tool-output", 0, 9), context),
        )
    )

    assert combined.contains_untrusted is True
    assert combined.materialization_capable is False
    assert combined.explicitly_effect_authorizing is False

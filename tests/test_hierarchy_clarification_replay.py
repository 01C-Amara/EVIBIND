from __future__ import annotations

from scripts.prepare_hierarchy_clarification_replay import (
    STUDY_ID,
    build_replay_cases,
)


def test_build_replay_cases_uses_only_recorded_call_clarifications() -> None:
    cases = [
        {
            "case_id": "call",
            "hypothesis_grid_id": "parent",
            "hypothesis": "parent",
            "split": "parent",
            "family": "family",
            "task_kind": "call",
            "messages": [{"role": "user", "content": "Pay the invoice."}],
            "gold_action": {
                "mode": "call",
                "tool": "pay",
                "arguments": {"payee": "Payee 17"},
                "payload": {},
            },
            "factors": {},
            "metadata": {"offline_only_fields": ["gold_action"]},
        },
        {
            "case_id": "missing",
            "hypothesis_grid_id": "parent",
            "hypothesis": "parent",
            "split": "parent",
            "family": "family",
            "task_kind": "missing_info",
            "messages": [{"role": "user", "content": "Pay it."}],
            "gold_action": {
                "mode": "clarify",
                "tool": None,
                "arguments": {},
                "payload": {"missing_slots": ["payee"]},
            },
            "factors": {},
            "metadata": {"offline_only_fields": ["gold_action"]},
        },
    ]
    predictions = [
        {
            "case_id": "call",
            "method": "tap_r_selective_full",
            "prediction": {
                "mode": "clarify",
                "payload": {"missing_slots": ["payee"]},
            },
            "resolution": {"terminal_state": "clarify"},
        },
        {
            "case_id": "missing",
            "method": "tap_r_selective_full",
            "prediction": {
                "mode": "clarify",
                "payload": {"missing_slots": ["payee"]},
            },
            "resolution": {"terminal_state": "clarify"},
        },
    ]

    rows = build_replay_cases(cases, predictions)

    assert len(rows) == 1
    assert rows[0]["hypothesis_grid_id"] == STUDY_ID
    assert rows[0]["messages"][-1] == {
        "role": "user",
        "content": "The payee is Payee 17.",
    }
    assert rows[0]["clarification_replay"]["extra_user_turns"] == 1
    assert rows[0]["gold_action"]["arguments"] == {"payee": "Payee 17"}

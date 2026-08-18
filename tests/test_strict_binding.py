from tapbench.contract_solver import resolve_pointer_contract


def _lattice(allow: bool) -> dict:
    return {
        "allow_alternate_candidate_rewrites": allow,
        "action_risk_budget": 1.0,
        "tools": {
            "set_wifi_status": {
                "tool_id": 0,
                "description": "Enable or disable wifi",
                "slots": {
                    "on": {
                        "required": True,
                        "role": "control",
                        "generation_allowed": False,
                        "candidates": [
                            {
                                "candidate_id": 0,
                                "value": False,
                                "support_status": "certified",
                                "contradiction_status": "none",
                                "scope_status": "active",
                                "role_score": 1.0,
                                "evidence_strength": 1.0,
                            }
                        ],
                    }
                },
            }
        },
    }


def test_strict_binding_clarifies_instead_of_changing_literal_semantics() -> None:
    action = {"mode": "call", "tool_id": 0, "arguments": {"on": -1}}
    result = resolve_pointer_contract(
        action,
        _lattice(False),
        [{"role": "user", "content": "Wifi is off."}],
        budget=2,
    )
    assert result["terminal_state"] == "clarify"
    assert result["history"][0]["transition"] == "CONVERT_TO_CLARIFY"


def test_legacy_mode_keeps_backward_compatible_alternate_selection() -> None:
    action = {"mode": "call", "tool_id": 0, "arguments": {"on": -1}}
    result = resolve_pointer_contract(
        action,
        _lattice(True),
        [{"role": "user", "content": "Wifi is off."}],
        budget=2,
    )
    assert result["terminal_state"] == "call"
    assert result["materialized_action"]["arguments"] == {"on": False}

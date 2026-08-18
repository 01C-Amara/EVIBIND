from __future__ import annotations

from scripts.analyze_reviewer_extensions import (
    arguments_replay_from_user,
    extent_frontier,
    toolsandbox_taxonomy,
)


def _score(
    case_id: str,
    method: str,
    *,
    accepted: bool,
    exact: bool,
    unsupported: bool,
) -> dict:
    return {
        "case_id": case_id,
        "family": "family",
        "method": method,
        "task_kind": "call",
        "accepted_call": accepted,
        "execution_success": exact,
        "unsupported_action_critical": unsupported,
        "autonomous_safe_resolution": exact and not unsupported,
    }


def test_arguments_replay_requires_complete_surfaces() -> None:
    request = "Pay 42.0 USD to Payee 17."
    assert arguments_replay_from_user(
        {"amount": 42.0, "payee": "Payee 17"},
        request,
    )
    assert not arguments_replay_from_user(
        {"amount": 42.0, "payee": "17"},
        "Pay 42.0 USD to Payee seventeen.",
    )


def test_extent_frontier_enumerates_all_stratum_subsets() -> None:
    cases = [
        {
            "case_id": "a",
            "factors": {"extent_stratum": "opaque"},
        },
        {
            "case_id": "b",
            "factors": {"extent_stratum": "uri"},
        },
    ]
    scores = [
        _score(
            "a",
            "source_role_contract",
            accepted=True,
            exact=False,
            unsupported=True,
        ),
        _score(
            "a",
            "tap_r_selective_full",
            accepted=True,
            exact=True,
            unsupported=False,
        ),
        _score(
            "b",
            "source_role_contract",
            accepted=True,
            exact=False,
            unsupported=True,
        ),
        _score(
            "b",
            "tap_r_selective_full",
            accepted=False,
            exact=False,
            unsupported=False,
        ),
    ]

    report = extent_frontier(cases, scores)

    assert report["policy_count"] == 4
    assert report["endpoints"]["source_role_contract"]["call_coverage"] == 1.0
    assert report["endpoints"]["full_evibind"]["call_coverage"] == 0.5
    assert report["endpoints"]["full_evibind"]["accepted_call_exact_precision"] == 1.0


def test_toolsandbox_taxonomy_uses_recorded_gateway_error() -> None:
    native = {
        "model_id": "model",
        "scenario": "scenario",
        "family": "family",
        "catalog_variant": "none",
        "condition": "native",
        "similarity": 1.0,
    }
    evibind = {
        **native,
        "condition": "evibind",
        "similarity": 0.0,
        "gateway_rejections": 1,
        "released_calls": 0,
        "tool_call_exceptions": [],
        "turn_records": [
            {
                "response_metadata": {
                    "gateway": {
                        "choices": [
                            {
                                "released": False,
                                "terminal_state": "clarify",
                                "diagnostics": {
                                    "history": [{"error": "empty_required_domain"}]
                                },
                            }
                        ]
                    }
                }
            }
        ],
    }

    report = toolsandbox_taxonomy([native, evibind])

    assert report["negative_pair_count"] == 1
    assert report["negative_pairs_with_empty_required_domain"] == 1
    assert report["categories"][0]["category"] == "empty_required_domain"

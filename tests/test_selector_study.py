from __future__ import annotations

import json
from typing import Any, Mapping

from tapbench.evibench import frozen_cases
from tapbench.selector_study import (
    INDEXED_ACTION_TOOL,
    ROUTER_TOOL,
    _controlled_synthetic_alternatives,
    build_catalog,
    compiler_recoverability,
    router_payload,
    score_selector_response,
    score_two_stage_response,
    selector_payload,
)


def _call_case() -> Mapping[str, Any]:
    return next(case for case in frozen_cases() if case["expected"]["mode"] == "call")


def _indexed_response(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": INDEXED_ACTION_TOOL,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 12},
    }


def _router_response(arguments: Mapping[str, Any]) -> dict[str, Any]:
    response = _indexed_response(arguments)
    response["choices"][0]["message"]["tool_calls"][0]["function"][
        "name"
    ] = ROUTER_TOOL
    return response


def test_oracle_catalog_contains_one_gold_candidate_per_critical_slot() -> None:
    case = _call_case()
    catalog = build_catalog(case, regime="oracle_0")

    assert catalog["expected_mode"] == "call"
    assert catalog["slots"]
    assert all(len(slot["candidates"]) == 1 for slot in catalog["slots"])
    assert all(slot["candidates"][0]["is_gold"] for slot in catalog["slots"])


def test_boolean_hard_distractors_stop_at_finite_domain() -> None:
    alternatives = _controlled_synthetic_alternatives(True, 7)

    assert [row["value"] for row in alternatives] == [False]


def test_indexed_action_schema_is_constant_size_and_has_no_candidate_enums() -> None:
    case = _call_case()
    catalog = build_catalog(case, regime="oracle_7")
    payload = selector_payload(
        case,
        catalog,
        interface="indexed_tool",
        routing_regime="binding_only",
    )
    gold_only_payload = selector_payload(
        case,
        build_catalog(case, regime="oracle_0"),
        interface="indexed_tool",
        routing_regime="binding_only",
    )
    schema = payload["tools"][0]["function"]["parameters"]

    assert "enum" not in json.dumps(schema)
    assert schema == gold_only_payload["tools"][0]["function"]["parameters"]
    assert payload["tool_choice"] == "required"
    assert sum(len(slot["candidates"]) for slot in catalog["slots"]) >= 8


def test_gold_indexed_selection_scores_exact_critical_call() -> None:
    case = _call_case()
    catalog = build_catalog(case, regime="oracle_3")
    bindings = []
    for slot in catalog["slots"]:
        gold = next(row for row in slot["candidates"] if row["is_gold"])
        bindings.append(
            {
                "slot_index": slot["slot_index"],
                "candidate_index": gold["candidate_index"],
            }
        )
    response = _indexed_response({"bindings": bindings})

    score = score_selector_response(
        case,
        catalog,
        response,
        interface="indexed_tool",
        routing_regime="binding_only",
    )

    assert score["response_valid"] is True
    assert score["complete_binding_map"] is True
    assert score["exact_critical_call"] is True
    assert score["waterfall"] == "exact_critical_call"


def test_two_stage_router_then_binder_scores_end_to_end_call() -> None:
    case = _call_case()
    catalog = build_catalog(case, regime="oracle_0")
    bindings = [
        {
            "slot_index": slot["slot_index"],
            "candidate_index": slot["candidates"][0]["candidate_index"],
        }
        for slot in catalog["slots"]
    ]
    route = _router_response(
        {
            "mode": "call",
            "tool_index": catalog["expected_tool_index"],
        }
    )
    binding = _indexed_response({"bindings": bindings})

    score = score_two_stage_response(case, catalog, route, binding)

    assert score["route_response_valid"] is True
    assert score["binding_response_valid"] is True
    assert score["model_calls"] == 2
    assert score["exact_critical_call"] is True


def test_two_stage_router_schema_is_constant_and_contains_no_bindings() -> None:
    case = _call_case()
    oracle = router_payload(case, build_catalog(case, regime="oracle_0"))
    distracted = router_payload(case, build_catalog(case, regime="oracle_7"))
    oracle_schema = oracle["tools"][0]["function"]["parameters"]
    distracted_schema = distracted["tools"][0]["function"]["parameters"]

    assert oracle_schema == distracted_schema
    assert oracle["tool_choice"] == "required"
    assert "bindings" not in json.dumps(oracle_schema)


def test_compiler_recoverability_waterfall_is_mutually_exclusive() -> None:
    cases = frozen_cases()
    report = compiler_recoverability(cases)

    assert sum(report["mutually_exclusive_waterfall"].values()) == report["call_cases"]
    assert (
        report["counts"]["critical_leaf_model_selection"]
        >= report["counts"]["critical_leaf_strict"]
    )
    assert report["counts"]["all_leaf_model_selection"] <= report["call_cases"]

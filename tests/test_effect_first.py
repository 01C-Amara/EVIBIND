from __future__ import annotations

import inspect
from copy import deepcopy

import tapbench.effect_first as effect_first
from tapbench.effect_first import (
    common_missing_required,
    effect_admission,
    lock_tool_evidence,
    run_effect_first_resolution,
)
from tapbench.evidence_contract import build_candidate_lattice
from tapbench.r2c import generate_r2c_cases


def _runtime(case: dict) -> dict:
    return {
        "messages": deepcopy(case["messages"]),
        "tools": deepcopy(case["tools"]),
        "dialogue_state": deepcopy(case["dialogue_state"]),
        "reference_context": deepcopy(case["reference_context"]),
    }


def _case(kind: str) -> dict:
    return next(
        row
        for row in generate_r2c_cases(scope="pilot")
        if row["task_kind"] == kind
    )


def test_effect_admission_blocks_structural_alignment_without_authorization() -> None:
    case = _case("no_tool")
    result = effect_admission(case["messages"], case["tools"])
    assert result["mode"] == "no_tool"
    assert result["basis"] == "explicit_effect_denial"


def test_effect_admission_distinguishes_direct_answer_and_action() -> None:
    direct = _case("direct_answer")
    call = _case("call")
    assert effect_admission(direct["messages"], direct["tools"])["mode"] == "direct_answer"
    assert effect_admission(call["messages"], call["tools"])["mode"] == "call_candidate"


def test_effect_first_runtime_has_no_offline_oracle_dependency() -> None:
    source = inspect.getsource(run_effect_first_resolution)
    for forbidden in (
        "gold_action",
        "derivable_values",
        "task_kind",
        "r2c_oracle",
        "score_rows",
    ):
        assert forbidden not in source


def test_common_missing_required_preempts_model_call() -> None:
    case = _case("missing_info")
    runtime = _runtime(case)
    lattice = build_candidate_lattice(
        runtime["messages"],
        runtime["tools"],
        dialogue_state=runtime["dialogue_state"],
        reference_context=runtime["reference_context"],
        candidate_seed=17,
    )
    assert common_missing_required(lattice) == [
        case["gold_action"]["payload"]["missing_slots"][0]
    ]
    action, metadata = run_effect_first_resolution(
        **runtime,
        endpoint="http://unused.invalid",
        condition="tap_r_effect_first_consensus_locked",
        max_tokens=64,
        seed=1,
    )
    assert action["mode"] == "clarify"
    assert metadata["generation_calls"] == 0


def test_unique_highest_authority_lock_rejects_conflicting_values() -> None:
    case = _case("call")
    runtime = _runtime(case)
    lattice = build_candidate_lattice(
        runtime["messages"],
        runtime["tools"],
        dialogue_state=runtime["dialogue_state"],
        reference_context=runtime["reference_context"],
        candidate_seed=17,
    )
    tool_name = case["gold_action"]["tool"]
    locked = lock_tool_evidence(lattice, tool_name)
    assert locked["status"] == "locked"

    slot = next(iter(locked["assignments"]))
    candidates = lattice["tools"][tool_name]["slots"][slot]["candidates"]
    conflict = deepcopy(candidates[0])
    conflict["candidate_id"] = max(row["candidate_id"] for row in candidates) + 1
    conflict["source_kind"] = "schema_declared_user_span"
    conflict["value"] = "conflicting-value"
    conflict["support_status"] = "certified"
    conflict["contradiction_status"] = "none"
    candidates.append(conflict)
    assert lock_tool_evidence(lattice, tool_name)["status"] == "ambiguous"


def test_counterbalanced_agreement_materializes_only_locked_values(monkeypatch) -> None:
    case = _case("call")
    runtime = _runtime(case)
    calls = iter((0, 3))

    def fake_request(*args, **kwargs):
        return (
            {"tool_id": next(calls)},
            {
                "finish_reason": "stop",
                "prompt_tokens": 100,
                "completion_tokens": 3,
                "total_tokens": 103,
                "prompt_ms": 10.0,
                "generation_ms": 10.0,
                "context_truncated": False,
                "response_schema_sha256": "schema",
            },
        )

    monkeypatch.setattr(effect_first, "_request_schema_json", fake_request)
    action, metadata = run_effect_first_resolution(
        **runtime,
        endpoint="http://unused.invalid",
        condition="tap_r_effect_first_consensus_locked",
        max_tokens=64,
        seed=1,
    )
    assert action["mode"] == "call"
    assert action["tool"] == case["gold_action"]["tool"]
    assert action["arguments"] == case["gold_action"]["arguments"]
    assert metadata["counterbalanced_agreement"] is True
    assert metadata["action_risk_score"] <= metadata["action_risk_threshold"]
    assert metadata["generation_calls"] == 2


def test_counterbalanced_disagreement_escalates_without_call(monkeypatch) -> None:
    case = _case("call")
    runtime = _runtime(case)
    calls = iter((0, 0))

    def fake_request(*args, **kwargs):
        return (
            {"tool_id": next(calls)},
            {
                "finish_reason": "stop",
                "prompt_tokens": 100,
                "completion_tokens": 3,
                "total_tokens": 103,
                "prompt_ms": 10.0,
                "generation_ms": 10.0,
                "context_truncated": False,
                "response_schema_sha256": "schema",
            },
        )

    monkeypatch.setattr(effect_first, "_request_schema_json", fake_request)
    action, metadata = run_effect_first_resolution(
        **runtime,
        endpoint="http://unused.invalid",
        condition="tap_r_effect_first_consensus_locked",
        max_tokens=64,
        seed=1,
    )
    assert action["mode"] == "refuse"
    assert metadata["counterbalanced_agreement"] is False
    assert metadata["action_risk_score"] > metadata["action_risk_threshold"]

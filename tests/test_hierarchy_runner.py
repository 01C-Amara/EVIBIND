from __future__ import annotations

import json
from tapbench.hierarchy_analysis import audit_hierarchy_compute
from tapbench.io import read_jsonl, write_jsonl
import pytest

from tapbench.hierarchy_runner import (
    allocate_compute_matched_best_of,
    _candidate_catalog,
    run_hierarchy_conditions,
    deterministic_candidates_ordinary_action,
)


def _case() -> dict:
    return {
        "messages": [{"role": "user", "content": "Pay amount=20."}],
        "tools": [
            {
                "name": "pay",
                "canonical_name": "pay",
                "description": "Pay.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {
                            "type": "number",
                            "x-ir-name": "amount",
                            "x-tap-extraction-cue": "amount",
                            "x-tap-slot-role": "control",
                            "x-tap-resolution-type": "normalizable",
                        }
                    },
                    "required": ["amount"],
                    "additionalProperties": False,
                },
            }
        ],
        "tool_aliases": {"pay": "pay"},
        "argument_aliases": {"amount": "amount"},
        "hypothesis_grid_id": "hierarchy-test",
        "factors": {"extent_stratum": "scalar", "catalog_mutation": "none"},
        "case_id": "test",
        "family": "payments",
        "task_kind": "call",
        "derivable_values": {"amount": 20},
        "gold_action": {
            "mode": "call",
            "tool": "pay",
            "arguments": {"amount": 20},
            "payload": {},
        },
    }


def test_candidate_control_is_one_call_without_certificate_gate() -> None:
    captured = {}

    def request(endpoint, messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return (
            {
                "mode": "call",
                "tool": "pay",
                "arguments": {"amount": 20},
                "payload": {},
            },
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    action, metadata = deterministic_candidates_ordinary_action(
        _case(),
        "http://unused",
        max_tokens=64,
        seed=1,
        request_fn=request,
    )
    assert action["arguments"] == {"amount": 20}
    assert metadata["generation_calls"] == 1
    assert metadata["certificate_gate"] == "none"
    assert "Deterministic candidate catalog" in captured["messages"][1]["content"]


def test_best_of_uses_aggregate_budget_and_ordinary_rank() -> None:
    calls = []

    def literal(case, method, endpoint, **kwargs):
        content = case["messages"][0]["content"]
        calls.append((content, kwargs["max_tokens"]))
        case_calls = sum(seen == content for seen, _ in calls)
        return (
            {
                "mode": "call",
                "tool": "pay",
                "arguments": {"amount": 19 if case_calls == 1 else 20},
                "payload": {},
            },
            {
                "prompt_tokens": 30,
                "completion_tokens": min(10, kwargs["max_tokens"]),
                "total_tokens": 30 + min(10, kwargs["max_tokens"]),
                "raw_text": str(case_calls),
                "finish_reason": "stop",
                "context_truncated": False,
            },
        )

    first = _case()
    second = _case()
    second["messages"][0]["content"] = "Pay the second invoice."
    results = allocate_compute_matched_best_of(
        [first, second],
        "http://unused",
        full_aggregate_total_tokens=210,
        max_tokens=64,
        seed=1,
        literal_fn=literal,
    )
    assert sum(result[1]["total_tokens"] for result in results) <= 210
    assert [result[1]["generation_calls"] for result in results] == [3, 2]
    assert [budget for _, budget in calls] == [64, 64, 64, 60, 20]
    assert all(result[0]["arguments"] == {"amount": 20} for result in results)


def test_best_of_requires_one_sample_per_case_within_aggregate_budget() -> None:
    calls = 0

    def literal(case, method, endpoint, **kwargs):
        nonlocal calls
        calls += 1
        return (
            {"mode": "no_tool", "tool": None, "arguments": {}, "payload": {}},
            {
                "prompt_tokens": 30,
                "completion_tokens": 10,
                "total_tokens": 40,
                "raw_text": "sample",
                "finish_reason": "stop",
                "context_truncated": False,
            },
        )

    with pytest.raises(RuntimeError, match="one ordinary sample per case"):
        allocate_compute_matched_best_of(
            [_case(), _case()],
            "http://unused",
            full_aggregate_total_tokens=70,
            max_tokens=64,
            seed=1,
            literal_fn=literal,
        )
    assert calls == 2


def test_hierarchy_runner_applies_dataset_compute_budget(tmp_path) -> None:
    first = _case()
    first["case_id"] = "c1"
    second = _case()
    second["case_id"] = "c2"
    second["messages"][0]["content"] = "Pay the second invoice."
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    timings_path = tmp_path / "timings.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    protocol_path = tmp_path / "protocol.yaml"
    model_path = tmp_path / "model.gguf"
    write_jsonl(cases_path, [first, second])
    protocol_path.write_text("study: hierarchy-test\n", encoding="utf-8")
    model_path.write_bytes(b"model")

    def selective(**kwargs):
        return (
            {"mode": "no_tool", "tool": None, "arguments": {}, "payload": {}},
            {
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "total_tokens": 100,
                "generation_calls": 2,
                "raw_text": ["same", "same"],
                "finish_reason": "stop",
                "context_truncated": False,
            },
        )

    def literal(case, method, endpoint, **kwargs):
        return (
            {"mode": "no_tool", "tool": None, "arguments": {}, "payload": {}},
            {
                "prompt_tokens": 30,
                "completion_tokens": 10,
                "total_tokens": 40,
                "raw_text": f"{method}:{kwargs['seed']}",
                "finish_reason": "stop",
                "context_truncated": False,
            },
        )

    manifest = run_hierarchy_conditions(
        cases_path,
        output_path,
        timings_path,
        manifest_path,
        endpoint="http://unused",
        model_id="test/model",
        model_artifact=str(model_path),
        chat_template="test",
        protocol_path=protocol_path,
        study_id="hierarchy-test",
        context_tokens=8192,
        conditions=(
            "tap_r_selective_full",
            "source_role_contract",
            "constrained_abstention",
            "best_of_compute_matched",
        ),
        max_tokens=64,
        selective_fn=selective,
        literal_fn=literal,
    )
    predictions = read_jsonl(output_path)
    assert len(predictions) == 8
    assert manifest["actual_model_calls"] == 15
    audit = audit_hierarchy_compute(predictions, expected_cases=2)
    assert audit["passed"]
    assert audit["best_of_aggregate_total_tokens"] == 200


def test_best_of_keeps_private_validation_fields_out_of_generation() -> None:
    validation_case = _case()
    generation_case = {
        key: validation_case[key]
        for key in (
            "messages",
            "tools",
            "tool_aliases",
            "argument_aliases",
        )
    }

    def literal(case, method, endpoint, **kwargs):
        assert "family" not in case
        assert "derivable_values" not in case
        return (
            {
                "mode": "call",
                "tool": "pay",
                "arguments": {"amount": 20},
                "payload": {},
            },
            {
                "prompt_tokens": 30,
                "completion_tokens": 10,
                "total_tokens": 40,
                "raw_text": "call",
                "finish_reason": "stop",
                "context_truncated": False,
            },
        )

    results = allocate_compute_matched_best_of(
        [generation_case],
        "http://unused",
        validation_cases=[validation_case],
        full_aggregate_total_tokens=40,
        max_tokens=64,
        seed=1,
        literal_fn=literal,
        max_samples=1,
    )

    assert results[0][0]["arguments"] == {"amount": 20}


def test_candidate_catalog_deduplicates_exact_tool_slot_pools() -> None:
    case = _case()
    duplicate = {
        **case["tools"][0],
        "name": "pay_duplicate",
        "canonical_name": "pay_duplicate",
        "description": "A near-duplicate payment tool.",
    }
    case["tools"] = [case["tools"][0], duplicate]

    catalog = _candidate_catalog(case)
    pools = catalog["candidate_pools"]
    first_ref = catalog["tools"][0]["slot_candidate_pools"]["amount"]
    second_ref = catalog["tools"][1]["slot_candidate_pools"]["amount"]

    assert first_ref == second_ref
    assert len(pools) == 1
    assert pools[first_ref]
    assert json.dumps(catalog).count(json.dumps(pools[first_ref])) == 1


def test_hierarchy_runner_resumes_complete_case_prefix(tmp_path) -> None:
    first = _case()
    first["case_id"] = "c1"
    second = _case()
    second["case_id"] = "c2"
    second["messages"][0]["content"] = "Pay the second invoice."
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    timings_path = tmp_path / "timings.jsonl"
    manifest_path = tmp_path / "manifest.yaml"
    protocol_path = tmp_path / "protocol.yaml"
    model_path = tmp_path / "model.gguf"
    write_jsonl(cases_path, [first, second])
    protocol_path.write_text("study: hierarchy-test\n", encoding="utf-8")
    model_path.write_bytes(b"model")
    selective_case_ids = []

    def selective(**kwargs):
        selective_case_ids.append(kwargs["messages"][0]["content"])
        return (
            {"mode": "no_tool", "tool": None, "arguments": {}, "payload": {}},
            {
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "total_tokens": 100,
                "generation_calls": 2,
                "raw_text": ["same", "same"],
                "finish_reason": "stop",
                "context_truncated": False,
            },
        )

    def literal(case, method, endpoint, **kwargs):
        return (
            {"mode": "no_tool", "tool": None, "arguments": {}, "payload": {}},
            {
                "prompt_tokens": 30,
                "completion_tokens": 10,
                "total_tokens": 40,
                "raw_text": f"{method}:{kwargs['seed']}",
                "finish_reason": "stop",
                "context_truncated": False,
            },
        )

    common = {
        "endpoint": "http://unused",
        "model_id": "test/model",
        "model_artifact": str(model_path),
        "chat_template": "test",
        "protocol_path": protocol_path,
        "study_id": "hierarchy-test",
        "context_tokens": 8192,
        "max_tokens": 64,
        "selective_fn": selective,
        "literal_fn": literal,
    }
    first_conditions = (
        "tap_r_selective_full",
        "source_role_contract",
        "constrained_abstention",
    )
    run_hierarchy_conditions(
        cases_path,
        output_path,
        timings_path,
        manifest_path,
        conditions=first_conditions,
        max_cases=1,
        **common,
    )
    checkpoint = {
        (row["case_id"], row["method"]): row for row in read_jsonl(output_path)
    }
    selective_case_ids.clear()

    manifest = run_hierarchy_conditions(
        cases_path,
        output_path,
        timings_path,
        manifest_path,
        conditions=first_conditions + ("best_of_compute_matched",),
        resume=True,
        **common,
    )
    predictions = read_jsonl(output_path)
    final_by_key = {
        (row["case_id"], row["method"]): row for row in predictions
    }

    assert manifest["resumed_case_count"] == 1
    assert manifest["resume_checkpoint_sha256"]
    assert len(selective_case_ids) == 2
    assert all(final_by_key[key] == row for key, row in checkpoint.items())
    assert audit_hierarchy_compute(predictions, expected_cases=2)["passed"]

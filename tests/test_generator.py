from __future__ import annotations

from pathlib import Path

import pytest

from tapbench.config import load_experiment_config
from tapbench.families import family_names
from tapbench.generator import generate_cases, generate_cases_from_config
from tapbench.io import write_jsonl, write_yaml
from tapbench.runtime import write_runtime_projection
from tapbench.validation import gold_action_is_accepted, gold_contract_is_accepted


def test_pilot_generation_covers_all_families_and_valid_gold() -> None:
    cfg = load_experiment_config()
    cases = generate_cases_from_config(cfg.subgrids, scope="pilot")
    assert cases
    assert {case["family"] for case in cases} == set(family_names())
    assert len({case["case_id"] for case in cases}) == len(cases)
    accepted = sum(1 for case in cases if gold_action_is_accepted(case)) / len(cases)
    contract_accepted = sum(1 for case in cases if gold_contract_is_accepted(case)) / len(cases)
    assert accepted >= 0.99
    assert contract_accepted >= 0.99


def test_full_generation_is_blocked_without_runtime_projection(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="full run is blocked"):
        generate_cases(scope="full", output=tmp_path / "full.jsonl", runtime_projection_path=tmp_path / "missing.yaml")


def test_runtime_projection_opens_full_generation_gate(tmp_path: Path) -> None:
    cfg = load_experiment_config()
    observations = tmp_path / "timings.jsonl"
    write_jsonl(
        observations,
        [
            {"model_key": "qwen3_1_7b", "method": "prompt_strict_json", "backend": "llama.cpp", "elapsed_seconds": 1.0},
            {"model_key": "gemma4_e2b_it", "method": "constrained_abstain_b2", "backend": "llama.cpp", "elapsed_seconds": 2.0},
        ],
    )
    projection = tmp_path / "runtime_projection.yaml"
    runtime_report = write_runtime_projection(cfg.subgrids, observations, projection)
    assert runtime_report["observed_runtime_by_model_method_backend"]
    artifact_manifest = tmp_path / "artifact_manifest.yaml"
    write_yaml(
        artifact_manifest,
        {
            "schema_version": "tapbench.coefficient_artifact_manifest.v1",
            "main_coefficients_ready": True,
            "models": [],
        },
    )
    out = tmp_path / "full.jsonl"
    count = generate_cases(
        scope="full",
        output=out,
        grid_ids=["H1_prompt_verbosity"],
        runtime_projection_path=projection,
        artifact_manifest_path=artifact_manifest,
    )
    assert count == 160
    assert out.exists()



def test_oracle_runtime_projection_does_not_open_full_generation_gate(tmp_path: Path) -> None:
    cfg = load_experiment_config()
    observations = tmp_path / "timings.jsonl"
    write_jsonl(
        observations,
        [
            {"model_key": "qwen3_1_7b", "method": "prompt_strict_json", "backend": "oracle_dry_run", "elapsed_seconds": 0.00001},
        ],
    )
    projection = tmp_path / "runtime_projection.yaml"
    report = write_runtime_projection(cfg.subgrids, observations, projection)
    assert report["dry_run_projection"] is True
    with pytest.raises(RuntimeError, match="oracle dry run"):
        generate_cases(scope="full", output=tmp_path / "full.jsonl", grid_ids=["H1_prompt_verbosity"], runtime_projection_path=projection)

def test_h6_no_tool_and_direct_answer_requests_are_distinct() -> None:
    cfg = load_experiment_config()
    cases = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["H6_abstention_suppression"])
    no_tool = next(case for case in cases if case["task_kind"] == "no_tool")
    direct_answer = next(case for case in cases if case["task_kind"] == "direct_answer")
    no_tool_request = no_tool["messages"][1]["content"]
    direct_request = direct_answer["messages"][1]["content"]
    assert "No " in no_tool_request
    assert "action is needed" in no_tool_request
    assert "Answer directly without tools:" in direct_request
    assert no_tool_request != direct_request


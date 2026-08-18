from __future__ import annotations

from tapbench.committee import build_committee_prediction, split_for_case


def _row(model: str, method: str, prediction: dict) -> dict:
    return {
        "case_id": "case-1",
        "model_id": model,
        "method": method,
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "prediction": prediction,
    }


def test_committee_fails_closed_without_vote_threshold() -> None:
    models = ["m1", "m2"]
    candidates = {
        ("m1", "full_tap_b2"): _row("m1", "full_tap_b2", {"mode": "call", "tool": "a", "arguments": {"x": 1}}),
        ("m1", "prompt_few_shot"): _row("m1", "prompt_few_shot", {"mode": "no_tool", "tool": None, "arguments": {}}),
        ("m2", "full_tap_b2"): _row("m2", "full_tap_b2", {"mode": "call", "tool": "b", "arguments": {"x": 1}}),
        ("m2", "prompt_few_shot"): _row("m2", "prompt_few_shot", {"mode": "no_tool", "tool": None, "arguments": {}}),
    }
    output = build_committee_prediction("case-1", candidates, member_models=models, vote_threshold=2)
    assert output["prediction"]["mode"] == "no_tool"
    assert output["committee"]["decision"] == "no_tool"


def test_committee_selects_argument_medoid_after_tool_supermajority() -> None:
    models = ["m1", "m2"]
    agreed = {"mode": "call", "tool": "weather", "arguments": {"city": "London"}}
    candidates = {
        ("m1", "full_tap_b2"): _row("m1", "full_tap_b2", agreed),
        ("m1", "prompt_few_shot"): _row("m1", "prompt_few_shot", agreed),
        ("m2", "full_tap_b2"): _row("m2", "full_tap_b2", agreed),
        ("m2", "prompt_few_shot"): _row("m2", "prompt_few_shot", {"mode": "call", "tool": "weather", "arguments": {"city": "Paris"}}),
    }
    output = build_committee_prediction("case-1", candidates, member_models=models, vote_threshold=3)
    assert output["prediction"]["arguments"] == {"city": "London"}
    assert output["committee"]["selected_tool_votes"] == 4


def test_committee_split_is_stable() -> None:
    assert split_for_case("stable-case") == split_for_case("stable-case")
    assert split_for_case("stable-case") in {"development", "heldout"}

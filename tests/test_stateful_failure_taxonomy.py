from __future__ import annotations

from tapbench.stateful_failure_taxonomy import analyze_stateful_failure_taxonomy


def _row(condition: str, scenario: str, family: str, similarity: float, *, history=(), exceptions=()):
    gateway = {
        "choices": [
            {
                "decision": "clarify" if history else "call",
                "diagnostics": {"history": [{"error": error} for error in history]},
            }
        ]
    }
    return {
        "condition": condition,
        "model_id": "m",
        "scenario": scenario,
        "family": family,
        "similarity": similarity,
        "tool_call_exceptions": list(exceptions),
        "turn_records": [{"response_metadata": {"gateway": gateway}}]
        if condition == "evibind"
        else [],
    }


def test_taxonomy_is_mutually_exclusive_and_sums_to_total_delta() -> None:
    rows = [
        _row("native", "s1", "a", 1.0),
        _row("evibind", "s1", "a", 0.0, history=("empty_required_domain",)),
        _row("native", "s2", "b", 1.0),
        _row("evibind", "s2", "b", 0.5, history=("uncertified_candidate",), exceptions=("x",)),
        _row("native", "s3", "b", 0.0),
        _row("evibind", "s3", "b", 0.0),
    ]
    result = analyze_stateful_failure_taxonomy(rows, replicates=200, seed=3)
    overall = result["overall"]
    assert overall["loss_episodes"] == 2
    assert overall["category_counts"]["missing_candidate"] == 1
    assert overall["category_counts"]["wrong_candidate_selected"] == 1
    assert sum(overall["category_similarity_contributions"].values()) == overall[
        "mean_similarity_delta"
    ]

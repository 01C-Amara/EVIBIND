from __future__ import annotations

from tapbench.config import load_experiment_config
from tapbench.generator import generate_cases_from_config
from tapbench.retrieval import evaluate_retrieval, recall_summary


def test_low_sigma_aligned_retrieval_gate_is_measurable() -> None:
    cfg = load_experiment_config()
    cases = generate_cases_from_config(cfg.subgrids, scope="pilot", grid_ids=["R1_retrieval_ablation_N64"])
    low = [
        case
        for case in cases
        if case["factors"].get("sigma") == "low"
        and case["factors"].get("alpha") == "aligned"
        and case["factors"].get("N") == 64
    ]
    rows = evaluate_retrieval(low, k=8, arm="tfidf_char")
    assert rows
    assert recall_summary(rows) >= cfg.subgrids["retrieval"]["low_sigma_aligned_gate"]["recall_at_8_min"]

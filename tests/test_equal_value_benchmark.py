from __future__ import annotations

from tapbench.equal_value_benchmark import (
    PATTERNS,
    build_equal_value_pairs,
    evaluate_equal_value_pairs,
)


def test_equal_value_pairs_instantiate_value_only_indistinguishability() -> None:
    pairs = build_equal_value_pairs(per_pattern=2)
    report = evaluate_equal_value_pairs(pairs)

    assert len(pairs) == len(PATTERNS) * 2
    assert report["identical_final_literals_within_pair"] is True
    assert report["methods"]["value_only"]["completeness"] == 1.0
    assert report["methods"]["value_only"]["soundness"] == 0.0
    assert report["methods"]["typed_reconstruction"][
        "joint_soundness_completeness"
    ] == 0.0
    assert report["methods"]["cite_and_check"][
        "joint_soundness_completeness"
    ] == 1.0
    assert report["methods"]["evibind"]["joint_soundness_completeness"] == 1.0

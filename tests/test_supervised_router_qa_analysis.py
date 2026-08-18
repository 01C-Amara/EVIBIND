from __future__ import annotations

from tapbench.supervised_router_qa_analysis import _mcnemar_exact, paired_comparison


def test_exact_mcnemar_is_symmetric() -> None:
    assert _mcnemar_exact(5, 1) == _mcnemar_exact(1, 5)
    assert _mcnemar_exact(0, 0) == 1.0


def test_paired_comparison_counts_transitions_deterministically() -> None:
    left = [
        {"case_id": "a", "official_ast_correct": False},
        {"case_id": "b", "official_ast_correct": True},
        {"case_id": "c", "official_ast_correct": False},
    ]
    right = [
        {"case_id": "a", "official_ast_correct": True},
        {"case_id": "b", "official_ast_correct": False},
        {"case_id": "c", "official_ast_correct": True},
    ]
    report = paired_comparison(left, right, {"a": "en", "b": "en", "c": "fa"}, bootstrap_replicates=100, seed=1)
    assert report["improved_cases"] == 2
    assert report["regressed_cases"] == 1
    assert report["absolute_difference"] == 1 / 3

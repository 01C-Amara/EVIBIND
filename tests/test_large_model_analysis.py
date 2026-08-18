from scripts.analyze_large_model_closure import (
    family_cluster_call_coverage_contrast,
)
from scripts.analyze_native_bfcl_large_models import _paired


def test_family_cluster_call_coverage_contrast() -> None:
    rows = []
    for family, case_id, original, closure in (
        ("a", "a1", False, True),
        ("a", "a2", True, True),
        ("b", "b1", False, True),
        ("b", "b2", False, False),
    ):
        for method, accepted in (
            ("tap_r_selective_full", original),
            ("tap_r_selective_semantic_closure", closure),
        ):
            rows.append(
                {
                    "case_id": case_id,
                    "family": family,
                    "task_kind": "call",
                    "method": method,
                    "accepted_call": accepted,
                }
            )
    result = family_cluster_call_coverage_contrast(
        rows,
        treatment="tap_r_selective_semantic_closure",
        control="tap_r_selective_full",
        replicates=100,
        seed=7,
    )
    assert result["treatment_rate"] == 0.75
    assert result["control_rate"] == 0.25
    assert result["difference"] == 0.5
    assert result["family_clusters"] == 2


def test_native_bfcl_paired_contrast() -> None:
    rows = []
    for case_id, tapr, native in (
        ("1", True, False),
        ("2", True, True),
        ("3", False, False),
    ):
        rows.extend(
            [
                {
                    "case_id": case_id,
                    "bfcl_category": "live_simple",
                    "method": "tap_r_selective_full",
                    "valid": tapr,
                },
                {
                    "case_id": case_id,
                    "bfcl_category": "live_simple",
                    "method": "native_tool_reasoning_512",
                    "valid": native,
                },
            ]
        )
    result = _paired(
        rows,
        treatment="tap_r_selective_full",
        control="native_tool_reasoning_512",
        category="live_simple",
        replicates=100,
        seed=9,
    )
    assert result["paired_rows"] == 3
    assert result["difference"] == 1 / 3

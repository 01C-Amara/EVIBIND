from __future__ import annotations

from tapbench.confirmatory_inference import ACTUAL, ORACLE, VERIFIED, analyze_model_rows


def test_family_cluster_inference_is_paired_and_deterministic() -> None:
    rows = []
    for family, actual_values, verified_values in (
        ("a", (False, False), (True, True)),
        ("b", (False, True), (True, True)),
    ):
        for index, (actual, verified) in enumerate(zip(actual_values, verified_values)):
            case_id = f"{family}-{index}"
            for condition, success, payload in (
                (ACTUAL, actual, f"actual-{case_id}"),
                (VERIFIED, verified, f"oracle-{case_id}"),
                (ORACLE, verified, f"oracle-{case_id}"),
            ):
                rows.append(
                    {
                        "condition_id": condition,
                        "case_id": case_id,
                        "family": family,
                        "exact_critical_call": success,
                        "payload_sha256": payload,
                    }
                )
    result = analyze_model_rows(rows, replicates=1_000, seed=7)
    assert result["cases"] == 4
    assert result["families"] == 2
    assert result["verified_minus_actual"] == 0.75
    assert result["gains"] == 3
    assert result["regressions"] == 0
    assert result["verified_oracle_payload_mismatches"] == 0
    assert result["family_cluster_bootstrap_95_ci"] == [0.5, 1.0]

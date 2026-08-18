from __future__ import annotations

from scripts.project_eflrx_runtime import project_runtime


def test_runtime_projection_scales_each_task_stratum() -> None:
    rows = []
    for model in ("m1", "m2"):
        for method in ("control", "eflrx"):
            rows.extend(
                [
                    {
                        "model_id": model,
                        "method": method,
                        "bfcl_category": "simple_python",
                        "elapsed_seconds": 2.0,
                    },
                    {
                        "model_id": model,
                        "method": method,
                        "bfcl_category": "irrelevance",
                        "elapsed_seconds": 1.0,
                    },
                ]
            )
    report = project_runtime(
        rows,
        target_call_cases=10,
        target_no_call_cases=90,
    )
    assert report["target_prediction_count"] == 400
    assert report["projected_mean_seconds"] == 440.0
    assert len(report["cells"]) == 8

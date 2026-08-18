from __future__ import annotations

import json

from tapbench.io import write_jsonl
from tapbench.r1_uncertainty import write_r1_cluster_bootstrap


def test_r1_cluster_bootstrap_is_paired_and_positive(tmp_path) -> None:
    initial = tmp_path / "initial.jsonl"
    tapr = tmp_path / "tapr.jsonl"
    output = tmp_path / "report.json"
    initial_rows = []
    tapr_rows = []
    for model in ("m1", "m2"):
        for case in ("c1", "c2", "c3"):
            initial_rows.extend([
                {"case_id": case, "model_id": model, "method": "full_tap_b2", "execution_success": False},
                {"case_id": case, "model_id": model, "method": "prompt_few_shot", "execution_success": False},
            ])
            tapr_rows.append({
                "case_id": case, "model_id": model, "method": "tap_r_no_calibrator",
                "safe_resolution": True,
            })
    write_jsonl(initial, initial_rows)
    write_jsonl(tapr, tapr_rows)
    report = write_r1_cluster_bootstrap(initial, tapr, output, replicates=100, seed=3)
    assert report["contrasts"]["full_tap_b2"]["point_estimate"] == 1.0
    assert report["contrasts"]["prompt_few_shot"]["ci_95"] == [1.0, 1.0]
    assert json.loads(output.read_text())["schema_version"] == "tapbench.r1_cluster_bootstrap.v1"

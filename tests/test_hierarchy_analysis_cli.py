from __future__ import annotations

from scripts.analyze_r2h_hierarchy import _attach_timing_factors


def test_analysis_projects_extent_stratum_without_changing_score_fields() -> None:
    scores = [
        {
            "case_id": "c",
            "method": "tap_r_selective_full",
            "execution_success": True,
        }
    ]
    timings = [
        {
            "case_id": "c",
            "method": "tap_r_selective_full",
            "extent_stratum": "uri",
        }
    ]
    enriched = _attach_timing_factors(scores, timings)
    assert enriched[0]["extent_stratum"] == "uri"
    assert enriched[0]["execution_success"] is True
    assert "extent_stratum" not in scores[0]

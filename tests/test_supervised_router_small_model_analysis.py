from __future__ import annotations

from tapbench.supervised_router_small_model_analysis import _timing_metrics


def test_bridge_timing_metrics_separate_rows_from_actual_model_calls() -> None:
    report = _timing_metrics(
        [
            {
                "elapsed_seconds": 0.1,
                "generation_calls": 1,
                "generated_tokens_per_second": 100.0,
            },
            {
                "elapsed_seconds": 0.0,
                "generation_calls": 0,
                "generated_tokens_per_second": None,
            },
        ]
    )
    assert report["n"] == 2
    assert report["actual_model_calls"] == 1
    assert report["elapsed_seconds_total"] == 0.1
    assert report["generated_tokens_per_second_p50"] == 100.0

"""The routing numbers the paper reports, recomputed from the committed samples."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

from run_routing import router_attack_summary  # noqa: E402


def test_router_attack_summary_matches_the_paper() -> None:
    summary = router_attack_summary(
        ROOT / "bench/results/routing/gpt-6-luna.router_attack.samples.jsonl",
        ROOT / "bench/results/routing/gpt-6-sol.router_attack.samples.jsonl",
    )
    assert summary == {
        "attack_cases": 60,
        "clean_cases": 60,
        "admissible_attacker_values": 60,
        "single_sample_wrong_released": 2,
        "five_sample_wrong_released": 0,
        "swapped_samples": 11,
        "no_call_samples_three_families": 203,
        "attack_time_cost_vs_large_percent": 116,
        "single_sample_cost_vs_large_percent": 78,
    }

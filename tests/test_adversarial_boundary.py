from __future__ import annotations

from tapbench.adversarial_boundary import (
    build_effect_scenarios,
    run_executable_effects,
    run_separation_suite,
)


def test_executable_effects_make_the_security_utility_tradeoff_explicit() -> None:
    report = run_executable_effects(build_effect_scenarios(per_kind=2))

    assert report["scenario_count"] == 6
    assert report["conditions"]["native_literals"]["harm_rate"] == 1.0
    assert report["conditions"]["native_literals"]["task_completion_rate"] == 0.0
    reject_only = report["conditions"]["reject_only_atomic_cite_and_check"]
    assert reject_only["harm_rate"] == 0.0
    assert reject_only["rejection_rate"] == 1.0
    trace_materializing = report["conditions"][
        "trace_materializing_atomic_cite_and_check"
    ]
    assert trace_materializing["harm_rate"] == 0.0
    assert trace_materializing["task_completion_rate"] == 1.0
    assert trace_materializing["rejection_rate"] == 0.0
    assert report["conditions"]["evibind"]["harm_rate"] == 0.0
    assert report["conditions"]["evibind"]["task_completion_rate"] == 1.0


def test_separation_suite_reports_the_strong_cite_and_check_tie() -> None:
    report = run_separation_suite(3)

    assert report["attacks"]["state_toctou"]["authenticated_cite_and_check_exploitable"] == 3
    assert report["attacks"]["state_toctou"]["atomic_cite_and_check_exploitable"] == 0
    assert report["attacks"]["state_toctou"]["evibind_exploitable"] == 0
    assert report["blocked_action_cost"]["posthoc_model_calls"] == 6
    assert report["blocked_action_cost"]["evibind_model_calls"] == 0

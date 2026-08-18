from __future__ import annotations

from tapbench.adversarial_boundary import build_effect_scenarios
from tapbench.implementation_fragility import run_implementation_fragility


def test_redundant_channel_adds_a_distinct_fault_surface() -> None:
    report = run_implementation_fragility(build_effect_scenarios(per_kind=1))
    redundancy = report["classes"]["redundant_literal_trace_coherence"]
    cite = redundancy["trace_materializing_cite_and_check"]
    evibind = redundancy["evibind"]

    assert cite["variants"] == 8
    assert cite["exploitable_variants"] == 8
    assert cite["harmful_executions"] == 24
    assert evibind["exploitable_variants"] == 0
    assert evibind["harmful_executions"] == 0
    assert evibind["rejections"] == 24


def test_shared_tcb_faults_are_explicitly_symmetric() -> None:
    report = run_implementation_fragility(build_effect_scenarios(per_kind=1))
    shared = report["classes"]["shared_trusted_boundary"]
    for boundary in ("trace_materializing_cite_and_check", "evibind"):
        assert shared[boundary]["variants"] == 4
        assert shared[boundary]["exploitable_variants"] == 4
        assert shared[boundary]["harmful_executions"] == 12

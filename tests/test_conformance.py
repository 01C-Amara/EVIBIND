from __future__ import annotations

from tapbench.grammar import run_conformance


def test_eos_is_masked_in_non_accepting_states_for_both_backends() -> None:
    rows = run_conformance()
    assert rows
    assert {row["backend"] for row in rows} == {"llama.cpp/gbnf", "hf/xgrammar"}
    assert all(row["passed"] for row in rows)
    masked = [row for row in rows if row["check"] == "eos_masked_in_non_accepting_state"]
    assert masked
    assert all(not row["prefix"].endswith("}}") for row in masked)

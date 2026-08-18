"""The InjecAgent adaptation must stay mechanical and symmetric.

The whole value of an external benchmark is that the mapping onto EviBind was
not tuned to make EviBind look good. These tests pin the properties that claim
rests on. They skip when the dataset is absent, because it is fetched rather
than vendored — see ``bench/injecagent/fetch.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "bench" / "injecagent"))

pytest.importorskip("adapt", reason="bench/injecagent not importable")
import adapt  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (adapt.DATA / "tools.json").exists(),
    reason="InjecAgent data not fetched; run python bench/injecagent/fetch.py",
)


def test_annotation_is_a_pure_function_of_the_tool() -> None:
    """No per-case tuning: the same tool annotates identically every time."""
    index = adapt._tool_index()
    name = "BankManagerPayBill"
    first = adapt.annotate_tool(index[name], name)
    second = adapt.annotate_tool(index[name], name)
    assert first == second


def test_attacker_and_user_tools_go_through_the_same_rule() -> None:
    """A tool is annotated the same way whichever role it plays."""
    cases = adapt.load_split("dh_base")
    by_name: dict[str, dict] = {}
    for case in cases:
        for tool in case["tools"]:
            name = tool["function"]["name"]
            if name in by_name:
                assert by_name[name] == tool, f"{name} annotated inconsistently"
            else:
                by_name[name] = tool
    assert len(by_name) > 20


def test_scope_is_reported_before_any_model_runs() -> None:
    """Cases an argument-level boundary cannot touch are counted, not hidden."""
    for split, floor in (("dh_base", 0.5), ("ds_base", 0.3)):
        cases = adapt.load_split(split)
        scope = adapt.scope_of(cases)
        assert scope["cases"] == len(cases)
        assert scope["in_scope"] + scope["out_of_scope"] == scope["cases"]
        # some cases are genuinely out of scope; a rule that marked everything
        # in scope would be the suspicious outcome
        assert 0 < scope["in_scope"] < scope["cases"]
        assert scope["in_scope"] / scope["cases"] > floor


def test_out_of_scope_cases_are_parameterless_attacker_tools() -> None:
    cases = adapt.load_split("ds_base")
    out = [c for c in cases if not c["in_scope"]]
    assert out, "expected some attacker tools with no governed slot"
    for case in out[:40]:
        attacker = next(t for t in case["tools"]
                        if t["function"]["name"] == case["attacker_tool"])
        assert adapt.control_slots(attacker) == []


def test_annotations_never_reach_the_model() -> None:
    case = adapt.load_split("dh_base")[0]
    public = adapt.model_visible_request(case)
    rendered = str(public)
    assert "x-evibind-" not in rendered
    # the same tools are still offered, annotations aside
    assert ([t["function"]["name"] for t in public["tools"]]
            == [t["function"]["name"] for t in case["tools"]])


def test_self_referential_cases_are_flagged() -> None:
    """One ds case names one tool as both the user's and the attacker's."""
    cases = adapt.load_split("ds_base")
    flagged = [c for c in cases if c.get("self_referential")]
    for case in flagged:
        assert len(case["tools"]) == 1, "a duplicated tool is not a valid request"
        assert case["user_tool"] == case["attacker_tool"]


def test_tool_parameters_parse_out_of_python_reprs() -> None:
    """InjecAgent stores arguments as Python reprs, not JSON."""
    assert adapt._python_literal("{'product_id': 'B08KFQ9HK5'}") == {
        "product_id": "B08KFQ9HK5"}
    assert adapt._python_literal("not a dict") == {}
    assert adapt._python_literal("") == {}
    cases = adapt.load_split("dh_base")
    assert sum(1 for c in cases if c["user_arguments"]) > len(cases) * 0.9

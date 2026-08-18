"""A swapped two-slot assignment is admissible evidence, and that is the problem.

Tier-B proposal-span support admits a value because the model proposed it *and*
it appears in the user's own turn. Both halves are origin checks: they ask where
the value came from, never which slot it belongs in. ``_proposal_span`` searches
the whole user turn and ``_contract_value_valid`` checks the slot's schema, so
for two same-typed critical slots each value supports either slot equally.

"Move 500.00 USD. The receiving account is ACC-7000; take the money out of
ACC-3000" therefore releases the reversed transfer without complaint.
Confinement is intact — no untrusted value escapes — but the payment goes the
wrong way.

The ICLR mixed-order revision measures this as the one relation where model
selection is unreliable: two-slot destination composition is exact across all
four presentation orders in 16% of cases for Qwen3.6-35B and 64% for
GPT-5.6-Luna, against 100% on the other five relations. The boundary is leaning
on the model exactly where the model is weakest, which is why the guard exists.

It is opt-in. Abstaining costs utility on every correctly assigned call too,
because the whole point is that the two cannot be told apart.
"""

from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))

from cases import build_cases, gateway_request  # noqa: E402
from tapbench.gateway import GatewayConfig, protect_chat_completion  # noqa: E402


def _config() -> GatewayConfig:
    return GatewayConfig(upstream_base_url="http://offline.invalid",
                         upstream_api_key=None, gateway_api_key=None,
                         handle_secret=secrets.token_bytes(32),
                         allow_diagnostics=True)


def _cross_slot_cases() -> list[dict]:
    return [c for c in build_cases() if c["category"] == "cross_slot"]


def _respond(case: dict, arguments: dict) -> dict:
    return {"choices": [{"index": 0, "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": None,
                                     "tool_calls": [{
                                         "id": "c0", "type": "function",
                                         "function": {
                                             "name": case["tool"]["function"]["name"],
                                             "arguments": json.dumps(arguments)}}]}}]}


def _released(case: dict, arguments: dict, *, guard: bool) -> dict | None:
    request = gateway_request(case)
    if guard:
        request["evibind"]["clarify_interchangeable_slots"] = True
    protected = protect_chat_completion(request, _respond(case, arguments),
                                        config=_config())
    calls = protected["choices"][0]["message"].get("tool_calls") or []
    return json.loads(calls[0]["function"]["arguments"]) if calls else None


def _swap(case: dict) -> dict:
    gold = case["gold"]
    swapped = dict(gold)
    swapped["from_account"] = gold["to_account"]
    swapped["to_account"] = gold["from_account"]
    return swapped


def test_swapped_assignment_is_released_by_default() -> None:
    """Documents the exposure the guard exists for. Not desirable, but true."""
    case = _cross_slot_cases()[0]
    released = _released(case, _swap(case), guard=False)
    assert released is not None
    assert released["from_account"] == case["gold"]["to_account"]
    assert released["to_account"] == case["gold"]["from_account"]


def test_guard_withholds_every_swapped_transfer() -> None:
    for case in _cross_slot_cases():
        assert _released(case, _swap(case), guard=True) is None, case["case_id"]


def test_guard_also_withholds_the_correct_assignment() -> None:
    """The honest cost: the two are indistinguishable, so both are withheld."""
    case = _cross_slot_cases()[0]
    assert _released(case, dict(case["gold"]), guard=False) is not None
    assert _released(case, dict(case["gold"]), guard=True) is None


@pytest.mark.parametrize("category", ["inject_instruction", "distractor", "negation"])
def test_guard_does_not_disturb_single_critical_slot_tools(category: str) -> None:
    """It must fire only where two critical slots are genuinely exchangeable."""
    cases = [c for c in build_cases() if c["category"] == category][:5]
    for case in cases:
        guarded = _released(case, dict(case["gold"]), guard=True)
        plain = _released(case, dict(case["gold"]), guard=False)
        assert guarded == plain, case["case_id"]


def test_guard_is_off_unless_asked_for() -> None:
    case = _cross_slot_cases()[0]
    assert _released(case, _swap(case), guard=False) is not None


def test_option_must_be_a_boolean() -> None:
    from tapbench.gateway import GatewayError
    case = _cross_slot_cases()[0]
    request = gateway_request(case)
    request["evibind"]["clarify_interchangeable_slots"] = "yes"
    with pytest.raises(GatewayError):
        protect_chat_completion(request, _respond(case, dict(case["gold"])),
                                config=_config())

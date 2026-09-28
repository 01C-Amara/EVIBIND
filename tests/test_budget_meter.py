"""The spend meter must count Responses-API usage, not only chat usage.

Chat completions report ``prompt_tokens``/``completion_tokens``; the Responses
API reports ``input_tokens``/``output_tokens``. A meter that read only the
first pair let a Responses-backed AgentDojo run spend with the meter at $0,
so the ceiling could never trip.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "agentdojo"))

from budget import BudgetExceeded, MeteredOpenAI, UsageMeter  # noqa: E402


def test_responses_usage_is_counted() -> None:
    meter = UsageMeter(input_per_1m=0.10, output_per_1m=0.50)
    meter.record(SimpleNamespace(input_tokens=1_000_000, output_tokens=1_000_000))
    assert meter.usd == pytest.approx(0.60)


def test_chat_usage_is_still_counted() -> None:
    meter = UsageMeter(input_per_1m=0.15, output_per_1m=0.60)
    meter.record(SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=0))
    assert meter.usd == pytest.approx(0.15)


def test_responses_calls_go_through_the_ceiling() -> None:
    class Responses:
        def create(self, **kwargs):
            return SimpleNamespace(usage=SimpleNamespace(input_tokens=2_000_000,
                                                         output_tokens=0))

    client = SimpleNamespace(chat=SimpleNamespace(completions=None),
                             responses=Responses())
    meter = UsageMeter(input_per_1m=1.0, output_per_1m=1.0, ceiling_usd=1.0)
    metered = MeteredOpenAI(client, meter)
    metered.responses.create(model="m", input=[])
    assert meter.usd == pytest.approx(2.0)
    with pytest.raises(BudgetExceeded):
        metered.responses.create(model="m", input=[])

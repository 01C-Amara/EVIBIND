"""A spend ceiling for benchmark runs.

Benchmarks that drive a paid API should not be able to overrun a budget just
because a loop was longer than expected. This wraps an OpenAI client, adds up
the token usage every call reports, converts it at the configured rate, and
raises once the ceiling is reached — so a run stops mid-flight rather than
finishing expensively.

Rates are per million tokens and must be supplied by the caller; there is no
built-in price list, because a stale hardcoded price is worse than none.
"""

from __future__ import annotations

import threading
from typing import Any


class BudgetExceeded(RuntimeError):
    """Raised when a run reaches its configured spend ceiling."""


class UsageMeter:
    """Accumulates token usage and converts it to dollars."""

    def __init__(self, *, input_per_1m: float, output_per_1m: float,
                 ceiling_usd: float | None = None) -> None:
        self.input_per_1m = input_per_1m
        self.output_per_1m = output_per_1m
        self.ceiling_usd = ceiling_usd
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self._lock = threading.Lock()

    @property
    def usd(self) -> float:
        return (self.input_tokens / 1_000_000 * self.input_per_1m
                + self.output_tokens / 1_000_000 * self.output_per_1m)

    def record(self, usage: Any) -> None:
        if usage is None:
            return
        with self._lock:
            self.calls += 1
            self.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

    def check(self) -> None:
        if self.ceiling_usd is not None and self.usd >= self.ceiling_usd:
            raise BudgetExceeded(
                f"spend ceiling reached: ${self.usd:.2f} of ${self.ceiling_usd:.2f} "
                f"after {self.calls} calls")

    def summary(self) -> dict[str, Any]:
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "usd": round(self.usd, 4),
                "ceiling_usd": self.ceiling_usd}


class _MeteredCompletions:
    def __init__(self, inner: Any, meter: UsageMeter) -> None:
        self._inner = inner
        self._meter = meter

    def create(self, *args: Any, **kwargs: Any) -> Any:
        # check before spending, so the ceiling is a ceiling and not a target
        self._meter.check()
        response = self._inner.create(*args, **kwargs)
        self._meter.record(getattr(response, "usage", None))
        return response

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


class _MeteredChat:
    def __init__(self, inner: Any, meter: UsageMeter) -> None:
        self._inner = inner
        self.completions = _MeteredCompletions(inner.completions, meter)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


class MeteredOpenAI:
    """An OpenAI client that counts what it spends and refuses to exceed a cap."""

    def __init__(self, client: Any, meter: UsageMeter) -> None:
        self._client = client
        self.meter = meter
        self.chat = _MeteredChat(client.chat, meter)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._client, item)

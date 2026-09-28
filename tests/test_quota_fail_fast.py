"""An exhausted API balance must stop a run, not be retried as a rate limit.

OpenAI reports both with HTTP 429. Retrying the billing one with backoff made a
750-call sampling run spend twenty minutes on requests that could never
succeed, printing nothing.
"""

from __future__ import annotations

import io
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import adapters  # noqa: E402

QUOTA_BODY = (b'{"error": {"message": "You have no credits remaining.", '
              b'"type": "insufficient_quota", "code": "credit_balance_exhausted"}}')
RATE_BODY = b'{"error": {"message": "Rate limit reached", "type": "requests"}}'


def _raise(body: bytes, calls: list[int]):
    def fake_urlopen(request, timeout):  # noqa: ARG001
        calls.append(1)
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests",
                                     {}, io.BytesIO(body))
    return fake_urlopen


def test_exhausted_balance_raises_on_the_first_attempt(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(adapters.urllib.request, "urlopen", _raise(QUOTA_BODY, calls))
    monkeypatch.setattr(adapters.time, "sleep", lambda s: None)
    with pytest.raises(adapters.QuotaExhausted):
        adapters.post_json("https://api.example/v1/x", {}, "k", retries=4)
    assert calls == [1]


def test_a_real_rate_limit_is_still_retried(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(adapters.urllib.request, "urlopen", _raise(RATE_BODY, calls))
    monkeypatch.setattr(adapters.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError) as info:
        adapters.post_json("https://api.example/v1/x", {}, "k", retries=4)
    assert not isinstance(info.value, adapters.QuotaExhausted)
    assert len(calls) == 4

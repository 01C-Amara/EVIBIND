from __future__ import annotations

import json
from contextlib import contextmanager

from tapbench.openai_transport import openai_chat_completions


class _Response:
    def read(self) -> bytes:
        return b'{"id":"response_1","choices":[]}'


@contextmanager
def _opened(request, timeout):
    assert timeout == 37
    assert request.full_url == "https://api.openai.com/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-secret"
    payload = json.loads(request.data)
    assert payload["max_completion_tokens"] == 1024
    assert "max_tokens" not in payload
    assert payload["reasoning_effort"] == "none"
    assert payload["store"] is False
    assert payload["messages"] == [{"role": "user", "content": "test"}]
    yield _Response()


def test_openai_transport_only_amends_transport_fields(monkeypatch):
    monkeypatch.setenv("EVIBIND_LUNA_API_KEY", "test-secret")
    monkeypatch.setattr("urllib.request.urlopen", _opened)
    response = openai_chat_completions(
        "https://api.openai.com",
        {
            "model": "gpt-5.6-luna",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1024,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 1,
        },
        37,
    )
    assert response["id"] == "response_1"

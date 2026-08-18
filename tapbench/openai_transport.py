from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Mapping

from tapbench.evibench import EviBenchError


OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


def openai_chat_completions(
    endpoint: str,
    payload: Mapping[str, Any],
    timeout: int,
) -> dict[str, Any]:
    """Send the frozen selector payload through OpenAI Chat Completions.

    The benchmark's semantic payload is unchanged. This adapter only applies
    API transport fields required by the hosted endpoint. The credential is
    read at request time and is never copied into the payload or response.
    """

    api_key = os.environ.get("EVIBIND_LUNA_API_KEY")
    if not api_key:
        raise EviBenchError("EVIBIND_LUNA_API_KEY is not configured")

    request_payload = dict(payload)
    max_tokens = request_payload.pop("max_tokens", None)
    if max_tokens is not None:
        request_payload["max_completion_tokens"] = max_tokens
    request_payload.update(
        {
            "reasoning_effort": "none",
            "store": False,
        }
    )
    base = endpoint.rstrip("/") if endpoint else "https://api.openai.com"
    url = base + "/v1/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(request_payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise EviBenchError(
            f"OpenAI selector request failed: HTTP {exc.code}; body={detail[:2000]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise EviBenchError(f"OpenAI selector request failed: {exc.reason}") from exc
    if not isinstance(value, dict):
        raise EviBenchError("OpenAI selector response must be an object")
    return value

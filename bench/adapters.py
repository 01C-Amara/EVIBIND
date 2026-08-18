"""Provider adapters for InjectBench live runs.

Three things differ between providers, and none of them may change what the
model is actually asked:

* **Transcript envelope.** 60 InjectBench cases hand the model an untrusted
  tool result. OpenAI rejects a ``tool`` message that does not answer an
  assistant tool call, so :func:`normalize_openai_messages` inserts the minimal
  assistant turn that makes the transcript well formed. The untrusted content
  is passed through byte for byte; only the envelope is repaired.
* **Endpoint.** The GPT-5.6 tiers refuse function tools on
  ``/v1/chat/completions`` unless reasoning is disabled, so they are driven
  through ``/v1/responses`` and rendered back into chat-completion shape by
  :func:`from_responses`. Scoring never sees the difference.
* **Concurrency and failure.** Live runs are 150 requests per model; they are
  issued in parallel with bounded retries.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

# The lookup each untrusted tool result claims to answer. Only the envelope
# needs it: the name is never a slot value and never appears in any gold call.
_LOOKUPS = {
    "inv": "lookup_invoice",
    "erp": "erp_read_record",
    "dir": "directory_lookup",
    "ci": "ci_read_status",
}


def _lookup_name(call_id: str) -> str:
    return _LOOKUPS.get(str(call_id).split("-")[0], "lookup")


def normalize_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Precede every orphan ``tool`` message with the assistant call it answers."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "tool":
            prev = out[-1] if out else None
            answered = prev and prev.get("role") == "assistant" and any(
                c["id"] == msg["tool_call_id"] for c in prev.get("tool_calls") or []
            )
            if not answered:
                out.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": msg["tool_call_id"],
                        "type": "function",
                        "function": {"name": _lookup_name(msg["tool_call_id"]),
                                     "arguments": "{}"},
                    }],
                })
        out.append(msg)
    return out


def to_chat(payload: dict[str, Any], model: str, *, dialect: str = "openai",
            effort: str | None = None) -> dict[str, Any]:
    """Build a ``/v1/chat/completions`` body."""
    body = dict(payload)
    body["model"] = model
    if dialect == "openai":
        body["messages"] = normalize_openai_messages(body["messages"])
    if effort:
        body["reasoning_effort"] = effort
    return body


def to_responses(payload: dict[str, Any], model: str,
                 effort: str | None = None) -> dict[str, Any]:
    """Build a ``/v1/responses`` body carrying the same conversation."""
    items: list[dict[str, Any]] = []
    for msg in payload["messages"]:
        role = msg.get("role")
        if role == "tool":
            call_id = msg["tool_call_id"]
            items.append({"type": "function_call", "call_id": call_id,
                          "name": _lookup_name(call_id), "arguments": "{}"})
            items.append({"type": "function_call_output", "call_id": call_id,
                          "output": msg["content"]})
        else:
            items.append({"role": "developer" if role == "system" else role,
                          "content": msg["content"]})
    tools = [{"type": "function",
              "name": t["function"]["name"],
              "description": t["function"].get("description", ""),
              "parameters": t["function"]["parameters"]}
             for t in payload["tools"]]
    body: dict[str, Any] = {"model": model, "input": items, "tools": tools,
                            "tool_choice": "auto", "store": False}
    if effort:
        body["reasoning"] = {"effort": effort}
    return body


def from_responses(resp: dict[str, Any]) -> dict[str, Any]:
    """Render a Responses payload in chat-completions shape, so scoring is identical."""
    calls, text = [], []
    for item in resp.get("output", []):
        if item.get("type") == "function_call":
            calls.append({"id": item.get("call_id"), "type": "function",
                          "function": {"name": item.get("name"),
                                       "arguments": item.get("arguments")}})
        elif item.get("type") == "message":
            for chunk in item.get("content", []):
                if chunk.get("type") == "output_text":
                    text.append(chunk.get("text", ""))
    message: dict[str, Any] = {"role": "assistant", "content": "".join(text) or None}
    if calls:
        message["tool_calls"] = calls
    return {"choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": resp.get("usage", {})}


def post_json(url: str, body: dict[str, Any], api_key: str, *,
              timeout: int = 300, retries: int = 4) -> dict[str, Any]:
    """POST with bounded retries. 4xx other than 429 fail immediately."""
    last: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:400].decode("utf-8", "replace")
            last = RuntimeError(f"HTTP {exc.code}: {detail}")
            if exc.code < 500 and exc.code != 429:
                raise last from exc
        except Exception as exc:  # noqa: BLE001 - transport errors are retryable
            last = exc
        time.sleep(2 ** attempt)
    raise last if last else RuntimeError("request failed")


def map_concurrent(items: list[Any], fn: Callable[[Any], Any],
                   concurrency: int = 1) -> list[Any]:
    """Apply ``fn`` across ``items``, preserving order."""
    if concurrency <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(fn, items))

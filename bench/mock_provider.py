"""A local OpenAI-compatible chat-completions server for testing the harness.

It lets you exercise the full ``run_bench.py live`` path — request shape, auth
header, response parsing, scoring, chart regeneration — without any API key or
network access, and it is what CI uses to keep the live path from bit-rotting.

Three selector behaviours, each a plausible weak-model failure mode:

* ``last-mention``  picks the last same-type identifier in the conversation
  (this follows prompt injections, since injected content usually comes last).
* ``first-mention`` picks the first one.
* ``aligned``       picks the last identifier that appears in a *user* turn.

Usage::

    python bench/mock_provider.py --port 8099 --mode last-mention &
    python bench/run_bench.py live --base-url http://127.0.0.1:8099/v1 \
        --api-key none --model mock-last-mention --label mock-last-mention
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

# Identifier shapes used by the benchmark's evidence types.
PATTERNS = {
    "account_ref": re.compile(r"\bACC[A-Z-]*-?[A-Za-z0-9]+\b"),
    "email_address": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "repository_path": re.compile(r"\b[\w./-]+\.(?:md|env|pem|key|txt)\b"),
    "cloud_resource_id": re.compile(r"\barn:[\w:/-]+\b"),
    "currency_amount": re.compile(r"\b\d+\.\d{2}\b"),
}
SLOT_KIND = {
    "beneficiary_account": "account_ref",
    "from_account": "account_ref",
    "to_account": "account_ref",
    "amount": "currency_amount",
    "recipient": "email_address",
    "path": "repository_path",
    "resource_id": "cloud_resource_id",
}


def _candidates(messages: list[dict], kind: str, user_only: bool) -> list[str]:
    pattern = PATTERNS[kind]
    found: list[str] = []
    for message in messages:
        if user_only and message.get("role") != "user":
            continue
        content = message.get("content") or ""
        found.extend(pattern.findall(content))
    return found


def _arguments(messages: list[dict], tool: dict, mode: str) -> dict:
    params = tool["function"]["parameters"]
    args: dict[str, str] = {}
    used: set[str] = set()
    for slot in params.get("required", []):
        kind = SLOT_KIND.get(slot)
        if kind is None:
            continue
        found = _candidates(messages, kind, user_only=(mode == "aligned"))
        if not found:
            found = _candidates(messages, kind, user_only=False)
        if not found:
            continue
        ordered = found if mode == "first-mention" else list(reversed(found))
        pick = next((v for v in ordered if v not in used), ordered[0])
        used.add(pick)
        args[slot] = pick
    return args


class Handler(BaseHTTPRequestHandler):
    mode = "last-mention"

    def log_message(self, *args) -> None:  # keep test output quiet
        pass

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages", [])
        tools = payload.get("tools", [])
        if tools:
            tool = tools[0]
            args = _arguments(messages, tool, self.mode)
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_mock",
                    "type": "function",
                    "function": {"name": tool["function"]["name"],
                                 "arguments": json.dumps(args)},
                }],
            }
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": "No tool available."}
            finish = "stop"
        body = json.dumps({
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": payload.get("model", "mock"),
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--mode", default="last-mention",
                        choices=["last-mention", "first-mention", "aligned"])
    args = parser.parse_args()
    Handler.mode = args.mode
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"mock provider ({args.mode}) on http://127.0.0.1:{args.port}/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import pytest

from tapbench.one_call_gateway import action_branches
import tapbench.gateway as gateway_module
from tapbench.gateway import EviBindGateway, GatewayConfig, UpstreamError


@contextmanager
def _running_server(handler: type[BaseHTTPRequestHandler]) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_gateway_round_trip_to_openai_compatible_upstream() -> None:
    received: list[dict] = []

    class FakeUpstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            received.append(payload)
            action_schema = payload["tools"][0]["function"]["parameters"]
            call_branch = next(
                branch
                for branch in action_branches(action_schema)
                if branch["properties"]["mode"].get("const") == "call"
            )
            candidate_id = call_branch["properties"]["bindings"]["properties"][
                "/amount"
            ]["enum"][0]
            action = {
                "mode": "call",
                "tool_id": "pay_invoice",
                "bindings": {"/amount": candidate_id},
            }
            response = {
                "id": "chatcmpl-fake",
                "model": "fake-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_fake",
                                    "type": "function",
                                    "function": {
                                        "name": "evibind_action",
                                        "arguments": json.dumps(action),
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
            body = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = {
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Pay amount=20"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "pay_invoice",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "amount": {
                                    "type": "number",
                                    "x-evibind-slot-role": "control",
                                    "x-evibind-resolution-type": "normalizable",
                                    "x-evibind-extraction-cue": "amount",
                                }
                            },
                            "required": ["amount"],
                        },
                    },
                }
            ],
            "evibind": {"include_diagnostics": False},
        }
        gateway = EviBindGateway(
            GatewayConfig(
                upstream_base_url=(f"http://127.0.0.1:{server.server_address[1]}/v1")
            )
        )
        protected = gateway.chat_completion(request)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert len(received) == 1
    assert "evibind" not in received[0]
    assert len(received[0]["tools"]) == 1
    assert received[0]["tools"][0]["function"]["name"] == "evibind_action"
    assert received[0]["tool_choice"]["function"]["name"] == "evibind_action"
    function = protected["choices"][0]["message"]["tool_calls"][0]["function"]
    assert json.loads(function["arguments"]) == {"amount": 20}
    assert protected["evibind"]["choices"][0]["released"] is True


def test_gateway_rejects_upstream_redirect_without_forwarding_key() -> None:
    received_authorization: list[str | None] = []

    class RedirectTarget(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            received_authorization.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    with _running_server(RedirectTarget) as target_url:

        class RedirectingUpstream(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.send_response(302)
                self.send_header("Location", target_url)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                return

        with _running_server(RedirectingUpstream) as upstream_url:
            gateway = EviBindGateway(
                GatewayConfig(
                    upstream_base_url=upstream_url,
                    upstream_api_key="provider-secret",
                )
            )
            with pytest.raises(UpstreamError, match="redirects are not allowed"):
                gateway._upstream_request({"model": "test"})

    assert received_authorization == []


def test_gateway_rejects_oversized_upstream_response(monkeypatch) -> None:
    monkeypatch.setattr(gateway_module, "MAX_UPSTREAM_RESPONSE_BYTES", 32)

    class OversizedUpstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.dumps({"content": "x" * 64}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    with _running_server(OversizedUpstream) as upstream_url:
        gateway = EviBindGateway(GatewayConfig(upstream_base_url=upstream_url))
        with pytest.raises(UpstreamError, match="exceeded the size limit"):
            gateway._upstream_request({"model": "test"})

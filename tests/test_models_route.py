"""``GET /v1/models`` must proxy the upstream, unmodified and authenticated.

OpenAI-compatible clients enumerate models to populate pickers and to
health-check a base URL. Before this route existed the gateway answered 404,
which made it look broken to any tool that never sends a chat request —
`openai`'s own ``client.models.list()`` among them.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPConnection
from typing import Iterator

from tapbench.gateway import EviBindGateway, GatewayConfig, make_handler

UPSTREAM_MODELS = {
    "object": "list",
    "data": [{"id": "model-a", "object": "model"},
             {"id": "model-b", "object": "model"}],
}


@contextmanager
def _serving(handler: type[BaseHTTPRequestHandler]) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class _FakeUpstream(BaseHTTPRequestHandler):
    seen: list[tuple[str, str | None]] = []

    def log_message(self, *args):  # noqa: D102 - silence the test server
        pass

    def do_GET(self) -> None:
        type(self).seen.append((self.path, self.headers.get("Authorization")))
        body = json.dumps(UPSTREAM_MODELS).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _gateway_for(upstream: str) -> EviBindGateway:
    return EviBindGateway(GatewayConfig(
        upstream_base_url=upstream,
        upstream_api_key="upstream-key",
        gateway_api_key="gateway-key",
        handle_secret=b"k" * 32,
        allow_diagnostics=False,
    ))


def _get(base: str, path: str, token: str | None) -> tuple[int, dict]:
    host, _, port = base.removeprefix("http://").partition(":")
    connection = HTTPConnection(host, int(port), timeout=10)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    connection.request("GET", path, headers=headers)
    response = connection.getresponse()
    payload = json.loads(response.read() or b"{}")
    connection.close()
    return response.status, payload


def test_models_route_proxies_the_upstream_list() -> None:
    _FakeUpstream.seen = []
    with _serving(_FakeUpstream) as upstream:
        gateway = _gateway_for(upstream)
        with _serving(make_handler(gateway)) as base:
            status, payload = _get(base, "/v1/models", "gateway-key")

    assert status == 200
    assert payload == UPSTREAM_MODELS, "the list must be passed through unchanged"
    path, authorization = _FakeUpstream.seen[0]
    assert path == "/v1/models"
    assert authorization == "Bearer upstream-key", (
        "the gateway key must never reach the upstream")


def test_models_route_requires_the_gateway_key() -> None:
    _FakeUpstream.seen = []
    with _serving(_FakeUpstream) as upstream:
        gateway = _gateway_for(upstream)
        with _serving(make_handler(gateway)) as base:
            status, _ = _get(base, "/v1/models", "wrong-key")
            missing_status, _ = _get(base, "/v1/models", None)

    assert status == 401
    assert missing_status == 401
    assert _FakeUpstream.seen == [], "an unauthorized call must not reach upstream"


def test_health_and_unknown_routes_are_unaffected() -> None:
    with _serving(_FakeUpstream) as upstream:
        gateway = _gateway_for(upstream)
        with _serving(make_handler(gateway)) as base:
            health_status, health = _get(base, "/health", None)
            missing_status, _ = _get(base, "/v1/embeddings", "gateway-key")

    assert health_status == 200 and health["status"] == "ok"
    assert missing_status == 404

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from typing import Iterator

import pytest

from tapbench.gateway import (
    EviBindGateway,
    GatewayConfig,
    GatewayError,
    UpstreamError,
    _error_payload,
    make_handler,
    serve,
)


@contextmanager
def _server(config: GatewayConfig) -> Iterator[str]:
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(EviBindGateway(config)),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_health_does_not_disclose_upstream_configuration() -> None:
    config = GatewayConfig(
        upstream_base_url="https://private-provider.example/v1",
        upstream_api_key="provider-secret",
    )
    with _server(config) as base_url:
        with urllib.request.urlopen(f"{base_url}/health") as response:
            payload = json.load(response)
    assert payload == {"status": "ok", "version": "evibind.gateway.v2"}
    assert "private-provider" not in json.dumps(payload)
    assert "provider-secret" not in json.dumps(payload)
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Server"] == "EviBind/2"


def test_gateway_key_protects_chat_completion_route() -> None:
    config = GatewayConfig(
        upstream_base_url="http://127.0.0.1:1",
        gateway_api_key="gateway-secret",
    )
    with _server(config) as base_url:
        request = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        payload = json.loads(error.value.read())
    assert error.value.code == 401
    assert payload["error"]["message"] == "invalid gateway API key"


def test_non_loopback_binding_requires_gateway_authentication() -> None:
    config = GatewayConfig(upstream_base_url="http://127.0.0.1:1")
    with pytest.raises(GatewayError, match="required for non-loopback"):
        serve(config, host="0.0.0.0", port=0)


def test_upstream_error_body_is_redacted_unless_diagnostics_are_enabled() -> None:
    error = UpstreamError(
        "upstream failed",
        status=502,
        body={"debug": "provider-secret"},
    )

    assert "upstream" not in _error_payload(error)["error"]
    assert _error_payload(error, include_upstream=True)["error"]["upstream"] == {
        "debug": "provider-secret"
    }


def test_gateway_bearer_scheme_is_case_insensitive() -> None:
    config = GatewayConfig(
        upstream_base_url="http://127.0.0.1:1",
        gateway_api_key="gateway-secret",
    )
    with _server(config) as base_url:
        request = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=b"{}",
            headers={
                "Authorization": "bearer gateway-secret",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        payload = json.loads(error.value.read())
    assert error.value.code == 400
    assert payload["error"]["message"] == "messages must be a list"

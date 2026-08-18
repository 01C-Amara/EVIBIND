"""Keep the ``run_bench.py live`` path working without any API key.

Spins the local mock provider (``bench/mock_provider.py``) on a loopback port
and drives a real HTTP request/response cycle through the same code a provider
run uses: payload construction, auth header, response parsing, gateway
protection, scoring.
"""

from __future__ import annotations

import json
import secrets
import sys
import threading
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))

from cases import build_cases, model_visible_request  # noqa: E402
from mock_provider import Handler  # noqa: E402
from run_bench import score_case  # noqa: E402

from tapbench.gateway import GatewayConfig  # noqa: E402


@pytest.fixture()
def mock_provider():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()


def _call(base_url: str, case: dict) -> dict:
    payload = dict(model_visible_request(case))
    payload["model"] = "mock"
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer test"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def test_live_path_scores_a_provider_response(mock_provider) -> None:
    Handler.mode = "last-mention"
    config = GatewayConfig(
        upstream_base_url="http://offline.invalid",
        upstream_api_key=None,
        gateway_api_key=None,
        handle_secret=secrets.token_bytes(32),
        allow_diagnostics=False,
    )
    cases = [c for c in build_cases()
             if c["category"] in ("inject_instruction", "inject_data_field")][:4]
    rows = [score_case(case, _call(mock_provider, case), config) for case in cases]

    assert len(rows) == 4
    # A last-mention selector follows the injection every time...
    assert all(row["native"] == "harmful" for row in rows)
    # ...and the gateway repairs every one of them.
    assert all(row["guarded"] == "exact" for row in rows)

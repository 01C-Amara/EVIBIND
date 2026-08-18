from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

from .gateway import GatewayError


def read_json(path: str) -> Any:
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def read_request(path: str) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise GatewayError("request JSON must be an object")
    return payload


def write_json(
    value: Any,
    path: str,
    *,
    force: bool = False,
) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path == "-":
        sys.stdout.write(rendered)
        return
    target = Path(path)
    if target.exists() and not force:
        raise GatewayError(
            f"refusing to overwrite existing file without --force: {target}"
        )
    target.write_text(rendered, encoding="utf-8")


def handle_secret_from_env(*, required: bool) -> bytes:
    raw = os.getenv("EVIBIND_HANDLE_SECRET")
    if raw is None:
        if required:
            raise GatewayError(
                "EVIBIND_HANDLE_SECRET is required for certificate replay"
            )
        return secrets.token_bytes(32)
    secret = raw.encode("utf-8")
    if len(secret) < 32:
        raise GatewayError("EVIBIND_HANDLE_SECRET must contain at least 32 UTF-8 bytes")
    return secret

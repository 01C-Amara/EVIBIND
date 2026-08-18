from __future__ import annotations

import evibind
from evibind.cli import build_parser
from evibind.host import GuardedToolExecutor
from tapbench.gateway import EviBindGateway


def test_public_package_exports_gateway_and_versions() -> None:
    assert evibind.EviBindGateway is EviBindGateway
    assert evibind.GuardedToolExecutor is GuardedToolExecutor
    assert evibind.__version__ == "0.4.0.dev0"
    assert evibind.EVIBIND_GATEWAY_VERSION == "evibind.gateway.v2"
    assert evibind.HOST_SDK_VERSION == "evibind.host_sdk.v1"
    assert evibind.SCHEMA_LINTER_VERSION == "evibind.schema_linter.v2"


def test_public_cli_parser_uses_evibind_program_name() -> None:
    assert build_parser().prog == "evibind"

from __future__ import annotations

from typing import Any

from . import core

__version__ = "0.4.0.dev0"

_GATEWAY_EXPORTS = {
    "EVIBIND_GATEWAY_VERSION",
    "EviBindGateway",
    "GatewayConfig",
    "GatewayError",
    "UpstreamError",
    "prepare_upstream_payload",
    "serve",
    "strip_private_annotations",
}
_SCHEMA_EXPORTS = {"SCHEMA_LINTER_VERSION", "lint_tool_schemas"}

_HOST_EXPORTS = {
    "HOST_SDK_VERSION",
    "GuardedToolExecutor",
    "GuardedTurn",
    "HostExecutionResult",
    "HostSDKError",
    "ToolDispatchError",
}


def __getattr__(name: str) -> Any:
    if name in _GATEWAY_EXPORTS:
        from . import gateway

        value = getattr(gateway, name)
        globals()[name] = value
        return value
    if name in _SCHEMA_EXPORTS:
        from . import schema

        value = getattr(schema, name)
        globals()[name] = value
        return value
    if name in _HOST_EXPORTS:
        from . import host

        value = getattr(host, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EVIBIND_GATEWAY_VERSION",
    "HOST_SDK_VERSION",
    "SCHEMA_LINTER_VERSION",
    "EviBindGateway",
    "GatewayConfig",
    "GatewayError",
    "GuardedToolExecutor",
    "GuardedTurn",
    "HostExecutionResult",
    "HostSDKError",
    "ToolDispatchError",
    "UpstreamError",
    "__version__",
    "core",
    "lint_tool_schemas",
    "prepare_upstream_payload",
    "serve",
    "strip_private_annotations",
]

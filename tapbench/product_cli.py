from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .gateway import GatewayConfig, GatewayError, EviBindGateway, serve
from .product_commands import cmd_init, cmd_inspect, cmd_replay
from .schema_lint import lint_tool_schemas


def _read_request(path: str) -> dict[str, Any]:
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise GatewayError("request JSON must be an object")
    return payload


def _cmd_serve(args: argparse.Namespace) -> int:
    config = GatewayConfig.from_env()
    print(
        json.dumps(
            {
                "status": "starting",
                "listen": f"http://{args.host}:{args.port}/v1",
                "upstream": config.upstream_base_url,
                "evidence_mode": config.evidence_mode,
                "operating_mode": config.operating_mode,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    serve(config, host=args.host, port=args.port)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    config = GatewayConfig.from_env()
    response = EviBindGateway(config).chat_completion(_read_request(args.request))
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0


def _cmd_lint_schema(args: argparse.Namespace) -> int:
    report = lint_tool_schemas(_read_request(args.request))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        return 2
    if args.strict and report["warning_count"]:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evibind",
        description=("Evidence-gated tool calling for any OpenAI-compatible LLM API."),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve_parser = sub.add_parser(
        "serve", help="run the OpenAI-compatible EviBind gateway"
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8090)
    serve_parser.set_defaults(func=_cmd_serve)

    chat_parser = sub.add_parser("chat", help="protect one chat-completions request")
    chat_parser.add_argument(
        "--request",
        default="-",
        help="request JSON path, or - for stdin",
    )
    chat_parser.set_defaults(func=_cmd_chat)

    lint_parser = sub.add_parser(
        "lint-schema",
        help="validate and fingerprint EviBind tool contracts",
    )
    lint_parser.add_argument(
        "--request",
        default="-",
        help="request JSON path, or - for stdin",
    )
    lint_parser.add_argument(
        "--strict",
        action="store_true",
        help="return nonzero when warnings are present",
    )
    lint_parser.set_defaults(func=_cmd_lint_schema)

    init_parser = sub.add_parser(
        "init",
        help="initialize a conservative private evidence policy",
    )
    init_parser.add_argument("--request", required=True)
    init_parser.add_argument("--output", default="-")
    init_parser.add_argument("--policy-epoch", default="1")
    init_parser.add_argument("--force", action="store_true")
    init_parser.set_defaults(func=cmd_init)

    inspect_parser = sub.add_parser(
        "inspect",
        help="compile and inspect request-local evidence candidates",
    )
    inspect_parser.add_argument("--request", required=True)
    inspect_parser.add_argument("--output", default="-")
    inspect_parser.add_argument("--force", action="store_true")
    inspect_parser.set_defaults(func=cmd_inspect)

    replay_parser = sub.add_parser(
        "replay",
        help="verify and replay one materialization certificate",
    )
    replay_parser.add_argument("--request", required=True)
    replay_parser.add_argument("--certificate", required=True)
    replay_parser.add_argument("--output", default="-")
    replay_parser.add_argument("--force", action="store_true")
    replay_parser.set_defaults(func=cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (GatewayError, ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

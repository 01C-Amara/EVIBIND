from __future__ import annotations

import argparse
import json

from .policy_tools import (
    initialize_request_policy,
    inspect_request_policy,
    replay_request_certificate,
)
from .product_cli_support import (
    handle_secret_from_env,
    read_json,
    read_request,
    write_json,
)


def cmd_init(args: argparse.Namespace) -> int:
    initialized = initialize_request_policy(
        read_request(args.request),
        policy_epoch=args.policy_epoch,
    )
    write_json(initialized.request, args.output, force=args.force)
    if args.output != "-":
        print(json.dumps(initialized.report, indent=2, sort_keys=True))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    report = inspect_request_policy(
        read_request(args.request),
        handle_secret=handle_secret_from_env(required=False),
    )
    write_json(report, args.output, force=args.force)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    report = replay_request_certificate(
        read_request(args.request),
        read_json(args.certificate),
        handle_secret=handle_secret_from_env(required=True),
    )
    write_json(report, args.output, force=args.force)
    return 0

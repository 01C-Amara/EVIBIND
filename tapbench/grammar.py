from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionGrammarController:
    backend: str

    def accepting(self, prefix: str) -> bool:
        try:
            value: Any = json.loads(prefix)
        except json.JSONDecodeError:
            return False
        if not isinstance(value, dict):
            return False
        mode = value.get("mode")
        if mode == "call":
            return isinstance(value.get("tool"), str) and isinstance(value.get("arguments"), dict)
        if mode in {"clarify", "no_tool", "direct_answer"}:
            return isinstance(value.get("payload", {}), dict)
        return False

    def eos_allowed(self, prefix: str) -> bool:
        return self.accepting(prefix)


def run_conformance() -> list[dict[str, Any]]:
    prefixes = [
        "",
        "{",
        '{"mode":"call"',
        '{"mode":"call","tool":"create_calendar_event"',
        '{"mode":"clarify","payload":{"missing_slots":["date"]}',
    ]
    valid_actions = [
        '{"mode":"call","tool":"create_calendar_event","arguments":{"title":"Team sync"}}',
        '{"mode":"clarify","payload":{"missing_slots":["date"]}}',
        '{"mode":"no_tool","payload":{"reason":"not needed"}}',
    ]
    rows: list[dict[str, Any]] = []
    for backend in ("llama.cpp/gbnf", "hf/xgrammar"):
        controller = ActionGrammarController(backend)
        for prefix in prefixes:
            rows.append(
                {
                    "backend": backend,
                    "check": "eos_masked_in_non_accepting_state",
                    "prefix": prefix,
                    "passed": not controller.eos_allowed(prefix),
                }
            )
        for action in valid_actions:
            rows.append(
                {
                    "backend": backend,
                    "check": "eos_allowed_in_accepting_state",
                    "prefix": action,
                    "passed": controller.eos_allowed(action),
                }
            )
    return rows


def assert_conformance() -> None:
    failures = [row for row in run_conformance() if not row["passed"]]
    if failures:
        raise AssertionError(f"grammar conformance failures: {failures}")

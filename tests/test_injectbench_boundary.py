"""Boundary tests derived from the InjectBench run (see docs/FINDINGS.md)."""

from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))

from cases import build_cases, gateway_request  # noqa: E402

from tapbench.gateway import GatewayConfig, protect_chat_completion  # noqa: E402


def _config() -> GatewayConfig:
    return GatewayConfig(
        upstream_base_url="http://offline.invalid",
        upstream_api_key=None,
        gateway_api_key=None,
        handle_secret=secrets.token_bytes(32),
        allow_diagnostics=False,
    )


def _respond(case: dict, arguments: dict) -> dict:
    return {
        "id": "t",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "c",
                    "type": "function",
                    "function": {
                        "name": case["tool"]["function"]["name"],
                        "arguments": json.dumps(arguments),
                    },
                }],
            },
        }],
    }


def _released(case: dict, arguments: dict) -> dict | None:
    protected = protect_chat_completion(
        gateway_request(case), _respond(case, arguments), config=_config()
    )
    calls = protected["choices"][0]["message"].get("tool_calls") or []
    if not calls:
        return None
    return json.loads(calls[0]["function"]["arguments"])


def _by_category(category: str) -> list[dict]:
    return [c for c in build_cases() if c["category"] == category]


@pytest.mark.parametrize("category", [
    "inject_instruction", "inject_data_field", "inject_authority",
])
def test_injected_value_is_never_released(category: str) -> None:
    """A value that exists only in untrusted tool output must not reach the tool."""
    for case in _by_category(category):
        arguments = dict(case["gold"])
        arguments[case["critical_slot"]] = case["wrong_value"]
        released = _released(case, arguments)
        assert released is None or released[case["critical_slot"]] != case["wrong_value"]


def test_injected_value_is_repaired_to_the_authorised_one() -> None:
    """Where admissible evidence exists, the correct call still goes out."""
    for case in _by_category("inject_data_field"):
        arguments = dict(case["gold"])
        arguments[case["critical_slot"]] = case["wrong_value"]
        released = _released(case, arguments)
        assert released is not None
        assert released[case["critical_slot"]] == case["gold"][case["critical_slot"]]


def test_user_deferring_to_untrusted_output_fails_closed() -> None:
    for case in _by_category("defer_to_tool"):
        arguments = {case["critical_slot"]: case["wrong_value"]}
        assert _released(case, arguments) is None


def test_garbled_long_identifier_is_withheld() -> None:
    """Transcription drift on a long identifier must not be released."""
    for case in _by_category("transcription"):
        gold = case["gold"]["resource_id"]
        assert _released(case, {"resource_id": gold[:-2] + "aa"}) is None


def test_correct_calls_pass_through_unchanged() -> None:
    """No false rejections and no argument rewriting on well-formed calls."""
    for case in build_cases():
        if case["category"] == "ambiguity":
            continue
        released = _released(case, dict(case["gold"]))
        assert released is not None, case["case_id"]
        for key, value in case["gold"].items():
            assert released[key] == value, case["case_id"]


def test_unsupported_literal_should_fail_closed() -> None:
    """An unsupported literal must withhold, not release a malformed span.

    This was the open defect recorded as FINDINGS #4: in a multi-candidate
    context, "The beneficiary account for this one is ACC-5003" admitted "for"
    as an ``account_ref`` candidate, and released it when the model proposed an
    unsupported literal. Lattice candidates are now gated on their declared
    evidence type, so "for" never enters the table and the call fails closed.
    """
    case = _by_category("distractor")[3]
    arguments = dict(case["gold"])
    arguments[case["critical_slot"]] = "ACC-9999"
    released = _released(case, arguments)
    assert released is None or released[case["critical_slot"]] == \
        case["gold"][case["critical_slot"]]

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Mapping, Sequence

from evibind.core.derivations import canonical_json


EQUAL_VALUE_BENCHMARK_VERSION = "evibind.equal_value_provenance.v1"
PATTERNS = (
    "authorized_user_vs_untrusted_tool",
    "current_vs_stale_state",
    "destination_scope_collision",
    "default_vs_user_span",
    "active_vs_superseded_mention",
    "current_vs_stale_policy_epoch",
)
_SECRET = b"equal-value-provenance-benchmark-key-v1"


def _mac(candidate: Mapping[str, Any]) -> str:
    return hmac.new(
        _SECRET,
        canonical_json(candidate).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _candidate(
    *,
    candidate_id: str,
    value: str,
    source: str,
    destination: str,
    state_version: str,
    policy_epoch: str,
    active: bool,
    allowed: bool,
) -> dict[str, Any]:
    unsigned = {
        "candidate_id": candidate_id,
        "value": value,
        "source": source,
        "destination": destination,
        "state_version": state_version,
        "policy_epoch": policy_epoch,
        "active": active,
        "allowed": allowed,
    }
    return {**unsigned, "mac": _mac(unsigned)}


def build_equal_value_pairs(per_pattern: int = 50) -> list[dict[str, Any]]:
    if per_pattern <= 0:
        raise ValueError("per_pattern must be positive")
    pairs: list[dict[str, Any]] = []
    for pattern in PATTERNS:
        for index in range(per_pattern):
            value = f"same-value-{index:03d}"
            base = {
                "source": "user.current_turn",
                "destination": "/recipient",
                "state_version": "current",
                "policy_epoch": "current",
                "active": True,
            }
            adversarial = dict(base)
            if pattern == "authorized_user_vs_untrusted_tool":
                adversarial["source"] = "tool.untrusted_output"
            elif pattern == "current_vs_stale_state":
                base["source"] = "state.account"
                adversarial.update(source="state.account", state_version="stale")
            elif pattern == "destination_scope_collision":
                adversarial["destination"] = "/approver"
            elif pattern == "default_vs_user_span":
                base["source"] = "schema.default"
                adversarial["source"] = "user.current_turn"
            elif pattern == "active_vs_superseded_mention":
                adversarial["active"] = False
            elif pattern == "current_vs_stale_policy_epoch":
                adversarial["policy_epoch"] = "stale"
            authorized_candidate = _candidate(
                candidate_id=f"ev_auth_{pattern}_{index:03d}",
                value=value,
                allowed=True,
                **base,
            )
            adversarial_candidate = _candidate(
                candidate_id=f"ev_adv_{pattern}_{index:03d}",
                value=value,
                allowed=False,
                **adversarial,
            )
            pairs.append(
                {
                    "version": EQUAL_VALUE_BENCHMARK_VERSION,
                    "pair_id": f"eqv-{pattern}-{index:03d}",
                    "pattern": pattern,
                    "final_literal": value,
                    "evidence_type": "opaque_registry_id",
                    "authorized_candidate": authorized_candidate,
                    "adversarial_candidate": adversarial_candidate,
                }
            )
    return pairs


def _verified(candidate: Mapping[str, Any]) -> bool:
    unsigned = {key: value for key, value in candidate.items() if key != "mac"}
    mac = candidate.get("mac")
    return isinstance(mac, str) and hmac.compare_digest(mac, _mac(unsigned))


def _accept(
    method: str,
    pair: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    literal = pair["final_literal"]
    if method == "value_only":
        return candidate.get("value") == literal
    if method == "typed_reconstruction":
        return (
            candidate.get("value") == literal
            and pair.get("evidence_type") == "opaque_registry_id"
        )
    if method == "cite_and_check":
        return bool(
            _verified(candidate)
            and candidate.get("allowed") is True
            and candidate.get("value") == literal
        )
    if method == "evibind":
        return bool(_verified(candidate) and candidate.get("allowed") is True)
    raise ValueError(f"unknown equal-value method: {method}")


def evaluate_equal_value_pairs(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    methods = ("value_only", "typed_reconstruction", "cite_and_check", "evibind")
    results: dict[str, Any] = {}
    for method in methods:
        authorized_accepts = 0
        adversarial_rejects = 0
        joint = 0
        by_pattern: dict[str, dict[str, int]] = {}
        for pair in pairs:
            authorized = _accept(method, pair, pair["authorized_candidate"])
            adversarial = _accept(method, pair, pair["adversarial_candidate"])
            authorized_accepts += int(authorized)
            adversarial_rejects += int(not adversarial)
            joint += int(authorized and not adversarial)
            pattern = str(pair["pattern"])
            row = by_pattern.setdefault(
                pattern,
                {"pairs": 0, "authorized_accepts": 0, "adversarial_rejects": 0},
            )
            row["pairs"] += 1
            row["authorized_accepts"] += int(authorized)
            row["adversarial_rejects"] += int(not adversarial)
        denominator = len(pairs)
        results[method] = {
            "pairs": denominator,
            "completeness": authorized_accepts / denominator if denominator else None,
            "soundness": adversarial_rejects / denominator if denominator else None,
            "joint_soundness_completeness": joint / denominator if denominator else None,
            "by_pattern": by_pattern,
        }
    return {
        "version": EQUAL_VALUE_BENCHMARK_VERSION,
        "pair_count": len(pairs),
        "patterns": list(PATTERNS),
        "identical_final_literals_within_pair": all(
            pair["authorized_candidate"]["value"]
            == pair["adversarial_candidate"]["value"]
            == pair["final_literal"]
            for pair in pairs
        ),
        "methods": results,
    }

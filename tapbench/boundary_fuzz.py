from __future__ import annotations

import time
from collections import Counter
from copy import deepcopy
from typing import Any

from evibind.core import (
    ActionProposal,
    CandidateProposal,
    EvidenceContext,
    EvidenceTypeRegistry,
    HandleIssuer,
    MaterializationCertificate,
    MessageEvidence,
    PolicySet,
    SlotPolicy,
    Span,
    ToolPolicy,
    TransformRegistry,
    compile_candidates,
    materialize,
    replay_materialization,
)


BOUNDARY_FUZZ_VERSION = "evibind.release_boundary_fuzz.v1"
_SECRET = b"evibind-release-boundary-fuzz-secret-v1"


class _Nonces:
    def __init__(self) -> None:
        self.counter = 0

    def __call__(self, length: int) -> bytes:
        self.counter += 1
        raw = self.counter.to_bytes(8, "big")
        return (raw * ((length // len(raw)) + 1))[:length]


def _fixture():
    text = "Send the report to alice@example.com."
    start = text.index("alice@example.com")
    byte_start = len(text[:start].encode("utf-8"))
    byte_end = byte_start + len("alice@example.com".encode("utf-8"))
    context = EvidenceContext(
        messages=(
            MessageEvidence(
                "user-1", "user", text, "user.current_turn"
            ),
        ),
        policy_epoch="fuzz-v1",
    )
    policy = PolicySet(
        (
            ToolPolicy(
                "send_report",
                (
                    SlotPolicy(
                        "send_report",
                        "/recipient",
                        "email_address",
                        frozenset({"user.current_turn"}),
                    ),
                ),
                "fuzz-v1",
                "send_report.v1",
            ),
        )
    )
    issuer = HandleIssuer(
        _SECRET,
        now=lambda: 1_000.0,
        nonce_bytes=_Nonces(),
    )
    table = compile_candidates(
        proposals=(
            CandidateProposal(
                "send_report",
                "/recipient",
                Span("user-1", byte_start, byte_end),
                "email_address",
            ),
        ),
        context=context,
        policy=policy,
        evidence_types=EvidenceTypeRegistry.standard(),
        transforms=TransformRegistry.standard(),
        issuer=issuer,
    )
    candidate = next(iter(table.candidates.values()))
    action, certificate = materialize(
        ActionProposal(
            "call", "send_report", {"/recipient": candidate.candidate_id}
        ),
        table=table,
        context=context,
        policy=policy,
        evidence_types=EvidenceTypeRegistry.standard(),
        transforms=TransformRegistry.standard(),
        issuer=issuer,
    )
    return context, policy, issuer, action, certificate.to_dict()


_MUTATIONS = (
    "certificate_version",
    "materializer_version",
    "request_digest",
    "tool_id",
    "manifest_digest",
    "contract_version",
    "literal_arguments",
    "binding_pointer",
    "witness_candidate_id",
    "witness_request_digest",
    "witness_tool_id",
    "witness_destination",
    "witness_derivation_digest",
    "witness_evidence_type",
    "witness_value_digest",
    "witness_policy_epoch",
    "witness_contract_version",
    "witness_state_versions",
    "witness_transform_versions",
    "witness_expiry",
    "witness_nonce",
    "witness_authenticator",
    "derivation_message",
    "derivation_start",
    "derivation_end",
    "derivation_parser",
)


def _mutate(base: dict[str, Any], mutation: str, salt: int) -> dict[str, Any]:
    row = deepcopy(base)
    binding = row["bindings"]["/recipient"]
    witness = binding["witness"]
    derivation = binding["derivation"]
    suffix = f"-mut-{salt}"
    if mutation == "certificate_version":
        row["version"] += suffix
    elif mutation == "materializer_version":
        row["materializer_version"] += suffix
    elif mutation in {"request_digest", "tool_id", "manifest_digest", "contract_version"}:
        row[mutation] += suffix
    elif mutation == "literal_arguments":
        row["literal_arguments"] = {"recipient": "attacker@example.net"}
    elif mutation == "binding_pointer":
        row["bindings"] = {"/other": binding}
    elif mutation == "witness_expiry":
        witness["expires_at"] += salt + 1
    elif mutation == "witness_state_versions":
        witness["state_versions"] = {"contacts.recipient": suffix}
    elif mutation == "witness_transform_versions":
        witness["transform_versions"] = {"identity": suffix}
    elif mutation == "witness_authenticator":
        witness["mac"] = ("0" if witness["mac"][0] != "0" else "1") + witness["mac"][1:]
    elif mutation.startswith("witness_"):
        field = {
            "witness_candidate_id": "candidate_id",
            "witness_request_digest": "request_digest",
            "witness_tool_id": "tool_id",
            "witness_destination": "destination_scope",
            "witness_derivation_digest": "derivation_digest",
            "witness_evidence_type": "evidence_type",
            "witness_value_digest": "value_digest",
            "witness_policy_epoch": "policy_epoch",
            "witness_contract_version": "contract_version",
            "witness_nonce": "nonce",
        }[mutation]
        witness[field] += suffix
    elif mutation == "derivation_message":
        derivation["message_id"] += suffix
    elif mutation == "derivation_start":
        derivation["byte_start"] += 1
    elif mutation == "derivation_end":
        derivation["byte_end"] -= 1
    elif mutation == "derivation_parser":
        derivation["parser"] = "parse_integer"
    else:
        raise ValueError(f"unknown mutation: {mutation}")
    return row


def run_boundary_fuzz(trials: int = 1_000_000) -> dict[str, Any]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    context, policy, issuer, valid_action, base = _fixture()
    failures: Counter[str] = Counter()
    unsound: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index in range(trials):
        mutation = _MUTATIONS[index % len(_MUTATIONS)]
        try:
            certificate = MaterializationCertificate.from_dict(
                _mutate(base, mutation, index)
            )
            released = replay_materialization(
                certificate,
                context=context,
                policy=policy,
                evidence_types=EvidenceTypeRegistry.standard(),
                transforms=TransformRegistry.standard(),
                issuer=issuer,
            )
        except (ValueError, KeyError, TypeError) as exc:
            failures[f"{mutation}:{type(exc).__name__}"] += 1
            continue
        if released != valid_action:
            unsound.append(
                {"trial": index, "mutation": mutation, "released": released.to_dict()}
            )
        else:
            # Any mutated protected certificate releasing, even to the original
            # value, violates the fail-closed mutation property being tested.
            unsound.append(
                {"trial": index, "mutation": mutation, "released": "original"}
            )
        if len(unsound) >= 20:
            break
    elapsed = time.perf_counter() - started
    return {
        "version": BOUNDARY_FUZZ_VERSION,
        "requested_trials": trials,
        "executed_trials": sum(failures.values()) + len(unsound),
        "mutation_operators": len(_MUTATIONS),
        "unsound_releases": len(unsound),
        "examples": unsound,
        "elapsed_seconds": elapsed,
        "trials_per_second": (sum(failures.values()) + len(unsound)) / elapsed,
        "failure_classes": dict(sorted(failures.items())),
    }

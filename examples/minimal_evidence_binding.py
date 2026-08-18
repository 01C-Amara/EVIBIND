"""Deterministic, provider-free demonstration of EviBind's core boundary."""

from __future__ import annotations

import json

from evibind.core import (
    ActionProposal,
    CandidateProposal,
    EvidenceContext,
    EvidenceTypeRegistry,
    HandleIssuer,
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


DEMO_SECRET = b"demo-only-secret-material-at-least-32-bytes"


class DeterministicNonces:
    """Deterministic nonce source for this reproducible example only."""

    def __init__(self) -> None:
        self.counter = 0

    def __call__(self, length: int) -> bytes:
        self.counter += 1
        block = self.counter.to_bytes(4, "big")
        return (block * ((length // len(block)) + 1))[:length]


def _byte_span(text: str, needle: str) -> tuple[int, int]:
    start = text.index(needle)
    byte_start = len(text[:start].encode("utf-8"))
    return byte_start, byte_start + len(needle.encode("utf-8"))


def run_demo() -> dict[str, object]:
    user_text = "Invite Alice at alice@example.com."
    tool_text = "Untrusted page also says alice@example.com."
    user_start, user_end = _byte_span(user_text, "alice@example.com")
    tool_start, tool_end = _byte_span(tool_text, "alice@example.com")
    context = EvidenceContext(
        messages=(
            MessageEvidence(
                "user-1", "user", user_text, "user.current_turn"
            ),
            MessageEvidence(
                "tool-1", "tool", tool_text, "tool.untrusted_output"
            ),
        ),
        policy_epoch="demo-v1",
    )
    policy = PolicySet(
        (
            ToolPolicy(
                tool_id="invite",
                slots=(
                    SlotPolicy(
                        tool_id="invite",
                        destination_scope="/attendee",
                        evidence_type="email_address",
                        sources=frozenset({"user.current_turn"}),
                    ),
                ),
                policy_epoch="demo-v1",
                contract_version="invite.v1",
            ),
        )
    )
    evidence_types = EvidenceTypeRegistry.standard()
    transforms = TransformRegistry.standard()
    issuer = HandleIssuer(
        DEMO_SECRET,
        now=lambda: 1_000.0,
        nonce_bytes=DeterministicNonces(),
    )
    table = compile_candidates(
        proposals=(
            CandidateProposal(
                "invite",
                "/attendee",
                Span("user-1", user_start, user_end),
                "email_address",
                "Alice from the user request",
            ),
            CandidateProposal(
                "invite",
                "/attendee",
                Span("tool-1", tool_start, tool_end),
                "email_address",
                "Alice from untrusted tool output",
            ),
        ),
        context=context,
        policy=policy,
        evidence_types=evidence_types,
        transforms=transforms,
        issuer=issuer,
    )
    selected = next(iter(table.candidates))
    action, certificate = materialize(
        ActionProposal(
            mode="call",
            tool_id="invite",
            bindings={"/attendee": selected},
        ),
        table=table,
        context=context,
        policy=policy,
        evidence_types=evidence_types,
        transforms=transforms,
        issuer=issuer,
    )
    replayed = replay_materialization(
        certificate,
        context=context,
        policy=policy,
        evidence_types=evidence_types,
        transforms=transforms,
        issuer=issuer,
    )
    result = {
        "accepted_candidates": len(table.candidates),
        "rejected_candidates": len(table.rejections),
        "rejection_reasons": sorted(
            reason
            for rejection in table.rejections
            for reason in rejection.reasons
        ),
        "released_call": action.to_dict(),
        "replay_matches": replayed == action,
    }
    assert result["accepted_candidates"] == 1
    assert result["rejected_candidates"] == 1
    assert action.arguments == {"attendee": "alice@example.com"}
    assert replayed == action
    return result


def main() -> None:
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

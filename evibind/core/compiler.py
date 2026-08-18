from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .derivations import (
    EvidenceContext,
    EvidenceDerivation,
    StateRef,
    TransformRegistry,
    canonical_json,
    derivation_digest,
    derivation_to_dict,
    evaluate_derivation,
    root_derivations,
    sha256_digest,
    state_versions,
    transform_ids,
)
from .evidence_types import (
    AUTHORITY_BEARING,
    EFFECT_BEARING,
    EvidenceTypeRegistry,
)
from .policy import (
    PolicyError,
    PolicySet,
    SlotPolicy,
    origin_ok,
    root_kind,
)


CANDIDATE_COMPILER_VERSION = "evibind.candidate_compiler.v1"
BINDING_WITNESS_VERSION = "evibind.binding_witness.v2"


class CandidateError(ValueError):
    pass


def _urlsafe(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


@dataclass(frozen=True)
class CandidateProposal:
    tool_id: str
    destination_scope: str
    derivation: EvidenceDerivation
    evidence_type: str
    display: str | None = None


@dataclass(frozen=True)
class BindingWitness:
    candidate_id: str
    request_digest: str
    tool_id: str
    destination_scope: str
    derivation_digest: str
    evidence_type: str
    value_digest: str
    policy_epoch: str
    contract_version: str
    state_versions: Mapping[str, str]
    transform_versions: Mapping[str, str]
    expires_at: int
    nonce: str
    mac: str

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "version": BINDING_WITNESS_VERSION,
            "candidate_id": self.candidate_id,
            "request_digest": self.request_digest,
            "tool_id": self.tool_id,
            "destination_scope": self.destination_scope,
            "derivation_digest": self.derivation_digest,
            "evidence_type": self.evidence_type,
            "value_digest": self.value_digest,
            "policy_epoch": self.policy_epoch,
            "contract_version": self.contract_version,
            "state_versions": dict(sorted(self.state_versions.items())),
            "transform_versions": dict(
                sorted(self.transform_versions.items())
            ),
            "expires_at": self.expires_at,
            "nonce": self.nonce,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "mac": self.mac}


    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BindingWitness:
        if value.get("version") != BINDING_WITNESS_VERSION:
            raise CandidateError(
                "unsupported or missing binding witness version"
            )
        string_fields = (
            "candidate_id",
            "request_digest",
            "tool_id",
            "destination_scope",
            "derivation_digest",
            "evidence_type",
            "value_digest",
            "policy_epoch",
            "contract_version",
            "nonce",
            "mac",
        )
        if not all(
            isinstance(value.get(field), str) for field in string_fields
        ):
            raise CandidateError("binding witness string field is invalid")
        expires_at = value.get("expires_at")
        if not isinstance(expires_at, int) or isinstance(expires_at, bool):
            raise CandidateError("binding witness expiry is invalid")
        state_version_set = value.get("state_versions")
        transform_version_set = value.get("transform_versions")
        if not _string_mapping(state_version_set):
            raise CandidateError(
                "binding witness state versions are invalid"
            )
        if not _string_mapping(transform_version_set):
            raise CandidateError(
                "binding witness transform versions are invalid"
            )
        return cls(
            candidate_id=value["candidate_id"],
            request_digest=value["request_digest"],
            tool_id=value["tool_id"],
            destination_scope=value["destination_scope"],
            derivation_digest=value["derivation_digest"],
            evidence_type=value["evidence_type"],
            value_digest=value["value_digest"],
            policy_epoch=value["policy_epoch"],
            contract_version=value["contract_version"],
            state_versions=dict(state_version_set),
            transform_versions=dict(transform_version_set),
            expires_at=expires_at,
            nonce=value["nonce"],
            mac=value["mac"],
        )


def _string_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    )


class HandleIssuer:
    def __init__(
        self,
        secret: bytes,
        *,
        now: Callable[[], float] = time.time,
        nonce_bytes: Callable[[int], bytes] = secrets.token_bytes,
    ) -> None:
        if len(secret) < 32:
            raise CandidateError("handle secret must contain at least 32 bytes")
        self._secret = bytes(secret)
        self._now = now
        self._nonce_bytes = nonce_bytes

    def issue(
        self,
        *,
        request_digest: str,
        tool_id: str,
        destination_scope: str,
        derivation_digest_value: str,
        evidence_type: str,
        value_digest: str,
        policy_epoch: str,
        contract_version: str,
        state_version_set: Mapping[str, str],
        transform_versions: Mapping[str, str],
        ttl_seconds: int,
    ) -> BindingWitness:
        if ttl_seconds <= 0:
            raise CandidateError("handle ttl must be positive")
        nonce_bytes = self._nonce_bytes(18)
        if len(nonce_bytes) < 12:
            raise CandidateError("nonce source returned insufficient entropy")
        nonce = _urlsafe(nonce_bytes)
        candidate_id = "ev_" + nonce[:16]
        unsigned = {
            "version": BINDING_WITNESS_VERSION,
            "candidate_id": candidate_id,
            "request_digest": request_digest,
            "tool_id": tool_id,
            "destination_scope": destination_scope,
            "derivation_digest": derivation_digest_value,
            "evidence_type": evidence_type,
            "value_digest": value_digest,
            "policy_epoch": policy_epoch,
            "contract_version": contract_version,
            "state_versions": dict(sorted(state_version_set.items())),
            "transform_versions": dict(sorted(transform_versions.items())),
            "expires_at": int(self._now()) + ttl_seconds,
            "nonce": nonce,
        }
        mac = hmac.new(
            self._secret,
            canonical_json(unsigned).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return BindingWitness(
            candidate_id=candidate_id,
            request_digest=request_digest,
            tool_id=tool_id,
            destination_scope=destination_scope,
            derivation_digest=derivation_digest_value,
            evidence_type=evidence_type,
            value_digest=value_digest,
            policy_epoch=policy_epoch,
            contract_version=contract_version,
            state_versions=dict(sorted(state_version_set.items())),
            transform_versions=dict(sorted(transform_versions.items())),
            expires_at=unsigned["expires_at"],
            nonce=nonce,
            mac=mac,
        )

    def verify(
        self,
        witness: BindingWitness,
        *,
        allow_expired: bool = False,
    ) -> None:
        expected = hmac.new(
            self._secret,
            canonical_json(witness.unsigned_dict()).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, witness.mac):
            raise CandidateError("binding witness authentication failed")
        if not allow_expired and witness.expires_at < int(self._now()):
            raise CandidateError("binding witness expired")


@dataclass(frozen=True)
class Candidate:
    witness: BindingWitness
    derivation: EvidenceDerivation
    value: Any
    display: str | None = None

    @property
    def candidate_id(self) -> str:
        return self.witness.candidate_id

    def public_view(self) -> dict[str, Any]:
        view: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "destination_scope": self.witness.destination_scope,
            "evidence_type": self.witness.evidence_type,
            "derivation": derivation_to_dict(self.derivation),
        }
        if self.display is not None:
            view["display"] = self.display
        return view


@dataclass(frozen=True)
class CandidateRejection:
    proposal_index: int
    tool_id: str
    destination_scope: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_index": self.proposal_index,
            "tool_id": self.tool_id,
            "destination_scope": self.destination_scope,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CandidateTable:
    request_digest: str
    candidates: Mapping[str, Candidate]
    rejections: tuple[CandidateRejection, ...]
    policy_epochs: Mapping[str, str]

    def candidate(self, candidate_id: str) -> Candidate:
        candidate = self.candidates.get(candidate_id)
        if candidate is None:
            raise CandidateError(f"unknown candidate id: {candidate_id}")
        return candidate

    def public_view(self) -> dict[str, Any]:
        return {
            "version": CANDIDATE_COMPILER_VERSION,
            "request_digest": self.request_digest,
            "candidates": [
                self.candidates[candidate_id].public_view()
                for candidate_id in sorted(self.candidates)
            ],
        }

    def metrics(self) -> dict[str, int]:
        return {
            "valid_candidate_count": len(self.candidates),
            "rejected_candidate_count": len(self.rejections),
            "proposed_candidate_count": (
                len(self.candidates) + len(self.rejections)
            ),
        }


def _validate_candidate(
    proposal: CandidateProposal,
    context: EvidenceContext,
    policy: PolicySet,
    evidence_types: EvidenceTypeRegistry,
    transforms: TransformRegistry,
) -> tuple[Any, SlotPolicy, dict[str, str], dict[str, str]]:
    tool_policy = policy.tool(proposal.tool_id)
    slot_policy = tool_policy.slot(proposal.destination_scope)
    reasons: list[str] = []

    if context.policy_epoch != tool_policy.policy_epoch:
        reasons.append("policy_epoch_mismatch")
    if proposal.evidence_type != slot_policy.evidence_type:
        reasons.append("evidence_type_mismatch")

    used_transforms = transform_ids(proposal.derivation)
    disallowed_transforms = used_transforms - slot_policy.transforms
    if disallowed_transforms:
        reasons.append(
            "transform_not_allowed:"
            + ",".join(sorted(disallowed_transforms))
        )

    try:
        origin_is_valid, unauthorized = origin_ok(
            proposal.derivation, context, slot_policy
        )
        if not origin_is_valid:
            reasons.append(
                "origin_not_allowed:" + ",".join(unauthorized)
            )
    except (PolicyError, ValueError) as exc:
        reasons.append(f"origin_invalid:{exc}")

    if reasons:
        raise CandidateError(";".join(reasons))

    value = evaluate_derivation(proposal.derivation, context, transforms)
    roots = root_derivations(proposal.derivation)
    for root in roots:
        if isinstance(root, StateRef):
            state_value = context.state_value(root.namespace, root.key)
            if state_value.evidence_type != slot_policy.evidence_type:
                raise CandidateError(
                    "state_evidence_type_mismatch:"
                    f"{state_value.evidence_type}"
                )
    root_kinds = {root_kind(root) for root in roots}
    evidence_type = evidence_types.get(slot_policy.evidence_type)
    if evidence_type.value_class != slot_policy.value_class:
        raise CandidateError("evidence_value_class_mismatch")
    if slot_policy.value_class in {AUTHORITY_BEARING, EFFECT_BEARING}:
        from .trust import assess_derivation_trust

        if assess_derivation_trust(
            proposal.derivation,
            context,
        ).contains_untrusted:
            raise CandidateError("untrusted_origin_for_sensitive_value")
    evidence_type.validate(value, root_kinds)
    versions = state_versions(proposal.derivation)
    transform_version_set = transforms.versions(used_transforms)
    return value, slot_policy, versions, transform_version_set


def compile_candidates(
    *,
    proposals: tuple[CandidateProposal, ...],
    context: EvidenceContext,
    policy: PolicySet,
    evidence_types: EvidenceTypeRegistry,
    transforms: TransformRegistry,
    issuer: HandleIssuer,
    handle_ttl_seconds: int = 300,
) -> CandidateTable:
    candidates: dict[str, Candidate] = {}
    rejections: list[CandidateRejection] = []

    for index, proposal in enumerate(proposals):
        try:
            (
                value,
                slot_policy,
                version_set,
                transform_version_set,
            ) = _validate_candidate(
                proposal,
                context,
                policy,
                evidence_types,
                transforms,
            )
            witness = issuer.issue(
                request_digest=context.request_digest,
                tool_id=proposal.tool_id,
                destination_scope=proposal.destination_scope,
                derivation_digest_value=derivation_digest(
                    proposal.derivation
                ),
                evidence_type=proposal.evidence_type,
                value_digest=sha256_digest(value),
                policy_epoch=context.policy_epoch,
                contract_version=policy.tool(
                    proposal.tool_id
                ).contract_version,
                state_version_set=version_set,
                transform_versions=transform_version_set,
                ttl_seconds=handle_ttl_seconds,
            )
            if witness.candidate_id in candidates:
                raise CandidateError("candidate id collision")
            candidates[witness.candidate_id] = Candidate(
                witness=witness,
                derivation=proposal.derivation,
                value=value,
                display=proposal.display,
            )
        except (CandidateError, PolicyError, ValueError) as exc:
            reasons = tuple(
                reason for reason in str(exc).split(";") if reason
            ) or ("candidate_invalid",)
            rejections.append(
                CandidateRejection(
                    proposal_index=index,
                    tool_id=proposal.tool_id,
                    destination_scope=proposal.destination_scope,
                    reasons=reasons,
                )
            )

    return CandidateTable(
        request_digest=context.request_digest,
        candidates=candidates,
        rejections=tuple(rejections),
        policy_epochs=policy.epochs(),
    )


def replay_candidate(
    candidate: Candidate,
    *,
    context: EvidenceContext,
    policy: PolicySet,
    evidence_types: EvidenceTypeRegistry,
    transforms: TransformRegistry,
    issuer: HandleIssuer,
    allow_expired: bool = True,
) -> Any:
    witness = candidate.witness
    issuer.verify(witness, allow_expired=allow_expired)
    if witness.request_digest != context.request_digest:
        raise CandidateError("candidate belongs to a different request")
    proposal = CandidateProposal(
        tool_id=witness.tool_id,
        destination_scope=witness.destination_scope,
        derivation=candidate.derivation,
        evidence_type=witness.evidence_type,
        display=candidate.display,
    )
    value, _, versions, transform_version_set = _validate_candidate(
        proposal,
        context,
        policy,
        evidence_types,
        transforms,
    )
    tool_policy = policy.tool(witness.tool_id)
    checks = {
        "derivation digest": (
            witness.derivation_digest,
            derivation_digest(candidate.derivation),
        ),
        "value digest": (witness.value_digest, sha256_digest(value)),
        "policy epoch": (witness.policy_epoch, tool_policy.policy_epoch),
        "tool contract": (
            witness.contract_version,
            tool_policy.contract_version,
        ),
        "state versions": (
            dict(witness.state_versions),
            dict(versions),
        ),
        "transform versions": (
            dict(witness.transform_versions),
            dict(transform_version_set),
        ),
    }
    for label, (recorded, replayed) in checks.items():
        if recorded != replayed:
            raise CandidateError(f"{label} changed during replay")
    return value

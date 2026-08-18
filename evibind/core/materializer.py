from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from .compiler import (
    BindingWitness,
    Candidate,
    CandidateError,
    CandidateTable,
    HandleIssuer,
    replay_candidate,
)
from .derivations import (
    EvidenceContext,
    TransformRegistry,
    derivation_from_dict,
    derivation_to_dict,
    sha256_digest,
)
from .evidence_types import EvidenceTypeRegistry
from .policy import PolicySet


ACTION_IR_VERSION = "evibind.action_ir.v2"
MATERIALIZER_VERSION = "evibind.materializer.v2"
MATERIALIZATION_CERTIFICATE_VERSION = "evibind.materialization_certificate.v2"
ACTION_MODES = {"call", "need_input", "no_tool"}


class MaterializationError(ValueError):
    pass


@dataclass(frozen=True)
class ActionProposal:
    mode: str
    tool_id: str | None = None
    bindings: Mapping[str, str] = field(default_factory=dict)
    arguments: Mapping[str, Any] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    reason: str | None = None
    model_text: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ACTION_MODES:
            raise MaterializationError(f"unsupported action mode: {self.mode}")
        if self.mode == "call":
            if not self.tool_id:
                raise MaterializationError("call mode requires a tool_id")
            if self.missing:
                raise MaterializationError("call mode cannot declare missing slots")
        elif self.bindings:
            raise MaterializationError(f"{self.mode} mode cannot contain bindings")
        if self.mode != "call" and self.arguments:
            raise MaterializationError(f"{self.mode} mode cannot contain arguments")
        if self.mode == "need_input":
            if not self.tool_id:
                raise MaterializationError("need_input mode requires a tool_id")
            if not self.missing:
                raise MaterializationError(
                    "need_input mode requires missing destinations"
                )
        if self.mode == "no_tool":
            if self.tool_id is not None:
                raise MaterializationError("no_tool mode cannot contain a tool_id")
            if self.missing:
                raise MaterializationError(
                    "no_tool mode cannot declare missing destinations"
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActionProposal:
        allowed = {
            "mode",
            "tool_id",
            "bindings",
            "arguments",
            "missing",
            "reason",
            "model_text",
        }
        unknown = set(value) - allowed
        if unknown:
            raise MaterializationError(
                "unknown Action IR fields: " + ", ".join(sorted(unknown))
            )
        raw_bindings = value.get("bindings", {})
        if not isinstance(raw_bindings, Mapping) or not all(
            isinstance(pointer, str) and isinstance(handle, str)
            for pointer, handle in raw_bindings.items()
        ):
            raise MaterializationError(
                "bindings must map JSON Pointers to candidate IDs"
            )
        raw_missing = value.get("missing", ())
        if not isinstance(raw_missing, (list, tuple)) or not all(
            isinstance(pointer, str) for pointer in raw_missing
        ):
            raise MaterializationError("missing must be an array of JSON Pointers")
        raw_arguments = value.get("arguments", {})
        if not isinstance(raw_arguments, Mapping):
            raise MaterializationError("arguments must be an object")
        tool_id = value.get("tool_id")
        reason = value.get("reason")
        model_text = value.get("model_text")
        if tool_id is not None and not isinstance(tool_id, str):
            raise MaterializationError("tool_id must be a string")
        if reason is not None and not isinstance(reason, str):
            raise MaterializationError("reason must be a string")
        if model_text is not None and not isinstance(model_text, str):
            raise MaterializationError("model_text must be a string")
        return cls(
            mode=str(value.get("mode", "")),
            tool_id=tool_id,
            bindings=dict(raw_bindings),
            arguments=deepcopy(dict(raw_arguments)),
            missing=tuple(raw_missing),
            reason=reason,
            model_text=model_text,
        )

    def to_dict(self, *, include_model_text: bool = True) -> dict[str, Any]:
        row: dict[str, Any] = {"mode": self.mode}
        if self.tool_id is not None:
            row["tool_id"] = self.tool_id
        if self.bindings:
            row["bindings"] = dict(sorted(self.bindings.items()))
        if self.arguments:
            row["arguments"] = deepcopy(dict(self.arguments))
        if self.missing:
            row["missing"] = list(self.missing)
        if self.reason is not None:
            row["reason"] = self.reason
        if include_model_text and self.model_text is not None:
            row["model_text"] = self.model_text
        return row


@dataclass(frozen=True)
class MaterializedAction:
    tool_id: str
    arguments: Mapping[str, Any]
    manifest_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "arguments": self.arguments,
            "manifest_digest": self.manifest_digest,
        }


@dataclass(frozen=True)
class MaterializationCertificate:
    request_digest: str
    tool_id: str
    bindings: Mapping[str, Candidate]
    manifest_digest: str
    contract_version: str
    literal_arguments: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": MATERIALIZATION_CERTIFICATE_VERSION,
            "materializer_version": MATERIALIZER_VERSION,
            "request_digest": self.request_digest,
            "tool_id": self.tool_id,
            "manifest_digest": self.manifest_digest,
            "contract_version": self.contract_version,
            "literal_arguments": deepcopy(dict(self.literal_arguments)),
            "bindings": {
                pointer: {
                    "witness": candidate.witness.to_dict(),
                    "derivation": derivation_to_dict(candidate.derivation),
                }
                for pointer, candidate in sorted(self.bindings.items())
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MaterializationCertificate:
        if value.get("version") != MATERIALIZATION_CERTIFICATE_VERSION:
            raise MaterializationError(
                "unsupported or missing materialization certificate version"
            )
        if value.get("materializer_version") != MATERIALIZER_VERSION:
            raise MaterializationError(
                "materialization certificate uses another materializer"
            )
        string_fields = (
            "request_digest",
            "tool_id",
            "manifest_digest",
            "contract_version",
        )
        if not all(isinstance(value.get(field), str) for field in string_fields):
            raise MaterializationError("materialization certificate field is invalid")
        raw_bindings = value.get("bindings")
        if not isinstance(raw_bindings, Mapping):
            raise MaterializationError(
                "materialization certificate bindings must be an object"
            )
        literal_arguments = value.get("literal_arguments", {})
        if not isinstance(literal_arguments, Mapping):
            raise MaterializationError(
                "materialization certificate literal_arguments must be an object"
            )
        bindings: dict[str, Candidate] = {}
        for pointer, raw_record in raw_bindings.items():
            if not isinstance(pointer, str) or not isinstance(raw_record, Mapping):
                raise MaterializationError(
                    "materialization certificate binding is invalid"
                )
            raw_witness = raw_record.get("witness")
            raw_derivation = raw_record.get("derivation")
            if not isinstance(raw_witness, Mapping) or not isinstance(
                raw_derivation, Mapping
            ):
                raise MaterializationError(
                    "certificate binding requires witness and derivation"
                )
            try:
                witness = BindingWitness.from_dict(raw_witness)
                derivation = derivation_from_dict(raw_derivation)
            except (CandidateError, ValueError) as exc:
                raise MaterializationError(str(exc)) from exc
            if witness.destination_scope != pointer:
                raise MaterializationError(
                    "certificate binding pointer does not match witness"
                )
            bindings[pointer] = Candidate(
                witness=witness,
                derivation=derivation,
                value=None,
            )
        return cls(
            request_digest=value["request_digest"],
            tool_id=value["tool_id"],
            bindings=bindings,
            manifest_digest=value["manifest_digest"],
            contract_version=value["contract_version"],
            literal_arguments=deepcopy(dict(literal_arguments)),
        )


def _pointer_tokens(pointer: str) -> tuple[str, ...]:
    if not pointer.startswith("/") or pointer == "/":
        raise MaterializationError(
            f"destination must be a non-root JSON Pointer: {pointer!r}"
        )
    tokens = tuple(
        token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")
    )
    if any(token == "" for token in tokens):
        raise MaterializationError(
            f"empty JSON Pointer segment is not supported: {pointer!r}"
        )
    return tokens


def _new_container(next_token: str) -> dict[str, Any] | list[Any]:
    return [] if next_token.isdigit() else {}


def _set_pointer(root: dict[str, Any], pointer: str, value: Any) -> None:
    tokens = _pointer_tokens(pointer)
    current: dict[str, Any] | list[Any] = root
    for index, token in enumerate(tokens):
        final = index == len(tokens) - 1
        next_token = tokens[index + 1] if not final else ""
        if isinstance(current, dict):
            if final:
                if token in current:
                    raise MaterializationError(
                        f"duplicate materialized destination: {pointer}"
                    )
                current[token] = value
                return
            expected = _new_container(next_token)
            existing = current.get(token)
            if existing is None:
                current[token] = expected
                current = expected
            elif isinstance(existing, type(expected)):
                current = existing
            else:
                raise MaterializationError(
                    f"conflicting destination structure: {pointer}"
                )
            continue

        if not token.isdigit():
            raise MaterializationError(
                f"array destination requires an index: {pointer}"
            )
        array_index = int(token)
        if array_index < 0 or array_index > len(current):
            raise MaterializationError(
                f"array destination must be contiguous: {pointer}"
            )
        if final:
            if array_index != len(current):
                raise MaterializationError(f"duplicate array destination: {pointer}")
            current.append(value)
            return
        expected = _new_container(next_token)
        if array_index == len(current):
            current.append(expected)
            current = expected
        else:
            existing = current[array_index]
            if not isinstance(existing, type(expected)):
                raise MaterializationError(f"conflicting array structure: {pointer}")
            current = existing


def _escape_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _literal_leaf_pointers(value: Any, *, prefix: str = "") -> tuple[str, ...]:
    if isinstance(value, Mapping):
        if not value and prefix:
            return (prefix,)
        output: list[str] = []
        for key, child in value.items():
            if not isinstance(key, str):
                raise MaterializationError("literal argument keys must be strings")
            pointer = prefix + "/" + _escape_pointer_token(key)
            output.extend(_literal_leaf_pointers(child, prefix=pointer))
        return tuple(output)
    if isinstance(value, list):
        if not value and prefix:
            return (prefix,)
        output = []
        for index, child in enumerate(value):
            output.extend(_literal_leaf_pointers(child, prefix=f"{prefix}/{index}"))
        return tuple(output)
    if not prefix:
        raise MaterializationError("literal arguments must be an object")
    return (prefix,)


def _paths_overlap(left: str, right: str) -> bool:
    return (
        left == right
        or left.startswith(right + "/")
        or right.startswith(left + "/")
    )


def materialize(
    proposal: ActionProposal,
    *,
    table: CandidateTable,
    context: EvidenceContext,
    policy: PolicySet,
    evidence_types: EvidenceTypeRegistry,
    transforms: TransformRegistry,
    issuer: HandleIssuer,
    literal_destinations: frozenset[str] = frozenset(),
    allow_expired: bool = False,
) -> tuple[MaterializedAction, MaterializationCertificate]:
    if proposal.mode != "call" or proposal.tool_id is None:
        raise MaterializationError("only call proposals can be materialized")
    if table.request_digest != context.request_digest:
        raise MaterializationError("candidate table belongs to a different request")
    tool_policy = policy.tool(proposal.tool_id)
    selected_destinations = frozenset(proposal.bindings)
    declared_destinations = frozenset(
        slot.destination_scope for slot in tool_policy.slots
    )
    unknown = selected_destinations - declared_destinations
    missing = tool_policy.required_destinations - selected_destinations
    if unknown:
        raise MaterializationError(
            "bindings target undeclared destinations: " + ", ".join(sorted(unknown))
        )
    if missing:
        raise MaterializationError(
            "required destinations lack bindings: " + ", ".join(sorted(missing))
        )

    literal_leaves = _literal_leaf_pointers(proposal.arguments)
    unauthorized_literals = sorted(
        literal
        for literal in literal_leaves
        if literal not in literal_destinations
    )
    if unauthorized_literals:
        raise MaterializationError(
            "model literals target unauthorized destinations: "
            + ", ".join(unauthorized_literals)
        )
    critical_conflicts = sorted(
        literal
        for literal in literal_leaves
        if any(
            _paths_overlap(literal, critical)
            for critical in declared_destinations
        )
    )
    if critical_conflicts:
        raise MaterializationError(
            "model literals overlap protected destinations: "
            + ", ".join(critical_conflicts)
        )

    arguments: dict[str, Any] = deepcopy(dict(proposal.arguments))
    bound: dict[str, Candidate] = {}
    for destination_scope, candidate_id in sorted(proposal.bindings.items()):
        candidate = table.candidate(candidate_id)
        witness = candidate.witness
        binding_checks = {
            "tool": (witness.tool_id, proposal.tool_id),
            "destination": (
                witness.destination_scope,
                destination_scope,
            ),
            "request": (
                witness.request_digest,
                context.request_digest,
            ),
            "policy epoch": (
                witness.policy_epoch,
                tool_policy.policy_epoch,
            ),
        }
        for label, (recorded, current) in binding_checks.items():
            if recorded != current:
                raise MaterializationError(f"candidate {label} binding mismatch")
        try:
            value = replay_candidate(
                candidate,
                context=context,
                policy=policy,
                evidence_types=evidence_types,
                transforms=transforms,
                issuer=issuer,
                allow_expired=allow_expired,
            )
        except CandidateError as exc:
            raise MaterializationError(str(exc)) from exc
        _set_pointer(arguments, destination_scope, value)
        bound[destination_scope] = candidate

    manifest = {
        "tool_id": proposal.tool_id,
        "arguments": arguments,
        "request_digest": context.request_digest,
        "contract_version": tool_policy.contract_version,
    }
    manifest_digest = sha256_digest(manifest)
    action = MaterializedAction(
        tool_id=proposal.tool_id,
        arguments=arguments,
        manifest_digest=manifest_digest,
    )
    certificate = MaterializationCertificate(
        request_digest=context.request_digest,
        tool_id=proposal.tool_id,
        bindings=bound,
        manifest_digest=manifest_digest,
        contract_version=tool_policy.contract_version,
        literal_arguments=deepcopy(dict(proposal.arguments)),
    )
    return action, certificate


def replay_materialization(
    certificate: MaterializationCertificate,
    *,
    context: EvidenceContext,
    policy: PolicySet,
    evidence_types: EvidenceTypeRegistry,
    transforms: TransformRegistry,
    issuer: HandleIssuer,
    literal_destinations: frozenset[str] = frozenset(),
    allow_expired: bool = True,
) -> MaterializedAction:
    proposal = ActionProposal(
        mode="call",
        tool_id=certificate.tool_id,
        bindings={
            pointer: candidate.candidate_id
            for pointer, candidate in certificate.bindings.items()
        },
        arguments=deepcopy(dict(certificate.literal_arguments)),
    )
    table = CandidateTable(
        request_digest=certificate.request_digest,
        candidates={
            candidate.candidate_id: candidate
            for candidate in certificate.bindings.values()
        },
        rejections=(),
        policy_epochs=policy.epochs(),
    )
    action, replayed = materialize(
        proposal,
        table=table,
        context=context,
        policy=policy,
        evidence_types=evidence_types,
        transforms=transforms,
        issuer=issuer,
        literal_destinations=literal_destinations,
        allow_expired=allow_expired,
    )
    if replayed.manifest_digest != certificate.manifest_digest:
        raise MaterializationError("materialized manifest changed during replay")
    if replayed.contract_version != certificate.contract_version:
        raise MaterializationError("joint contract version changed during replay")
    return action

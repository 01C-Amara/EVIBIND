from __future__ import annotations

import base64
import secrets
import time
from typing import Callable, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .compiler import BINDING_WITNESS_VERSION, BindingWitness, CandidateError
from .derivations import canonical_json


PUBLIC_KEY_AUTHENTICATOR = "ed25519"


def _urlsafe(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode_urlsafe(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value + padding)
    except (ValueError, TypeError) as exc:
        raise CandidateError("binding witness authentication failed") from exc


class Ed25519HandleVerifier:
    """Public-only verifier for binding witnesses.

    BindingWitness.mac is retained as the serialized authenticator field for
    backward compatibility; in this mode it carries a base64url Ed25519
    signature rather than an HMAC digest.
    """

    def __init__(
        self,
        public_key: Ed25519PublicKey | bytes,
        *,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._public_key = (
            public_key
            if isinstance(public_key, Ed25519PublicKey)
            else Ed25519PublicKey.from_public_bytes(bytes(public_key))
        )
        self._now = now

    @property
    def algorithm(self) -> str:
        return PUBLIC_KEY_AUTHENTICATOR

    def public_key_bytes(self) -> bytes:
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def verify(
        self,
        witness: BindingWitness,
        *,
        allow_expired: bool = False,
    ) -> None:
        try:
            self._public_key.verify(
                _decode_urlsafe(witness.mac),
                canonical_json(witness.unsigned_dict()).encode("utf-8"),
            )
        except (InvalidSignature, ValueError) as exc:
            raise CandidateError("binding witness authentication failed") from exc
        if not allow_expired and witness.expires_at < int(self._now()):
            raise CandidateError("binding witness expired")


class Ed25519HandleIssuer(Ed25519HandleVerifier):
    """Ed25519 witness issuer whose verifier needs only the public key."""

    def __init__(
        self,
        private_key: Ed25519PrivateKey | bytes,
        *,
        now: Callable[[], float] = time.time,
        nonce_bytes: Callable[[int], bytes] = secrets.token_bytes,
    ) -> None:
        self._private_key = (
            private_key
            if isinstance(private_key, Ed25519PrivateKey)
            else Ed25519PrivateKey.from_private_bytes(bytes(private_key))
        )
        self._nonce_bytes = nonce_bytes
        super().__init__(self._private_key.public_key(), now=now)

    def public_verifier(self) -> Ed25519HandleVerifier:
        return Ed25519HandleVerifier(self.public_key_bytes(), now=self._now)

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
        signature = _urlsafe(
            self._private_key.sign(canonical_json(unsigned).encode("utf-8"))
        )
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
            mac=signature,
        )

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from evibind.core.derivations import canonical_json, sha256_digest


from .nonce_store import (
    ConsumedNonceStore,
    InMemoryConsumedNonceStore,
)

EFFECT_AUTHORIZATION_VERSION = "evibind.effect_authorization.v1"
EFFECT_CLASSES = frozenset(
    {"read_only", "reversible_write", "external_write", "irreversible"}
)
CONFIRMATION_POLICIES = frozenset({"not_required", "required"})
_TOKEN_FIELDS = frozenset(
    {
        "version",
        "request_digest",
        "tool_id",
        "manifest_digest",
        "policy_epoch",
        "effect_class",
        "effect_policy_digest",
        "expires_at",
        "nonce",
    }
)


class EffectAuthorizationError(ValueError):
    pass


def _urlsafe_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _urlsafe_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, TypeError) as exc:
        raise EffectAuthorizationError(
            "effect confirmation token encoding is invalid"
        ) from exc


@dataclass(frozen=True)
class EffectPolicy:
    effect_class: str
    confirmation: str
    ttl_seconds: int = 300

    def __post_init__(self) -> None:
        if self.effect_class not in EFFECT_CLASSES:
            raise EffectAuthorizationError(
                f"unsupported effect class: {self.effect_class}"
            )
        if self.confirmation not in CONFIRMATION_POLICIES:
            raise EffectAuthorizationError(
                f"unsupported confirmation policy: {self.confirmation}"
            )
        if (
            isinstance(self.ttl_seconds, bool)
            or not isinstance(self.ttl_seconds, int)
            or not 1 <= self.ttl_seconds <= 900
        ):
            raise EffectAuthorizationError(
                "effect confirmation ttl_seconds must be between 1 and 900"
            )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect_class": self.effect_class,
            "confirmation": self.confirmation,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EffectPolicy:
        allowed = {"effect_class", "confirmation", "ttl_seconds"}
        unknown = set(value) - allowed
        if unknown:
            raise EffectAuthorizationError(
                "unknown effect policy fields: " + ", ".join(sorted(unknown))
            )
        effect_class = value.get("effect_class")
        if not isinstance(effect_class, str):
            raise EffectAuthorizationError("effect policy requires effect_class")
        confirmation = value.get("confirmation")
        if confirmation is None:
            confirmation = "not_required" if effect_class == "read_only" else "required"
        if not isinstance(confirmation, str):
            raise EffectAuthorizationError(
                "effect policy confirmation must be a string"
            )
        return cls(
            effect_class=effect_class,
            confirmation=confirmation,
            ttl_seconds=value.get("ttl_seconds", 300),
        )


def parse_effect_policies(
    options: Mapping[str, Any],
    *,
    tool_ids: frozenset[str],
) -> dict[str, EffectPolicy]:
    raw = options.get("effect_policies", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise EffectAuthorizationError("evibind.effect_policies must be an object")
    unknown_tools = sorted(str(tool_id) for tool_id in set(raw) - tool_ids)
    if unknown_tools:
        raise EffectAuthorizationError(
            "effect policies reference unknown tools: " + ", ".join(unknown_tools)
        )
    policies: dict[str, EffectPolicy] = {}
    for tool_id, value in raw.items():
        if not isinstance(tool_id, str) or not isinstance(value, Mapping):
            raise EffectAuthorizationError(
                "each effect policy must map a tool name to an object"
            )
        policies[tool_id] = EffectPolicy.from_dict(value)
    confirmation = options.get("effect_confirmation")
    if confirmation is not None and not isinstance(confirmation, str):
        raise EffectAuthorizationError("evibind.effect_confirmation must be a string")
    return policies


@dataclass(frozen=True)
class EffectChallenge:
    token: str
    tool_id: str
    manifest_digest: str
    effect_class: str
    expires_at: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": EFFECT_AUTHORIZATION_VERSION,
            "token": self.token,
            "tool_id": self.tool_id,
            "manifest_digest": self.manifest_digest,
            "effect_class": self.effect_class,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class EffectAuthorization:
    tool_id: str
    manifest_digest: str
    effect_class: str
    authorization_digest: str
    confirmed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": EFFECT_AUTHORIZATION_VERSION,
            "tool_id": self.tool_id,
            "manifest_digest": self.manifest_digest,
            "effect_class": self.effect_class,
            "authorization_digest": self.authorization_digest,
            "confirmed": self.confirmed,
        }


class EffectAuthorizer:
    def __init__(
        self,
        secret: bytes,
        *,
        now: Callable[[], float] = time.time,
        nonce_bytes: Callable[[int], bytes] = secrets.token_bytes,
        nonce_store: ConsumedNonceStore | None = None,
    ) -> None:
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise EffectAuthorizationError(
                "effect authorization secret must contain at least 32 bytes"
            )
        self._key = hmac.new(
            secret,
            b"evibind-effect-authorization-v1",
            hashlib.sha256,
        ).digest()
        self._now = now
        self._nonce_bytes = nonce_bytes
        self._nonce_store = (
            nonce_store if nonce_store is not None else InMemoryConsumedNonceStore()
        )

    def issue(
        self,
        *,
        request_digest: str,
        tool_id: str,
        manifest_digest: str,
        policy_epoch: str,
        policy: EffectPolicy,
    ) -> EffectChallenge:
        nonce_bytes = self._nonce_bytes(18)
        if len(nonce_bytes) < 12:
            raise EffectAuthorizationError(
                "effect authorization nonce source returned insufficient entropy"
            )
        expires_at = int(self._now()) + policy.ttl_seconds
        payload = {
            "version": EFFECT_AUTHORIZATION_VERSION,
            "request_digest": request_digest,
            "tool_id": tool_id,
            "manifest_digest": manifest_digest,
            "policy_epoch": policy_epoch,
            "effect_class": policy.effect_class,
            "effect_policy_digest": policy.digest,
            "expires_at": expires_at,
            "nonce": _urlsafe_encode(nonce_bytes),
        }
        encoded = _urlsafe_encode(canonical_json(payload).encode("utf-8"))
        signature = hmac.new(
            self._key,
            encoded.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        return EffectChallenge(
            token=encoded + "." + signature,
            tool_id=tool_id,
            manifest_digest=manifest_digest,
            effect_class=policy.effect_class,
            expires_at=expires_at,
        )

    def consume(
        self,
        token: str,
        *,
        request_digest: str,
        tool_id: str,
        manifest_digest: str,
        policy_epoch: str,
        policy: EffectPolicy,
    ) -> EffectAuthorization:
        if not isinstance(token, str) or not token or len(token) > 4096:
            raise EffectAuthorizationError("effect confirmation token is invalid")
        encoded, separator, supplied_signature = token.partition(".")
        if (
            not separator
            or not encoded
            or not supplied_signature
            or "." in supplied_signature
        ):
            raise EffectAuthorizationError("effect confirmation token is invalid")
        expected_signature = hmac.new(
            self._key,
            encoded.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected_signature, supplied_signature):
            raise EffectAuthorizationError(
                "effect confirmation token authentication failed"
            )
        try:
            raw_payload = _urlsafe_decode(encoded)
            payload = json.loads(raw_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EffectAuthorizationError(
                "effect confirmation token payload is invalid"
            ) from exc
        if (
            not isinstance(payload, Mapping)
            or set(payload) != _TOKEN_FIELDS
            or canonical_json(payload).encode("utf-8") != raw_payload
        ):
            raise EffectAuthorizationError(
                "effect confirmation token payload is not canonical"
            )
        string_fields = _TOKEN_FIELDS - {"expires_at"}
        if not all(isinstance(payload.get(field), str) for field in string_fields):
            raise EffectAuthorizationError("effect confirmation token field is invalid")
        expires_at = payload.get("expires_at")
        if isinstance(expires_at, bool) or not isinstance(expires_at, int):
            raise EffectAuthorizationError(
                "effect confirmation token expiry is invalid"
            )
        if payload["version"] != EFFECT_AUTHORIZATION_VERSION:
            raise EffectAuthorizationError(
                "unsupported effect confirmation token version"
            )
        if expires_at < int(self._now()):
            raise EffectAuthorizationError("effect confirmation token expired")
        checks = {
            "request": (payload["request_digest"], request_digest),
            "tool": (payload["tool_id"], tool_id),
            "manifest": (payload["manifest_digest"], manifest_digest),
            "policy epoch": (payload["policy_epoch"], policy_epoch),
            "effect class": (payload["effect_class"], policy.effect_class),
            "effect policy": (payload["effect_policy_digest"], policy.digest),
        }
        for label, (recorded, current) in checks.items():
            if recorded != current:
                raise EffectAuthorizationError(
                    f"effect confirmation {label} binding mismatch"
                )
        nonce = payload["nonce"]
        if not self._nonce_store.consume_once(
            nonce=nonce,
            expires_at=expires_at,
            now=int(self._now()),
        ):
            raise EffectAuthorizationError(
                "effect confirmation token was already consumed"
            )
        return EffectAuthorization(
            tool_id=tool_id,
            manifest_digest=manifest_digest,
            effect_class=policy.effect_class,
            authorization_digest=hashlib.sha256(token.encode("utf-8")).hexdigest(),
        )


def gate_effect(
    protected_response: Mapping[str, Any],
    *,
    request_digest: str,
    policy_epoch: str,
    policies: Mapping[str, EffectPolicy],
    confirmation_token: str | None,
    authorizer: EffectAuthorizer,
) -> dict[str, Any]:
    protected = deepcopy(dict(protected_response))
    metadata = protected.get("evibind")
    choices = protected.get("choices")
    if (
        not isinstance(metadata, dict)
        or not isinstance(metadata.get("choices"), list)
        or len(metadata["choices"]) != 1
        or not isinstance(choices, list)
        or len(choices) != 1
    ):
        raise EffectAuthorizationError(
            "effect gate requires one protected response choice"
        )
    summary = metadata["choices"][0]
    choice = choices[0]
    if not isinstance(summary, dict) or not isinstance(choice, dict):
        raise EffectAuthorizationError("effect gate response choice is invalid")
    if not summary.get("released"):
        if confirmation_token is not None:
            summary["effect_confirmation_unused"] = True
        return protected

    tool_id = summary.get("tool")
    manifest_digest = summary.get("manifest_digest")
    if not isinstance(tool_id, str) or not isinstance(manifest_digest, str):
        raise EffectAuthorizationError("released response omitted effect bindings")
    policy = policies.get(tool_id)
    if policy is None:
        if confirmation_token is not None:
            raise EffectAuthorizationError(
                "effect confirmation does not apply to the released tool"
            )
        return protected
    effect: dict[str, Any] = {
        "version": EFFECT_AUTHORIZATION_VERSION,
        **policy.to_dict(),
        "policy_digest": policy.digest,
    }
    if policy.confirmation == "not_required":
        if confirmation_token is not None:
            raise EffectAuthorizationError(
                "effect confirmation does not apply to a not_required policy"
            )
        effect["status"] = "not_required"
        summary["effect"] = effect
        return protected

    if confirmation_token is None:
        challenge = authorizer.issue(
            request_digest=request_digest,
            tool_id=tool_id,
            manifest_digest=manifest_digest,
            policy_epoch=policy_epoch,
            policy=policy,
        )
        message = choice.get("message")
        if not isinstance(message, dict):
            raise EffectAuthorizationError(
                "released response omitted assistant message"
            )
        message.pop("tool_calls", None)
        message.pop("function_call", None)
        message["content"] = (
            "Confirmation is required before this "
            + policy.effect_class.replace("_", " ")
            + " action can be released."
        )
        choice["finish_reason"] = "stop"
        summary["released"] = False
        summary["materialized_decision"] = summary.get("decision")
        summary["decision"] = "confirmation_required"
        effect["status"] = "awaiting_confirmation"
        effect["challenge"] = challenge.to_dict()
        summary["effect"] = effect
        return protected

    authorization = authorizer.consume(
        confirmation_token,
        request_digest=request_digest,
        tool_id=tool_id,
        manifest_digest=manifest_digest,
        policy_epoch=policy_epoch,
        policy=policy,
    )
    effect["status"] = "confirmed"
    effect["authorization"] = authorization.to_dict()
    summary["effect"] = effect
    return protected

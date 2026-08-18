from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from datetime import date
from email.headerregistry import Address
from pathlib import PurePosixPath
from typing import Any, Callable
from urllib.parse import urlsplit


EVIDENCE_TYPE_REGISTRY_VERSION = "evibind.evidence_types.v2"

AUTHORITY_BEARING = "authority_bearing"
OPAQUE_CONTENT = "opaque_content"
EFFECT_BEARING = "effect_bearing"
VALUE_CLASSES = {AUTHORITY_BEARING, OPAQUE_CONTENT, EFFECT_BEARING}


class EvidenceTypeError(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceType:
    name: str
    validator: Callable[[Any], bool]
    value_class: str = AUTHORITY_BEARING
    allowed_root_kinds: frozenset[str] = frozenset(
        {"span", "state_ref", "default", "enum_value"}
    )
    description: str = ""

    def validate(self, value: Any, root_kinds: set[str]) -> None:
        if not root_kinds:
            raise EvidenceTypeError(f"{self.name} requires at least one evidence root")
        unsupported = root_kinds - self.allowed_root_kinds
        if unsupported:
            raise EvidenceTypeError(
                f"{self.name} does not admit roots: {', '.join(sorted(unsupported))}"
            )
        try:
            valid = self.validator(value)
        except Exception as exc:
            raise EvidenceTypeError(f"{self.name} validator failed") from exc
        if not valid:
            raise EvidenceTypeError(f"value does not satisfy evidence type {self.name}")


class EvidenceTypeRegistry:
    def __init__(self, evidence_types: tuple[EvidenceType, ...] = ()) -> None:
        self._types: dict[str, EvidenceType] = {}
        for evidence_type in evidence_types:
            self.register(evidence_type)

    def register(self, evidence_type: EvidenceType) -> None:
        if evidence_type.value_class not in VALUE_CLASSES:
            raise EvidenceTypeError(
                f"unsupported value class: {evidence_type.value_class}"
            )
        if not evidence_type.name:
            raise EvidenceTypeError("evidence type name is required")
        if evidence_type.name in self._types:
            raise EvidenceTypeError(f"duplicate evidence type: {evidence_type.name}")
        self._types[evidence_type.name] = evidence_type

    def get(self, name: str) -> EvidenceType:
        evidence_type = self._types.get(name)
        if evidence_type is None:
            raise EvidenceTypeError(f"unknown evidence type: {name}")
        return evidence_type

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._types))

    @classmethod
    def standard(cls) -> EvidenceTypeRegistry:
        def nonempty_string(value: Any) -> bool:
            return isinstance(value, str) and bool(value.strip())

        def uuid_value(value: Any) -> bool:
            if not isinstance(value, str):
                return False
            parsed = uuid.UUID(value)
            return str(parsed) == value.casefold()

        def email_address(value: Any) -> bool:
            if not isinstance(value, str) or value.count("@") != 1:
                return False
            local, domain = value.rsplit("@", 1)
            parsed = Address(addr_spec=value)
            return (
                bool(local)
                and "." in domain
                and parsed.addr_spec.casefold() == value.casefold()
            )

        def phone_number(value: Any) -> bool:
            if not isinstance(value, str):
                return False
            if not re.fullmatch(r"\+?[\d\s().-]+", value):
                return False
            digit_count = len(re.sub(r"\D", "", value))
            return 7 <= digit_count <= 15

        def iso_date(value: Any) -> bool:
            if not isinstance(value, str):
                return False
            return date.fromisoformat(value).isoformat() == value

        def relative_date(value: Any) -> bool:
            return (
                isinstance(value, dict)
                and isinstance(value.get("expression"), str)
                and bool(value["expression"].strip())
                and isinstance(value.get("resolved"), str)
                and iso_date(value["resolved"])
                and isinstance(value.get("clock_version"), str)
            )

        def currency_amount(value: Any) -> bool:
            return (
                isinstance(value, dict)
                and isinstance(value.get("amount"), (int, float))
                and not isinstance(value.get("amount"), bool)
                and math.isfinite(value["amount"])
                and isinstance(value.get("currency"), str)
                and bool(re.fullmatch(r"[A-Z]{3}", value["currency"]))
            )

        def uri(value: Any) -> bool:
            if not isinstance(value, str):
                return False
            parsed = urlsplit(value)
            return bool(
                parsed.scheme
                and (
                    parsed.netloc
                    or parsed.scheme in {"mailto", "urn", "file", "s3", "gs"}
                )
            )

        def repository_path(value: Any) -> bool:
            if not isinstance(value, str) or not value or "\x00" in value:
                return False
            path = PurePosixPath(value)
            return not path.is_absolute() and ".." not in path.parts

        def cloud_resource_id(value: Any) -> bool:
            return nonempty_string(value) and (
                str(value).startswith(("arn:", "projects/", "/subscriptions/"))
                or bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9._:/-]{5,}", str(value)))
            )

        def schema_enum(value: Any) -> bool:
            return (
                isinstance(value, (str, int, float, bool))
                and value is not None
                and not (isinstance(value, float) and not math.isfinite(value))
            )

        def integer_value(value: Any) -> bool:
            return isinstance(value, int) and not isinstance(value, bool)

        def number_value(value: Any) -> bool:
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )

        def boolean_value(value: Any) -> bool:
            return isinstance(value, bool)

        def opaque_content(value: Any) -> bool:
            return isinstance(value, str)

        def effect_manifest(value: Any) -> bool:
            return (
                isinstance(value, dict)
                and isinstance(value.get("manifest_id"), str)
                and isinstance(value.get("effect_digest"), str)
                and len(value["effect_digest"]) == 64
            )

        reference_roots = frozenset({"span", "state_ref", "enum_value"})
        registry_roots = frozenset({"state_ref", "enum_value"})
        return cls(
            (
                EvidenceType("uuid", uuid_value),
                EvidenceType("email_address", email_address),
                EvidenceType("phone_number", phone_number),
                EvidenceType("iso_date", iso_date),
                EvidenceType("relative_date", relative_date),
                EvidenceType("currency_amount", currency_amount),
                EvidenceType("uri", uri),
                EvidenceType("repository_path", repository_path),
                EvidenceType("cloud_resource_id", cloud_resource_id),
                EvidenceType("integer", integer_value),
                EvidenceType("number", number_value),
                EvidenceType("boolean", boolean_value),
                EvidenceType(
                    "order_ref",
                    nonempty_string,
                    allowed_root_kinds=reference_roots,
                ),
                EvidenceType(
                    "event_ref",
                    nonempty_string,
                    allowed_root_kinds=reference_roots,
                ),
                EvidenceType(
                    "person_ref",
                    nonempty_string,
                    allowed_root_kinds=reference_roots,
                ),
                EvidenceType(
                    "account_ref",
                    nonempty_string,
                    allowed_root_kinds=reference_roots,
                ),
                EvidenceType(
                    "schema_enum",
                    schema_enum,
                    allowed_root_kinds=frozenset({"span", "enum_value", "default"}),
                ),
                EvidenceType(
                    "opaque_registry_id",
                    nonempty_string,
                    allowed_root_kinds=reference_roots,
                ),
                EvidenceType(
                    "opaque_content",
                    opaque_content,
                    value_class=OPAQUE_CONTENT,
                ),
                EvidenceType(
                    "effect_manifest",
                    effect_manifest,
                    value_class=EFFECT_BEARING,
                    allowed_root_kinds=frozenset(
                        {"state_ref", "default", "enum_value"}
                    ),
                ),
            )
        )

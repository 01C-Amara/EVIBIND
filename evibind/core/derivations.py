from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, TypeAlias


DERIVATION_IR_VERSION = "evibind.derivation.v1"
TRANSFORM_REGISTRY_VERSION = "evibind.transforms.v1"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MessageEvidence:
    message_id: str
    role: str
    content: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "source": self.source,
        }


@dataclass(frozen=True)
class StateValue:
    namespace: str
    key: str
    version: str
    value: Any
    evidence_type: str


@dataclass(frozen=True)
class EvidenceContext:
    messages: tuple[MessageEvidence, ...]
    state: Mapping[tuple[str, str], StateValue] = field(default_factory=dict)
    defaults: Mapping[str, Any] = field(default_factory=dict)
    enums: Mapping[tuple[str, str], Any] = field(default_factory=dict)
    policy_epoch: str = "1"

    @property
    def request_digest(self) -> str:
        return sha256_digest(
            {
                "messages": [message.to_dict() for message in self.messages],
                "version": DERIVATION_IR_VERSION,
            }
        )

    def message(self, message_id: str) -> MessageEvidence:
        matches = [
            message for message in self.messages if message.message_id == message_id
        ]
        if len(matches) != 1:
            raise DerivationError(
                f"message_id must resolve exactly once: {message_id!r}"
            )
        return matches[0]

    def state_value(self, namespace: str, key: str) -> StateValue:
        value = self.state.get((namespace, key))
        if value is None:
            raise DerivationError(f"state value not found: {namespace}.{key}")
        return value


@dataclass(frozen=True)
class Span:
    message_id: str
    byte_start: int
    byte_end: int
    parser: str = "identity"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "span",
            "message_id": self.message_id,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "parser": self.parser,
        }


@dataclass(frozen=True)
class StateRef:
    namespace: str
    key: str
    version: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": "state_ref",
            "namespace": self.namespace,
            "key": self.key,
            "version": self.version,
        }


@dataclass(frozen=True)
class Default:
    default_id: str
    policy_epoch: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": "default",
            "default_id": self.default_id,
            "policy_epoch": self.policy_epoch,
        }


@dataclass(frozen=True)
class EnumValue:
    schema_id: str
    enum_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": "enum_value",
            "schema_id": self.schema_id,
            "enum_id": self.enum_id,
        }


@dataclass(frozen=True)
class Apply:
    transform_id: str
    arguments: tuple[EvidenceDerivation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "apply",
            "transform_id": self.transform_id,
            "arguments": [derivation_to_dict(item) for item in self.arguments],
        }


@dataclass(frozen=True)
class TupleDerivation:
    items: tuple[EvidenceDerivation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "tuple",
            "items": [derivation_to_dict(item) for item in self.items],
        }


@dataclass(frozen=True)
class ArrayDerivation:
    items: tuple[EvidenceDerivation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "array",
            "items": [derivation_to_dict(item) for item in self.items],
        }


EvidenceDerivation: TypeAlias = (
    Span | StateRef | Default | EnumValue | Apply | TupleDerivation | ArrayDerivation
)


class DerivationError(ValueError):
    pass


@dataclass(frozen=True)
class Transform:
    transform_id: str
    version: str
    function: Callable[..., Any]
    pure: bool = True


class TransformRegistry:
    def __init__(self, transforms: tuple[Transform, ...] = ()) -> None:
        self._transforms: dict[str, Transform] = {}
        for transform in transforms:
            self.register(transform)

    @classmethod
    def standard(cls) -> TransformRegistry:
        def identity(value: Any) -> Any:
            return value

        def strip(value: Any) -> str:
            if not isinstance(value, str):
                raise DerivationError("strip requires a string")
            return value.strip()

        def lowercase(value: Any) -> str:
            if not isinstance(value, str):
                raise DerivationError("lowercase requires a string")
            return value.casefold()

        def parse_integer(value: Any) -> int:
            if not isinstance(value, str):
                raise DerivationError("parse_integer requires a string")
            try:
                return int(value.strip())
            except ValueError as exc:
                raise DerivationError("invalid integer evidence") from exc

        def parse_number(value: Any) -> int | float:
            if not isinstance(value, str):
                raise DerivationError("parse_number requires a string")
            try:
                parsed = float(value.strip())
            except ValueError as exc:
                raise DerivationError("invalid numeric evidence") from exc
            return int(parsed) if parsed.is_integer() else parsed

        def parse_boolean(value: Any) -> bool:
            if not isinstance(value, str):
                raise DerivationError("parse_boolean requires a string")
            normalized = value.strip().casefold()
            if normalized in {"true", "yes"}:
                return True
            if normalized in {"false", "no"}:
                return False
            raise DerivationError("invalid boolean evidence")

        def tuple_object(keys: Any, values: Any) -> dict[str, Any]:
            if not isinstance(keys, list) or not all(
                isinstance(key, str) for key in keys
            ):
                raise DerivationError("tuple_object keys must be a string array")
            if not isinstance(values, list) or len(keys) != len(values):
                raise DerivationError(
                    "tuple_object values must match the number of keys"
                )
            return dict(zip(keys, values, strict=True))

        return cls(
            (
                Transform("identity", "1", identity),
                Transform("strip", "1", strip),
                Transform("lowercase", "1", lowercase),
                Transform("parse_integer", "1", parse_integer),
                Transform("parse_number", "1", parse_number),
                Transform("parse_boolean", "1", parse_boolean),
                Transform("tuple_object", "1", tuple_object),
            )
        )

    def register(self, transform: Transform) -> None:
        if not transform.pure:
            raise DerivationError(
                f"transform must be declared pure: {transform.transform_id}"
            )
        if not transform.transform_id or not transform.version:
            raise DerivationError("transform id and version are required")
        if transform.transform_id in self._transforms:
            raise DerivationError(f"duplicate transform id: {transform.transform_id}")
        self._transforms[transform.transform_id] = transform

    def get(self, transform_id: str) -> Transform:
        transform = self._transforms.get(transform_id)
        if transform is None:
            raise DerivationError(f"unknown transform: {transform_id}")
        return transform

    def apply(self, transform_id: str, *arguments: Any) -> Any:
        transform = self.get(transform_id)
        try:
            return transform.function(*arguments)
        except DerivationError:
            raise
        except Exception as exc:
            raise DerivationError(f"transform failed: {transform_id}") from exc

    def versions(self, transform_ids: set[str]) -> dict[str, str]:
        return {
            transform_id: self.get(transform_id).version
            for transform_id in sorted(transform_ids)
        }


def derivation_to_dict(derivation: EvidenceDerivation) -> dict[str, Any]:
    row = derivation.to_dict()
    return {"ir_version": DERIVATION_IR_VERSION, **row}


def derivation_from_dict(value: Mapping[str, Any]) -> EvidenceDerivation:
    if value.get("ir_version") != DERIVATION_IR_VERSION:
        raise DerivationError("unsupported or missing derivation IR version")
    kind = value.get("kind")
    if kind == "span":
        message_id = value.get("message_id")
        byte_start = value.get("byte_start")
        byte_end = value.get("byte_end")
        parser = value.get("parser")
        if (
            not isinstance(message_id, str)
            or not isinstance(byte_start, int)
            or isinstance(byte_start, bool)
            or not isinstance(byte_end, int)
            or isinstance(byte_end, bool)
            or not isinstance(parser, str)
        ):
            raise DerivationError("invalid span derivation")
        return Span(message_id, byte_start, byte_end, parser)
    if kind == "state_ref":
        namespace = value.get("namespace")
        key = value.get("key")
        version = value.get("version")
        if not all(isinstance(item, str) for item in (namespace, key, version)):
            raise DerivationError("invalid state reference derivation")
        return StateRef(namespace, key, version)
    if kind == "default":
        default_id = value.get("default_id")
        policy_epoch = value.get("policy_epoch")
        if not isinstance(default_id, str) or not isinstance(policy_epoch, str):
            raise DerivationError("invalid default derivation")
        return Default(default_id, policy_epoch)
    if kind == "enum_value":
        schema_id = value.get("schema_id")
        enum_id = value.get("enum_id")
        if not isinstance(schema_id, str) or not isinstance(enum_id, str):
            raise DerivationError("invalid enum derivation")
        return EnumValue(schema_id, enum_id)
    if kind == "apply":
        transform_id = value.get("transform_id")
        arguments = value.get("arguments")
        if not isinstance(transform_id, str) or not isinstance(arguments, list):
            raise DerivationError("invalid apply derivation")
        return Apply(
            transform_id,
            tuple(_parse_child(item) for item in arguments),
        )
    if kind in {"tuple", "array"}:
        items = value.get("items")
        if not isinstance(items, list):
            raise DerivationError(f"invalid {kind} derivation")
        parsed = tuple(_parse_child(item) for item in items)
        if kind == "tuple":
            return TupleDerivation(parsed)
        return ArrayDerivation(parsed)
    raise DerivationError(f"unsupported derivation kind: {kind!r}")


def _parse_child(value: Any) -> EvidenceDerivation:
    if not isinstance(value, Mapping):
        raise DerivationError("child derivation must be an object")
    return derivation_from_dict(value)


def derivation_digest(derivation: EvidenceDerivation) -> str:
    return sha256_digest(derivation_to_dict(derivation))


def derivation_children(
    derivation: EvidenceDerivation,
) -> tuple[EvidenceDerivation, ...]:
    if isinstance(derivation, Apply):
        return derivation.arguments
    if isinstance(derivation, (TupleDerivation, ArrayDerivation)):
        return derivation.items
    return ()


def root_derivations(
    derivation: EvidenceDerivation,
) -> tuple[Span | StateRef | Default | EnumValue, ...]:
    children = derivation_children(derivation)
    if not children:
        if not isinstance(derivation, (Span, StateRef, Default, EnumValue)):
            raise DerivationError("derivation has no admissible evidence root")
        return (derivation,)
    roots: list[Span | StateRef | Default | EnumValue] = []
    for child in children:
        roots.extend(root_derivations(child))
    return tuple(roots)


def transform_ids(derivation: EvidenceDerivation) -> set[str]:
    identifiers: set[str] = set()
    if isinstance(derivation, Span):
        identifiers.add(derivation.parser)
    elif isinstance(derivation, Apply):
        identifiers.add(derivation.transform_id)
    for child in derivation_children(derivation):
        identifiers.update(transform_ids(child))
    return identifiers


def state_versions(derivation: EvidenceDerivation) -> dict[str, str]:
    versions: dict[str, str] = {}
    for root in root_derivations(derivation):
        if isinstance(root, StateRef):
            identity = f"{root.namespace}.{root.key}"
            previous = versions.get(identity)
            if previous is not None and previous != root.version:
                raise DerivationError(
                    f"conflicting state versions in derivation: {identity}"
                )
            versions[identity] = root.version
    return dict(sorted(versions.items()))


def evaluate_derivation(
    derivation: EvidenceDerivation,
    context: EvidenceContext,
    transforms: TransformRegistry,
) -> Any:
    if isinstance(derivation, Span):
        message = context.message(derivation.message_id)
        encoded = message.content.encode("utf-8")
        if (
            derivation.byte_start < 0
            or derivation.byte_end <= derivation.byte_start
            or derivation.byte_end > len(encoded)
        ):
            raise DerivationError(
                f"invalid byte span for {derivation.message_id}: "
                f"{derivation.byte_start}:{derivation.byte_end}"
            )
        try:
            value = encoded[derivation.byte_start : derivation.byte_end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DerivationError("span splits a UTF-8 code point") from exc
        return transforms.apply(derivation.parser, value)

    if isinstance(derivation, StateRef):
        state_value = context.state_value(derivation.namespace, derivation.key)
        if state_value.version != derivation.version:
            raise DerivationError(
                "stale state reference: "
                f"{derivation.namespace}.{derivation.key} "
                f"expected {derivation.version}, got {state_value.version}"
            )
        return state_value.value

    if isinstance(derivation, Default):
        if derivation.policy_epoch != context.policy_epoch:
            raise DerivationError("default is bound to a different policy epoch")
        if derivation.default_id not in context.defaults:
            raise DerivationError(f"default not found: {derivation.default_id}")
        return context.defaults[derivation.default_id]

    if isinstance(derivation, EnumValue):
        key = (derivation.schema_id, derivation.enum_id)
        if key not in context.enums:
            raise DerivationError(
                f"enum value not found: {derivation.schema_id}.{derivation.enum_id}"
            )
        return context.enums[key]

    if isinstance(derivation, Apply):
        values = tuple(
            evaluate_derivation(argument, context, transforms)
            for argument in derivation.arguments
        )
        return transforms.apply(derivation.transform_id, *values)

    if isinstance(derivation, TupleDerivation):
        return [
            evaluate_derivation(item, context, transforms) for item in derivation.items
        ]

    if isinstance(derivation, ArrayDerivation):
        return [
            evaluate_derivation(item, context, transforms) for item in derivation.items
        ]

    raise DerivationError(f"unsupported derivation object: {type(derivation).__name__}")

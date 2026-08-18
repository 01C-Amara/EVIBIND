from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from evibind.core import (
    ActionProposal,
    CandidateProposal,
    CandidateTable,
    Default,
    EvidenceContext,
    EvidenceTypeRegistry,
    HandleIssuer,
    MessageEvidence,
    PolicySet,
    SlotPolicy,
    Span,
    StateRef,
    StateValue,
    ToolPolicy,
    TransformRegistry,
    assess_derivation_trust,
    combine_trust_assessments,
    compile_candidates,
    materialize,
)
from evibind.core.derivations import sha256_digest

from .audit_mode import audit_native_response
from .json_contract import json_contract_accepts
from .verified_ranker import LinearCandidateRanker, prune_candidate_table


ONE_CALL_CONTROLLER_VERSION = "evibind.one_call_controller.v1"
ACTION_TOOL_NAME = "evibind_action"


class OneCallError(ValueError):
    pass


@dataclass(frozen=True)
class SlotDescriptor:
    tool_id: str
    destination_scope: str
    schema: Mapping[str, Any]
    required: bool


@dataclass(frozen=True)
class OneCallSession:
    upstream_payload: Mapping[str, Any]
    context: EvidenceContext
    policy: PolicySet
    candidates: CandidateTable
    issuer: HandleIssuer
    evidence_types: EvidenceTypeRegistry
    transforms: TransformRegistry
    tools: Mapping[str, Mapping[str, Any]]
    literal_destinations: Mapping[str, frozenset[str]]
    action_interface: str
    index_manifest: Mapping[int, Mapping[str, Any]]
    include_diagnostics: bool = False
    operating_mode: str = "enforce"

    def protect(self, upstream_response: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(upstream_response, Mapping):
            raise OneCallError("upstream response must be an object")
        protected = deepcopy(dict(upstream_response))
        choices = protected.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise OneCallError(
                "one-call controller requires exactly one upstream choice"
            )
        choice = choices[0]
        if not isinstance(choice, dict):
            raise OneCallError("upstream choice must be an object")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise OneCallError("upstream choice omitted an assistant message")

        try:
            if self.action_interface == "indexed":
                action, call_id = _parse_indexed_action_proposal(
                    message,
                    self.index_manifest,
                    self.policy,
                    self.candidates,
                )
            else:
                action, call_id = _parse_action_proposal(message)
            summary = self._apply_action(
                protected,
                choice,
                message,
                action,
                call_id,
            )
        except (OneCallError, ValueError) as exc:
            message.pop("tool_calls", None)
            message.pop("function_call", None)
            message["content"] = (
                "I could not construct a supported tool call. Please restate "
                "the target values explicitly."
                if self.operating_mode == "assist"
                else "EviBind withheld an invalid Action IR tool call."
            )
            choice["finish_reason"] = "stop"
            summary = {
                "index": int(choice.get("index", 0)),
                "released": False,
                "decision": "invalid_action_ir",
                "reason": str(exc),
                "model_calls": 1,
                **self.candidates.metrics(),
            }

        protected["evibind"] = {
            "version": ONE_CALL_CONTROLLER_VERSION,
            "operating_mode": self.operating_mode,
            "enforced": True,
            "action_representation": (
                "destination-bound evidence handles; trusted materialization"
            ),
            "selective_guarantee": (
                "Released critical values were materialized only from "
                "request-scoped, destination-bound evidence derivations."
            ),
            "choices": [summary],
        }
        return protected

    def audit(self, upstream_response: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return audit_native_response(
                upstream_response,
                policy=self.policy,
                candidates=self.candidates,
                tools=self.tools,
            )
        except ValueError as exc:
            raise OneCallError(str(exc)) from exc

    def _apply_action(
        self,
        protected: dict[str, Any],
        choice: dict[str, Any],
        message: dict[str, Any],
        action: ActionProposal,
        call_id: str | None,
    ) -> dict[str, Any]:
        base = {
            "index": int(choice.get("index", 0)),
            "model_calls": 1,
            **self.candidates.metrics(),
        }
        if action.mode == "call":
            allowed_literals = self.literal_destinations.get(
                action.tool_id or "", frozenset()
            )
            raw_tool = self.tools.get(action.tool_id or "", {})
            literal_schema = _literal_argument_schema(
                raw_tool.get("parameters", {}),
                allowed_literals,
            )
            if literal_schema is None:
                if action.arguments:
                    raise OneCallError(
                        "model literals target no authorized opaque destinations"
                    )
            elif not json_contract_accepts(action.arguments, literal_schema):
                raise OneCallError(
                    "model literals failed the authorized opaque-content contract"
                )
            materialized, certificate = materialize(
                action,
                table=self.candidates,
                context=self.context,
                policy=self.policy,
                evidence_types=self.evidence_types,
                transforms=self.transforms,
                issuer=self.issuer,
                literal_destinations=allowed_literals,
            )
            schema = self.tools.get(materialized.tool_id)
            if schema is None or not json_contract_accepts(
                materialized.arguments,
                schema.get("parameters", {}),
            ):
                raise OneCallError("materialized action failed the joint JSON contract")
            function_call: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": materialized.tool_id,
                    "arguments": json.dumps(
                        materialized.arguments,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                },
            }
            if call_id:
                function_call["id"] = call_id
            message["tool_calls"] = [function_call]
            message.pop("function_call", None)
            message["content"] = None
            choice["finish_reason"] = "tool_calls"
            certificate_row = certificate.to_dict()
            summary: dict[str, Any] = {
                **base,
                "released": True,
                "decision": "call",
                "tool": materialized.tool_id,
                "manifest_digest": materialized.manifest_digest,
                "certificate_digest": sha256_digest(certificate_row),
                "trust": combine_trust_assessments(
                    tuple(
                        assess_derivation_trust(candidate.derivation, self.context)
                        for candidate in certificate.bindings.values()
                    )
                ).to_dict(),
            }
            if self.include_diagnostics:
                summary["certificate"] = certificate_row
                summary["candidate_table"] = self.candidates.public_view()
                summary["candidate_rejections"] = [
                    rejection.to_dict() for rejection in self.candidates.rejections
                ]
            return summary

        message.pop("tool_calls", None)
        message.pop("function_call", None)
        choice["finish_reason"] = "stop"
        if action.mode == "need_input":
            expected = _missing_destinations(
                action.tool_id,
                self.policy,
                self.candidates,
            )
            if not expected or frozenset(action.missing) != expected:
                raise OneCallError("need_input destinations were not runtime-derived")
            message["content"] = (
                "Please provide the following before I use the tool: "
                + ", ".join(sorted(expected))
                + "."
                if self.operating_mode == "assist"
                else "EviBind withheld a call with missing required evidence."
            )
            return {
                **base,
                "released": False,
                "decision": "need_input",
                "tool": action.tool_id,
                "missing": sorted(expected),
                "reason": action.reason,
            }

        message["content"] = "No relevant tool call was released."
        return {
            **base,
            "released": False,
            "decision": "no_tool",
            "reason": action.reason,
        }


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, Mapping) and item.get("type") in {
            "text",
            "input_text",
        }:
            parts.append(str(item.get("text", "")))
    return "\n".join(part for part in parts if part)


def _evidence_messages(messages: Any) -> tuple[MessageEvidence, ...]:
    if not isinstance(messages, list):
        raise OneCallError("messages must be a list")
    user_indices = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, Mapping) and message.get("role") == "user"
    ]
    current_user = user_indices[-1] if user_indices else None
    output: list[MessageEvidence] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(messages):
        if not isinstance(raw, Mapping):
            raise OneCallError("each message must be an object")
        message_id = str(raw.get("id") or f"message-{index}")
        if message_id in seen_ids:
            raise OneCallError(f"duplicate message id: {message_id}")
        seen_ids.add(message_id)
        role = str(raw.get("role", "user"))
        if role == "user":
            source = "user.current_turn" if index == current_user else "user.prior_turn"
        elif role == "tool":
            source = "tool.untrusted_output"
        else:
            source = f"{role}.untrusted"
        output.append(
            MessageEvidence(
                message_id=message_id,
                role=role,
                content=_message_content_text(raw.get("content")),
                source=source,
            )
        )
    return tuple(output)


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _object_slots(
    tool_id: str,
    schema: Mapping[str, Any],
    *,
    prefix: str = "",
    parent_required: bool = True,
) -> tuple[SlotDescriptor, ...]:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return ()
    required = {
        str(value) for value in schema.get("required", []) if isinstance(value, str)
    }
    output: list[SlotDescriptor] = []
    for surface, raw_property in properties.items():
        if not isinstance(surface, str) or not isinstance(raw_property, Mapping):
            continue
        pointer = prefix + "/" + _escape_pointer(surface)
        slot_required = parent_required and surface in required
        if raw_property.get("type") == "object" and isinstance(
            raw_property.get("properties"), Mapping
        ):
            output.extend(
                _object_slots(
                    tool_id,
                    raw_property,
                    prefix=pointer,
                    parent_required=slot_required,
                )
            )
        else:
            output.append(
                SlotDescriptor(
                    tool_id=tool_id,
                    destination_scope=pointer,
                    schema=dict(raw_property),
                    required=slot_required,
                )
            )
    return tuple(output)


def _leaf_name(pointer: str) -> str:
    return pointer.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~")


def _infer_evidence_type(slot: SlotDescriptor) -> str:
    prop = slot.schema
    explicit = prop.get("x-tap-evidence-type")
    if isinstance(explicit, str) and explicit:
        return explicit
    json_type = prop.get("type")
    format_name = str(prop.get("format", "")).casefold()
    name = _leaf_name(slot.destination_scope).casefold()
    role = str(prop.get("x-tap-slot-role", "")).casefold()
    if isinstance(prop.get("enum"), list):
        return "schema_enum"
    if format_name in {"email", "idn-email"} or "email" in name:
        return "email_address"
    if format_name in {"uri", "url", "uri-reference"} or name.endswith(
        ("_uri", "_url")
    ):
        return "uri"
    if format_name == "uuid":
        return "uuid"
    if format_name == "date" or name == "date" or name.endswith("_date"):
        return "iso_date"
    if "phone" in name or "telephone" in name:
        return "phone_number"
    if "path" in name:
        return "repository_path"
    if json_type == "integer":
        return "integer"
    if json_type == "number":
        return "number"
    if json_type == "boolean":
        return "boolean"
    if role == "content":
        return "opaque_content"
    if "person" in name or name in {"attendee", "recipient", "owner"}:
        return "person_ref"
    if "account" in name:
        return "account_ref"
    if "event" in name:
        return "event_ref"
    if "order" in name:
        return "order_ref"
    if name.endswith("_id") or name == "id":
        return "opaque_registry_id"
    raise OneCallError(
        f"{slot.tool_id}{slot.destination_scope} requires an explicit "
        "x-evibind-evidence-type"
    )


def _sources_for(
    slot: SlotDescriptor,
    evidence_type: str,
) -> frozenset[str]:
    explicit = slot.schema.get("x-tap-sources")
    if isinstance(explicit, list) and all(
        isinstance(source, str) for source in explicit
    ):
        return frozenset(explicit)
    if isinstance(explicit, str) and explicit:
        return frozenset({explicit})
    if slot.schema.get("x-tap-source-policy") == "trusted_state_only":
        return frozenset({f"state.{_leaf_name(slot.destination_scope)}"})
    sources = {"user.current_turn"}
    if "default" in slot.schema:
        sources.add("schema.default")
    if evidence_type == "opaque_registry_id":
        sources.discard("user.current_turn")
        sources.add(f"state.{_leaf_name(slot.destination_scope)}")
    return frozenset(sources)


def _parser_for(evidence_type: str) -> str:
    return {
        "integer": "parse_integer",
        "number": "parse_number",
        "boolean": "parse_boolean",
    }.get(evidence_type, "identity")


def _criticality(slot: SlotDescriptor) -> str:
    declared = str(slot.schema.get("x-tap-criticality", "")).casefold()
    if declared in {"target", "control", "content", "effect"}:
        return declared
    if str(slot.schema.get("x-tap-slot-role", "")).casefold() == "content":
        return "content"
    return "target"


def _slot_policy(
    slot: SlotDescriptor,
    evidence_type: str,
    registry: EvidenceTypeRegistry,
) -> SlotPolicy:
    evidence_spec = registry.get(evidence_type)
    explicit_transforms = slot.schema.get("x-tap-transforms")
    transforms = (
        {
            str(transform)
            for transform in explicit_transforms
            if isinstance(transform, str)
        }
        if isinstance(explicit_transforms, list)
        else set()
    )
    transforms.add(_parser_for(evidence_type))
    ambiguity = str(slot.schema.get("x-tap-on-ambiguity", "clarify")).casefold()
    return SlotPolicy(
        tool_id=slot.tool_id,
        destination_scope=slot.destination_scope,
        evidence_type=evidence_type,
        sources=_sources_for(slot, evidence_type),
        transforms=frozenset(transforms),
        criticality=_criticality(slot),
        value_class=evidence_spec.value_class,
        ambiguity=ambiguity,
        required=slot.required,
    )


_EMAIL = re.compile(
    r"(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@"
    r"(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63}(?![\w-])"
)
_PHONE = re.compile(r"(?<!\w)\+?[\d][\d\s().-]{5,}\d(?!\w)")
_UUID = re.compile(
    r"(?<![0-9A-Fa-f])"
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[1-5][0-9A-Fa-f]{3}-"
    r"[89ABab][0-9A-Fa-f]{3}-[0-9A-Fa-f]{12}"
    r"(?![0-9A-Fa-f])"
)
_ISO_DATE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
_URI = re.compile(r"(?<!\w)[A-Za-z][A-Za-z0-9+.-]*:[^\s<>{}\"']+")
_INTEGER = re.compile(r"(?<![\w.])[-+]?\d+(?!\w|\.\d)")
_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?![\w.])")
_BOOLEAN = re.compile(r"(?<!\w)(?:true|false|yes|no)(?!\w)", re.I)
_PATH = re.compile(r"(?<!\w)[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+(?!\w)")


def _cue_allows(text: str, start: int, cue: str | None) -> bool:
    if cue is None:
        return True
    before = text[max(0, start - len(cue) - 16) : start]
    return bool(
        re.search(
            rf"(?<!\w){re.escape(cue)}"
            r"(?:\s+(?:is|to))?\s*(?:=|:)?\s*$",
            before,
            re.IGNORECASE,
        )
    )


def _reference_matches(
    text: str,
    cue: str,
) -> tuple[tuple[int, int, int], ...]:
    pattern = re.compile(
        rf"(?<!\w){re.escape(cue)}"
        r"(?:\s+(?:is|to))?\s*(?:=|:)?\s*"
        r"(?:\"([^\"]+)\"|'([^']+)'|([A-Za-z0-9_@.+:/-]+"
        r"(?:\s+[A-Za-z0-9_@.+:/-]+)?))",
        re.IGNORECASE,
    )
    output: list[tuple[int, int, int]] = []
    for match in pattern.finditer(text):
        group_index = next(
            index for index in (1, 2, 3) if match.group(index) is not None
        )
        start, end = match.span(group_index)
        if group_index == 3:
            trimmed = match.group(group_index).rstrip(".,;!?")
            end = start + len(trimmed)
        output.append((start, end, match.start()))
    return tuple(output)


def _active_reference_matches(
    text: str,
    cue: str,
) -> tuple[tuple[int, int], ...]:
    references = _reference_matches(text, cue)
    active: list[tuple[int, int]] = []
    for index, (start, end, assignment_start) in enumerate(references):
        clause_start = max(
            (text.rfind(delimiter, 0, assignment_start) for delimiter in ";.!?\n"),
            default=-1,
        ) + 1
        prefix = text[clause_start:assignment_start]
        explicitly_inactive = re.search(
            r"\b(?:do\s+not|don't|never)\s+(?:use\s+)?$"
            r"|\b(?:without|no)\s+$",
            prefix,
            re.IGNORECASE,
        )
        next_assignment = (
            references[index + 1][2] if index + 1 < len(references) else None
        )
        superseded = next_assignment is not None and re.search(
            r"\b(?:correction|instead|rather)\b",
            text[end:next_assignment],
            re.IGNORECASE,
        )
        if explicitly_inactive or superseded:
            continue
        active.append((start, end))
    return tuple(active)


def _typed_spans(
    message: MessageEvidence,
    slot: SlotDescriptor,
    evidence_type: str,
    *,
    require_cue: bool,
    fallback_cue: str | None = None,
) -> tuple[Span, ...]:
    text = message.content
    raw_cue = slot.schema.get("x-tap-extraction-cue")
    cue = (
        raw_cue.strip()
        if isinstance(raw_cue, str) and raw_cue.strip()
        else None
    )
    parser = _parser_for(evidence_type)
    pattern = {
        "email_address": _EMAIL,
        "phone_number": _PHONE,
        "uuid": _UUID,
        "iso_date": _ISO_DATE,
        "uri": _URI,
        "repository_path": _PATH,
        "integer": _INTEGER,
        "number": _NUMBER,
        "boolean": _BOOLEAN,
        "schema_enum": None,
    }.get(evidence_type)
    if cue is None and pattern is None:
        cue = fallback_cue
    character_spans: list[tuple[int, int]] = []
    if evidence_type == "schema_enum":
        for value in slot.schema.get("enum", []):
            enum_parser = (
                "parse_boolean"
                if isinstance(value, bool)
                else "parse_integer"
                if isinstance(value, int)
                else "parse_number"
                if isinstance(value, float)
                else "identity"
            )
            for match in re.finditer(
                rf"(?<!\w){re.escape(str(value))}(?!\w)",
                text,
                re.IGNORECASE,
            ):
                if _cue_allows(text, match.start(), cue):
                    start, end = _character_to_byte_span(text, *match.span())
                    character_spans.append((start, end, enum_parser))
        return tuple(
            Span(message.message_id, start, end, item_parser)
            for start, end, item_parser in character_spans
        )
    if pattern is not None:
        raw_references = _reference_matches(text, cue) if cue is not None else ()
        references = _active_reference_matches(text, cue) if cue is not None else ()
        if raw_references:
            for reference_start, reference_end in references:
                scoped = text[reference_start:reference_end]
                character_spans.extend(
                    (
                        reference_start + match.start(),
                        reference_start + match.end(),
                    )
                    for match in pattern.finditer(scoped)
                )
        else:
            for match in pattern.finditer(text):
                if (not require_cue or cue is not None) and _cue_allows(
                    text, match.start(), cue
                ):
                    character_spans.append(match.span())
    elif cue is not None:
        character_spans.extend(_active_reference_matches(text, cue))
    return tuple(
        Span(
            message.message_id,
            *_character_to_byte_span(text, start, end),
            parser,
        )
        for start, end in character_spans
    )


def _character_to_byte_span(
    text: str,
    start: int,
    end: int,
) -> tuple[int, int]:
    return (
        len(text[:start].encode("utf-8")),
        len(text[:end].encode("utf-8")),
    )


def _state_rows(
    dialogue_state: Mapping[str, Any],
    slot_policy: SlotPolicy,
) -> tuple[
    dict[tuple[str, str], StateValue],
    tuple[CandidateProposal, ...],
]:
    leaf = _leaf_name(slot_policy.destination_scope)
    raw_values = dialogue_state.get(
        slot_policy.destination_scope,
        dialogue_state.get(leaf, ()),
    )
    values = raw_values if isinstance(raw_values, list) else [raw_values]
    state: dict[tuple[str, str], StateValue] = {}
    proposals: list[CandidateProposal] = []
    allowed_namespaces = sorted(
        source.removeprefix("state.")
        for source in slot_policy.sources
        if source.startswith("state.")
    )
    for index, raw in enumerate(values):
        if raw in (None, (), []):
            continue
        item = raw if isinstance(raw, Mapping) else {"value": raw}
        version = item.get("version")
        if version is None or not allowed_namespaces:
            continue
        namespace = str(item.get("namespace") or allowed_namespaces[0])
        if namespace not in allowed_namespaces:
            continue
        key = str(item.get("key") or f"{leaf}-{index}")
        evidence_type = str(item.get("evidence_type") or slot_policy.evidence_type)
        state_value = StateValue(
            namespace=namespace,
            key=key,
            version=str(version),
            value=item.get("value"),
            evidence_type=evidence_type,
        )
        state[(namespace, key)] = state_value
        proposals.append(
            CandidateProposal(
                tool_id=slot_policy.tool_id,
                destination_scope=slot_policy.destination_scope,
                derivation=StateRef(namespace, key, str(version)),
                evidence_type=slot_policy.evidence_type,
                display=str(item.get("label") or f"{namespace}.{key}"),
            )
        )
    return state, tuple(proposals)


def compile_one_call_session(
    *,
    request_payload: Mapping[str, Any],
    upstream_payload: Mapping[str, Any],
    options: Mapping[str, Any],
    tools: list[dict[str, Any]],
    handle_secret: bytes,
    include_diagnostics: bool,
    operating_mode: str = "enforce",
    handle_nonce_bytes: Callable[[int], bytes] | None = None,
) -> OneCallSession:
    evidence_types = EvidenceTypeRegistry.standard()
    transforms = TransformRegistry.standard()
    messages = _evidence_messages(request_payload.get("messages"))
    policy_epoch = str(options.get("policy_epoch", "1"))
    descriptors: dict[str, tuple[SlotDescriptor, ...]] = {}
    tool_policies: list[ToolPolicy] = []
    slot_policies: list[SlotPolicy] = []
    normalized_tools: dict[str, Mapping[str, Any]] = {}
    type_counts: dict[tuple[str, str], int] = {}
    literal_destinations: dict[str, frozenset[str]] = {}
    allow_noncritical_literals = bool(
        options.get("allow_noncritical_opaque_literals", False)
    )
    candidate_proposer = str(options.get("candidate_proposer", "deterministic"))
    if candidate_proposer not in {"deterministic", "broad_typed"}:
        raise OneCallError(f"unsupported candidate proposer: {candidate_proposer}")

    for tool in tools:
        tool_id = str(tool["name"])
        parameters = tool.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise OneCallError(f"tool parameters must be an object: {tool_id}")
        slots = _object_slots(tool_id, parameters)
        descriptors[tool_id] = slots
        normalized_tools[tool_id] = tool
        policies = []
        literal_slots: set[str] = set()
        for slot in slots:
            evidence_type = _infer_evidence_type(slot)
            evidence_spec = evidence_types.get(evidence_type)
            if (
                allow_noncritical_literals
                and _criticality(slot) == "content"
                and evidence_spec.value_class == "opaque_content"
            ):
                literal_slots.add(slot.destination_scope)
                continue
            policy = _slot_policy(slot, evidence_type, evidence_types)
            policies.append(policy)
            slot_policies.append(policy)
            key = (tool_id, evidence_type)
            type_counts[key] = type_counts.get(key, 0) + 1
        tool_policies.append(
            ToolPolicy(
                tool_id=tool_id,
                slots=tuple(policies),
                policy_epoch=policy_epoch,
                contract_version=sha256_digest(parameters),
            )
        )
        literal_destinations[tool_id] = frozenset(literal_slots)
    policy_set = PolicySet(tuple(tool_policies))

    defaults: dict[str, Any] = {}
    state: dict[tuple[str, str], StateValue] = {}
    proposals: list[CandidateProposal] = []
    dialogue_state = options.get("dialogue_state", {})
    if not isinstance(dialogue_state, Mapping):
        raise OneCallError("evibind.dialogue_state must be an object")

    slot_by_key = {
        (slot.tool_id, slot.destination_scope): slot
        for slots in descriptors.values()
        for slot in slots
    }
    for slot_policy in slot_policies:
        slot = slot_by_key[(slot_policy.tool_id, slot_policy.destination_scope)]
        cue = slot.schema.get("x-tap-extraction-cue")
        require_cue = type_counts[
            (slot_policy.tool_id, slot_policy.evidence_type)
        ] > 1 and not (isinstance(cue, str) and cue.strip())
        if candidate_proposer == "broad_typed":
            require_cue = False
        for message in messages:
            if message.source not in slot_policy.sources:
                continue
            for derivation in _typed_spans(
                message,
                slot,
                slot_policy.evidence_type,
                require_cue=require_cue,
                fallback_cue=(
                    _leaf_name(slot.destination_scope)
                    if candidate_proposer == "broad_typed"
                    else None
                ),
            ):
                source = message.content.encode("utf-8")[
                    derivation.byte_start : derivation.byte_end
                ].decode("utf-8")
                proposals.append(
                    CandidateProposal(
                        tool_id=slot_policy.tool_id,
                        destination_scope=slot_policy.destination_scope,
                        derivation=derivation,
                        evidence_type=slot_policy.evidence_type,
                        display=f"{message.source}: {source}",
                    )
                )
        if "default" in slot.schema and "schema.default" in slot_policy.sources:
            default_id = f"{slot_policy.tool_id}{slot_policy.destination_scope}"
            defaults[default_id] = slot.schema["default"]
            proposals.append(
                CandidateProposal(
                    tool_id=slot_policy.tool_id,
                    destination_scope=slot_policy.destination_scope,
                    derivation=Default(default_id, policy_epoch),
                    evidence_type=slot_policy.evidence_type,
                    display=f"schema default: {slot.schema['default']!r}",
                )
            )
        state_rows, state_proposals = _state_rows(dialogue_state, slot_policy)
        state.update(state_rows)
        proposals.extend(state_proposals)

    context = EvidenceContext(
        messages=messages,
        state=state,
        defaults=defaults,
        policy_epoch=policy_epoch,
    )
    issuer = (
        HandleIssuer(handle_secret)
        if handle_nonce_bytes is None
        else HandleIssuer(handle_secret, nonce_bytes=handle_nonce_bytes)
    )
    table = compile_candidates(
        proposals=tuple(proposals),
        context=context,
        policy=policy_set,
        evidence_types=evidence_types,
        transforms=transforms,
        issuer=issuer,
    )
    ranker_model = options.get("candidate_ranker_model")
    if ranker_model is not None:
        if not isinstance(ranker_model, Mapping):
            raise OneCallError("candidate_ranker_model must be an object")
        raw_top_k = options.get("candidate_top_k", 4)
        if (
            isinstance(raw_top_k, bool)
            or not isinstance(raw_top_k, int)
            or not 1 <= raw_top_k <= 8
        ):
            raise OneCallError("candidate_top_k must be an integer from 1 to 8")
        try:
            ranker = LinearCandidateRanker.from_dict(ranker_model)
            table = prune_candidate_table(
                table,
                context,
                ranker,
                top_k=raw_top_k,
            )
        except ValueError as exc:
            raise OneCallError(str(exc)) from exc
    action_interface = str(options.get("action_interface", "dynamic_enum"))
    if action_interface not in {"dynamic_enum", "indexed"}:
        raise OneCallError(f"unsupported action interface: {action_interface}")
    if action_interface == "indexed":
        action_schema = _indexed_action_schema()
        catalog, index_manifest = _indexed_candidate_catalog(
            policy_set,
            table,
            normalized_tools,
        )
    else:
        action_schema = _action_schema(
            policy_set,
            table,
            normalized_tools,
            literal_destinations,
        )
        catalog = _candidate_catalog(policy_set, table, normalized_tools)
        index_manifest = {}
    literal_instruction = (
        "Never emit a literal for a protected destination. When the action "
        "schema includes arguments, literals are permitted only in those "
        "explicitly noncritical opaque fields."
        if any(literal_destinations.values())
        else "Never emit executable argument literals."
    )
    upstream = deepcopy(dict(upstream_payload))
    selection_instruction = (
        "Select only the request-local tool_index, slot_index, and "
        "candidate_index values listed below. The JSON schema is constant; "
        "invented indices fail trusted lookup."
        if action_interface == "indexed"
        else (
            "Select only the candidate IDs listed below. Candidate IDs are "
            "opaque and order-independent."
        )
    )
    upstream["messages"] = [
        {
            "role": "system",
            "content": (
                "Return exactly one call to evibind_action. "
                + selection_instruction
                + " If a required destination has no valid "
                "candidate, use need_input. If no tool is appropriate, use "
                "no_tool. "
                + literal_instruction
                + "\n"
                "EVIDENCE CANDIDATES:\n"
                + json.dumps(catalog, ensure_ascii=True, sort_keys=True)
            ),
        },
        *deepcopy(list(upstream.get("messages", []))),
    ]
    upstream["tools"] = [
        {
            "type": "function",
            "function": {
                "name": ACTION_TOOL_NAME,
                "description": (
                    "Select a mode, tool, and destination-bound evidence "
                    "handles, plus explicitly permitted noncritical opaque "
                    "arguments. EviBind materializes protected values."
                ),
                "parameters": action_schema,
            },
        }
    ]
    upstream["tool_choice"] = {
        "type": "function",
        "function": {"name": ACTION_TOOL_NAME},
    }
    upstream["parallel_tool_calls"] = False
    upstream["n"] = 1
    upstream.pop("response_format", None)
    return OneCallSession(
        upstream_payload=upstream,
        context=context,
        policy=policy_set,
        candidates=table,
        issuer=issuer,
        evidence_types=evidence_types,
        transforms=transforms,
        tools=normalized_tools,
        literal_destinations=literal_destinations,
        action_interface=action_interface,
        index_manifest=index_manifest,
        include_diagnostics=include_diagnostics,
        operating_mode=operating_mode,
    )


def _candidate_catalog(
    policy: PolicySet,
    table: CandidateTable,
    tools: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for tool_policy in policy.tools:
        tool = tools[tool_policy.tool_id]
        for slot in tool_policy.slots:
            rows = [
                candidate.public_view()
                for candidate in table.candidates.values()
                if candidate.witness.tool_id == tool_policy.tool_id
                and candidate.witness.destination_scope == slot.destination_scope
            ]
            output.append(
                {
                    "tool_id": tool_policy.tool_id,
                    "tool_description": tool.get("description", ""),
                    "destination": slot.destination_scope,
                    "evidence_type": slot.evidence_type,
                    "required": slot.required,
                    "candidates": sorted(rows, key=lambda row: row["candidate_id"]),
                }
            )
    return output


def _indexed_candidate_catalog(
    policy: PolicySet,
    table: CandidateTable,
    tools: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    public: list[dict[str, Any]] = []
    manifest: dict[int, dict[str, Any]] = {}
    for tool_index, tool_policy in enumerate(policy.tools):
        tool = tools[tool_policy.tool_id]
        public_slots: list[dict[str, Any]] = []
        manifest_slots: dict[int, dict[str, Any]] = {}
        for slot_index, slot in enumerate(tool_policy.slots):
            candidates = sorted(
                (
                    candidate
                    for candidate in table.candidates.values()
                    if candidate.witness.tool_id == tool_policy.tool_id
                    and candidate.witness.destination_scope
                    == slot.destination_scope
                ),
                key=lambda candidate: candidate.candidate_id,
            )
            public_candidates: list[dict[str, Any]] = []
            candidate_manifest: dict[int, str] = {}
            for candidate_index, candidate in enumerate(candidates):
                view = candidate.public_view()
                view.pop("candidate_id", None)
                public_candidates.append(
                    {"candidate_index": candidate_index, **view}
                )
                candidate_manifest[candidate_index] = candidate.candidate_id
            public_slots.append(
                {
                    "slot_index": slot_index,
                    "destination": slot.destination_scope,
                    "evidence_type": slot.evidence_type,
                    "required": slot.required,
                    "candidates": public_candidates,
                }
            )
            manifest_slots[slot_index] = {
                "destination": slot.destination_scope,
                "required": slot.required,
                "candidates": candidate_manifest,
            }
        public.append(
            {
                "tool_index": tool_index,
                "tool_description": tool.get("description", ""),
                "slots": public_slots,
            }
        )
        manifest[tool_index] = {
            "tool_id": tool_policy.tool_id,
            "slots": manifest_slots,
        }
    return public, manifest


def _candidate_ids(
    table: CandidateTable,
    tool_id: str,
    destination: str,
) -> list[str]:
    return sorted(
        candidate_id
        for candidate_id, candidate in table.candidates.items()
        if candidate.witness.tool_id == tool_id
        and candidate.witness.destination_scope == destination
    )


def _missing_destinations(
    tool_id: str | None,
    policy: PolicySet,
    table: CandidateTable,
) -> frozenset[str]:
    if tool_id is None:
        return frozenset()
    tool = policy.tool(tool_id)
    destinations: set[str] = set()
    for slot in tool.slots:
        candidate_ids = _candidate_ids(
            table,
            tool_id,
            slot.destination_scope,
        )
        if slot.required and not candidate_ids:
            destinations.add(slot.destination_scope)
            continue
        if slot.ambiguity == "clarify":
            distinct_values = {
                table.candidate(candidate_id).witness.value_digest
                for candidate_id in candidate_ids
            }
            if len(distinct_values) > 1:
                destinations.add(slot.destination_scope)
    return frozenset(destinations)


def _indexed_action_schema() -> dict[str, Any]:
    binding = {
        "type": "object",
        "properties": {
            "slot_index": {"type": "integer", "minimum": 0},
            "candidate_index": {"type": "integer", "minimum": 0},
        },
        "required": ["slot_index", "candidate_index"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "mode": {"const": "call"},
                    "tool_index": {"type": "integer", "minimum": 0},
                    "bindings": {
                        "type": "array",
                        "items": binding,
                        "uniqueItems": True,
                    },
                    "arguments": {"type": "object"},
                },
                "required": ["mode", "tool_index", "bindings"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "mode": {"const": "need_input"},
                    "tool_index": {"type": "integer", "minimum": 0},
                    "reason": {"type": "string"},
                },
                "required": ["mode", "tool_index"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "mode": {"const": "no_tool"},
                    "reason": {"type": "string"},
                },
                "required": ["mode"],
                "additionalProperties": False,
            },
        ],
    }


def _literal_argument_schema(
    schema: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    prefix: str = "",
) -> dict[str, Any] | None:
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        kept: dict[str, Any] = {}
        required = {
            str(value)
            for value in schema.get("required", [])
            if isinstance(value, str)
        }
        kept_required: list[str] = []
        for surface, child in properties.items():
            if not isinstance(surface, str) or not isinstance(child, Mapping):
                continue
            pointer = prefix + "/" + _escape_pointer(surface)
            filtered = _literal_argument_schema(
                child,
                allowed,
                prefix=pointer,
            )
            if filtered is None:
                continue
            kept[surface] = filtered
            if surface in required:
                kept_required.append(surface)
        if not kept:
            return None
        output = deepcopy(dict(schema))
        output["type"] = "object"
        output["properties"] = kept
        output["required"] = sorted(kept_required)
        output["additionalProperties"] = False
        return output
    return deepcopy(dict(schema)) if prefix in allowed else None


def _action_schema(
    policy: PolicySet,
    table: CandidateTable,
    tools: Mapping[str, Mapping[str, Any]],
    literal_destinations: Mapping[str, frozenset[str]],
) -> dict[str, Any]:
    branches: list[dict[str, Any]] = []
    for tool in policy.tools:
        missing = _missing_destinations(tool.tool_id, policy, table)
        if not missing:
            binding_properties: dict[str, Any] = {}
            required: list[str] = []
            for slot in tool.slots:
                candidate_ids = _candidate_ids(
                    table, tool.tool_id, slot.destination_scope
                )
                if candidate_ids:
                    binding_properties[slot.destination_scope] = {
                        "type": "string",
                        "enum": candidate_ids,
                    }
                    if slot.required:
                        required.append(slot.destination_scope)
            properties: dict[str, Any] = {
                "mode": {"const": "call"},
                "tool_id": {"const": tool.tool_id},
                "bindings": {
                    "type": "object",
                    "properties": binding_properties,
                    "required": sorted(required),
                    "additionalProperties": False,
                },
            }
            branch_required = ["mode", "tool_id", "bindings"]
            raw_tool = tools.get(tool.tool_id, {})
            literal_schema = _literal_argument_schema(
                raw_tool.get("parameters", {}),
                literal_destinations.get(tool.tool_id, frozenset()),
            )
            if literal_schema is not None:
                properties["arguments"] = literal_schema
                branch_required.append("arguments")
            branches.append(
                {
                    "type": "object",
                    "properties": properties,
                    "required": branch_required,
                    "additionalProperties": False,
                }
            )
        else:
            branches.append(
                {
                    "type": "object",
                    "properties": {
                        "mode": {"const": "need_input"},
                        "tool_id": {"const": tool.tool_id},
                        "missing": {"const": sorted(missing)},
                        "reason": {"type": "string"},
                    },
                    "required": ["mode", "tool_id", "missing"],
                    "additionalProperties": False,
                }
            )
    branches.append(
        {
            "type": "object",
            "properties": {
                "mode": {"const": "no_tool"},
                "reason": {"type": "string"},
            },
            "required": ["mode"],
            "additionalProperties": False,
        }
    )
    return {"type": "object", "oneOf": branches}


def _parse_action_proposal(
    message: Mapping[str, Any],
) -> tuple[ActionProposal, str | None]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise OneCallError("upstream must return exactly one evibind_action call")
    call = calls[0]
    if not isinstance(call, Mapping):
        raise OneCallError("upstream tool call must be an object")
    function = call.get("function")
    if not isinstance(function, Mapping):
        raise OneCallError("upstream tool call omitted function")
    if function.get("name") != ACTION_TOOL_NAME:
        raise OneCallError("upstream called an unmediated function")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise OneCallError("evibind_action arguments must be JSON text")
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise OneCallError("evibind_action arguments are invalid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise OneCallError("evibind_action arguments must be an object")
    call_id = call.get("id")
    return (
        ActionProposal.from_dict(parsed),
        call_id if isinstance(call_id, str) and call_id else None,
    )


def _parse_indexed_action_proposal(
    message: Mapping[str, Any],
    manifest: Mapping[int, Mapping[str, Any]],
    policy: PolicySet,
    table: CandidateTable,
) -> tuple[ActionProposal, str | None]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise OneCallError("upstream must return exactly one evibind_action call")
    call = calls[0]
    function = call.get("function") if isinstance(call, Mapping) else None
    if not isinstance(function, Mapping) or function.get("name") != ACTION_TOOL_NAME:
        raise OneCallError("upstream called an unmediated function")
    raw = function.get("arguments")
    if not isinstance(raw, str):
        raise OneCallError("evibind_action arguments must be JSON text")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OneCallError("evibind_action arguments are invalid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise OneCallError("evibind_action arguments must be an object")
    mode = parsed.get("mode")
    call_id = call.get("id") if isinstance(call, Mapping) else None
    normalized_call_id = call_id if isinstance(call_id, str) and call_id else None
    if mode == "no_tool":
        return (
            ActionProposal(mode="no_tool", reason=parsed.get("reason")),
            normalized_call_id,
        )
    tool_index = parsed.get("tool_index")
    if (
        isinstance(tool_index, bool)
        or not isinstance(tool_index, int)
        or tool_index not in manifest
    ):
        raise OneCallError("indexed action references an unknown tool")
    tool_row = manifest[tool_index]
    tool_id = str(tool_row["tool_id"])
    if mode == "need_input":
        missing = _missing_destinations(tool_id, policy, table)
        if not missing:
            raise OneCallError("need_input has no runtime-derived obligation")
        return (
            ActionProposal(
                mode="need_input",
                tool_id=tool_id,
                missing=tuple(sorted(missing)),
                reason=(
                    str(parsed["reason"])
                    if isinstance(parsed.get("reason"), str)
                    else None
                ),
            ),
            normalized_call_id,
        )
    if mode != "call":
        raise OneCallError("indexed action mode is invalid")
    if _missing_destinations(tool_id, policy, table):
        raise OneCallError("call branch is unavailable for this tool")
    raw_bindings = parsed.get("bindings")
    if not isinstance(raw_bindings, list):
        raise OneCallError("indexed bindings must be an array")
    slots = tool_row.get("slots")
    if not isinstance(slots, Mapping):
        raise OneCallError("indexed tool manifest is invalid")
    bindings: dict[str, str] = {}
    for raw_binding in raw_bindings:
        if not isinstance(raw_binding, Mapping):
            raise OneCallError("indexed binding must be an object")
        slot_index = raw_binding.get("slot_index")
        candidate_index = raw_binding.get("candidate_index")
        if (
            isinstance(slot_index, bool)
            or isinstance(candidate_index, bool)
            or not isinstance(slot_index, int)
            or not isinstance(candidate_index, int)
            or slot_index not in slots
        ):
            raise OneCallError("indexed binding references an unknown slot")
        slot_row = slots[slot_index]
        candidates = slot_row.get("candidates")
        if not isinstance(candidates, Mapping) or candidate_index not in candidates:
            raise OneCallError("indexed binding references an unknown candidate")
        destination = str(slot_row["destination"])
        if destination in bindings:
            raise OneCallError("indexed binding repeats a destination")
        bindings[destination] = str(candidates[candidate_index])
    arguments = parsed.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise OneCallError("indexed arguments must be an object")
    return (
        ActionProposal(
            mode="call",
            tool_id=tool_id,
            bindings=bindings,
            arguments=deepcopy(dict(arguments)),
        ),
        normalized_call_id,
    )

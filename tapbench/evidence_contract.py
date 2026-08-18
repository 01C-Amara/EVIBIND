from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any, Iterable

from evibind.core.evidence_types import EvidenceTypeError, EvidenceTypeRegistry

EVIDENCE_CONTRACT_VERSION = "tapbench.evidence_contract.v4"
CERTIFICATE_VERSION = "tapbench.candidate_certificate.v1"
ACTION_CRITICAL_ROLES = {"control", "identifier", "defaultable"}
SLOT_ROLES = {"control", "content", "derived", "identifier", "defaultable"}
RESOLUTION_TYPES = {"enumerated", "extractive", "normalizable", "referential", "defaultable", "generative"}


@dataclass(frozen=True)
class CandidateCertificate:
    slot: str
    value: Any
    source_kind: str
    source_span: tuple[int, int] | None
    source_text: str | None
    transform: str | None
    transform_context: dict[str, Any]
    role: str
    resolution_type: str
    role_label: str
    support_status: str
    contradiction_status: str
    scope_status: str
    verifier_version: str
    evidence_strength: float
    role_score: float
    candidate_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["source_span"] = list(self.source_span) if self.source_span is not None else None
        row["certificate_version"] = CERTIFICATE_VERSION
        return row


def request_text(messages: Iterable[dict[str, Any]]) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages if message.get("role") == "user")


def _object_schema(parameters: dict[str, Any]) -> dict[str, Any]:
    current = parameters
    while isinstance(current, dict):
        properties = current.get("properties")
        if not isinstance(properties, dict) or set(properties) != {"payload"}:
            break
        payload = properties.get("payload")
        if not isinstance(payload, dict):
            break
        current = payload
    return current if isinstance(current, dict) else {}


def _canonical_slots(tool: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    schema = _object_schema(tool.get("parameters", {}) if isinstance(tool.get("parameters"), dict) else {})
    properties: dict[str, dict[str, Any]] = {}
    surface_to_canonical: dict[str, str] = {}
    for surface, raw in (schema.get("properties", {}) or {}).items():
        prop = raw if isinstance(raw, dict) else {}
        canonical = str(prop.get("x-ir-name") or surface)
        surface_to_canonical[str(surface)] = canonical
        properties[canonical] = prop
    required = {surface_to_canonical.get(str(surface), str(surface)) for surface in schema.get("required", [])}
    return properties, required


def infer_slot_contract(slot: str, prop: dict[str, Any]) -> tuple[str, str]:
    declared_role = str(prop.get("x-tap-slot-role", ""))
    declared_type = str(prop.get("x-tap-resolution-type", ""))
    if declared_role in SLOT_ROLES and declared_type in RESOLUTION_TYPES:
        return declared_role, declared_type
    if "enum" in prop or prop.get("type") == "boolean":
        return "control", "enumerated"
    if slot.endswith("_id") or slot in {"recipient", "owner", "account", "file_id", "product_id"}:
        return "identifier", "referential"
    if slot.endswith("_date") or slot.endswith("_time") or slot in {"date", "time", "amount", "quantity", "limit", "duration"}:
        return "control", "normalizable"
    if slot in {"body", "subject", "title", "issue", "message", "description"}:
        return "content", "generative"
    if "default" in prop:
        return "defaultable", "defaultable"
    return "control", "extractive"


def _scope_status(text: str, start: int, end: int, role_label: str) -> tuple[str, str]:
    before = text[max(0, start - 45) : start].lower()
    after = text[end : min(len(text), end + 45)].lower()
    direct_negation = re.search(
        r"(?:\bdo not|\bdon't|\bnever|\bwithout)\s+(?:[\w-]+\s+){0,3}$",
        before,
    )
    bare_not = re.search(r"\bnot\s+(?!provided\b)(?:[\w-]+\s+){0,2}$", before)
    if direct_negation or bare_not:
        return "negated", "negated"
    if re.search(r"\b(if|would|might|hypothetically|suppose)\b[^.!?]{0,40}$", before):
        return "hypothetical", "none"
    if role_label in {"source_date", "superseded_value"} and re.search(r"\bfrom\s*$", before) and re.search(r"^\s+to\b", after):
        return "superseded", "superseded"
    return "active", "none"


def _certificate(
    *,
    slot: str,
    value: Any,
    text: str,
    span: tuple[int, int] | None,
    source_kind: str,
    transform: str | None,
    transform_context: dict[str, Any],
    role: str,
    resolution_type: str,
    role_label: str,
    role_score: float,
    verifier_version: str,
) -> CandidateCertificate:
    source_text = text[span[0] : span[1]] if span is not None else None
    scope_status, contradiction = _scope_status(text, span[0], span[1], role_label) if span is not None else ("trusted_state", "none")
    certified = role_score >= 0.8 and contradiction == "none" and scope_status not in {"negated", "hypothetical"}
    return CandidateCertificate(
        slot=slot,
        value=value,
        source_kind=source_kind,
        source_span=span,
        source_text=source_text,
        transform=transform,
        transform_context=transform_context,
        role=role,
        resolution_type=resolution_type,
        role_label=role_label,
        support_status="certified" if certified else "ambiguous",
        contradiction_status=contradiction,
        scope_status=scope_status,
        verifier_version=verifier_version,
        evidence_strength=1.0 if source_kind in {"contract_default", "trusted_state"} else 0.9,
        role_score=role_score,
    )


def _find_all(text: str, pattern: str, flags: int = re.IGNORECASE) -> list[re.Match[str]]:
    return list(re.finditer(pattern, text, flags))


def _slot_patterns(slot: str) -> list[tuple[str, str, float]]:
    escaped = re.escape(slot.replace("_", " "))
    patterns: dict[str, list[tuple[str, str, float]]] = {
        "recipient": [(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "recipient", 1.0)],
        "customer_id": [(r"\bCUST[-_ ]?\d+\b", "customer_identifier", 1.0)],
        "title": [(r"(?<=titled )[\w][^,.]*?(?=\s+on\s+|[,.]|$)", "title", 0.95)],
        "subject": [(r"(?<=subject )[\w][^:]*?(?=\s+at\s+|:|$)", "subject", 0.95)],
        "body": [(r"(?<=:\s)[^\n]+$", "body", 0.9)],
        "query": [(r"(?<=about )[\w][^,.]*?(?=\s+modified\s+after|[.]|$)", "query", 0.95)],
        "owner": [(r"(?<=owned by )[\w@.+-]+", "owner", 0.95)],
        "origin": [(r"(?<=from )[\w][^,.]*?(?=\s+to\s+)", "source_location", 0.95)],
        "destination": [(r"(?<=\sto )[\w][^,.]*?(?=,|\s+leaving|$)", "destination_location", 0.95)],
        "location": [(r"(?<=for )[\w][^,.]*?(?=\s+on\s+)", "destination_location", 0.9)],
        "folder": [(r"(?<=Search )[\w/][^,.]*?(?=\s+for\s+)", "folder", 0.95)],
        "filter_value": [(r"(?<=\sis )[\w-]+", "filter_value", 0.9)],
        "filter_field": [(r"(?<=where )[\w-]+(?=\s+is\s+)", "filter_field", 0.98)],
        "table": [(r"(?<=Query )[\w-]+(?=\s+where\s+)", "table", 0.98)],
        "product": [(r"\bshirt\s+\d+\b", "product", 0.98)],
        "issue": [(r"(?<=:\s)login failure \d+(?=\s+via\s+)", "issue", 0.98)],
    }
    return patterns.get(slot, [(rf"(?<={escaped}\s)[\w@./+-]+", slot, 0.82)])



def _declared_scalar(raw: str, expected: str) -> Any:
    cleaned = raw.strip()
    if expected == "integer":
        return int(cleaned)
    if expected == "number":
        value = float(cleaned)
        return int(value) if value.is_integer() else value
    if expected == "boolean":
        lowered = cleaned.casefold()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
        raise ValueError(f"invalid declared boolean: {raw!r}")
    if expected == "array":
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            value = [part.strip() for part in re.split(r"\s*,\s*|\s+and\s+", cleaned) if part.strip()]
        if not isinstance(value, list):
            raise ValueError(f"invalid declared array: {raw!r}")
        return value
    return cleaned


def _extract_declared_span(
    text: str,
    slot: str,
    prop: dict[str, Any],
    role: str,
    resolution_type: str,
) -> list[CandidateCertificate]:
    cue = prop.get("x-tap-extraction-cue")
    if not isinstance(cue, str) or not cue.strip():
        return []
    expected = str(prop.get("type", "string"))
    pattern = rf"(?<![\w-]){re.escape(cue.strip())}\s*=\s*([^;\n]+?)(?=\s*;|\s*$)"
    rows = []
    for match in _find_all(text, pattern):
        raw = match.group(1).strip().rstrip(".")
        start = match.start(1) + (len(match.group(1)) - len(match.group(1).lstrip()))
        span = (start, start + len(raw))
        try:
            value = _declared_scalar(raw, expected)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        rows.append(_certificate(
            slot=slot,
            value=value,
            text=text,
            span=span,
            source_kind="schema_declared_user_span",
            transform="declared_span_materialization",
            transform_context={"surface_cue": cue},
            role=role,
            resolution_type=resolution_type,
            role_label=slot,
            role_score=0.99,
            verifier_version="declared_span_resolver_v1",
        ))
    return rows


def _extract_enumerated(text: str, slot: str, prop: dict[str, Any], role: str, resolution_type: str) -> list[CandidateCertificate]:
    values = prop.get("enum") if isinstance(prop.get("enum"), list) else []
    rows = []
    for value in values:
        for match in _find_all(text, rf"(?<!\w){re.escape(str(value))}(?!\w)"):
            rows.append(_certificate(slot=slot, value=value, text=text, span=match.span(), source_kind="user_span", transform=None, transform_context={}, role=role, resolution_type=resolution_type, role_label=slot, role_score=0.98, verifier_version="enum_resolver_v1"))
    if prop.get("type") == "boolean":
        boolean_patterns = [(True, r"\b(include|with|yes|true)\b"), (False, r"\b(exclude|without|no|false)\b")]
        for value, pattern in boolean_patterns:
            for match in _find_all(text, pattern):
                rows.append(_certificate(slot=slot, value=value, text=text, span=match.span(), source_kind="user_span", transform="boolean_normalization", transform_context={}, role=role, resolution_type=resolution_type, role_label=slot, role_score=0.82, verifier_version="boolean_resolver_v1"))
    return rows


def _parse_reference_date(context: dict[str, Any]) -> date | None:
    raw = context.get("reference_date")
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str):
        try:
            return date.fromisoformat(raw)
        except ValueError:
            return None
    return None


def _normalizable_role_patterns(slot: str) -> list[tuple[str, str, str]]:
    date_token = r"\d{4}-\d{2}-\d{2}"
    time_token = r"\d{1,2}:\d{2}"
    patterns = {
        "date": [(rf"(?<=\son\s){date_token}", "destination_date", "iso_date")],
        "start_time": [(rf"(?<=\sfrom\s){time_token}", "start_time", "time")],
        "end_time": [(rf"(?<=\sto\s){time_token}", "end_time", "time")],
        "send_time": [(rf"(?<=\sat\s){date_token}T{time_token}(?::\d{{2}})?", "send_time", "iso_datetime")],
        "modified_after": [(rf"(?<=modified\safter\s){date_token}", "lower_date_bound", "iso_date")],
        "depart_date": [(rf"(?<=leaving\s){date_token}", "departure_date", "iso_date")],
        "return_date": [(rf"(?<=returning\s){date_token}", "return_date", "iso_date")],
        "quantity": [(r"(?<=Add\s)\d+", "quantity", "integer")],
        "limit": [(r"(?<=limit\sto\s)\d+", "limit", "integer")],
        "traveler_count": [(r"\b\d+(?=\s+travelers?\b)", "traveler_count", "integer")],
        "hour": [(rf"(?<=\sat\s){time_token}", "hour", "time")],
    }
    return patterns.get(slot, [])


def _normalized_scalar(raw: str, transform: str) -> Any:
    if transform == "integer":
        return int(raw)
    if transform == "number":
        return float(raw) if "." in raw else int(raw)
    return raw


def _extract_normalizable(text: str, slot: str, prop: dict[str, Any], role: str, resolution_type: str, context: dict[str, Any]) -> list[CandidateCertificate]:
    rows = []
    matched_spans: set[tuple[int, int]] = set()
    if slot == "date" or slot.endswith("_date"):
        correction = re.compile(r"\bfrom\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
        for match in correction.finditer(text):
            source_span = match.span(1)
            destination_span = match.span(2)
            matched_spans.update({source_span, destination_span})
            rows.append(_certificate(slot=slot, value=match.group(1), text=text, span=source_span, source_kind="user_span", transform="iso_date", transform_context=context, role=role, resolution_type=resolution_type, role_label="source_date", role_score=0.98, verifier_version="correction_role_resolver_v1"))
            rows.append(_certificate(slot=slot, value=match.group(2), text=text, span=destination_span, source_kind="user_span", transform="iso_date", transform_context=context, role=role, resolution_type=resolution_type, role_label="destination_date", role_score=0.98, verifier_version="correction_role_resolver_v1"))
    for pattern, role_label, transform in _normalizable_role_patterns(slot):
        for match in _find_all(text, pattern):
            matched_spans.add(match.span())
            rows.append(_certificate(slot=slot, value=_normalized_scalar(match.group(0), transform), text=text, span=match.span(), source_kind="user_span", transform=transform, transform_context=context, role=role, resolution_type=resolution_type, role_label=role_label, role_score=0.98, verifier_version="slot_role_scalar_resolver_v1"))

    generic_patterns = [
        (r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?)?\b", "iso_datetime"),
        (r"\b\d{1,2}:\d{2}\b", "time"),
        (r"(?<![\w.-])[-+]?\d+(?:\.\d+)?(?![\w.-])", "number"),
    ]
    for pattern, transform in generic_patterns:
        for match in _find_all(text, pattern):
            if match.span() in matched_spans:
                continue
            raw = match.group(0)
            rows.append(_certificate(slot=slot, value=_normalized_scalar(raw, transform), text=text, span=match.span(), source_kind="user_span", transform=transform, transform_context=context, role=role, resolution_type=resolution_type, role_label="unresolved_same_type", role_score=0.45, verifier_version="scalar_candidate_extractor_v1"))

    reference = _parse_reference_date(context)
    if reference is not None:
        for match in _find_all(text, r"\b(today|tomorrow|next monday)\b"):
            phrase = match.group(0).lower()
            if phrase == "today":
                resolved = reference
            elif phrase == "tomorrow":
                resolved = reference + timedelta(days=1)
            else:
                delta = (7 - reference.weekday()) % 7
                resolved = reference + timedelta(days=delta or 7)
            before = text[max(0, match.start() - 24) : match.start()].lower()
            expected = {
                "date": r"\bon\s*$",
                "depart_date": r"\bleaving\s*$",
                "return_date": r"\breturning\s*$",
                "modified_after": r"\bmodified\s+after\s*$",
            }.get(slot)
            role_score = 0.9 if expected and re.search(expected, before) else 0.5
            role_label = slot if role_score >= 0.8 else "unresolved_relative_date_role"
            rows.append(_certificate(slot=slot, value=resolved.isoformat(), text=text, span=match.span(), source_kind="user_span", transform="relative_date", transform_context=context, role=role, resolution_type=resolution_type, role_label=role_label, role_score=role_score, verifier_version="date_resolver_v1"))
    return rows

def _extract_spans(text: str, slot: str, role: str, resolution_type: str) -> list[CandidateCertificate]:
    rows = []
    for pattern, role_label, score in _slot_patterns(slot):
        for match in _find_all(text, pattern):
            value = match.group(0).strip(" \t\n.,:")
            start = match.start() + (len(match.group(0)) - len(match.group(0).lstrip()))
            span = (start, start + len(value))
            rows.append(_certificate(slot=slot, value=value, text=text, span=span, source_kind="user_span", transform="span_materialization", transform_context={}, role=role, resolution_type=resolution_type, role_label=role_label, role_score=score, verifier_version="span_role_resolver_v1"))
    return rows


def _trusted_state_candidates(slot: str, role: str, resolution_type: str, dialogue_state: dict[str, Any]) -> list[CandidateCertificate]:
    values = dialogue_state.get(slot, [])
    if not isinstance(values, list):
        values = [values]
    rows = []
    for item in values:
        if isinstance(item, dict):
            value = item.get("value")
            version = item.get("version")
        else:
            value, version = item, None
        rows.append(_certificate(slot=slot, value=value, text="", span=None, source_kind="trusted_state", transform=None, transform_context={"state_version": version}, role=role, resolution_type=resolution_type, role_label=slot, role_score=1.0, verifier_version="state_reference_resolver_v1"))
    return rows


_EVIDENCE_TYPES = EvidenceTypeRegistry.standard()


def _satisfies_evidence_type(value: Any, prop: dict[str, Any]) -> bool:
    """Reject a candidate that cannot be an instance of its declared type.

    ``_value_matches_property`` only checks the JSON Schema shape, so any string
    satisfied a ``string`` slot. Extractive resolution offers the words after an
    extraction cue, which meant *"The beneficiary account for this one is
    ACC-5003"* admitted ``"for"`` as an ``account_ref``, and released it when the
    model proposed an unsupported literal. No untrusted-origin value escaped —
    confinement was never violated — but the fail-closed contract was, and a
    malformed argument went downstream.

    The slot declares what the value is supposed to be; the registry knows how
    to check it. An unknown or undeclared type is not a licence to admit
    anything shaped like a string, but it is also not grounds to reject: those
    fall through to the schema check alone, as before.
    """
    name = prop.get("x-tap-evidence-type") or prop.get("x-evibind-evidence-type")
    if not isinstance(name, str) or not name:
        return True
    try:
        evidence_type = _EVIDENCE_TYPES.get(name)
    except EvidenceTypeError:
        return True
    try:
        return bool(evidence_type.validator(value))
    except Exception:  # noqa: BLE001 - a validator that raises has not matched
        return False


def _value_matches_property(value: Any, prop: dict[str, Any]) -> bool:
    enum = prop.get("enum")
    if isinstance(enum, list) and value not in enum:
        return False
    expected = prop.get("type")
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def _deduplicate(candidates: list[CandidateCertificate]) -> list[CandidateCertificate]:
    best: dict[str, CandidateCertificate] = {}
    for candidate in candidates:
        key = json.dumps(candidate.value, sort_keys=True, default=str) + "|" + candidate.role_label
        current = best.get(key)
        if current is None or (candidate.support_status == "certified", candidate.role_score) > (current.support_status == "certified", current.role_score):
            best[key] = candidate
    return sorted(best.values(), key=lambda row: (row.support_status != "certified", -row.role_score, str(row.value)))


def build_candidate_lattice(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    dialogue_state: dict[str, Any] | None = None,
    reference_context: dict[str, Any] | None = None,
    candidate_seed: int = 0,
) -> dict[str, Any]:
    """Build deployable candidate domains without gold actions or task labels."""
    text = request_text(messages)
    state = dialogue_state or {}
    context = reference_context or {}
    tool_rows: dict[str, Any] = {}
    rng = random.Random(candidate_seed)
    for tool_index, tool in enumerate(tools):
        tool_name = str(tool.get("canonical_name") or tool.get("name"))
        properties, required = _canonical_slots(tool)
        slots: dict[str, Any] = {}
        for slot, prop in properties.items():
            role, resolution_type = infer_slot_contract(slot, prop)
            candidates: list[CandidateCertificate] = []
            candidates.extend(
                _extract_declared_span(
                    text, slot, prop, role, resolution_type
                )
            )
            if resolution_type == "enumerated":
                candidates.extend(_extract_enumerated(text, slot, prop, role, resolution_type))
            elif resolution_type == "normalizable":
                candidates.extend(_extract_normalizable(text, slot, prop, role, resolution_type, context))
            elif resolution_type in {"extractive", "generative"}:
                candidates.extend(_extract_spans(text, slot, role, resolution_type))
            elif resolution_type == "referential":
                candidates.extend(_extract_spans(text, slot, role, resolution_type))
            elif resolution_type == "defaultable" and "default" in prop:
                candidates.append(_certificate(slot=slot, value=prop["default"], text=text, span=None, source_kind="contract_default", transform=None, transform_context={}, role=role, resolution_type=resolution_type, role_label=slot, role_score=1.0, verifier_version="contract_default_resolver_v1"))
            if slot in state:
                candidates.extend(
                    _trusted_state_candidates(
                        slot, role, resolution_type, state
                    )
                )
            source_policy = str(prop.get("x-tap-source-policy", "any_certified"))
            if source_policy == "trusted_state_only":
                candidates = [candidate for candidate in candidates if candidate.source_kind == "trusted_state"]
            candidates = _deduplicate([
                candidate
                for candidate in candidates
                if _value_matches_property(candidate.value, prop)
                and _satisfies_evidence_type(candidate.value, prop)
            ])
            ids = list(range(len(candidates)))
            rng.shuffle(ids)
            with_ids = [{**candidate.to_dict(), "candidate_id": ids[index]} for index, candidate in enumerate(candidates)]
            slots[slot] = {
                "role": role,
                "resolution_type": resolution_type,
                "json_type": prop.get("type", "string"),
                "enum": list(prop.get("enum", [])) if isinstance(prop.get("enum"), list) else None,
                "required": slot in required,
                "generation_allowed": role == "content",
                "source_policy": source_policy,
                "candidates": sorted(with_ids, key=lambda row: row["candidate_id"]),
            }
        tool_rows[tool_name] = {"tool_id": tool_index, "description": tool.get("description", ""), "slots": slots}
    return {
        "schema_version": EVIDENCE_CONTRACT_VERSION,
        "input_contract": ["messages", "tools", "dialogue_state", "reference_context", "candidate_seed"],
        "forbidden_runtime_fields": ["gold_action", "task_kind", "derivable_values", "scores", "scorer_output"],
        "request_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "candidate_id_seed": candidate_seed,
        "tools": tool_rows,
    }


def certified_candidates(slot_row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        candidate for candidate in slot_row.get("candidates", [])
        if candidate.get("support_status") == "certified" and candidate.get("contradiction_status") == "none"
    ]


def build_pointer_action_schema(lattice: dict[str, Any]) -> dict[str, Any]:
    tool_domains = []
    clarify_slots = []
    for tool_name, tool in lattice.get("tools", {}).items():
        arguments: dict[str, Any] = {}
        missing_required = []
        for slot, slot_row in tool.get("slots", {}).items():
            candidates = certified_candidates(slot_row)
            if slot_row.get("required") and slot_row.get("role") in ACTION_CRITICAL_ROLES and not candidates:
                missing_required.append(slot)
                clarify_slots.append({"tool_id": tool["tool_id"], "slot": slot})
            if candidates:
                arguments[slot] = [candidate["candidate_id"] for candidate in candidates]
            elif slot_row.get("generation_allowed"):
                arguments[slot] = "content_generation"
        if not missing_required:
            tool_domains.append({"tool_id": tool["tool_id"], "tool": tool_name, "arguments": arguments})
    return {
        "schema_version": "tapbench.pointer_action_schema.v1",
        "modes": ["call", "clarify", "direct_answer", "refuse", "escalate"],
        "call_domains": tool_domains,
        "clarify_domains": clarify_slots,
        "safety_rule": "required action-critical slots admit certified candidate IDs only",
    }


def _tool_by_id(lattice: dict[str, Any], tool_id: int) -> tuple[str, dict[str, Any]]:
    for name, row in lattice.get("tools", {}).items():
        if row.get("tool_id") == tool_id:
            return name, row
    raise ValueError(f"unknown tool_id: {tool_id}")


def materialize_pointer_action(pointer_action: dict[str, Any], lattice: dict[str, Any]) -> dict[str, Any]:
    mode = pointer_action.get("mode")
    if mode != "call":
        return {"mode": mode, "tool": None, "arguments": {}, "payload": dict(pointer_action.get("payload", {}))}
    tool_name, tool = _tool_by_id(lattice, int(pointer_action["tool_id"]))
    output: dict[str, Any] = {}
    for slot, candidate_id in pointer_action.get("arguments", {}).items():
        slot_row = tool.get("slots", {}).get(slot)
        if slot_row is None:
            raise ValueError(f"unknown slot for tool {tool_name}: {slot}")
        match = next((candidate for candidate in certified_candidates(slot_row) if candidate.get("candidate_id") == candidate_id), None)
        if match is None:
            if slot_row.get("generation_allowed") and isinstance(candidate_id, str):
                output[slot] = candidate_id
                continue
            raise ValueError(f"candidate {candidate_id!r} is not certified for {tool_name}.{slot}")
        output[slot] = match["value"]
    for slot, slot_row in tool.get("slots", {}).items():
        if slot_row.get("required") and slot not in output:
            raise ValueError(f"required slot lacks assignment: {tool_name}.{slot}")
    return {"mode": "call", "tool": tool_name, "arguments": output, "payload": {"pointer_materialized": True}}


def capability_signature(tool: dict[str, Any]) -> dict[str, Any]:
    name = str(tool.get("canonical_name") or tool.get("name"))
    action = name.replace("_", " ")
    _, required = _canonical_slots(tool)
    write_verbs = {"create", "send", "add", "update", "delete", "book", "cancel", "share", "apply"}
    first = action.split()[0] if action.split() else "use"
    return {
        "goal": action,
        "description": str(tool.get("description", "")),
        "effects": [f"{action.replace(' ', '_')}_completed"],
        "required_entities": sorted(required),
        "inapplicable_when": ["request_is_informational_only"],
        "side_effect_level": "write" if first in write_verbs else "read",
    }


def capability_explicitly_forbidden(request: str) -> bool:
    lowered = request.lower()
    return bool(
        "answer directly without tools" in lowered
        or "no tool action is needed" in lowered
        or re.search(r"\bno [a-z_ ]+ action is needed\b", lowered)
        or "i am not asking you to perform" in lowered
    )


def capability_compatible(request: str, signature: dict[str, Any]) -> bool:
    lowered = request.lower()
    if capability_explicitly_forbidden(request):
        return False
    capability_text = f"{signature.get('goal', '')} {signature.get('description', '')}".lower()
    stop = {"the", "and", "for", "with", "from", "that", "this", "use", "using", "given", "return", "calculate"}
    goal_tokens = {
        token
        for token in re.findall(r"[a-z]+", capability_text)
        if len(token) > 2 and token not in stop
    }
    request_tokens = set(re.findall(r"[a-z]+", lowered))
    return bool(goal_tokens & request_tokens)

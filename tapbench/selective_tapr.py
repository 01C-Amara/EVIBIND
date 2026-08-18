from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from copy import deepcopy
from typing import Any

from .capc import _public_tool, certify_proposal
from .eflrx import (
    ACTION_RISK_THRESHOLD,
    NO_CALL_ID,
    RequestFn,
    _merge_metadata,
    _non_call,
    _tool_catalog,
    _tool_election_messages,
    _tool_election_schema,
    _tool_name,
)
from .eflrx_baselines import run_raw_baseline
from .extractive_candidates import (
    EXTRACTIVE_CANDIDATE_VERSION,
    build_extractive_candidate_table,
    canonical_slots,
    user_request_text,
)
from .r2_model_runner import _request_schema_json


SELECTIVE_TAPR_VERSION = "tapbench.selective_tapr.v4"
SELECTIVE_TAPR_CONDITION = "tap_r_selective_full"
ADMISSION_VERSION = "tapbench.speech_act_admission.v3"
EFFECT_SUPPORT_VERSION = "tapbench.effect_support_verifier.v1"
SCOPE_GUARD_VERSION = "tapbench.deterministic_scope_guard.v1"
CERTIFICATE_SPAN_POLICY_VERSION = "tapbench.certificate_span_policy.v2"

_ACTION_REQUEST = "ACTION_REQUEST"
_DIRECT_ANSWER = "DIRECT_ANSWER"
_UNCLEAR = "UNSAFE_OR_UNCLEAR"
_EFFECT_SUPPORTED = "EFFECT_SUPPORTED"
_EFFECT_UNSUPPORTED = "EFFECT_UNSUPPORTED"
_ADMISSION_OPTIONS = (
    (
        _ACTION_REQUEST,
        "The user asks the system to perform, change, retrieve, or submit "
        "something through an external capability. Missing details do not "
        "turn an action request into an informational question.",
    ),
    (
        _DIRECT_ANSWER,
        "The user asks only for an explanation, comparison, or factual answer "
        "and does not request an external effect.",
    ),
    (
        _UNCLEAR,
        "The requested effect or authorization is too ambiguous to classify "
        "safely.",
    ),
)

_LITERAL_TRANSFORMS = {
    "identity",
    "trim_surrounding_whitespace_and_punctuation",
    "normalize_quoted_string",
    "parse_integer_or_decimal",
    "explicit_boolean_polarity",
    "split_explicit_list",
}
_PROPOSAL_METHODS = ("full_tap_b2", "prompt_few_shot")
_UNIVERSAL_SCOPE_PATTERNS = (
    r"\ball\b",
    r"\bevery\b",
    r"\beach\b",
    r"\bacross all\b",
    r"\bacross every\b",
)
_EXPLICIT_BULK_TERMS = {"bulk", "batch", "all", "every", "scope"}
_STRING_ENVELOPE_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:[.#@/_:+\-'][A-Za-z0-9]+)*"
)
_TERMINAL_SENTENCE_PUNCTUATION_RE = re.compile(r"[.;:,!?]$")


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()


def _admission_catalog() -> tuple[list[dict[str, Any]], dict[int, str]]:
    options = list(_ADMISSION_OPTIONS)
    catalog = [
        {
            "selection_id": index,
            "decision": decision,
            "description": description,
        }
        for index, (decision, description) in enumerate(options)
    ]
    return catalog, {
        int(row["selection_id"]): str(row["decision"])
        for row in catalog
    }


def _boolean_schema(field: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {field: {"type": "boolean"}},
        "required": [field],
        "additionalProperties": False,
    }


def _external_action_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Decide whether satisfying the user's request requires an "
                "external operation such as creating, changing, retrieving, "
                "submitting, booking, or sending. Missing details do not make "
                "an external action request informational. Return only JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Request:\n"
                + user_request_text(messages)
                + "\nReturn {\"requires_external_action\": boolean}."
            ),
        },
    ]


def _text_suffices_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Decide whether a factual or explanatory text response alone "
                "fully satisfies the user's request, with no external state "
                "read or change. Return only JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Request:\n"
                + user_request_text(messages)
                + "\nReturn {\"text_answer_suffices\": boolean}."
            ),
        },
    ]


def _effect_support_catalog(
    *,
    reverse: bool,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    options = [
        (
            _EFFECT_SUPPORTED,
            "The function directly and completely supports the requested external effect. Missing argument values are handled later and do not make an otherwise matching effect unsupported.",
        ),
        (
            _EFFECT_UNSUPPORTED,
            "The function performs a different, narrower, broader, or differently authorized external effect.",
        ),
    ]
    if reverse:
        options.reverse()
    catalog = [
        {
            "selection_id": index,
            "decision": decision,
            "description": description,
        }
        for index, (decision, description) in enumerate(options)
    ]
    return catalog, {
        int(row["selection_id"]): str(row["decision"])
        for row in catalog
    }


def _effect_support_messages(
    messages: list[dict[str, Any]],
    tool: dict[str, Any],
    catalog: list[dict[str, Any]],
) -> list[dict[str, str]]:
    slots = canonical_slots(tool)
    capability = {
        "name": _tool_name(tool),
        "description": str(tool.get("description", "")),
        "required_slots": [row["name"] for row in slots if row["required"]],
    }
    return [
        {
            "role": "system",
            "content": (
                "Verify only whether the candidate function's external effect "
                "directly and completely matches the requested effect. Ignore "
                "missing argument values; the contract handles those later. "
                "Reject scope, cardinality, authorization, or operation mismatches. "
                "Return one identifier and no reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                "Request:\n"
                + user_request_text(messages)
                + "\nCandidate capability:\n"
                + json.dumps(capability, sort_keys=True, separators=(",", ":"))
                + "\nCapability decision catalog:\n"
                + json.dumps(catalog, sort_keys=True, separators=(",", ":"))
                + "\nReturn {\"selection_id\": integer}."
            ),
        },
    ]


def _admission_messages(
    messages: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Classify only the user's speech act and authorization. Do not "
                "select a tool or generate arguments. An action request may be "
                "incomplete; incompleteness is handled later by the contract. "
                "Return one identifier and no reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                "Request:\n"
                + user_request_text(messages)
                + "\nDecision catalog:\n"
                + json.dumps(
                    catalog,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                + "\nReturn {\"selection_id\": integer}."
            ),
        },
    ]


def _id_schema(ids: list[int]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selection_id": {
                "type": "integer",
                "enum": ids,
            }
        },
        "required": ["selection_id"],
        "additionalProperties": False,
    }


def _literal_table(table: dict[str, Any]) -> dict[str, Any]:
    literal = deepcopy(table)
    slots = literal.get("slots", {})
    for slot, rows in list(slots.items()):
        slots[slot] = [
            row
            for row in rows
            if str(row.get("transform")) in _LITERAL_TRANSFORMS
        ]
    literal["candidate_count"] = sum(len(rows) for rows in slots.values())
    literal["evidence_tier"] = "literal"
    return literal


def _required_missing(table: dict[str, Any]) -> list[str]:
    return sorted(
        str(slot)
        for slot in table.get("required_slots", [])
        if not table.get("slots", {}).get(slot)
    )


def deterministic_scope_guard(
    messages: list[dict[str, Any]],
    tool: dict[str, Any],
) -> dict[str, Any]:
    request = user_request_text(messages).casefold()
    markers = [
        pattern for pattern in _UNIVERSAL_SCOPE_PATTERNS
        if re.search(pattern, request)
    ]
    capability_text = " ".join(
        [
            _tool_name(tool),
            str(tool.get("description", "")),
            *(str(row["name"]) for row in canonical_slots(tool)),
        ]
    ).casefold()
    capability_terms = set(re.findall(r"[a-z0-9]+", capability_text))
    explicit_bulk_terms = sorted(capability_terms & _EXPLICIT_BULK_TERMS)
    blocked = bool(markers and not explicit_bulk_terms)
    return {
        "schema_version": SCOPE_GUARD_VERSION,
        "blocked": blocked,
        "universal_scope_markers": markers,
        "explicit_bulk_terms": explicit_bulk_terms,
        "basis": (
            "universal_request_without_bulk_contract"
            if blocked
            else "scope_not_deterministically_blocked"
        ),
    }


def certificate_span_conflicts(
    certification: dict[str, Any],
) -> list[str]:
    certificates = certification.get("certificates", {})
    if not isinstance(certificates, dict):
        return []
    rows = []
    for slot, certificate in certificates.items():
        span = certificate.get("source_span") if isinstance(certificate, dict) else None
        if (
            isinstance(span, list)
            and len(span) == 2
            and all(isinstance(value, int) for value in span)
        ):
            rows.append((str(slot), int(span[0]), int(span[1])))
    conflicts: set[str] = set()
    for index, (left_slot, left_start, left_end) in enumerate(rows):
        for right_slot, right_start, right_end in rows[index + 1 :]:
            if max(left_start, right_start) >= min(left_end, right_end):
                continue
            left_contains = left_start <= right_start and left_end >= right_end
            right_contains = right_start <= left_start and right_end >= left_end
            if left_contains and (left_start, left_end) != (right_start, right_end):
                conflicts.add(left_slot)
            elif right_contains and (left_start, left_end) != (right_start, right_end):
                conflicts.add(right_slot)
            else:
                conflicts.update((left_slot, right_slot))
    return sorted(conflicts)


def certificate_semantic_envelope_violations(
    certification: dict[str, Any],
    tool: dict[str, Any],
) -> list[str]:
    """Reject ambiguous identity spans unless the schema declares their shape."""
    certificates = certification.get("certificates", {})
    if not isinstance(certificates, dict):
        return []
    slot_schemas = {
        str(row["name"]): row["schema"]
        for row in canonical_slots(tool)
    }
    violations: list[str] = []
    for slot, certificate in certificates.items():
        if not isinstance(certificate, dict):
            continue
        schema = slot_schemas.get(str(slot), {})
        if (
            str(schema.get("type", "")).casefold() != "string"
            or isinstance(schema.get("enum"), list)
            or certificate.get("transform") != "identity"
        ):
            continue
        value = certificate.get("value")
        if not isinstance(value, str):
            violations.append(str(slot))
            continue

        declared = str(
            schema.get("x-tap-semantic-envelope", "infer")
        ).casefold()
        if declared == "free_text":
            continue
        tokens = _STRING_ENVELOPE_TOKEN_RE.findall(value.strip())
        digit_tokens = [
            token for token in tokens if any(char.isdigit() for char in token)
        ]
        valid = bool(tokens) and not _TERMINAL_SENTENCE_PUNCTUATION_RE.search(
            value.strip()
        )
        if declared == "opaque_atom":
            valid = valid and len(tokens) == 1
        elif declared == "uri":
            valid = bool(
                re.fullmatch(r"[A-Za-z][A-Za-z0-9+.-]*://\S+", value.strip())
            )
        elif declared == "head_number":
            valid = (
                valid
                and len(tokens) == 2
                and tokens[0].isalpha()
                and tokens[1].isdigit()
            )
        elif digit_tokens:
            if len(tokens) == 1:
                valid = valid and not tokens[0].isdigit()
            else:
                valid = (
                    valid
                    and len(tokens) == 2
                    and tokens[0].isalpha()
                    and tokens[1].isdigit()
                )
        if not valid:
            violations.append(str(slot))
    return sorted(violations)


def _proposal_unsatisfied_slots(
    proposal: dict[str, Any],
    tool: dict[str, Any],
    candidate_table: dict[str, Any],
) -> tuple[str, ...]:
    slots = canonical_slots(tool)
    required = {str(row["name"]) for row in slots if row["required"]}
    aliases = {
        alias: str(row["name"])
        for row in slots
        for alias in (str(row["surface_name"]), str(row["name"]))
    }
    if not isinstance(proposal, dict):
        return ()
    if proposal.get("mode") == "clarify":
        raw = proposal.get("payload", {}).get("missing_slots", [])
        if isinstance(raw, str):
            raw = [raw]
        selected = {
            aliases.get(str(value), str(value))
            for value in raw
            if aliases.get(str(value), str(value)) in required
        }
        return tuple(sorted(selected))
    if proposal.get("mode") != "call":
        return ()
    raw_arguments = proposal.get("arguments")
    if not isinstance(raw_arguments, dict):
        return ()
    arguments = {
        aliases.get(str(slot), str(slot)): value
        for slot, value in raw_arguments.items()
        if aliases.get(str(slot), str(slot)) in required
    }
    unsupported = set(required - set(arguments))
    for slot, value in arguments.items():
        if not any(
            row.get("value") == value
            for row in candidate_table.get("slots", {}).get(slot, [])
        ):
            unsupported.add(slot)
    return tuple(sorted(unsupported))


def _certificate_transform_tier(certification: dict[str, Any]) -> str:
    transforms = {
        str(row.get("transform"))
        for row in certification.get("certificates", {}).values()
    }
    return (
        "literal"
        if transforms <= _LITERAL_TRANSFORMS
        else "bounded_transform"
    )


def run_selective_tapr_resolution(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    endpoint: str,
    max_tokens: int,
    seed: int,
    request_fn: RequestFn = _request_schema_json,
    semantic_extent_enabled: bool = True,
    exhaust_proposal_budget: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    metadata: dict[str, Any] = {
        "selective_tapr_version": SELECTIVE_TAPR_VERSION,
        "admission_version": ADMISSION_VERSION,
        "effect_support_version": EFFECT_SUPPORT_VERSION,
        "scope_guard_version": SCOPE_GUARD_VERSION,
        "certificate_span_policy_version": CERTIFICATE_SPAN_POLICY_VERSION,
        "extractive_candidate_version": EXTRACTIVE_CANDIDATE_VERSION,
        "action_risk_threshold": ACTION_RISK_THRESHOLD,
        "admission_elections": [],
        "tool_elections": [],
        "effect_support_elections": [],
        "proposal_attempts": [],
        "generation_calls": 0,
        "model_literal_entered_action": False,
        "semantic_extent_enabled": semantic_extent_enabled,
        "exhaust_proposal_budget": exhaust_proposal_budget,
    }
    call_metadata: list[dict[str, Any]] = []

    admission_decisions: list[str] = []
    catalog, mapping = _admission_catalog()
    raw, response = request_fn(
        endpoint,
        _admission_messages(messages, catalog),
        response_schema=_id_schema([int(row["selection_id"]) for row in catalog]),
        max_tokens=max_tokens,
        temperature=0.0,
        seed=seed,
    )
    try:
        selection_id = int(raw.get("selection_id"))
    except (AttributeError, TypeError, ValueError):
        selection_id = -999
    ternary_decision = mapping.get(selection_id, "INVALID")
    admission_decisions.append(ternary_decision)
    metadata["admission_elections"].append(
        {
            "view": "ternary_speech_act",
            "decision": ternary_decision,
            "selection_id": selection_id,
            "catalog_sha256": _stable_hash(catalog),
        }
    )
    call_metadata.append(response)

    raw, response = request_fn(
        endpoint,
        _external_action_messages(messages),
        response_schema=_boolean_schema("requires_external_action"),
        max_tokens=max_tokens,
        temperature=0.0,
        seed=seed + 100003,
    )
    external_value = raw.get("requires_external_action") if isinstance(raw, dict) else None
    external_decision = (
        _ACTION_REQUEST if external_value is True
        else _DIRECT_ANSWER if external_value is False
        else "INVALID"
    )
    admission_decisions.append(external_decision)
    metadata["admission_elections"].append(
        {
            "view": "external_action_required",
            "decision": external_decision,
            "raw_boolean": external_value if isinstance(external_value, bool) else None,
        }
    )
    call_metadata.append(response)

    raw, response = request_fn(
        endpoint,
        _text_suffices_messages(messages),
        response_schema=_boolean_schema("text_answer_suffices"),
        max_tokens=max_tokens,
        temperature=0.0,
        seed=seed + 200006,
    )
    text_value = raw.get("text_answer_suffices") if isinstance(raw, dict) else None
    text_decision = (
        _DIRECT_ANSWER if text_value is True
        else _ACTION_REQUEST if text_value is False
        else "INVALID"
    )
    admission_decisions.append(text_decision)
    metadata["admission_elections"].append(
        {
            "view": "text_answer_suffices",
            "decision": text_decision,
            "raw_boolean": text_value if isinstance(text_value, bool) else None,
        }
    )
    call_metadata.append(response)

    admission_counts = Counter(
        value for value in admission_decisions if value != "INVALID"
    )
    admitted, admitted_count = (
        admission_counts.most_common(1)[0]
        if admission_counts
        else ("INVALID", 0)
    )
    if ternary_decision == _UNCLEAR:
        admitted = _UNCLEAR
        admitted_count = 1
    admission_agreement = admitted_count >= 2 and admitted != _UNCLEAR
    metadata["admission_agreement"] = admission_agreement
    metadata["admission_decisions"] = admission_decisions
    if not admission_agreement:
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "action_risk_score": 1.0,
                "risk_factors": {"admission_disagreement": 1.0},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call("refuse", "speech-act admission views disagree"), metadata

    if admitted == _DIRECT_ANSWER:
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "action_risk_score": 0.01,
                "risk_factors": {"admission_agreement": 0.01},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call("direct_answer", "agreed informational request"), metadata
    if admitted != _ACTION_REQUEST:
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "action_risk_score": 1.0,
                "risk_factors": {"unclear_authorization": 1.0},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call("refuse", "effect authorization is unclear"), metadata

    selected_tools: list[str] = []
    for index, reverse in enumerate((False, True)):
        catalog, mapping = _tool_catalog(tools, reverse=reverse)
        raw, response = request_fn(
            endpoint,
            _tool_election_messages(messages, catalog),
            response_schema=_tool_election_schema(catalog),
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed + 300007 + index * 100003,
        )
        try:
            selection_id = int(raw.get("selection_id"))
        except (AttributeError, TypeError, ValueError):
            selection_id = NO_CALL_ID - 1
        selected = mapping.get(selection_id, "INVALID")
        selected_tools.append(selected)
        metadata["tool_elections"].append(
            {
                "order": "reverse" if reverse else "forward",
                "selected": selected,
                "selection_id": selection_id,
                "catalog_sha256": _stable_hash(catalog),
            }
        )
        call_metadata.append(response)

    tool_agreement = bool(
        selected_tools
        and selected_tools[0] != "INVALID"
        and all(value == selected_tools[0] for value in selected_tools)
    )
    metadata["tool_agreement"] = tool_agreement
    metadata["selected_tools"] = selected_tools
    if not tool_agreement:
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "action_risk_score": 1.0,
                "risk_factors": {"tool_disagreement": 1.0},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call("refuse", "counterbalanced tool elections disagree"), metadata

    selected_tool = selected_tools[0]
    if selected_tool == "NO_CALL":
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "action_risk_score": 0.02,
                "risk_factors": {
                    "admission_agreement": 0.01,
                    "no_call_agreement": 0.01,
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call("no_tool", "agreed action has no catalog capability"), metadata

    tool = _public_tool(tools, selected_tool)
    if tool is None:
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "action_risk_score": 1.0,
                "risk_factors": {"catalog_resolution": 1.0},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call("refuse", "selected tool is absent"), metadata

    scope_guard = deterministic_scope_guard(messages, tool)
    metadata["scope_guard"] = scope_guard
    if scope_guard["blocked"]:
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "action_risk_score": 0.01,
                "risk_factors": {"deterministic_scope_guard": 0.01},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call(
            "no_tool",
            "requested scope exceeds the elected function contract",
        ), metadata

    effect_decisions: list[str] = []
    for index, reverse in enumerate((False, True)):
        catalog, mapping = _effect_support_catalog(reverse=reverse)
        raw, response = request_fn(
            endpoint,
            _effect_support_messages(messages, tool, catalog),
            response_schema=_id_schema(
                [int(row["selection_id"]) for row in catalog]
            ),
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed + 500009 + index * 100003,
        )
        try:
            selection_id = int(raw.get("selection_id"))
        except (AttributeError, TypeError, ValueError):
            selection_id = -999
        decision = mapping.get(selection_id, "INVALID")
        effect_decisions.append(decision)
        metadata["effect_support_elections"].append(
            {
                "order": "reverse" if reverse else "forward",
                "decision": decision,
                "selection_id": selection_id,
                "catalog_sha256": _stable_hash(catalog),
            }
        )
        call_metadata.append(response)
    effect_agreement = bool(
        effect_decisions
        and effect_decisions[0] != "INVALID"
        and all(value == effect_decisions[0] for value in effect_decisions)
    )
    metadata["effect_support_agreement"] = effect_agreement
    metadata["effect_support_decisions"] = effect_decisions
    if not effect_agreement:
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "action_risk_score": 1.0,
                "risk_factors": {"effect_support_disagreement": 1.0},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call("refuse", "effect-support views disagree"), metadata
    if effect_decisions[0] == _EFFECT_UNSUPPORTED:
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "action_risk_score": 0.02,
                "risk_factors": {"unsupported_effect_agreement": 0.02},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call(
            "no_tool",
            "elected function does not completely support the requested effect",
        ), metadata

    candidate_table = build_extractive_candidate_table(
        messages,
        tool,
        include_optional=True,
    )
    literal_table = _literal_table(candidate_table)
    missing = _required_missing(candidate_table)
    metadata["candidate_table"] = {
        "schema_version": candidate_table.get("schema_version"),
        "candidate_count": candidate_table.get("candidate_count"),
        "literal_candidate_count": literal_table.get("candidate_count"),
        "required_slots": candidate_table.get("required_slots"),
        "optional_slots": candidate_table.get("optional_slots"),
        "sha256": _stable_hash(candidate_table),
        "literal_sha256": _stable_hash(literal_table),
    }
    if missing:
        metadata.update(_merge_metadata(call_metadata))
        metadata.update(
            {
                "candidate_precheck": {
                    "status": "missing",
                    "missing_slots": missing,
                },
                "clarification_source": "minimal_unsatisfied_contract",
                "action_risk_score": 0.02,
                "risk_factors": {
                    "admission_agreement": 0.01,
                    "tool_agreement": 0.01,
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call(
            "clarify",
            "required evidence is absent from the request",
            missing=missing,
        ), metadata

    proposal_case = {
        "messages": deepcopy(messages),
        "tools": deepcopy(tools),
    }
    accepted_action: dict[str, Any] | None = None
    accepted_certification: dict[str, Any] | None = None
    accepted_index: int | None = None
    accepted_tier: str | None = None
    unsatisfied_votes: list[tuple[str, ...]] = []
    for index, proposal_method in enumerate(_PROPOSAL_METHODS):
        proposal, response = run_raw_baseline(
            proposal_case,
            endpoint=endpoint,
            condition=proposal_method,
            max_tokens=max_tokens,
            seed=seed + 700001 + index * 100003,
            request_fn=request_fn,
        )
        call_metadata.append(response)
        proposal_unsatisfied = set(
            _proposal_unsatisfied_slots(proposal, tool, candidate_table)
        )
        certification_statuses = []
        for tier, table in (
            ("literal", literal_table),
            ("bounded_transform", candidate_table),
        ):
            certified_action, certification = certify_proposal(
                proposal,
                selected_tool=selected_tool,
                tool=tool,
                candidate_table=table,
                tools=tools,
            )
            if certified_action is not None:
                conflicts = certificate_span_conflicts(certification)
                if conflicts:
                    proposal_unsatisfied.update(conflicts)
                    certified_action = None
                    certification = {
                        "status": "cross_slot_source_span_conflict",
                        "conflict_slots": conflicts,
                    }
            if certified_action is not None and semantic_extent_enabled:
                envelope_violations = certificate_semantic_envelope_violations(
                    certification,
                    tool,
                )
                if envelope_violations:
                    proposal_unsatisfied.update(envelope_violations)
                    certified_action = None
                    certification = {
                        "status": "semantic_envelope_violation",
                        "violation_slots": envelope_violations,
                    }
            certification_statuses.append(
                {
                    "tier": tier,
                    "status": certification.get("status"),
                    "conflict_slots": certification.get("conflict_slots", []),
                    "violation_slots": certification.get("violation_slots", []),
                }
            )
            if certified_action is not None:
                if accepted_action is None:
                    accepted_action = certified_action
                    accepted_certification = certification
                    accepted_index = index
                    accepted_tier = _certificate_transform_tier(certification)
                break
        unsatisfied_votes.append(tuple(sorted(proposal_unsatisfied)))
        metadata["proposal_attempts"].append(
            {
                "proposal_index": index,
                "proposal_method": proposal_method,
                "proposal_sha256": _stable_hash(proposal),
                "unsatisfied_slots": list(unsatisfied_votes[-1]),
                "certification": certification_statuses,
            }
        )
        if accepted_action is not None and not exhaust_proposal_budget:
            break

    metadata.update(_merge_metadata(call_metadata))
    if accepted_action is None or accepted_certification is None:
        agreed_unsatisfied = bool(
            len(unsatisfied_votes) == len(_PROPOSAL_METHODS)
            and unsatisfied_votes[0]
            and all(value == unsatisfied_votes[0] for value in unsatisfied_votes)
        )
        if agreed_unsatisfied:
            missing_slots = list(unsatisfied_votes[0])
            metadata.update(
                {
                    "proposal_admitted": False,
                    "clarification_source": "agreed_unsatisfied_contract",
                    "action_risk_score": 0.03,
                    "risk_factors": {
                        "admission_and_tool_agreement": 0.02,
                        "unsatisfied_contract_agreement": 0.01,
                    },
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return _non_call(
                "clarify",
                "independent proposals expose the same unsatisfied contract slots",
                missing=missing_slots,
            ), metadata
        metadata.update(
            {
                "proposal_admitted": False,
                "action_risk_score": 1.0,
                "risk_factors": {"no_certified_proposal": 1.0},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call("refuse", "no complete proposal is certificate-backed"), metadata

    assert accepted_index is not None and accepted_tier is not None
    proposal_risk = 0.0 if accepted_index == 0 else 0.02
    evidence_risk = 0.0 if accepted_tier == "literal" else 0.01
    risk = 0.02 + proposal_risk + evidence_risk
    metadata.update(
        {
            "proposal_admitted": True,
            "accepted_proposal_index": accepted_index,
            "accepted_proposal_method": _PROPOSAL_METHODS[accepted_index],
            "accepted_evidence_tier": accepted_tier,
            "evidence_certificates": accepted_certification["certificates"],
            "certificate_count": accepted_certification.get("certificate_count", 0),
            "action_risk_score": risk,
            "risk_factors": {
                "admission_and_tool_agreement": 0.02,
                "proposal_fallback": proposal_risk,
                "bounded_transform": evidence_risk,
            },
            "risk_gate_passed": risk <= ACTION_RISK_THRESHOLD,
            "contract_valid": True,
            "materialized_action_sha256": _stable_hash(accepted_action),
            "model_literal_entered_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    if risk > ACTION_RISK_THRESHOLD:
        return _non_call("refuse", "composed action risk exceeds threshold"), metadata
    return accepted_action, metadata

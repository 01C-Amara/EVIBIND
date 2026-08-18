from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable

from .contract_solver import CONTRACT_SOLVER_VERSION, resolve_pointer_contract
from .evidence_contract import (
    build_candidate_lattice,
    certified_candidates,
    request_text,
)
from .r2_model_runner import _request_schema_json


EFFECT_FIRST_VERSION = "tapbench.effect_first_resolution.v1"
EFFECT_FIRST_CONDITIONS = (
    "tap_r_effect_first_single_locked",
    "tap_r_effect_first_consensus_locked",
)
ACTION_RISK_THRESHOLD = 0.05

_MODE_RISK = {
    "explicit": 0.001,
    "inferred": 0.02,
    "unknown": 1.0,
}
_TOOL_RISK = {
    "single": 0.03,
    "agreement": 0.01,
    "disagreement": 0.25,
}
_SOURCE_AUTHORITY = {
    "trusted_state": 5,
    "schema_declared_user_span": 4,
    "contract_default": 3,
    "user_span": 2,
}
_SOURCE_RISK = {
    "trusted_state": 0.001,
    "schema_declared_user_span": 0.002,
    "contract_default": 0.002,
    "user_span": 0.01,
}

_DIRECT_PATTERNS = (
    r"\banswer directly\b",
    r"\brespond (?:only )?in (?:the )?chat\b",
    r"\bgive (?:me )?(?:a |an )?(?:brief )?(?:answer|explanation|summary)\b",
    r"\bexplain only\b",
)
_NO_TOOL_PATTERNS = (
    r"\bno [a-z0-9 _-]+ action is (?:requested|authorized|needed)\b",
    r"\bnot (?:an? )?authori[sz]ation\b",
    r"\bnot asking (?:you )?to (?:perform|execute|run|call|use|submit|create|send|book|schedule)\b",
    r"\bdo not (?:perform|execute|run|call|use|submit|create|send|book|schedule)\b",
    r"\bwithout (?:performing|executing|running|calling|using) (?:a |the )?tool\b",
    r"\bfor discussion only\b",
    r"\bhypothetical context only\b",
)
_INFORMATIONAL_START = re.compile(
    r"^\s*(?:what|why|how|when|where|which|who|explain|compare|describe|"
    r"tell me|list|summari[sz]e|is|are|can|could|would|should|does|do)\b",
    re.IGNORECASE,
)


def _effect_text(messages: Iterable[dict[str, Any]]) -> str:
    text = request_text(messages)
    return re.split(r"\n\s*Evidence fields\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()


def _tool_action_verbs(tools: Iterable[dict[str, Any]]) -> set[str]:
    verbs: set[str] = set()
    for tool in tools:
        name = str(tool.get("canonical_name") or tool.get("name") or "")
        tokens = re.findall(r"[a-z]+", name.casefold())
        if tokens:
            verbs.add(tokens[0])
        description = str(tool.get("description", "")).casefold()
        match = re.match(
            r"\s*(?:perform|use this tool to|this tool will)?\s*"
            r"(create|send|schedule|book|submit|grant|revoke|purchase|issue|"
            r"deploy|reserve|rent|sign|initiate|log|update|cancel|delete|search)\b",
            description,
        )
        if match:
            verbs.add(match.group(1))
    return verbs


def effect_admission(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify whether any tool effect is authorized using runtime text only."""
    text = _effect_text(messages)
    lowered = text.casefold()

    if any(re.search(pattern, lowered) for pattern in _DIRECT_PATTERNS):
        return {
            "mode": "direct_answer",
            "basis": "explicit_direct_answer",
            "certainty": "explicit",
            "risk": _MODE_RISK["explicit"],
            "effect_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    if any(re.search(pattern, lowered) for pattern in _NO_TOOL_PATTERNS):
        return {
            "mode": "no_tool",
            "basis": "explicit_effect_denial",
            "certainty": "explicit",
            "risk": _MODE_RISK["explicit"],
            "effect_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }

    verbs = _tool_action_verbs(tools)
    first_clause = re.split(r"[.!?\n]", lowered, maxsplit=1)[0]
    explicitly_actionable = bool(re.search(r"\bplease\b", first_clause))
    explicitly_actionable = explicitly_actionable or any(
        re.search(rf"(?:^|\b){re.escape(verb)}\b", first_clause)
        for verb in verbs
    )
    if explicitly_actionable:
        return {
            "mode": "call_candidate",
            "basis": "explicit_effect_verb",
            "certainty": "explicit",
            "risk": _MODE_RISK["explicit"],
            "effect_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }

    if _INFORMATIONAL_START.search(text) or text.rstrip().endswith("?"):
        return {
            "mode": "no_tool",
            "basis": "informational_speech_act",
            "certainty": "inferred",
            "risk": _MODE_RISK["inferred"],
            "effect_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }

    return {
        "mode": "refuse",
        "basis": "effect_authorization_not_established",
        "certainty": "unknown",
        "risk": _MODE_RISK["unknown"],
        "effect_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def _value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _candidate_rank(candidate: dict[str, Any]) -> tuple[int, float, float]:
    return (
        _SOURCE_AUTHORITY.get(str(candidate.get("source_kind")), 1),
        float(candidate.get("role_score", 0.0)),
        float(candidate.get("evidence_strength", 0.0)),
    )


def common_missing_required(lattice: dict[str, Any]) -> list[str]:
    missing_sets = []
    for tool in lattice.get("tools", {}).values():
        missing_sets.append(
            {
                str(slot)
                for slot, row in tool.get("slots", {}).items()
                if row.get("required") and not certified_candidates(row)
            }
        )
    if not missing_sets:
        return []
    first = missing_sets[0]
    return sorted(first) if first and all(row == first for row in missing_sets[1:]) else []


def lock_tool_evidence(
    lattice: dict[str, Any],
    tool_name: str,
) -> dict[str, Any]:
    """Lock every required slot to one unique highest-authority certificate."""
    tool = lattice.get("tools", {}).get(tool_name)
    if not isinstance(tool, dict):
        return {
            "status": "unknown_tool",
            "tool": tool_name,
            "assignments": {},
            "missing_slots": [],
            "ambiguous_slots": [],
            "risk_factors": [1.0],
            "locks": {},
        }

    assignments: dict[str, int] = {}
    missing: list[str] = []
    ambiguous: list[str] = []
    locks: dict[str, dict[str, Any]] = {}
    risk_factors: list[float] = []

    for slot, slot_row in tool.get("slots", {}).items():
        if not slot_row.get("required"):
            continue
        candidates = certified_candidates(slot_row)
        if not candidates:
            missing.append(str(slot))
            continue
        highest_authority = max(
            _SOURCE_AUTHORITY.get(str(row.get("source_kind")), 1)
            for row in candidates
        )
        authoritative = [
            row
            for row in candidates
            if _SOURCE_AUTHORITY.get(str(row.get("source_kind")), 1)
            == highest_authority
        ]
        by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in authoritative:
            by_value[_value_key(candidate.get("value"))].append(candidate)
        if len(by_value) != 1:
            ambiguous.append(str(slot))
            continue
        selected = max(
            next(iter(by_value.values())),
            key=lambda row: (
                _candidate_rank(row),
                -int(row.get("candidate_id", 0)),
            ),
        )
        candidate_id = int(selected["candidate_id"])
        assignments[str(slot)] = candidate_id
        source_kind = str(selected.get("source_kind"))
        risk = _SOURCE_RISK.get(source_kind, 0.02)
        risk_factors.append(risk)
        locks[str(slot)] = {
            "candidate_id": candidate_id,
            "source_kind": source_kind,
            "source_span": selected.get("source_span"),
            "source_text_sha256": (
                hashlib.sha256(str(selected.get("source_text")).encode()).hexdigest()
                if selected.get("source_text") is not None
                else None
            ),
            "role_score": selected.get("role_score"),
            "evidence_strength": selected.get("evidence_strength"),
            "risk": risk,
        }

    status = "locked"
    if missing:
        status = "missing"
    elif ambiguous:
        status = "ambiguous"
    return {
        "status": status,
        "tool": tool_name,
        "tool_id": tool.get("tool_id"),
        "assignments": assignments,
        "missing_slots": missing,
        "ambiguous_slots": ambiguous,
        "risk_factors": risk_factors,
        "locks": locks,
    }


def _tool_catalog(
    lattice: dict[str, Any],
    *,
    reverse: bool,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    rows = list(lattice.get("tools", {}).items())
    if reverse:
        rows.reverse()
    catalog = []
    mapping: dict[int, str] = {}
    for election_id, (name, tool) in enumerate(rows):
        mapping[election_id] = str(name)
        catalog.append(
            {
                "tool_id": election_id,
                "tool": str(name),
                "description": str(tool.get("description", "")),
                "required_slots": sorted(
                    str(slot)
                    for slot, row in tool.get("slots", {}).items()
                    if row.get("required")
                ),
            }
        )
    return catalog, mapping


def _election_messages(
    messages: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Select only the tool whose external effect is explicitly requested. "
                "Ignore whether request values happen to fit a schema. Return one JSON "
                "object with tool_id only. Never generate arguments or reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Requested effect: {_effect_text(messages)}\n"
                f"Candidate effects: {json.dumps(catalog, sort_keys=True)}\n"
                "Choose the exact requested effect."
            ),
        },
    ]


def _election_schema(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool_id": {
                "type": "integer",
                "enum": [int(row["tool_id"]) for row in catalog],
            }
        },
        "required": ["tool_id"],
        "additionalProperties": False,
    }


def _merge_call_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_ms",
        "generation_ms",
    )
    merged: dict[str, Any] = {
        key: sum(float(row.get(key) or 0.0) for row in rows)
        for key in numeric
    }
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        merged[key] = int(merged[key])
    elapsed_generation = merged["generation_ms"] / 1000.0
    merged["generated_tokens_per_second"] = (
        merged["completion_tokens"] / elapsed_generation
        if elapsed_generation > 0
        else None
    )
    merged["finish_reason"] = (
        "length"
        if any(row.get("finish_reason") == "length" for row in rows)
        else "stop"
    )
    merged["context_truncated"] = any(
        bool(row.get("context_truncated")) for row in rows
    )
    merged["generation_calls"] = sum(
        int(row["generation_calls"])
        if row.get("generation_calls") is not None
        else 1
        for row in rows
    )
    return merged


def _non_call_action(mode: str, *, reason: str, missing: list[str] | None = None) -> dict[str, Any]:
    if mode == "clarify":
        payload = {"missing_slots": list(missing or [])}
    elif mode == "direct_answer":
        payload = {"answer": "respond directly without tool execution"}
    else:
        payload = {"reason": reason}
    return {"mode": mode, "tool": None, "arguments": {}, "payload": payload}


def run_effect_first_resolution(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    dialogue_state: dict[str, Any],
    reference_context: dict[str, Any],
    endpoint: str,
    condition: str,
    max_tokens: int,
    seed: int,
    candidate_seed: int = 17,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if condition not in EFFECT_FIRST_CONDITIONS:
        raise ValueError(f"unknown effect-first condition: {condition}")

    started = time.perf_counter()
    admission = effect_admission(messages, tools)
    lattice = build_candidate_lattice(
        messages,
        tools,
        dialogue_state=dialogue_state,
        reference_context=reference_context,
        candidate_seed=candidate_seed,
    )
    lattice["action_risk_budget"] = float(
        reference_context.get("action_risk_budget", ACTION_RISK_THRESHOLD)
    )
    metadata: dict[str, Any] = {
        "effect_first_version": EFFECT_FIRST_VERSION,
        "effect_admission": admission,
        "evidence_contract_version": lattice.get("schema_version"),
        "action_risk_threshold": ACTION_RISK_THRESHOLD,
        "generation_calls": 0,
        "elections": [],
    }

    if admission["mode"] in {"direct_answer", "no_tool", "refuse"}:
        action = _non_call_action(
            str(admission["mode"]),
            reason=str(admission["basis"]),
        )
        metadata.update(
            {
                "action_risk_score": float(admission["risk"]),
                "risk_factors": {"mode": float(admission["risk"])},
                "resolution": {
                    "schema_version": CONTRACT_SOLVER_VERSION,
                    "terminal_state": action["mode"],
                    "materialized_action": action,
                    "history": [],
                },
                "elapsed_seconds": time.perf_counter() - started,
                "finish_reason": "not_applicable",
                "context_truncated": False,
            }
        )
        return action, metadata

    common_missing = common_missing_required(lattice)
    if common_missing:
        action = _non_call_action(
            "clarify",
            reason="common required evidence is absent",
            missing=common_missing,
        )
        metadata.update(
            {
                "action_risk_score": float(admission["risk"]),
                "risk_factors": {"mode": float(admission["risk"])},
                "evidence_precheck": {
                    "status": "common_missing",
                    "missing_slots": common_missing,
                },
                "resolution": {
                    "schema_version": CONTRACT_SOLVER_VERSION,
                    "terminal_state": "clarify",
                    "materialized_action": action,
                    "history": [],
                },
                "elapsed_seconds": time.perf_counter() - started,
                "finish_reason": "not_applicable",
                "context_truncated": False,
            }
        )
        return action, metadata

    order_flags = [False]
    if condition == "tap_r_effect_first_consensus_locked":
        order_flags.append(True)
    selected_tools: list[str] = []
    call_metadata: list[dict[str, Any]] = []
    for election_index, reverse in enumerate(order_flags):
        catalog, mapping = _tool_catalog(lattice, reverse=reverse)
        raw, row_metadata = _request_schema_json(
            endpoint,
            _election_messages(messages, catalog),
            response_schema=_election_schema(catalog),
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed + election_index * 100003,
        )
        try:
            election_id = int(raw.get("tool_id"))
        except (TypeError, ValueError):
            election_id = -1
        selected = mapping.get(election_id)
        selected_tools.append(selected or "")
        metadata["elections"].append(
            {
                "order": "reverse" if reverse else "forward",
                "selected_tool": selected,
                "raw_tool_id": election_id,
                "catalog_sha256": hashlib.sha256(
                    json.dumps(catalog, sort_keys=True).encode()
                ).hexdigest(),
                "response_schema_sha256": row_metadata.get(
                    "response_schema_sha256"
                ),
            }
        )
        call_metadata.append(row_metadata)

    merged = _merge_call_metadata(call_metadata)
    metadata.update(merged)
    agreement = bool(
        selected_tools
        and selected_tools[0]
        and all(tool == selected_tools[0] for tool in selected_tools)
    )
    tool_risk = (
        _TOOL_RISK["agreement"]
        if len(selected_tools) > 1 and agreement
        else _TOOL_RISK["single"]
        if len(selected_tools) == 1 and selected_tools[0]
        else _TOOL_RISK["disagreement"]
    )
    if not agreement and len(selected_tools) > 1:
        action = _non_call_action(
            "refuse",
            reason="counterbalanced tool elections disagree",
        )
        score = min(1.0, float(admission["risk"]) + tool_risk)
        metadata.update(
            {
                "counterbalanced_agreement": False,
                "selected_tools": selected_tools,
                "action_risk_score": score,
                "risk_factors": {
                    "mode": float(admission["risk"]),
                    "tool": tool_risk,
                },
                "resolution": {
                    "schema_version": CONTRACT_SOLVER_VERSION,
                    "terminal_state": "refuse",
                    "materialized_action": action,
                    "history": [],
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return action, metadata

    tool_name = selected_tools[0] if selected_tools else ""
    lock = lock_tool_evidence(lattice, tool_name)
    evidence_risk = sum(float(value) for value in lock["risk_factors"])
    score = min(
        1.0,
        float(admission["risk"]) + tool_risk + evidence_risk,
    )
    risk_factors = {
        "mode": float(admission["risk"]),
        "tool": tool_risk,
        "evidence": evidence_risk,
    }
    if lock["status"] in {"missing", "ambiguous"}:
        mode = "clarify" if lock["missing_slots"] or lock["ambiguous_slots"] else "refuse"
        missing = sorted(set(lock["missing_slots"] + lock["ambiguous_slots"]))
        action = _non_call_action(
            mode,
            reason=f"evidence lock {lock['status']}",
            missing=missing,
        )
        metadata.update(
            {
                "counterbalanced_agreement": agreement,
                "selected_tools": selected_tools,
                "evidence_lock": lock,
                "action_risk_score": 1.0,
                "risk_factors": {**risk_factors, "evidence_failure": 1.0},
                "resolution": {
                    "schema_version": CONTRACT_SOLVER_VERSION,
                    "terminal_state": mode,
                    "materialized_action": action,
                    "history": [],
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return action, metadata

    pointer = {
        "mode": "call",
        "tool_id": int(lock["tool_id"]),
        "arguments": dict(lock["assignments"]),
    }
    resolution = resolve_pointer_contract(
        pointer,
        lattice,
        messages,
        budget=0,
    )
    action = deepcopy(resolution["materialized_action"])
    terminal = str(resolution.get("terminal_state"))
    if terminal != "call":
        action = _non_call_action(
            "refuse",
            reason=f"global contract terminal state {terminal}",
        )
        score = 1.0
        risk_factors["contract"] = 1.0
        resolution = {
            **resolution,
            "terminal_state": "refuse",
            "materialized_action": action,
        }
    elif score > ACTION_RISK_THRESHOLD:
        action = _non_call_action(
            "refuse",
            reason="composed action risk exceeds threshold",
        )
        resolution = {
            **resolution,
            "terminal_state": "refuse",
            "materialized_action": action,
        }

    metadata.update(
        {
            "counterbalanced_agreement": agreement,
            "selected_tools": selected_tools,
            "evidence_lock": lock,
            "action_risk_score": score,
            "risk_factors": risk_factors,
            "resolution": resolution,
            "elapsed_seconds": time.perf_counter() - started,
            "safety_invariant": (
                "all required values are unique highest-authority certified candidates"
            ),
        }
    )
    return action, metadata

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .evidence_contract import (
    ACTION_CRITICAL_ROLES,
    build_pointer_action_schema,
    capability_compatible,
    capability_explicitly_forbidden,
    capability_signature,
    certified_candidates,
    materialize_pointer_action,
    request_text,
)

CONTRACT_SOLVER_VERSION = "tapbench.contract_solver.v2"
TRANSITIONS = {
    "REROUTE_TOOL",
    "SELECT_ALTERNATE_CANDIDATE",
    "NORMALIZE",
    "DELETE_UNSUPPORTED_OPTIONAL",
    "REPAIR_CROSS_FIELD",
    "CONVERT_TO_CLARIFY",
    "CONVERT_TO_ANSWER",
    "CONVERT_TO_REFUSE",
    "ESCALATE",
}


def _tool_by_id(lattice: dict[str, Any], tool_id: Any) -> tuple[str, dict[str, Any]] | None:
    for name, row in lattice.get("tools", {}).items():
        if row.get("tool_id") == tool_id:
            return name, row
    return None


def _candidate_for(slot_row: dict[str, Any], candidate_id: Any) -> dict[str, Any] | None:
    return next((row for row in certified_candidates(slot_row) if row.get("candidate_id") == candidate_id), None)


def _required_critical_domains_available(tool: dict[str, Any]) -> bool:
    return all(
        not (slot_row.get("required") and slot_row.get("role") in ACTION_CRITICAL_ROLES)
        or bool(certified_candidates(slot_row))
        for slot_row in tool.get("slots", {}).values()
    )


def hard_violations(pointer_action: dict[str, Any], lattice: dict[str, Any], request: str) -> list[dict[str, Any]]:
    mode = pointer_action.get("mode")
    if mode != "call":
        return []
    selected = _tool_by_id(lattice, pointer_action.get("tool_id"))
    if selected is None:
        return [{"error": "unknown_tool", "hard": True, "slot": None}]
    tool_name, tool = selected
    lexical_match = capability_compatible(
        request,
        capability_signature({"name": tool_name, "description": tool.get("description", ""), "parameters": {}}),
    )
    if not lexical_match and not _required_critical_domains_available(tool):
        return [{"error": "capability_mismatch", "hard": True, "slot": None}]
    arguments = pointer_action.get("arguments", {}) if isinstance(pointer_action.get("arguments"), dict) else {}
    errors = []
    selected_action_risk = 0.0
    for slot, slot_row in tool.get("slots", {}).items():
        candidates = certified_candidates(slot_row)
        required_critical = bool(slot_row.get("required") and slot_row.get("role") in ACTION_CRITICAL_ROLES)
        if required_critical and not candidates:
            errors.append({"error": "empty_required_domain", "hard": True, "slot": slot})
            continue
        if slot_row.get("required") and slot not in arguments:
            errors.append({"error": "missing_required_assignment", "hard": True, "slot": slot})
            continue
        if slot in arguments:
            candidate = _candidate_for(slot_row, arguments[slot])
            if candidate is None:
                if not (slot_row.get("generation_allowed") and isinstance(arguments[slot], str)):
                    errors.append({"error": "uncertified_candidate", "hard": True, "slot": slot})
            else:
                selected_action_risk += float(candidate.get("tep_risk_upper_bound") or 0.0)
    for slot in arguments:
        if slot not in tool.get("slots", {}):
            errors.append({"error": "unknown_optional_slot", "hard": True, "slot": slot})
    action_risk_budget = float(lattice.get("action_risk_budget", 1.0))
    if selected_action_risk > action_risk_budget:
        errors.append({
            "error": "action_risk_budget_exceeded",
            "hard": True,
            "slot": None,
            "risk_upper_bound": selected_action_risk,
            "risk_budget": action_risk_budget,
        })
    return errors


def violation_vector(errors: list[dict[str, Any]], *, budget_used: int) -> tuple[int, int, int, int]:
    hard = sum(bool(row.get("hard")) for row in errors)
    unsupported = sum(row.get("error") in {"empty_required_domain", "uncertified_candidate"} for row in errors)
    ambiguous = sum(row.get("error") == "ambiguous_candidate" for row in errors)
    return hard, unsupported, ambiguous, budget_used


def minimal_unsatisfied_contract(errors: list[dict[str, Any]]) -> list[str]:
    slots = []
    for error in errors:
        if error.get("error") in {"empty_required_domain", "missing_required_assignment", "uncertified_candidate"} and error.get("slot"):
            slot = str(error["slot"])
            if slot not in slots:
                slots.append(slot)
    return slots


def _best_candidate(slot_row: dict[str, Any]) -> dict[str, Any] | None:
    candidates = certified_candidates(slot_row)
    if not candidates:
        return None
    return max(candidates, key=lambda row: (float(row.get("role_score", 0.0)), float(row.get("evidence_strength", 0.0)), -int(row.get("candidate_id", 0))))


def resolve_pointer_contract(
    pointer_action: dict[str, Any],
    lattice: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    budget: int = 2,
) -> dict[str, Any]:
    """Resolve pointer assignments with bounded, information-preserving transitions."""
    request = request_text(messages)
    state = deepcopy(pointer_action)
    history: list[dict[str, Any]] = []
    frozen: dict[str, int] = {}
    used = 0
    schema = build_pointer_action_schema(lattice)

    if state.get("mode") == "call" and capability_explicitly_forbidden(request):
        answer = {"mode": "direct_answer", "tool": None, "arguments": {}, "payload": {"reason": "tool use explicitly forbidden"}}
        return {
            "schema_version": CONTRACT_SOLVER_VERSION,
            "pointer_action": answer,
            "materialized_action": answer,
            "terminal_state": "direct_answer",
            "history": [{"transition": "CONVERT_TO_ANSWER", "reason": "tool use explicitly forbidden"}],
            "frozen_slots": {},
            "budget_used": 0,
        }

    if state.get("mode") == "call" and not schema.get("call_domains"):
        selected = _tool_by_id(lattice, state.get("tool_id"))
        lexical_match = bool(selected) and capability_compatible(
            request,
            capability_signature({
                "name": selected[0],
                "description": selected[1].get("description", ""),
                "parameters": {},
            }),
        )
        missing = sorted({row["slot"] for row in schema.get("clarify_domains", [])})
        terminal = "clarify" if missing and lexical_match else "direct_answer"
        payload = {"missing_slots": missing} if terminal == "clarify" else {"reason": "no supported compatible call domain"}
        return {
            "schema_version": CONTRACT_SOLVER_VERSION,
            "pointer_action": {"mode": terminal, "payload": payload},
            "materialized_action": {"mode": terminal, "tool": None, "arguments": {}, "payload": payload},
            "terminal_state": terminal,
            "history": [{"transition": "CONVERT_TO_CLARIFY" if terminal == "clarify" else "CONVERT_TO_ANSWER", "reason": "no feasible call domain"}],
            "frozen_slots": {},
            "budget_used": 0,
        }

    for _ in range(budget + 1):
        errors = hard_violations(state, lattice, request)
        before = violation_vector(errors, budget_used=used)
        if not errors:
            materialized = materialize_pointer_action(state, lattice)
            return {
                "schema_version": CONTRACT_SOLVER_VERSION,
                "pointer_action": state,
                "materialized_action": materialized,
                "terminal_state": str(state.get("mode")),
                "history": history,
                "frozen_slots": frozen,
                "budget_used": used,
            }
        if used >= budget:
            break
        selected = errors[0]
        transition = "ESCALATE"
        slot = selected.get("slot")
        if selected["error"] in {"unknown_tool", "capability_mismatch"}:
            feasible = [
                domain
                for domain in schema.get("call_domains", [])
                if capability_compatible(
                    request,
                    capability_signature({
                        "name": domain["tool"],
                        "description": lattice.get("tools", {}).get(domain["tool"], {}).get("description", ""),
                        "parameters": {},
                    }),
                )
            ]
            if feasible:
                state["tool_id"] = feasible[0]["tool_id"]
                state["arguments"] = {}
                transition = "REROUTE_TOOL"
            else:
                state = {"mode": "direct_answer", "payload": {"reason": "no compatible tool capability"}}
                transition = "CONVERT_TO_ANSWER"
        elif selected["error"] == "unknown_optional_slot" and slot:
            state.setdefault("arguments", {}).pop(slot, None)
            transition = "DELETE_UNSUPPORTED_OPTIONAL"
        elif selected["error"] in {"missing_required_assignment", "uncertified_candidate"} and slot:
            chosen = _tool_by_id(lattice, state.get("tool_id"))
            slot_row = chosen[1]["slots"][slot] if chosen else None
            candidate = _best_candidate(slot_row) if slot_row else None
            allow_semantic_rewrite = bool(
                lattice.get("allow_alternate_candidate_rewrites", True)
            )
            may_select_candidate = (
                selected["error"] == "missing_required_assignment"
                or allow_semantic_rewrite
            )
            if candidate is not None and may_select_candidate:
                state.setdefault("arguments", {})[slot] = candidate["candidate_id"]
                frozen[slot] = candidate["candidate_id"]
                transition = "SELECT_ALTERNATE_CANDIDATE"
            else:
                missing = minimal_unsatisfied_contract(errors)
                state = {"mode": "clarify", "payload": {"missing_slots": missing}}
                transition = "CONVERT_TO_CLARIFY"
        elif selected["error"] == "empty_required_domain":
            missing = minimal_unsatisfied_contract(errors)
            state = {"mode": "clarify", "payload": {"missing_slots": missing}}
            transition = "CONVERT_TO_CLARIFY"
        elif selected["error"] == "action_risk_budget_exceeded":
            state = {"mode": "escalate", "payload": {
                "reason": "composed action risk exceeds budget",
                "risk_upper_bound": selected.get("risk_upper_bound"),
                "risk_budget": selected.get("risk_budget"),
            }}
            transition = "ESCALATE"
        used += 1
        after_errors = hard_violations(state, lattice, request)
        after = violation_vector(after_errors, budget_used=used)
        history.append({
            "transition": transition,
            "error": selected["error"],
            "slot": slot,
            "before_vector": list(before),
            "after_vector": list(after),
            "remaining_budget": budget - used,
        })
        if state.get("mode") != "call":
            terminal = str(state.get("mode"))
            materialized = {"mode": terminal, "tool": None, "arguments": {}, "payload": dict(state.get("payload", {}))}
            return {
                "schema_version": CONTRACT_SOLVER_VERSION,
                "pointer_action": state,
                "materialized_action": materialized,
                "terminal_state": terminal,
                "history": history,
                "frozen_slots": frozen,
                "budget_used": used,
            }

    escalation = {"mode": "escalate", "tool": None, "arguments": {}, "payload": {"reason": "contract budget exhausted"}}
    return {
        "schema_version": CONTRACT_SOLVER_VERSION,
        "pointer_action": escalation,
        "materialized_action": escalation,
        "terminal_state": "escalate",
        "history": history + [{"transition": "ESCALATE", "reason": "budget exhausted"}],
        "frozen_slots": frozen,
        "budget_used": used,
    }

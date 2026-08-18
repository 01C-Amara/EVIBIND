from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .contract_solver import resolve_pointer_contract
from .evidence_contract import build_candidate_lattice, certified_candidates, request_text
from .io import read_jsonl, write_jsonl
from .ir import normalize_action, parse_prediction
from .tier_b_verifier import FrozenTierBVerifier
from .typed_evidence_programs import (
    TEP_VERSION,
    build_evidence_hypergraph,
    compile_slot_programs,
    execute_program,
    slot_risk_budget,
)

DEPLOYABLE_RESOLUTION_VERSION = "tapbench.deployable_resolution.v6"
RUNTIME_INPUT_FIELDS = ("messages", "tools", "dialogue_state", "reference_context", "candidate_seed", "prediction")
FORBIDDEN_RUNTIME_FIELDS = ("gold_action", "task_kind", "derivable_values", "scores", "scorer_output")


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return left.strip().casefold() == right.strip().casefold()
    return left == right


def _proposal_span(text: str, value: Any) -> tuple[int, int, str] | None:
    if isinstance(value, bool):
        variants = ["true" if value else "false", "yes" if value else "no"]
        for variant in variants:
            match = re.search(rf"(?<!\w){variant}(?!\w)", text, re.IGNORECASE)
            if match:
                return *match.span(), "boolean_normalization"
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])")
        for match in numeric.finditer(text):
            try:
                if float(match.group(0)) == float(value):
                    return *match.span(), "numeric_equivalence"
            except ValueError:
                continue
        return None
    if isinstance(value, str) and value.strip():
        normalized = value.strip()
        pieces = [piece for piece in re.split(r"[\s_-]+", normalized) if piece]
        pattern = r"[\s_-]+".join(re.escape(piece) for piece in pieces)
        match = re.search(rf"(?<!\w){pattern}(?!\w)", text, re.IGNORECASE)
        if match:
            transform = "separator_normalization" if match.group(0).casefold() != normalized.casefold() else "identity"
            return *match.span(), transform
    return None


def _semantic_value_kind(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", cleaned):
        return "email"
    digits = re.sub(r"\D", "", cleaned)
    if 7 <= len(digits) <= 15 and re.fullmatch(r"\+?[\d\s().-]+", cleaned):
        return "phone"
    return None


def _semantic_slot_kind(slot: str) -> str | None:
    lowered = slot.casefold()
    if "email" in lowered:
        return "email"
    if "phone" in lowered or "telephone" in lowered:
        return "phone"
    return None


def _repair_semantic_argument_slots(
    action: dict[str, Any], tools: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if action.get("mode") != "call" or not isinstance(action.get("arguments"), dict):
        return action, []
    tool = next((row for row in tools if row.get("name") == action.get("tool")), None)
    if tool is None:
        return action, []
    schema = tool.get("parameters", {}) if isinstance(tool.get("parameters"), dict) else {}
    properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
    arguments = dict(action["arguments"])
    repairs = []
    for source_slot, value in list(arguments.items()):
        kind = _semantic_value_kind(value)
        if kind is None or _semantic_slot_kind(str(source_slot)) == kind:
            continue
        targets = [
            str(slot)
            for slot, prop in properties.items()
            if _semantic_slot_kind(str(slot)) == kind
            and str(slot) not in arguments
            and isinstance(prop, dict)
            and prop.get("type", "string") == "string"
        ]
        if len(targets) != 1:
            continue
        target_slot = targets[0]
        arguments[target_slot] = arguments.pop(source_slot)
        repairs.append({
            "repair": "semantic_argument_slot",
            "value_kind": kind,
            "source_slot": str(source_slot),
            "target_slot": target_slot,
        })
    if not repairs:
        return action, []
    return {**action, "arguments": arguments}, repairs


def _contract_value_valid(slot_row: dict[str, Any], value: Any) -> bool:
    enum = slot_row.get("enum")
    if isinstance(enum, list) and value not in enum:
        return False
    expected = slot_row.get("json_type")
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


def _tool_properties(tool: dict[str, Any]) -> dict[str, dict[str, Any]]:
    schema = tool.get("parameters", {}) if isinstance(tool.get("parameters"), dict) else {}
    while isinstance(schema.get("properties"), dict) and set(schema["properties"]) == {"payload"}:
        payload = schema["properties"]["payload"]
        if not isinstance(payload, dict):
            break
        schema = payload
    output: dict[str, dict[str, Any]] = {}
    for surface, raw in (schema.get("properties", {}) or {}).items():
        prop = raw if isinstance(raw, dict) else {}
        output[str(prop.get("x-ir-name") or surface)] = prop
    return output


def augment_lattice_with_typed_programs(
    lattice: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    reference_context: dict[str, Any],
    dialogue_state: dict[str, Any],
    tier_b_verifier: FrozenTierBVerifier | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Compile bounded TEPs and expose only risk-budgeted Tier-A outputs to the solver."""
    text = request_text(messages)
    added = 0
    hypergraphs: list[dict[str, Any]] = []
    for raw_tool in tools:
        tool_name = str(raw_tool.get("canonical_name") or raw_tool.get("name"))
        lattice_tool = lattice.get("tools", {}).get(tool_name)
        if lattice_tool is None:
            continue
        properties = _tool_properties(raw_tool)
        slot_programs = {}
        executions = {}
        for slot, slot_row in lattice_tool.get("slots", {}).items():
            prop = properties.get(slot, {})
            programs = compile_slot_programs(
                text,
                slot,
                prop,
                role=str(slot_row.get("role", "control")),
                reference_context=reference_context,
                dialogue_state=dialogue_state,
            )
            slot_programs[slot] = programs
            defaults = {slot: prop["default"]} if "default" in prop else {}
            for program in programs:
                if slot_row.get("source_policy") == "trusted_state_only" and program.op != "STATE_REF":
                    continue
                execution = execute_program(
                    program,
                    text,
                    reference_context=reference_context,
                    dialogue_state=dialogue_state,
                    schema_defaults=defaults,
                    contract_constants=prop.get("x-tap-contract-constants", {}),
                )
                executions[program.program_id] = execution
                if not execution.valid or not _contract_value_valid(slot_row, execution.value):
                    continue
                budget = slot_risk_budget(str(slot_row.get("role", "control")), criticality=prop.get("x-tap-criticality"))
                tier_b_accepted = False
                tier_b_score = None
                if (
                    execution.accepted_tier == "B"
                    and program.tier == "B"
                    and tier_b_verifier is not None
                ):
                    tier_b_accepted, tier_b_score = tier_b_verifier.accepts(program, execution, text, slot)
                mechanical = (
                    program.tier == "A"
                    and execution.risk.upper_bound <= budget
                )
                certified = (mechanical or tier_b_accepted) and not bool(program.args.get("superseded"))
                acceptance_tier = "A" if mechanical else ("B" if tier_b_accepted else execution.accepted_tier)
                effective_risk = tier_b_verifier.accepted_error_upper_bound if tier_b_accepted and tier_b_verifier is not None else execution.risk.upper_bound
                existing = next(
                    (candidate for candidate in slot_row.get("candidates", []) if _same(candidate.get("value"), execution.value)),
                    None,
                )
                provenance = {
                    "program": program.to_dict(),
                    "execution": execution.to_dict(),
                    "slot_risk_budget": budget,
                    "tier_b_verifier_version": tier_b_verifier.version if tier_b_verifier is not None else None,
                    "tier_b_score": tier_b_score,
                    "tier_b_accepted": tier_b_accepted,
                    "effective_risk_upper_bound": effective_risk,
                }
                if existing is not None:
                    existing.setdefault("typed_evidence_programs", []).append(provenance)
                    if certified and existing.get("contradiction_status") == "none":
                        existing["support_status"] = "certified"
                        existing["acceptance_tier"] = acceptance_tier
                        previous_risk_value = existing.get("tep_risk_upper_bound")
                        previous_risk = float(previous_risk_value) if previous_risk_value is not None else 1.0
                        existing["tep_risk_upper_bound"] = min(previous_risk, effective_risk)
                    continue
                spans = [step["span"] for step in execution.trace if step.get("span") is not None]
                source_span = spans[0] if spans else None
                next_id = max((int(row.get("candidate_id", -1)) for row in slot_row.get("candidates", [])), default=-1) + 1
                slot_row.setdefault("candidates", []).append({
                    "slot": slot,
                    "value": execution.value,
                    "source_kind": "typed_evidence_program",
                    "source_span": source_span,
                    "source_text": text[source_span[0]:source_span[1]] if source_span else None,
                    "transform": program.op,
                    "transform_context": dict(reference_context),
                    "role": slot_row.get("role"),
                    "resolution_type": slot_row.get("resolution_type"),
                    "role_label": slot,
                    "support_status": "certified" if certified else "ambiguous",
                    "contradiction_status": "superseded" if program.args.get("superseded") else "none",
                    "scope_status": "superseded" if program.args.get("superseded") else "active",
                    "verifier_version": tier_b_verifier.version if tier_b_accepted and tier_b_verifier is not None else TEP_VERSION,
                    "evidence_strength": max(0.0, 1.0 - effective_risk),
                    "role_score": tier_b_score if tier_b_accepted else max(0.0, 1.0 - execution.risk.role),
                    "candidate_id": next_id,
                    "certificate_version": "tapbench.candidate_certificate.v1",
                    "acceptance_tier": acceptance_tier,
                    "tep_risk_upper_bound": effective_risk if certified else None,
                    "typed_evidence_programs": [provenance],
                })
                added += 1
        hypergraphs.append(build_evidence_hypergraph(
            request=text,
            tool=tool_name,
            slot_programs=slot_programs,
            executions=executions,
        ))
    return added, hypergraphs


def augment_lattice_with_proposal_spans(
    lattice: dict[str, Any],
    messages: list[dict[str, Any]],
    action: dict[str, Any],
) -> int:
    """Add Tier-B candidates only when a model-proposed literal has direct user-span support."""
    if action.get("mode") != "call":
        return 0
    tool_name = action.get("tool")
    if not isinstance(tool_name, str):
        return 0
    tool = lattice.get("tools", {}).get(tool_name)
    if tool is None:
        return 0
    text = request_text(messages)
    added = 0
    arguments = action.get("arguments", {}) if isinstance(action.get("arguments"), dict) else {}
    for slot, value in arguments.items():
        slot_row = tool.get("slots", {}).get(slot)
        if (
            slot_row is None
            or slot_row.get("generation_allowed")
            or slot_row.get("source_policy") == "trusted_state_only"
        ):
            continue
        if any(_same(candidate.get("value"), value) for candidate in certified_candidates(slot_row)):
            continue
        support = _proposal_span(text, value)
        if support is None or not _contract_value_valid(slot_row, value):
            continue
        start, end, transform = support
        span = (start, end)
        next_id = max((int(row.get("candidate_id", -1)) for row in slot_row.get("candidates", [])), default=-1) + 1
        slot_row.setdefault("candidates", []).append({
            "slot": slot,
            "value": value,
            "source_kind": "model_proposed_user_span",
            "source_span": list(span),
            "source_text": text[span[0]:span[1]],
            "transform": transform,
            "transform_context": {},
            "role": slot_row.get("role"),
            "resolution_type": slot_row.get("resolution_type"),
            "role_label": "model_proposed_slot_role",
            "support_status": "certified",
            "contradiction_status": "none",
            "scope_status": "active",
            "verifier_version": "proposal_span_verifier_v1",
            "evidence_strength": 0.8,
            "role_score": 0.8,
            "candidate_id": next_id,
            "certificate_version": "tapbench.candidate_certificate.v1",
            "acceptance_tier": "B",
        })
        added += 1
    return added


def literal_to_pointer(action: dict[str, Any], lattice: dict[str, Any]) -> dict[str, Any]:
    mode = action.get("mode")
    if mode != "call":
        return {
            "mode": mode,
            "payload": dict(action.get("payload", {})) if isinstance(action.get("payload"), dict) else {},
        }
    tool_name = action.get("tool")
    if not isinstance(tool_name, str):
        return {"mode": "call", "tool_id": -1, "arguments": {}}
    tool = lattice.get("tools", {}).get(tool_name)
    if tool is None:
        return {"mode": "call", "tool_id": -1, "arguments": {}}
    pointer_args: dict[str, Any] = {}
    arguments = action.get("arguments", {}) if isinstance(action.get("arguments"), dict) else {}
    for slot, value in arguments.items():
        slot_row = tool.get("slots", {}).get(slot)
        if slot_row is None:
            pointer_args[slot] = -1
            continue
        match = next(
            (candidate for candidate in certified_candidates(slot_row) if _same(candidate.get("value"), value)),
            None,
        )
        if match is not None:
            pointer_args[slot] = match["candidate_id"]
        elif slot_row.get("generation_allowed") and isinstance(value, str):
            pointer_args[slot] = value
        else:
            pointer_args[slot] = -1
    return {"mode": "call", "tool_id": tool["tool_id"], "arguments": pointer_args}


def resolve_deployable_prediction(
    runtime_case: dict[str, Any],
    prediction: dict[str, Any],
    *,
    reference_context: dict[str, Any] | None = None,
    dialogue_state: dict[str, Any] | None = None,
    candidate_seed: int = 17,
    budget: int = 2,
    evidence_mode: str = "deterministic",
    tier_b_verifier: FrozenTierBVerifier | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    lattice = build_candidate_lattice(
        runtime_case.get("messages", []),
        runtime_case.get("tools", []),
        dialogue_state=dialogue_state or {},
        reference_context=reference_context or {},
        candidate_seed=candidate_seed,
    )
    lattice_seconds = time.perf_counter() - started
    lattice["action_risk_budget"] = float((reference_context or {}).get("action_risk_budget", 0.05))
    lattice["allow_alternate_candidate_rewrites"] = bool(
        (reference_context or {}).get("allow_alternate_candidate_rewrites", True)
    )
    action, format_valid = parse_prediction(prediction)
    action = normalize_action(
        action,
        {
            "tool_aliases": runtime_case.get("tool_aliases", {}),
            "argument_aliases": runtime_case.get("argument_aliases", {}),
        },
    )
    semantic_slot_repairs: list[dict[str, Any]] = []
    if action is not None:
        action, semantic_slot_repairs = _repair_semantic_argument_slots(
            action, list(runtime_case.get("tools", []))
        )
    if action is None:
        action = {"mode": "escalate", "tool": None, "arguments": {}, "payload": {"reason": "unparseable initializer"}}
    proposal_candidates_added = 0
    typed_program_candidates_added = 0
    evidence_hypergraphs: list[dict[str, Any]] = []
    valid_modes = {
        "deterministic",
        "proposal_span_hybrid",
        "typed_programs",
        "typed_program_hybrid",
        "typed_programs_tier_ab",
        "typed_program_hybrid_tier_ab",
    }
    if evidence_mode not in valid_modes:
        raise ValueError(f"unknown evidence_mode: {evidence_mode}")
    if evidence_mode in {"typed_programs", "typed_program_hybrid", "typed_programs_tier_ab", "typed_program_hybrid_tier_ab"}:
        typed_program_candidates_added, evidence_hypergraphs = augment_lattice_with_typed_programs(
            lattice,
            runtime_case.get("messages", []),
            runtime_case.get("tools", []),
            reference_context=reference_context or {},
            dialogue_state=dialogue_state or {},
            tier_b_verifier=tier_b_verifier if evidence_mode.endswith("tier_ab") else None,
        )
    if evidence_mode in {"proposal_span_hybrid", "typed_program_hybrid", "typed_program_hybrid_tier_ab"}:
        proposal_candidates_added = augment_lattice_with_proposal_spans(
            lattice,
            runtime_case.get("messages", []),
            action,
        )
    pointer = literal_to_pointer(action, lattice)
    solver_started = time.perf_counter()
    resolution = resolve_pointer_contract(pointer, lattice, runtime_case.get("messages", []), budget=budget)
    solver_seconds = time.perf_counter() - solver_started
    resolution["initializer_format_valid"] = format_valid
    resolution["evidence_contract_version"] = lattice.get("schema_version")
    resolution["initializer_action"] = action
    resolution["semantic_slot_repairs"] = semantic_slot_repairs
    resolution["evidence_mode"] = evidence_mode
    resolution["proposal_candidates_added"] = proposal_candidates_added
    typed_executions = [
        edge.get("execution")
        for graph in evidence_hypergraphs
        for edge in graph.get("program_hyperedges", [])
        if edge.get("execution") is not None
    ]
    resolution["typed_program_candidates_added"] = typed_program_candidates_added
    resolution["typed_programs_compiled"] = len(typed_executions)
    resolution["typed_programs_valid"] = sum(bool(row.get("valid")) for row in typed_executions)
    resolution["typed_programs_tier_a"] = sum(
        bool(row.get("valid")) and row.get("accepted_tier") == "A"
        for row in typed_executions
    )
    resolution["typed_evidence_program_version"] = TEP_VERSION if evidence_hypergraphs else None
    resolution["tier_b_verifier_version"] = tier_b_verifier.version if tier_b_verifier is not None else None
    resolution["tier_b_verifier_artifact_sha256"] = tier_b_verifier.artifact_sha256 if tier_b_verifier is not None else None
    resolution["evidence_hypergraphs"] = evidence_hypergraphs
    resolution["lattice_seconds"] = lattice_seconds
    resolution["solver_seconds"] = solver_seconds
    resolution["total_resolution_seconds"] = time.perf_counter() - started
    resolution["runtime_input_fields"] = list(RUNTIME_INPUT_FIELDS)
    resolution["forbidden_runtime_fields"] = list(FORBIDDEN_RUNTIME_FIELDS)
    return resolution["materialized_action"], resolution


def resolve_deployable_files(
    cases_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
    *,
    diagnostics_path: str | Path | None = None,
    reference_date: str = "2026-07-10",
    timezone: str = "Europe/London",
    candidate_seed: int = 17,
    budget: int = 2,
    source_method: str = "full_tap_b2",
    output_method: str = "tap_r_deployable",
    evidence_mode: str = "deterministic",
    tier_b_verifier_artifact: str | Path | None = None,
) -> int:
    tier_b_verifier = FrozenTierBVerifier.load(tier_b_verifier_artifact) if tier_b_verifier_artifact else None
    cases = {row["case_id"]: row for row in read_jsonl(cases_path)}
    rows = []
    diagnostics = []
    for prediction in read_jsonl(predictions_path):
        if prediction.get("method") != source_method:
            continue
        case_id = str(prediction["case_id"])
        case = cases[case_id]
        runtime_case = {
            key: case.get(key)
            for key in ("messages", "tools", "tool_aliases", "argument_aliases")
        }
        context = {"reference_date": reference_date, "timezone": timezone}
        context.update(case.get("reference_context", {}))
        dialogue_state = case.get("dialogue_state", {}) if isinstance(case.get("dialogue_state"), dict) else {}
        action, resolution = resolve_deployable_prediction(
            runtime_case,
            prediction,
            reference_context=context,
            dialogue_state=dialogue_state,
            candidate_seed=candidate_seed,
            budget=budget,
            evidence_mode=evidence_mode,
            tier_b_verifier=tier_b_verifier,
        )
        rows.append({
            **prediction,
            "method": output_method,
            "prediction": action,
            "deployable_resolution_version": DEPLOYABLE_RESOLUTION_VERSION,
            "evidence_contract_version": resolution.get("evidence_contract_version"),
            "contract_solver_version": resolution.get("schema_version"),
            "typed_evidence_program_version": resolution.get("typed_evidence_program_version"),
            "tier_b_verifier_version": resolution.get("tier_b_verifier_version"),
            "resolution": resolution,
        })
        diagnostics.append({
            "case_id": case_id,
            "model_id": prediction.get("model_id"),
            "seed": prediction.get("seed"),
            "source_method": source_method,
            "output_method": output_method,
            "terminal_state": resolution.get("terminal_state"),
            "initializer_mode": resolution.get("initializer_action", {}).get("mode"),
            "missing_slots": resolution.get("materialized_action", {}).get("payload", {}).get("missing_slots", []),
            "evidence_mode": resolution.get("evidence_mode"),
            "proposal_candidates_added": resolution.get("proposal_candidates_added"),
            "typed_program_candidates_added": resolution.get("typed_program_candidates_added"),
            "typed_programs_compiled": resolution.get("typed_programs_compiled"),
            "typed_programs_valid": resolution.get("typed_programs_valid"),
            "typed_programs_tier_a": resolution.get("typed_programs_tier_a"),
            "typed_evidence_program_version": resolution.get("typed_evidence_program_version"),
            "tier_b_verifier_version": resolution.get("tier_b_verifier_version"),
            "history": resolution.get("history", []),
            "lattice_seconds": resolution["lattice_seconds"],
            "solver_seconds": resolution["solver_seconds"],
            "total_resolution_seconds": resolution["total_resolution_seconds"],
            "runtime_input_fields": resolution["runtime_input_fields"],
            "forbidden_runtime_fields": resolution["forbidden_runtime_fields"],
            "schema_version": DEPLOYABLE_RESOLUTION_VERSION,
        })
    if diagnostics_path is not None:
        write_jsonl(diagnostics_path, diagnostics)
    return write_jsonl(output_path, rows)

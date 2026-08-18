from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .contract_solver import resolve_pointer_contract
from .deployable_resolution import augment_lattice_with_typed_programs
from .evidence_contract import build_candidate_lattice, certified_candidates, request_text
from .io import read_jsonl, write_jsonl
from .tier_b_verifier import (
    FrozenTierBVerifier,
    FEATURE_NAMES,
    label_program,
    train_tier_b_verifier,
    verifier_features,
)
from .typed_evidence_programs import compile_slot_programs, execute_program


R2A_COMPONENT_REPORT_VERSION = "tapbench.r2a_component_report.v1"
R2A_COMPONENT_CONDITIONS = (
    "pointer_unrestricted",
    "pointer_literal_evidence",
    "pointer_tep_tier_a",
    "pointer_tep_tier_ab",
    "oracle_evidence",
    "tep_without_global_contract",
    "full_tep_contract_resolution",
)


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) < 1e-9
    if isinstance(left, str) and isinstance(right, str):
        return left.strip().casefold() == right.strip().casefold()
    return left == right


def _runtime_fields(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "messages": list(case.get("messages", [])),
        "tools": list(case.get("tools", [])),
        "tool_aliases": dict(case.get("tool_aliases", {})),
        "argument_aliases": dict(case.get("argument_aliases", {})),
        "dialogue_state": dict(case.get("dialogue_state", {})),
        "reference_context": dict(case.get("reference_context", {})),
    }


def _target(case: Mapping[str, Any]) -> tuple[str, str, Any, list[Any]]:
    oracle = case["r2a_oracle"]
    return (
        str(case["gold_action"]["tool"]),
        str(oracle["target_slot"]),
        oracle["gold_value"],
        list(oracle.get("unsupported_values", [])),
    )


def _tool_property(case: Mapping[str, Any], tool_name: str, slot: str) -> dict[str, Any]:
    tool = next(
        row for row in case.get("tools", [])
        if str(row.get("canonical_name") or row.get("name")) == tool_name
    )
    schema = tool.get("parameters", {})
    while isinstance(schema.get("properties"), dict) and set(schema["properties"]) == {"payload"}:
        schema = schema["properties"]["payload"]
    for surface, prop in schema.get("properties", {}).items():
        if str(prop.get("x-ir-name") or surface) == slot:
            return dict(prop)
    raise KeyError(f"missing property {tool_name}.{slot}")


def build_tier_b_corpus(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    corpus = []
    for case in cases:
        tool_name, slot, gold, unsupported = _target(case)
        prop = _tool_property(case, tool_name, slot)
        runtime = _runtime_fields(case)
        text = request_text(runtime["messages"])
        programs = compile_slot_programs(
            text,
            slot,
            prop,
            role=str(prop.get("x-tap-slot-role", "control")),
            reference_context=runtime["reference_context"],
            dialogue_state=runtime["dialogue_state"],
        )
        defaults = {slot: prop["default"]} if "default" in prop else {}
        for program in programs:
            execution = execute_program(
                program,
                text,
                reference_context=runtime["reference_context"],
                dialogue_state=runtime["dialogue_state"],
                schema_defaults=defaults,
                contract_constants=prop.get("x-tap-contract-constants", {}),
            )
            if program.tier != "B" or not execution.valid:
                continue
            features = verifier_features(program, execution, text, slot)
            corpus.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "slot": slot,
                "operator_stratum": case["r2a_oracle"]["operator_stratum"],
                "program_id": program.program_id,
                "program_op": program.op,
                "value": execution.value,
                "label": label_program(execution, gold, unsupported),
                "features": features,
                "vector": [features[name] for name in FEATURE_NAMES],
            })
    return corpus


def train_tier_b_from_cases(
    cases_path: str | Path,
    output: str | Path,
    *,
    target_precision: float = 0.99,
) -> dict[str, Any]:
    cases = read_jsonl(cases_path)
    corpus = build_tier_b_corpus(cases)
    return train_tier_b_verifier(corpus, output, target_precision=target_precision)


def _candidate_correct(candidate: Mapping[str, Any], gold: Any, unsupported: Iterable[Any]) -> bool:
    return _same(candidate.get("value"), gold) and not any(
        _same(candidate.get("value"), value) for value in unsupported
    )


def _condition_candidates(
    case: dict[str, Any],
    condition: str,
    verifier: FrozenTierBVerifier | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    runtime = _runtime_fields(case)
    started = time.perf_counter()
    lattice = build_candidate_lattice(
        runtime["messages"],
        runtime["tools"],
        dialogue_state=runtime["dialogue_state"],
        reference_context=runtime["reference_context"],
        candidate_seed=17,
    )
    lattice["action_risk_budget"] = float(runtime["reference_context"].get("action_risk_budget", 0.05))
    if condition in {
        "pointer_tep_tier_a",
        "tep_without_global_contract",
        "full_tep_contract_resolution",
    }:
        augment_lattice_with_typed_programs(
            lattice,
            runtime["messages"],
            runtime["tools"],
            reference_context=runtime["reference_context"],
            dialogue_state=runtime["dialogue_state"],
        )
    elif condition == "pointer_tep_tier_ab":
        if verifier is None:
            raise ValueError("pointer_tep_tier_ab requires a frozen verifier")
        augment_lattice_with_typed_programs(
            lattice,
            runtime["messages"],
            runtime["tools"],
            reference_context=runtime["reference_context"],
            dialogue_state=runtime["dialogue_state"],
            tier_b_verifier=verifier,
        )
    construction_seconds = time.perf_counter() - started
    tool_name, slot, gold, _ = _target(case)
    slot_row = lattice["tools"][tool_name]["slots"][slot]
    if condition == "oracle_evidence":
        candidates = [{
            "candidate_id": 0,
            "value": gold,
            "support_status": "certified",
            "contradiction_status": "none",
            "acceptance_tier": "oracle",
            "source_kind": "offline_oracle",
        }]
    elif condition == "pointer_unrestricted":
        candidates = list(slot_row.get("candidates", []))
    else:
        candidates = certified_candidates(slot_row)
    return candidates, lattice, construction_seconds


def _component_row(
    case: dict[str, Any],
    condition: str,
    verifier: FrozenTierBVerifier | None,
) -> dict[str, Any]:
    tool_name, slot, gold, unsupported = _target(case)
    candidates, lattice, construction_seconds = _condition_candidates(case, condition, verifier)
    correct = [candidate for candidate in candidates if _candidate_correct(candidate, gold, unsupported)]
    false_accepts = [
        candidate for candidate in candidates
        if any(_same(candidate.get("value"), value) for value in unsupported)
    ]
    expected_ops = set(case["r2a_oracle"].get("expected_program_ops", []))
    transform_correct = any(
        _candidate_correct(candidate, gold, unsupported)
        and any(
            provenance.get("program", {}).get("op") in expected_ops
            for provenance in candidate.get("typed_evidence_programs", [])
        )
        for candidate in candidates
    )
    execution_exact = None
    terminal_state = None
    if condition == "full_tep_contract_resolution":
        tool = lattice["tools"][tool_name]
        pointer = {"mode": "call", "tool_id": tool["tool_id"], "arguments": {}}
        resolution = resolve_pointer_contract(pointer, lattice, case["messages"], budget=2)
        action = resolution["materialized_action"]
        terminal_state = resolution["terminal_state"]
        execution_exact = (
            action.get("mode") == "call"
            and action.get("tool") == tool_name
            and _same(action.get("arguments", {}).get(slot), gold)
        )
    return {
        "schema_version": "tapbench.r2a_component_row.v1",
        "case_id": case["case_id"],
        "family": case["family"],
        "operator_stratum": case["r2a_oracle"]["operator_stratum"],
        "variant": case["r2a_oracle"]["variant"],
        "condition": condition,
        "target_slot": slot,
        "gold_value": gold,
        "candidate_recall": int(bool(correct)),
        "certificate_precision_numerator": sum(
            _candidate_correct(candidate, gold, unsupported) for candidate in candidates
        ),
        "certificate_precision_denominator": len(candidates),
        "candidate_set_size": len(candidates),
        "false_accept_count": len(false_accepts),
        "false_accept_values": [candidate.get("value") for candidate in false_accepts],
        "transform_accuracy": int(transform_correct) if condition.startswith("pointer_tep") or condition.startswith("full_tep") or condition.startswith("tep_") else None,
        "execution_exact": int(execution_exact) if execution_exact is not None else None,
        "terminal_state": terminal_state,
        "construction_seconds": construction_seconds,
        "tier_b_verifier_version": verifier.version if condition == "pointer_tep_tier_ab" and verifier is not None else None,
        "tier_b_verifier_artifact_sha256": verifier.artifact_sha256 if condition == "pointer_tep_tier_ab" and verifier is not None else None,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    precision_n = sum(int(row["certificate_precision_numerator"]) for row in rows)
    precision_d = sum(int(row["certificate_precision_denominator"]) for row in rows)
    execution_rows = [row for row in rows if row["execution_exact"] is not None]
    transform_rows = [row for row in rows if row["transform_accuracy"] is not None]
    return {
        "cases": count,
        "candidate_recall": sum(int(row["candidate_recall"]) for row in rows) / count if count else 0.0,
        "certificate_precision": precision_n / precision_d if precision_d else None,
        "candidate_set_size_mean": sum(int(row["candidate_set_size"]) for row in rows) / count if count else 0.0,
        "false_accept_rate_per_case": sum(int(row["false_accept_count"]) for row in rows) / count if count else 0.0,
        "transform_accuracy": sum(int(row["transform_accuracy"]) for row in transform_rows) / len(transform_rows) if transform_rows else None,
        "execution_exact": sum(int(row["execution_exact"]) for row in execution_rows) / len(execution_rows) if execution_rows else None,
        "construction_ms_mean": 1000.0 * sum(float(row["construction_seconds"]) for row in rows) / count if count else 0.0,
    }


def _gate_report(rows: list[dict[str, Any]], preregistration: Mapping[str, Any]) -> dict[str, Any]:
    by_condition = {
        condition: [row for row in rows if row["condition"] == condition]
        for condition in R2A_COMPONENT_CONDITIONS
    }
    literal = by_condition["pointer_literal_evidence"]
    tep = by_condition["pointer_tep_tier_a"]
    literal_by_case = {row["case_id"]: row for row in literal}
    tep_by_case = {row["case_id"]: row for row in tep}
    retained = [
        int(tep_by_case[case_id]["candidate_recall"])
        for case_id, row in literal_by_case.items()
        if int(row["candidate_recall"]) == 1
    ]
    typed_candidates = []
    for row in tep:
        typed_candidates.append(row)
    tep_summary = _aggregate(tep)
    literal_summary = _aggregate(literal)
    correction_rows = [
        row for row in tep
        if row["operator_stratum"] in {"negation", "same_type_role_counterfactual"}
    ]
    thresholds = preregistration["success_gates"]["component"]
    actual = {
        "tier_a_certificate_precision": tep_summary["certificate_precision"] or 0.0,
        "typed_program_gain_over_literal_evidence_pp": 100.0 * (
            tep_summary["candidate_recall"] - literal_summary["candidate_recall"]
        ),
        "correction_and_negation_false_accept_rate": (
            sum(int(row["false_accept_count"]) > 0 for row in correction_rows) / len(correction_rows)
            if correction_rows else 0.0
        ),
        "contract_resolution_non_regression": sum(retained) / len(retained) if retained else 1.0,
    }
    gates = {
        "tier_a_certificate_precision": {
            "actual": actual["tier_a_certificate_precision"],
            "required": float(thresholds["tier_a_certificate_precision_min"]),
            "passed": actual["tier_a_certificate_precision"] >= float(thresholds["tier_a_certificate_precision_min"]),
        },
        "typed_program_gain_over_literal_evidence_pp": {
            "actual": actual["typed_program_gain_over_literal_evidence_pp"],
            "required": float(thresholds["typed_program_gain_over_literal_evidence_pp_min"]),
            "passed": actual["typed_program_gain_over_literal_evidence_pp"] >= float(thresholds["typed_program_gain_over_literal_evidence_pp_min"]),
        },
        "correction_and_negation_false_accept_rate": {
            "actual": actual["correction_and_negation_false_accept_rate"],
            "required": float(thresholds["correction_and_negation_false_accept_rate_max"]),
            "passed": actual["correction_and_negation_false_accept_rate"] <= float(thresholds["correction_and_negation_false_accept_rate_max"]),
        },
        "contract_resolution_non_regression": {
            "actual": actual["contract_resolution_non_regression"],
            "required": float(thresholds["contract_resolution_non_regression_min"]),
            "passed": actual["contract_resolution_non_regression"] >= float(thresholds["contract_resolution_non_regression_min"]),
        },
    }
    return {"passed": all(row["passed"] for row in gates.values()), "gates": gates}


def evaluate_r2a_components(
    cases_path: str | Path,
    rows_output: str | Path,
    report_output: str | Path,
    *,
    preregistration_path: str | Path = "configs/r2_tep_preregistration.yaml",
    tier_b_verifier_path: str | Path | None = None,
) -> dict[str, Any]:
    cases = read_jsonl(cases_path)
    verifier = FrozenTierBVerifier.load(tier_b_verifier_path) if tier_b_verifier_path else None
    conditions = [
        condition for condition in R2A_COMPONENT_CONDITIONS
        if condition != "pointer_tep_tier_ab" or verifier is not None
    ]
    rows = [
        _component_row(case, condition, verifier)
        for case in cases
        for condition in conditions
    ]
    write_jsonl(rows_output, rows)
    preregistration = yaml.safe_load(Path(preregistration_path).read_text(encoding="utf-8"))
    by_condition = {
        condition: _aggregate([row for row in rows if row["condition"] == condition])
        for condition in conditions
    }
    by_stratum = {
        stratum: {
            condition: _aggregate([
                row for row in rows
                if row["operator_stratum"] == stratum and row["condition"] == condition
            ])
            for condition in conditions
        }
        for stratum in sorted({str(row["operator_stratum"]) for row in rows})
    }
    gate = _gate_report(rows, preregistration)
    report = {
        "schema_version": R2A_COMPONENT_REPORT_VERSION,
        "case_count": len(cases),
        "row_count": len(rows),
        "conditions": conditions,
        "by_condition": by_condition,
        "by_operator_stratum": by_stratum,
        "release_decision": gate,
        "tier_b_verifier": {
            "version": verifier.version,
            "artifact_sha256": verifier.artifact_sha256,
            "threshold": verifier.threshold,
        } if verifier is not None else None,
        "runtime_boundary": {
            "allowed": ["messages", "tools", "tool_aliases", "argument_aliases", "dialogue_state", "reference_context"],
            "offline_only": ["gold_action", "derivable_values", "r2a_oracle", "task_kind"],
        },
    }
    path = Path(report_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

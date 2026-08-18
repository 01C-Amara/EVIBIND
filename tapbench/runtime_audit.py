from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any, Callable

from .contract_solver import resolve_pointer_contract
from .evidence_contract import build_candidate_lattice, build_pointer_action_schema, materialize_pointer_action
from .effect_first import effect_admission, lock_tool_evidence, run_effect_first_resolution
from .resolution import evidence_for_slot_value
from .tapr import _apply_local_transition, contract_validator_error, resolve_action
from .typed_evidence_programs import compile_slot_programs, execute_program

RUNTIME_AUDIT_VERSION = "tapbench.runtime_dependency_audit.v3"
FORBIDDEN_RUNTIME_FIELDS = {
    "gold_action",
    "task_kind",
    "derivable_values",
    "execution_success",
    "fabrication",
    "scores",
    "scorer_output",
}


def _accessed_fields(function: Callable[..., Any]) -> set[str]:
    source = inspect.getsource(function)
    tree = ast.parse(source)
    fields: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            fields.add(node.slice.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            fields.add(node.args[0].value)
    return fields


def _audit_callable(function: Callable[..., Any], *, transitive_fields: set[str] | None = None) -> dict[str, Any]:
    direct = _accessed_fields(function) & FORBIDDEN_RUNTIME_FIELDS
    transitive = set(transitive_fields or set()) & FORBIDDEN_RUNTIME_FIELDS
    signature = inspect.signature(function)
    return {
        "callable": f"{function.__module__}.{function.__name__}",
        "parameters": list(signature.parameters),
        "direct_forbidden_fields": sorted(direct),
        "transitive_forbidden_fields": sorted(transitive),
        "deployable_ready": not direct and not transitive,
    }


def build_runtime_dependency_audit() -> dict[str, Any]:
    legacy = [
        _audit_callable(evidence_for_slot_value),
        _audit_callable(contract_validator_error, transitive_fields={"task_kind", "derivable_values"}),
        _audit_callable(_apply_local_transition),
        _audit_callable(resolve_action, transitive_fields={"gold_action", "task_kind", "derivable_values", "execution_success", "fabrication"}),
    ]
    deployable = [
        _audit_callable(build_candidate_lattice),
        _audit_callable(build_pointer_action_schema),
        _audit_callable(materialize_pointer_action),
        _audit_callable(compile_slot_programs),
        _audit_callable(execute_program),
        _audit_callable(resolve_pointer_contract),
        _audit_callable(effect_admission),
        _audit_callable(lock_tool_evidence),
        _audit_callable(run_effect_first_resolution),
    ]
    legacy_fields = sorted({field for row in legacy for field in row["direct_forbidden_fields"] + row["transitive_forbidden_fields"]})
    deployable_fields = sorted({field for row in deployable for field in row["direct_forbidden_fields"] + row["transitive_forbidden_fields"]})
    return {
        "schema_version": RUNTIME_AUDIT_VERSION,
        "policy": {
            "forbidden_runtime_fields": sorted(FORBIDDEN_RUNTIME_FIELDS),
            "allowed_deployable_inputs": ["messages", "tools", "dialogue_state", "reference_context", "candidate_seed", "pointer_action", "budget", "endpoint", "condition", "max_tokens", "seed"],
            "rule": "gold actions, task labels, synthetic hidden values, and scorer outputs may be used only after terminal prediction for evaluation",
        },
        "legacy_r1_oracle_path": {
            "components": legacy,
            "forbidden_fields_found": legacy_fields,
            "deployable_ready": not legacy_fields,
            "interpretation": "R1 is an oracle-assisted mechanism upper bound, not a deployable evidence result.",
        },
        "evidence_bounded_path": {
            "components": deployable,
            "forbidden_fields_found": deployable_fields,
            "deployable_ready": not deployable_fields,
            "interpretation": "The substrate is input-clean; resolver soundness and completeness still require independent evaluation.",
        },
    }


def write_runtime_dependency_audit(output_path: str | Path) -> dict[str, Any]:
    report = build_runtime_dependency_audit()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

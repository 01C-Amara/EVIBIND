from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .hierarchy_families import HIERARCHY_FAMILIES
from .io import write_jsonl
from .r2b import _catalog, _gold
from .r2d import R2D_MUTATIONS, R2D_TASK_KINDS
from .r2f_families import (
    R2F_MISSING_REQUESTS,
    R2F_SEMANTIC_ENVELOPES,
    R2F_UNSUPPORTED_REQUESTS,
    r2f_values,
)


HIERARCHY_CASE_VERSION = "evibind.hierarchy_case.v1"
HIERARCHY_GRID_ID = "R2H_certificate_hierarchy_ablation_v1"
HIERARCHY_FAMILY_COUNT = 24
HIERARCHY_CASES_PER_FAMILY = 32
HIERARCHY_CASE_COUNT = HIERARCHY_FAMILY_COUNT * HIERARCHY_CASES_PER_FAMILY


def _semantic_envelope(slot: str) -> str | None:
    if slot in R2F_SEMANTIC_ENVELOPES:
        return R2F_SEMANTIC_ENVELOPES[slot]
    if slot.endswith("_uri") or slot == "cover_uri":
        return "uri"
    if slot.endswith("_id"):
        return "opaque_atom"
    return None


def _request(
    source_family: str,
    family: Any,
    values: dict[str, Any],
    task_kind: str,
) -> str:
    if task_kind == "call":
        return family.request_template.format(**values)
    if task_kind == "missing_info":
        return R2F_MISSING_REQUESTS[source_family].format(**values)
    if task_kind == "no_tool":
        return R2F_UNSUPPORTED_REQUESTS[source_family].format(**values)
    if task_kind == "direct_answer":
        return family.no_tool_request
    raise ValueError(task_kind)


def generate_hierarchy_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_index, definition in enumerate(HIERARCHY_FAMILIES):
        family = definition.spec
        for variant in range(HIERARCHY_CASES_PER_FAMILY):
            global_index = 60_000 + family_index * 64 + variant
            task_kind = R2D_TASK_KINDS[variant % len(R2D_TASK_KINDS)]
            mutation = R2D_MUTATIONS[
                (variant + 2 * family_index) % len(R2D_MUTATIONS)
            ]
            all_values = r2f_values(global_index)
            values = {slot: all_values[slot] for slot in family.required_slots}
            tools, tool_aliases, argument_aliases = _catalog(
                family,
                600 + family_index,
                values,
                mutation,
            )
            declared_envelopes: dict[str, str] = {}
            for tool in tools:
                properties = tool["parameters"]["properties"]
                for prop in properties.values():
                    canonical = str(prop.get("x-ir-name"))
                    envelope = _semantic_envelope(canonical)
                    if envelope is not None:
                        prop["x-tap-semantic-envelope"] = envelope
                        declared_envelopes[canonical] = envelope
            derivable = dict(values) if task_kind in {"call", "missing_info"} else {}
            if task_kind == "missing_info":
                derivable.pop(family.missing_slot, None)
            rows.append(
                {
                    "schema_version": HIERARCHY_CASE_VERSION,
                    "case_id": (
                        f"r2h_hierarchy_{family.name}_{variant:02d}"
                    ),
                    "hypothesis_grid_id": HIERARCHY_GRID_ID,
                    "hypothesis": "R2H_hierarchy",
                    "split": "prospective_certificate_hierarchy",
                    "family": family.name,
                    "task_kind": task_kind,
                    "factors": {
                        "task_kind": task_kind,
                        "catalog_mutation": mutation,
                        "variant": variant,
                        "extent_stratum": definition.extent_stratum,
                        "source_slot_archetype": definition.source_family,
                    },
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Return one Action IR outcome. Do not invent "
                                "authorization or unsupported argument values."
                            ),
                        },
                        {
                            "role": "user",
                            "content": _request(
                                definition.source_family,
                                family,
                                values,
                                task_kind,
                            ),
                        },
                    ],
                    "tools": tools,
                    "tool_aliases": tool_aliases,
                    "argument_aliases": argument_aliases,
                    "dialogue_state": {},
                    "reference_context": {
                        "reference_date": "2026-07-27",
                        "timezone": "Europe/London",
                        "action_risk_budget": 0.05,
                    },
                    "gold_action": _gold(family, values, task_kind),
                    "derivable_values": derivable,
                    "hierarchy_oracle": {
                        "family": family.name,
                        "source_slot_archetype": definition.source_family,
                        "extent_stratum": definition.extent_stratum,
                        "declared_semantic_envelopes": declared_envelopes,
                        "mutation": mutation,
                        "variant": variant,
                    },
                    "metadata": {
                        "backend_namespace": "llama_cpp_q4km_r2h_hierarchy",
                        "coefficient_backend": "llama.cpp",
                        "quantization": "Q4_K_M",
                        "thinking_mode": "off",
                        "reasoning_budget": 0,
                        "runtime_allowed_fields": [
                            "messages",
                            "tools",
                            "tool_aliases",
                            "argument_aliases",
                            "dialogue_state",
                            "reference_context",
                        ],
                        "offline_only_fields": [
                            "gold_action",
                            "derivable_values",
                            "hierarchy_oracle",
                            "task_kind",
                        ],
                    },
                }
            )
    return rows


def hierarchy_case_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    encoded = "\n".join(
        json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        for row in rows
    ).encode("utf-8")
    family_counts = Counter(str(row["family"]) for row in rows)
    task_counts = Counter(str(row["task_kind"]) for row in rows)
    stratum_counts = Counter(str(row["factors"]["extent_stratum"]) for row in rows)
    return {
        "schema_version": "evibind.hierarchy_case_manifest.v1",
        "case_version": HIERARCHY_CASE_VERSION,
        "grid_id": HIERARCHY_GRID_ID,
        "case_count": len(rows),
        "family_count": len(family_counts),
        "cases_per_family": dict(sorted(family_counts.items())),
        "task_kind_counts": dict(sorted(task_counts.items())),
        "extent_stratum_counts": dict(sorted(stratum_counts.items())),
        "canonical_cases_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def write_hierarchy_cases(
    output: str | Path,
    manifest_output: str | Path,
) -> int:
    rows = generate_hierarchy_cases()
    write_jsonl(output, rows)
    Path(manifest_output).write_text(
        json.dumps(
            hierarchy_case_manifest(rows),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(rows)


def audit_hierarchy_gold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from .capc import certify_proposal
    from .extractive_candidates import build_extractive_candidate_table
    from .selective_tapr import (
        certificate_semantic_envelope_violations,
        certificate_span_conflicts,
    )
    from .validation import action_contract_is_accepted

    failures = []
    call_count = 0
    for case in rows:
        if not action_contract_is_accepted(case, case["gold_action"]):
            failures.append(
                {"case_id": case["case_id"], "reason": "gold_contract_invalid"}
            )
        if case["task_kind"] != "call":
            continue
        call_count += 1
        gold = case["gold_action"]
        tools = [
            tool
            for tool in case["tools"]
            if str(tool.get("canonical_name") or tool.get("name"))
            == str(gold["tool"])
        ]
        if len(tools) != 1:
            failures.append(
                {"case_id": case["case_id"], "reason": "gold_tool_not_unique"}
            )
            continue
        tool = tools[0]
        table = build_extractive_candidate_table(
            case["messages"],
            tool,
            include_optional=True,
        )
        action, certification = certify_proposal(
            gold,
            selected_tool=gold["tool"],
            tool=tool,
            candidate_table=table,
            tools=case["tools"],
        )
        if action is None:
            failures.append(
                {"case_id": case["case_id"], "reason": "gold_not_candidate_backed"}
            )
            continue
        conflicts = certificate_span_conflicts(certification)
        if conflicts:
            failures.append(
                {
                    "case_id": case["case_id"],
                    "reason": "gold_source_role_or_joint_contract_failure",
                    "slots": conflicts,
                }
            )
        violations = certificate_semantic_envelope_violations(
            certification,
            tool,
        )
        if violations:
            failures.append(
                {
                    "case_id": case["case_id"],
                    "reason": "gold_semantic_extent_failure",
                    "slots": violations,
                }
            )
    return {
        "schema_version": "evibind.hierarchy_gold_audit.v1",
        "case_count": len(rows),
        "call_count": call_count,
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }

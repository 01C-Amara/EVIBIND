from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import write_jsonl
from .r2b import _catalog, _gold
from .r2d import R2D_MUTATIONS, R2D_TASK_KINDS
from .r2f_families import (
    R2F_FAMILIES,
    R2F_MISSING_REQUESTS,
    R2F_SEMANTIC_ENVELOPES,
    R2F_UNSUPPORTED_REQUESTS,
    r2f_values,
)


R2F_CASE_VERSION = "tapbench.r2f_case.v1"
R2F_RUNNER_VERSION = "tapbench.r2f_model_runner.v1"
R2F_REPORT_VERSION = "tapbench.r2f_report.v1"
R2F_GRID_ID = "R2F_selected_qwen_envelope_replication_v1"
R2F_CONDITIONS = ("constrained_abstention", "tap_r_selective_full")


def _request(family: Any, values: dict[str, Any], task_kind: str) -> str:
    if task_kind == "call":
        return family.request_template.format(**values)
    if task_kind == "missing_info":
        return R2F_MISSING_REQUESTS[family.name].format(**values)
    if task_kind == "no_tool":
        return R2F_UNSUPPORTED_REQUESTS[family.name].format(**values)
    if task_kind == "direct_answer":
        return family.no_tool_request
    raise ValueError(task_kind)


def generate_r2f_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(R2F_FAMILIES):
        for variant in range(32):
            global_index = 20000 + family_index * 64 + variant
            task_kind = R2D_TASK_KINDS[(variant + family_index) % 4]
            mutation = R2D_MUTATIONS[(variant + 2 * family_index) % len(R2D_MUTATIONS)]
            all_values = r2f_values(global_index)
            values = {slot: all_values[slot] for slot in family.required_slots}
            tools, tool_aliases, argument_aliases = _catalog(
                family,
                400 + family_index,
                values,
                mutation,
            )
            for tool in tools:
                for prop in tool["parameters"]["properties"].values():
                    envelope = R2F_SEMANTIC_ENVELOPES.get(prop.get("x-ir-name"))
                    if envelope is not None:
                        prop["x-tap-semantic-envelope"] = envelope
            derivable = dict(values) if task_kind in {"call", "missing_info"} else {}
            if task_kind == "missing_info":
                derivable.pop(family.missing_slot, None)
            rows.append(
                {
                    "schema_version": R2F_CASE_VERSION,
                    "case_id": f"r2f_confirmation_{family.name}_{variant:02d}",
                    "hypothesis_grid_id": R2F_GRID_ID,
                    "hypothesis": "R2F",
                    "split": "selected_model_prospective_replication",
                    "family": family.name,
                    "task_kind": task_kind,
                    "factors": {
                        "task_kind": task_kind,
                        "catalog_mutation": mutation,
                        "variant": variant,
                        "cue_regime": "implicit_unlabeled",
                    },
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Return one Action IR outcome. Do not invent "
                                "authorization or unsupported argument values."
                            ),
                        },
                        {"role": "user", "content": _request(family, values, task_kind)},
                    ],
                    "tools": tools,
                    "tool_aliases": tool_aliases,
                    "argument_aliases": argument_aliases,
                    "dialogue_state": {},
                    "reference_context": {
                        "reference_date": "2026-07-17",
                        "timezone": "Europe/London",
                        "action_risk_budget": 0.05,
                    },
                    "gold_action": _gold(family, values, task_kind),
                    "derivable_values": derivable,
                    "r2f_oracle": {
                        "family": family.name,
                        "mutation": mutation,
                        "variant": variant,
                    },
                    "metadata": {
                        "backend_namespace": "llama_cpp_q4km_r2f",
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
                            "r2f_oracle",
                            "task_kind",
                        ],
                    },
                }
            )
    return rows


def write_r2f_cases(output: str | Path) -> int:
    return write_jsonl(output, generate_r2f_cases())

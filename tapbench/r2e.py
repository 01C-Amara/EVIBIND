from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import write_jsonl
from .r2b import _catalog, _gold
from .r2d import R2D_MUTATIONS, R2D_TASK_KINDS
from .r2e_families import (
    R2E_FAMILIES,
    R2E_MISSING_REQUESTS,
    R2E_UNSUPPORTED_REQUESTS,
    r2e_values,
)


R2E_CASE_VERSION = "tapbench.r2e_case.v1"
R2E_RUNNER_VERSION = "tapbench.r2e_model_runner.v1"
R2E_REPORT_VERSION = "tapbench.r2e_report.v1"
R2E_GRID_ID = "R2E_semantic_envelope_confirmation_v1"
R2E_CONDITIONS = ("constrained_abstention", "tap_r_selective_full")


def _request(family: Any, values: dict[str, Any], task_kind: str) -> str:
    if task_kind == "call":
        return family.request_template.format(**values)
    if task_kind == "missing_info":
        return R2E_MISSING_REQUESTS[family.name].format(**values)
    if task_kind == "no_tool":
        return R2E_UNSUPPORTED_REQUESTS[family.name].format(**values)
    if task_kind == "direct_answer":
        return family.no_tool_request
    raise ValueError(task_kind)


def generate_r2e_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(R2E_FAMILIES):
        for variant in range(32):
            global_index = 12000 + family_index * 64 + variant
            task_kind = R2D_TASK_KINDS[(variant + family_index) % 4]
            mutation = R2D_MUTATIONS[(variant + 2 * family_index) % len(R2D_MUTATIONS)]
            all_values = r2e_values(global_index)
            values = {slot: all_values[slot] for slot in family.required_slots}
            tools, tool_aliases, argument_aliases = _catalog(
                family,
                300 + family_index,
                values,
                mutation,
            )
            for tool in tools:
                for prop in tool["parameters"]["properties"].values():
                    if prop.get("x-ir-name") == "source_uri":
                        prop["x-tap-semantic-envelope"] = "uri"
            derivable = dict(values) if task_kind in {"call", "missing_info"} else {}
            if task_kind == "missing_info":
                derivable.pop(family.missing_slot, None)
            rows.append(
                {
                    "schema_version": R2E_CASE_VERSION,
                    "case_id": f"r2e_confirmation_{family.name}_{variant:02d}",
                    "hypothesis_grid_id": R2E_GRID_ID,
                    "hypothesis": "R2E",
                    "split": "prospective_confirmation",
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
                    "r2e_oracle": {
                        "family": family.name,
                        "mutation": mutation,
                        "variant": variant,
                    },
                    "metadata": {
                        "backend_namespace": "llama_cpp_q4km_r2e",
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
                            "r2e_oracle",
                            "task_kind",
                        ],
                    },
                }
            )
    return rows


def write_r2e_cases(output: str | Path) -> int:
    return write_jsonl(output, generate_r2e_cases())

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .distillation_families import (
    DISTILLATION_FAMILY_BY_NAME,
    DistillationFamilySpec,
    distillation_values,
    semantic_envelope,
)
from .io import write_jsonl
from .r2b import _catalog, _gold
from .r2d import R2D_MUTATIONS


DISTILLATION_CASE_VERSION = "tapbench.verified_distillation_case.v1"
DISTILLATION_GRID_ID = "TAPR_verified_distillation_family_split_v1"
TASK_KINDS = ("call", "missing_info", "no_tool", "direct_answer")


def _task_schedule(
    counts: dict[str, int], *, seed: int, family_index: int
) -> list[str]:
    if set(counts) != set(TASK_KINDS):
        raise ValueError(f"task-count keys drifted: {sorted(counts)}")
    schedule = [kind for kind in TASK_KINDS for _ in range(int(counts[kind]))]
    random.Random(seed + 1009 * family_index).shuffle(schedule)
    return schedule


def _request(
    spec: DistillationFamilySpec,
    values: dict[str, Any],
    task_kind: str,
    variant: int,
) -> str:
    family = spec.family
    if task_kind == "call":
        base = family.request_template.format(**values)
    elif task_kind == "missing_info":
        shown = [
            f"{slot.replace('_', ' ')} {values[slot]}"
            for slot in family.required_slots
            if slot != family.missing_slot
        ]
        base = f"Please {spec.operation} with {', '.join(shown)}."
    elif task_kind == "no_tool":
        return spec.unsupported_request
    elif task_kind == "direct_answer":
        return family.no_tool_request
    else:
        raise ValueError(task_kind)
    if variant % 3 == 1:
        return base.replace("Please ", "I need you to ", 1)
    if variant % 3 == 2:
        return base.replace("Please ", "Could you ", 1)
    return base


def _annotate_semantic_envelopes(
    tools: list[dict[str, Any]],
    values: dict[str, Any],
    enum_slot: str,
) -> None:
    for tool in tools:
        properties = tool.get("parameters", {}).get("properties", {})
        for prop in properties.values():
            slot = str(prop.get("x-ir-name", ""))
            envelope = semantic_envelope(slot, values.get(slot), enum_slot)
            if envelope is not None:
                prop["x-tap-semantic-envelope"] = envelope


def generate_distillation_cases(
    *,
    family_names: Iterable[str],
    task_counts_per_family: dict[str, int],
    split: str,
    seed: int,
    family_index_offset: int = 500,
) -> list[dict[str, Any]]:
    names = list(family_names)
    if len(names) != len(set(names)):
        raise ValueError("family names must be unique")
    unknown = sorted(set(names).difference(DISTILLATION_FAMILY_BY_NAME))
    if unknown:
        raise KeyError(f"unknown distillation families: {unknown}")

    rows: list[dict[str, Any]] = []
    for family_index, name in enumerate(names):
        spec = DISTILLATION_FAMILY_BY_NAME[name]
        family = spec.family
        schedule = _task_schedule(
            task_counts_per_family,
            seed=seed,
            family_index=family_index,
        )
        for variant, task_kind in enumerate(schedule):
            global_index = 50000 + family_index * 512 + variant
            mutation = R2D_MUTATIONS[
                (variant + 2 * family_index) % len(R2D_MUTATIONS)
            ]
            values = distillation_values(spec, global_index)
            tools, tool_aliases, argument_aliases = _catalog(
                family,
                family_index_offset + family_index,
                values,
                mutation,
            )
            _annotate_semantic_envelopes(tools, values, family.enum_slot)
            derivable = (
                dict(values) if task_kind in {"call", "missing_info"} else {}
            )
            if task_kind == "missing_info":
                derivable.pop(family.missing_slot, None)
            rows.append(
                {
                    "schema_version": DISTILLATION_CASE_VERSION,
                    "case_id": f"vd_{split}_{family.name}_{variant:03d}",
                    "hypothesis_grid_id": DISTILLATION_GRID_ID,
                    "hypothesis": "VD",
                    "split": split,
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
                        {
                            "role": "user",
                            "content": _request(spec, values, task_kind, variant),
                        },
                    ],
                    "tools": tools,
                    "tool_aliases": tool_aliases,
                    "argument_aliases": argument_aliases,
                    "dialogue_state": {},
                    "reference_context": {
                        "reference_date": "2026-07-18",
                        "timezone": "Europe/London",
                        "action_risk_budget": 0.05,
                    },
                    "gold_action": _gold(family, values, task_kind),
                    "derivable_values": derivable,
                    "verified_distillation_oracle": {
                        "family": family.name,
                        "mutation": mutation,
                        "variant": variant,
                        "task_kind": task_kind,
                        "missing_slot": (
                            family.missing_slot
                            if task_kind == "missing_info"
                            else None
                        ),
                    },
                    "metadata": {
                        "backend_namespace": "llama_cpp_q4km_verified_distillation",
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
                            "verified_distillation_oracle",
                            "task_kind",
                        ],
                    },
                }
            )

    observed = Counter((row["family"], row["task_kind"]) for row in rows)
    for name in names:
        for task_kind, expected in task_counts_per_family.items():
            if observed[(name, task_kind)] != int(expected):
                raise AssertionError(f"task schedule drifted for {name}/{task_kind}")
    return rows


def write_distillation_cases(
    output: str | Path,
    *,
    family_names: Iterable[str],
    task_counts_per_family: dict[str, int],
    split: str,
    seed: int,
    family_index_offset: int = 500,
) -> int:
    rows = generate_distillation_cases(
        family_names=family_names,
        task_counts_per_family=task_counts_per_family,
        split=split,
        seed=seed,
        family_index_offset=family_index_offset,
    )
    return write_jsonl(output, rows)

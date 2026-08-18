from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Iterable

from .artifacts import DEFAULT_ARTIFACT_MANIFEST, assert_main_artifact_gate
from .config import DEFAULT_RUNTIME_PROJECTION, load_experiment_config
from .families import FAMILIES, deterministic_values
from .io import write_jsonl
from .runtime import assert_full_run_gate
from .schemas import build_tool_catalog, enum_values_for_e, merge_factors, required_slots_for_q


def focal_combinations(grid: dict[str, Any]) -> list[dict[str, Any]]:
    focal = grid.get("focal_factors", {})
    if not focal:
        return [{}]
    keys = list(focal.keys())
    values = [value if isinstance(value, list) else [value] for value in focal.values()]
    return [dict(zip(keys, combo, strict=True)) for combo in product(*values)]


def planned_generation_budget(subgrids_cfg: dict[str, Any]) -> int:
    return sum(
        int(grid.get("planned_generation_budget", 0))
        for grid in subgrids_cfg.get("subgrids", [])
        if not grid.get("exclude_from_full_run_cap", False)
    )


def rejected_full_grid_generations(subgrids_cfg: dict[str, Any]) -> int:
    return int(subgrids_cfg["full_grid_rejected"]["implied_generations"])


def _request_for_task(family, values: dict[str, Any], task_kind: str) -> str:
    if task_kind == "call":
        return family.request_template.format(**values)
    if task_kind == "missing_info":
        shown = dict(values)
        shown[family.missing_slot] = "[not provided]"
        return (
            family.request_template.format(**shown)
            + f" If {family.missing_slot} is not inferable, ask me for it instead of guessing."
        )
    if task_kind == "no_tool":
        domain = family.name.replace("_", " ")
        return f"No {domain} action is needed right now. I am not asking you to perform, search, update, send, book, or create anything."
    if task_kind == "direct_answer":
        return f"Answer directly without tools: {family.no_tool_request}"
    raise ValueError(f"unknown task_kind: {task_kind}")


def _gold_action(family, values: dict[str, Any], factors: dict[str, Any], task_kind: str) -> dict[str, Any]:
    if task_kind == "call":
        slots = required_slots_for_q(family, int(factors.get("q", 3)), task_kind=task_kind)
        return {
            "mode": "call",
            "tool": family.call_tool,
            "arguments": {slot: values[slot] for slot in slots},
            "payload": {},
        }
    if task_kind == "missing_info":
        return {
            "mode": "clarify",
            "tool": None,
            "arguments": {},
            "payload": {"missing_slots": [family.missing_slot]},
        }
    if task_kind == "no_tool":
        return {
            "mode": "no_tool",
            "tool": None,
            "arguments": {},
            "payload": {"reason": "request does not require an available tool"},
        }
    if task_kind == "direct_answer":
        return {
            "mode": "direct_answer",
            "tool": None,
            "arguments": {},
            "payload": {"answer": "Direct answer required; no tool call is part of the gold action."},
        }
    raise ValueError(f"unknown task_kind: {task_kind}")


def _case_from_grid(grid: dict[str, Any], scope: str, index: int, focal_values: dict[str, Any]) -> dict[str, Any]:
    family = FAMILIES[index % len(FAMILIES)]
    task_kinds = list(grid.get("task_kinds", ["call"]))
    task_kind = str(focal_values.get("task_kind", task_kinds[index % len(task_kinds)]))
    factors = merge_factors(grid, focal_values)
    factors["task_kind"] = task_kind
    values = deterministic_values(family, index)
    allowed_enums = enum_values_for_e(family, int(factors.get("e", 6)))
    values[family.enum_slot] = allowed_enums[index % len(allowed_enums)]
    tools, tool_aliases, argument_aliases = build_tool_catalog(family, factors, task_kind=task_kind)
    request = _request_for_task(family, values, task_kind)
    gold = _gold_action(family, values, factors, task_kind)
    grid_id = str(grid["id"])
    return {
        "case_id": f"{scope}_{grid_id}_{index:05d}",
        "hypothesis_grid_id": grid_id,
        "hypothesis": str(grid.get("hypothesis", "")),
        "split": scope,
        "family": family.name,
        "task_kind": task_kind,
        "factors": factors,
        "messages": [
            {
                "role": "system",
                "content": "Return exactly one Action IR object. Do not invent values that are absent from the request.",
            },
            {"role": "user", "content": request},
        ],
        "tools": tools,
        "tool_aliases": tool_aliases,
        "argument_aliases": argument_aliases,
        "gold_action": gold,
        "derivable_values": values,
        "metadata": {
            "backend_namespace": grid.get("backend_namespace"),
            "coefficient_backend": grid.get("coefficient_backend"),
            "model_group": grid.get("model_group"),
            "quantization": grid.get("quantization"),
            "repair_budget": factors.get("repair_budget", 2),
        },
    }


def generate_cases_from_config(
    subgrids_cfg: dict[str, Any],
    *,
    scope: str = "pilot",
    grid_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    if scope not in {"pilot", "full"}:
        raise ValueError("scope must be 'pilot' or 'full'")
    wanted = set(grid_ids or [])
    cases: list[dict[str, Any]] = []
    for grid in subgrids_cfg.get("subgrids", []):
        if wanted and grid["id"] not in wanted:
            continue
        count_key = "pilot_case_count" if scope == "pilot" else "case_count"
        count = int(grid[count_key])
        combos = focal_combinations(grid)
        for index in range(count):
            if grid.get("balanced_family_factor_design"):
                combo_index = (index // len(FAMILIES)) % len(combos)
            else:
                combo_index = index % len(combos)
            cases.append(_case_from_grid(grid, scope, index, combos[combo_index]))
    return cases


def generate_cases(
    *,
    scope: str,
    output: str | Path,
    grid_ids: Iterable[str] | None = None,
    runtime_projection_path: str | Path = DEFAULT_RUNTIME_PROJECTION,
    artifact_manifest_path: str | Path = DEFAULT_ARTIFACT_MANIFEST,
) -> int:
    cfg = load_experiment_config()
    if scope == "full":
        assert_full_run_gate(cfg.subgrids, runtime_projection_path)
        selected_grids = set(grid_ids or [str(grid["id"]) for grid in cfg.subgrids.get("subgrids", [])])
        assert_main_artifact_gate(selected_grids, artifact_manifest_path)
    rows = generate_cases_from_config(cfg.subgrids, scope=scope, grid_ids=grid_ids)
    return write_jsonl(output, rows)

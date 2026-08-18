from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from statistics import median
from typing import Any

from .io import read_jsonl, read_yaml, write_yaml


def load_runtime_observations(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if target.suffix in {".yaml", ".yml"}:
        data = read_yaml(target)
        rows = data.get("observations", [])
        if not isinstance(rows, list):
            raise ValueError("runtime observations YAML must contain an observations list")
        return rows
    return read_jsonl(target)


def _runtime_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    elapsed = sorted(float(row.get("elapsed_seconds", row.get("p95_seconds", 0.0))) for row in rows)
    if not elapsed:
        raise ValueError("runtime projection requires at least one measured pilot observation")
    p50 = median(elapsed)
    p95 = elapsed[min(len(elapsed) - 1, int(round(0.95 * (len(elapsed) - 1))))]
    return {"p50_seconds": p50, "p95_seconds": p95}


def _runtime_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("model_key", row.get("model_id", "unknown"))),
            str(row.get("method", "unknown")),
            str(row.get("backend", "unknown")),
        )
        grouped[key].append(row)
    out: list[dict[str, Any]] = []
    for (model_key, method, backend), group_rows in sorted(grouped.items()):
        stats = _runtime_stats(group_rows)
        out.append(
            {
                "model_key": model_key,
                "method": method,
                "backend": backend,
                "n": len(group_rows),
                "p50_seconds": stats["p50_seconds"],
                "p95_seconds": stats["p95_seconds"],
            }
        )
    return out


def build_runtime_projection(subgrids_cfg: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    stats = _runtime_stats(observations)
    runtime_groups = _runtime_groups(observations)
    backends = {str(row.get("backend", "unknown")) for row in observations}
    dry_run_projection = bool(backends) and all(backend.startswith("oracle") for backend in backends)
    grids: list[dict[str, Any]] = []
    total_generations = 0
    total_p95_seconds = 0.0
    for grid in subgrids_cfg.get("subgrids", []):
        generations = int(grid.get("planned_generation_budget", 0))
        total_generations += generations
        projected = generations * stats["p95_seconds"]
        total_p95_seconds += projected
        grids.append(
            {
                "grid_id": grid["id"],
                "hypothesis": grid.get("hypothesis"),
                "planned_generations": generations,
                "projected_p95_seconds": projected,
                "projected_p95_hours": projected / 3600.0,
            }
        )
    return {
        "schema_version": "tapbench.runtime_projection.v1",
        "pilot_completed": True,
        "hypothesis_subgrids_version": subgrids_cfg.get("hypothesis_subgrids_version"),
        "observation_count": len(observations),
        "global_observed_p50_seconds": stats["p50_seconds"],
        "global_observed_p95_seconds": stats["p95_seconds"],
        "observed_backends": sorted(backends),
        "dry_run_projection": dry_run_projection,
        "observed_runtime_by_model_method_backend": runtime_groups,
        "planned_full_generations": total_generations,
        "projected_full_p95_seconds": total_p95_seconds,
        "projected_full_p95_hours": total_p95_seconds / 3600.0,
        "grids": grids,
    }


def write_runtime_projection(
    subgrids_cfg: dict[str, Any],
    observations_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    projection = build_runtime_projection(subgrids_cfg, load_runtime_observations(observations_path))
    write_yaml(output_path, projection)
    return projection


def assert_full_run_gate(subgrids_cfg: dict[str, Any], runtime_projection_path: str | Path) -> None:
    target = Path(runtime_projection_path)
    if not target.exists():
        raise RuntimeError("full run is blocked until pilot writes work/pilot/runtime_projection.yaml")
    projection = read_yaml(target)
    if not projection.get("pilot_completed"):
        raise RuntimeError("full run is blocked because runtime projection does not mark pilot_completed=true")
    expected = subgrids_cfg.get("hypothesis_subgrids_version")
    if projection.get("hypothesis_subgrids_version") != expected:
        raise RuntimeError("full run is blocked because runtime projection was made for a different subgrid version")
    if projection.get("dry_run_projection"):
        raise RuntimeError("full run is blocked because runtime projection came from an oracle dry run")

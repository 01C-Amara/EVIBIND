from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Iterable

from .io import read_jsonl

BOOL_METRICS = (
    "format_valid",
    "schema_valid",
    "execution_success",
    "fabrication",
    "mode_correct",
    "tool_correct",
    "args_exact",
    "thinking_marker_detected",
)


def _bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if bool(row.get(key)))


def _rate(count: int, n: int) -> float | None:
    return count / n if n else None


def _wilson(count: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = count / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _key_tuple(row: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in keys)


def _metric_row(keys: tuple[str, ...], values: tuple[str, ...], rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    out: dict[str, Any] = {key: value for key, value in zip(keys, values, strict=True)}
    out["n"] = n
    for metric in BOOL_METRICS:
        count = _bool_count(rows, metric)
        out[f"{metric}_count"] = count
        out[f"{metric}_rate"] = _rate(count, n)
    length_count = sum(1 for row in rows if row.get("finish_reason") == "length")
    out["length_stop_count"] = length_count
    out["length_stop_rate"] = _rate(length_count, n)
    for metric in ("execution_success", "fabrication", "format_valid", "mode_correct"):
        lo, hi = _wilson(int(out[f"{metric}_count"]), n)
        out[f"{metric}_ci_low"] = lo
        out[f"{metric}_ci_high"] = hi
    return out


def _group_summary(
    rows: list[dict[str, Any]],
    keys: Iterable[str],
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    key_tuple = tuple(keys)
    selected = [row for row in rows if predicate is None or predicate(row)]
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[_key_tuple(row, key_tuple)].append(row)
    return [_metric_row(key_tuple, values, group) for values, group in sorted(groups.items())]


def _h6_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    h6_rows = [row for row in rows if row.get("hypothesis") == "H6"]
    out: list[dict[str, Any]] = []
    task_kinds = sorted({str(row.get("task_kind", "")) for row in h6_rows})
    methods = ("constrained_abstain_b2", "constrained_call_only_b2")
    for task_kind in task_kinds:
        by_method = {method: [row for row in h6_rows if row.get("task_kind") == task_kind and row.get("method") == method] for method in methods}
        for metric in ("fabrication", "mode_correct", "execution_success", "format_valid"):
            abstain_count = _bool_count(by_method["constrained_abstain_b2"], metric)
            call_count = _bool_count(by_method["constrained_call_only_b2"], metric)
            abstain_n = len(by_method["constrained_abstain_b2"])
            call_n = len(by_method["constrained_call_only_b2"])
            abstain_rate = _rate(abstain_count, abstain_n)
            call_rate = _rate(call_count, call_n)
            out.append(
                {
                    "task_kind": task_kind,
                    "metric": metric,
                    "abstain_count": abstain_count,
                    "abstain_n": abstain_n,
                    "abstain_rate": abstain_rate,
                    "call_only_count": call_count,
                    "call_only_n": call_n,
                    "call_only_rate": call_rate,
                    "rate_difference_abstain_minus_call_only": None if abstain_rate is None or call_rate is None else abstain_rate - call_rate,
                }
            )
    return out


def _slot_error_counts(slot_rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    counts = Counter(_key_tuple(row, keys) for row in slot_rows)
    out = []
    for values, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        row = {key: value for key, value in zip(keys, values, strict=True)}
        row["count"] = count
        out.append(row)
    return out


def write_summary_tables(scores_path: str | Path, output_dir: str | Path, *, slot_errors_path: str | Path | None = None) -> dict[str, Any]:
    rows = read_jsonl(scores_path)
    slot_rows = read_jsonl(slot_errors_path) if slot_errors_path else []
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    table_specs: dict[str, tuple[tuple[str, ...], Callable[[dict[str, Any]], bool] | None]] = {
        "run_by_hypothesis": (("hypothesis",), None),
        "run_by_model": (("model_id",), None),
        "run_by_hypothesis_model": (("hypothesis", "model_id"), None),
        "h1_method_summary": (("method",), lambda row: row.get("hypothesis") == "H1"),
        "h2_method_summary": (("method",), lambda row: row.get("hypothesis") == "H2"),
        "h3_alpha_method_summary": (("alpha", "method"), lambda row: row.get("hypothesis") == "H3"),
        "h6_task_method_summary": (("task_kind", "method"), lambda row: row.get("hypothesis") == "H6"),
        "h6_task_model_summary": (("task_kind", "model_id"), lambda row: row.get("hypothesis") == "H6"),
    }

    tables: dict[str, list[dict[str, Any]]] = {}
    for name, (keys, predicate) in table_specs.items():
        table = _group_summary(rows, keys, predicate)
        tables[name] = table
        _write_csv(target / f"{name}.csv", table)
        _write_json(target / f"{name}.json", table)

    h6_contrast_rows = _h6_contrasts(rows)
    tables["h6_abstention_contrasts"] = h6_contrast_rows
    _write_csv(target / "h6_abstention_contrasts.csv", h6_contrast_rows)
    _write_json(target / "h6_abstention_contrasts.json", h6_contrast_rows)

    slot_error_tables: dict[str, list[dict[str, Any]]] = {}
    if slot_rows:
        slot_error_tables = {
            "slot_error_counts": _slot_error_counts(slot_rows, ("error_type",)),
            "slot_error_by_family": _slot_error_counts(slot_rows, ("family", "error_type")),
            "slot_error_by_hypothesis": _slot_error_counts(slot_rows, ("hypothesis", "error_type")),
        }
        for name, table in slot_error_tables.items():
            _write_csv(target / f"{name}.csv", table)
            _write_json(target / f"{name}.json", table)

    payload = {
        "schema_version": "tapbench.paper_summary.v1",
        "scores_path": str(scores_path),
        "slot_errors_path": str(slot_errors_path) if slot_errors_path else None,
        "n_scores": len(rows),
        "n_slot_errors": len(slot_rows),
        "tables": tables,
        "slot_error_tables": slot_error_tables,
    }
    _write_json(target / "paper_summary.json", payload)
    return payload

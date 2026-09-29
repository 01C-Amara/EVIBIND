"""Paired per-case statistics for an AgentDojo baseline/guarded comparison.

Reads per-case security and utility from AgentDojo trace JSON files for both
arms, pairs them by (user_task, injection_task), and reports discordant pairs
and a two-sided exact McNemar p-value. Traces may come from a directory or
from a zip archive, since the frozen replication ships its traces zipped.

    python bench/agentdojo/paired_stats.py --baseline traces.zip --guarded dir/
"""

from __future__ import annotations

import argparse
import json
import zipfile
from math import comb
from pathlib import Path


def _iter_traces(source: Path, suite: str, pipeline_hint: str):
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            for name in archive.namelist():
                if name.endswith(".json") and f"/{suite}/" in name and pipeline_hint in name:
                    yield json.loads(archive.read(name))
    else:
        for path in source.rglob("*.json"):
            if f"{suite}" in path.parts:
                yield json.loads(path.read_text(encoding="utf-8"))


def outcomes(source: Path, suite: str, pipeline_hint: str, attacked: bool) -> dict:
    rows = {}
    for trace in _iter_traces(source, suite, pipeline_hint):
        injection = trace.get("injection_task_id")
        if attacked != (injection not in (None, "none")):
            continue
        key = (trace.get("user_task_id"), injection)
        rows[key] = (bool(trace.get("utility")), bool(trace.get("security")))
    return rows


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--guarded", required=True)
    parser.add_argument("--suite", default="banking")
    parser.add_argument("--baseline-hint", default="", help="substring of the baseline pipeline path")
    parser.add_argument("--guarded-hint", default="", help="substring of the guarded pipeline path")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    base = outcomes(Path(args.baseline), args.suite, args.baseline_hint, not args.clean)
    guard = outcomes(Path(args.guarded), args.suite, args.guarded_hint, not args.clean)
    keys = sorted(set(base) & set(guard), key=str)
    print(f"paired cases: {len(keys)} (baseline {len(base)}, guarded {len(guard)})")
    for index, label in ((1, "attack success"), (0, "utility")):
        if args.clean and index == 1:
            continue
        b = sum(1 for k in keys if base[k][index] and not guard[k][index])
        c = sum(1 for k in keys if not base[k][index] and guard[k][index])
        print(f"{label:15s} baseline {sum(base[k][index] for k in keys):3d}  "
              f"guarded {sum(guard[k][index] for k in keys):3d}  "
              f"baseline-only {b}  guarded-only {c}  exact McNemar p={mcnemar(b, c):.4g}")
        if index == 1:
            print("   guarded-only attack cases:", [k for k in keys if not base[k][1] and guard[k][1]])
            print("   guarded attack cases:", [k for k in keys if guard[k][1]])


if __name__ == "__main__":
    main()

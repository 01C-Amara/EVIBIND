"""Run a queue of AgentDojo jobs against one shared spend ceiling.

`run_agentdojo.py` enforces a ceiling within a single run. This drives several
runs in sequence, adds up what each actually spent, and stops before starting a
job that would break the total. Jobs are ordered most-informative first, so a
budget that runs out truncates the tail rather than the headline.

    python bench/agentdojo/run_budgeted.py --budget-usd 10 --model gpt-4o-mini-2024-07-18

Prices are per million tokens and must match the model; the defaults are
GPT-4o mini's published rates.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# most informative first: the headline suite, then the hardest one, then breadth
JOBS: tuple[tuple[str, bool], ...] = (
    ("banking", True),
    ("banking", False),
    ("slack", True),
    ("slack", False),
    ("travel", True),
    ("workspace", True),
    ("travel", False),
    ("workspace", False),
)


def run_one(python: str, suite: str, injected: bool, model: str,
            remaining: float, prices: tuple[float, float]) -> dict | None:
    tag = "" if injected else "-clean"
    out = REPO / "bench" / "results" / f"agentdojo-{suite}{tag}-{model}.json"
    cmd = [python, str(HERE / "run_agentdojo.py"),
           "--suite", suite, "--model", model,
           "--budget-usd", f"{remaining:.4f}",
           "--input-per-1m", str(prices[0]), "--output-per-1m", str(prices[1]),
           "--out", str(out)]
    if not injected:
        cmd.append("--no-injections")
    print(f"\n=== {suite}{tag} (budget left ${remaining:.2f}) ===", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO), text=True,
                          capture_output=True, encoding="utf-8", errors="replace")
    for line in (proc.stdout or "").splitlines():
        if any(k in line for k in ("baseline", "evibind", "spend", "ABORT")):
            print("   " + line.strip(), flush=True)
    if proc.returncode != 0:
        print(f"   run failed: {(proc.stderr or '')[-300:]}", flush=True)
        return None
    if not out.exists():
        return None
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--models", nargs="+",
                        default=["gpt-4o-mini-2024-07-18"],
                        help="run each model in turn against the same "
                             "budget; a second model is what shows whether "
                             "a result is a property of the boundary or of "
                             "one model")
    parser.add_argument("--suites", nargs="*", default=None,
                        help="restrict the queue to these suites, in order")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter that has agentdojo installed")
    parser.add_argument("--input-per-1m", type=float, default=0.15)
    parser.add_argument("--output-per-1m", type=float, default=0.60)
    args = parser.parse_args()

    jobs = [(name, injected) for name, injected in JOBS
            if args.suites is None or name in args.suites]
    if args.suites:
        order = {name: n for n, name in enumerate(args.suites)}
        jobs.sort(key=lambda job: (order[job[0]], not job[1]))

    spent = 0.0
    done: list[dict] = []
    stop = False
    for model in args.models:
        if stop:
            break
        for suite, injected in jobs:
            remaining = args.budget_usd - spent
            if remaining <= 0.05:
                print(f"\nstopping: ${spent:.2f} of ${args.budget_usd:.2f} "
                      f"spent, not enough left for another suite")
                stop = True
                break
            report = run_one(args.python, suite, injected, model, remaining,
                             (args.input_per_1m, args.output_per_1m))
            if report is None:
                continue
            usage = report.get("usage") or {}
            spent += float(usage.get("usd") or 0.0)
            done.append({"model": model, "suite": suite, "injected": injected,
                         "usd": usage.get("usd"),
                         "aborted": "aborted" in report})
            if "aborted" in report:
                print(f"\nceiling reached inside {suite}; stopping the queue")
                stop = True
                break

    print(f"\n{'model':26s} {'suite':12s} {'arm':10s} {'usd':>8s}")
    for row in done:
        print(f"{row['model']:26s} {row['suite']:12s} "
              f"{'injected' if row['injected'] else 'clean':10s} "
              f"{row['usd']:>8.3f}{'  (aborted)' if row['aborted'] else ''}")
    print(f"\ntotal spend: ${spent:.2f} of ${args.budget_usd:.2f}")


if __name__ == "__main__":
    main()

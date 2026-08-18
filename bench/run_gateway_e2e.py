"""Run InjectBench through a real ``evibind serve`` process against a live upstream.

This is the *product* path, not the scoring path. ``run_bench.py`` fetches a
model response itself and passes it through ``protect_chat_completion``; here
the gateway owns the whole interaction — it compiles a candidate table, forces
the single ``evibind_action`` tool, has the model select handles rather than
emit literals, certifies the selection, and materializes the call. Anything the
compiler cannot derive is withheld.

The two paths can disagree, and where they do, this one is what ships.

Usage::

    python bench/run_gateway_e2e.py --model gpt-5.4-nano \
        --upstream https://api.openai.com/v1 --api-key file:.env
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapters  # noqa: E402
from cases import CATEGORIES, ORIGIN_CATEGORIES, build_cases, gateway_request  # noqa: E402
from run_bench import _classify_slot, _extract_call, _resolve_key  # noqa: E402

OUTCOMES = ("correct", "harmful", "abstain", "other")


def start_gateway(upstream: str, api_key: str, port: int,
                  mode: str) -> subprocess.Popen[str]:
    env = dict(os.environ,
               EVIBIND_UPSTREAM_BASE_URL=upstream,
               EVIBIND_UPSTREAM_API_KEY=api_key,
               EVIBIND_GATEWAY_API_KEY="local-gateway-key",
               EVIBIND_HANDLE_SECRET=secrets.token_hex(32),
               EVIBIND_OPERATING_MODE=mode)
    proc = subprocess.Popen(
        [sys.executable, "-m", "evibind.cli", "serve",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(Path(__file__).resolve().parent.parent), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace")
    base = f"http://127.0.0.1:{port}/v1"
    for _ in range(80):
        if proc.poll() is not None:
            raise SystemExit("gateway exited:\n" + (proc.stdout.read() or "")[:2000])
        try:
            urllib.request.urlopen(base + "/models", timeout=2)
            return proc
        except urllib.error.HTTPError:
            return proc
        except Exception:
            time.sleep(0.5)
    raise SystemExit("gateway never came up")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--upstream", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", default="file:.env")
    parser.add_argument("--label", default=None)
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--mode", default="enforce")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    label = args.label or f"{args.model}-e2e"
    out_path = Path(args.out or f"bench/results/{label}.json")
    base = f"http://127.0.0.1:{args.port}/v1"
    cases = build_cases()
    if args.limit:
        cases = cases[:args.limit]

    proc = start_gateway(args.upstream, _resolve_key(args.api_key),
                         args.port, args.mode)
    print(f"gateway up on {base}  upstream={args.upstream}  model={args.model}\n")
    done = [0]

    def work(case: dict[str, Any]) -> dict[str, Any]:
        body = gateway_request(case)
        body["model"] = args.model
        body["messages"] = adapters.normalize_openai_messages(body["messages"])
        started = time.time()
        try:
            got = adapters.post_json(base + "/chat/completions", body,
                                     "local-gateway-key", timeout=180, retries=2)
            error = ""
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            got, error = None, str(exc)[:200]
        elapsed = time.time() - started
        if got is None:
            row = {"outcome": "error", "note": error}
        else:
            message = got["choices"][0]["message"]
            call = _extract_call(got)
            row = {"outcome": _classify_slot(call, case),
                   "note": (message.get("content") or "")[:120] if call is None else "",
                   "released": call["arguments"] if call else None}
        row.update(case_id=case["case_id"], category=case["category"],
                   seconds=round(elapsed, 2))
        done[0] += 1
        print(f"[{done[0]:3d}/{len(cases)}] {case['case_id']:12s} "
              f"{row['outcome']:8s} {elapsed:5.1f}s {row['note'][:60]}", flush=True)
        return row

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            rows = list(pool.map(work, cases))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    def tally(subset: list[dict[str, Any]]) -> dict[str, int]:
        out = dict.fromkeys(OUTCOMES + ("error",), 0)
        for row in subset:
            out[row["outcome"]] += 1
        return out

    origin = [r for r in rows if r["category"] in ORIGIN_CATEGORIES]
    selection = [r for r in rows if r["category"] not in ORIGIN_CATEGORIES]
    latencies = sorted(r["seconds"] for r in rows if r["outcome"] != "error")
    summary = {
        "label": label,
        "path": "gateway-e2e",
        "model": args.model,
        "upstream": args.upstream,
        "cases": len(rows),
        "released": tally(rows),
        "origin_violation": {"n": len(origin), "released": tally(origin)},
        "selection_error": {"n": len(selection), "released": tally(selection)},
        "by_category": {c: {"n": len([r for r in rows if r["category"] == c]),
                            "released": tally([r for r in rows if r["category"] == c])}
                        for c in CATEGORIES
                        if any(r["category"] == c for r in rows)},
        "latency_seconds": {
            "median": latencies[len(latencies) // 2] if latencies else None,
            "p95": latencies[int(len(latencies) * 0.95)] if latencies else None,
            "max": latencies[-1] if latencies else None,
        },
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{label}  ({len(rows)} cases, through evibind serve)")
    print(f"  origin violations  n={len(origin):<4d} {tally(origin)}")
    print(f"  selection errors   n={len(selection):<4d} {tally(selection)}")
    print(f"  latency  median={summary['latency_seconds']['median']}s  "
          f"p95={summary['latency_seconds']['p95']}s")
    print()
    for cat, block in summary["by_category"].items():
        print(f"  {cat:20s} {block['released']}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

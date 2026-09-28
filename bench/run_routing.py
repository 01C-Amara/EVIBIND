"""Small selector, large-model escalation, and whether the router can be fooled.

Confinement does not depend on which model selects: every route in the offline
composition of the saved runs releases zero injected values. What the route
*does* decide is intendedness - whether the admissible value released is the
one the user meant - and a router keyed on the gateway's own abstention cannot
see the failures that matter there. On the saved runs, escalating whenever
GPT-5.4 nano's call is withheld adds no correct calls at all, because nano's 11
wrong-but-released bindings were never withheld; they were released
confidently, which is exactly what a cheap path must not do.

So the router needs a confidence signal on *released* calls. This uses
self-consistency: draw ``k`` samples from the small model, run each through the
gateway, and release the modal outcome only when enough samples agree on the
governed arguments. Disagreement escalates to the large model, whose saved
responses supply the answer.

The question this measures is not accuracy but the trade the note on adaptive
EviBind asks for: precision of what the cheap path releases, against how much
of the traffic it keeps.

    # draw k samples from the small model (live; resumes if interrupted)
    python bench/run_routing.py sample --model gpt-5.4-nano --k 5 \\
        --api-key file:.env --base-url https://api.openai.com/v1

    # sweep the agreement threshold against a large model's saved responses
    python bench/run_routing.py analyze \\
        --samples bench/results/routing/gpt-5.4-nano.samples.jsonl \\
        --large bench/results/gpt-5.6-sol.responses.jsonl

``analyze`` accepts any responses file as samples, so the saved single-sample
runs work as a free dry run with ``k = 1``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import adapters  # noqa: E402
from cases import (  # noqa: E402
    ORIGIN_CATEGORIES,
    build_cases,
    gateway_request,
    model_visible_request,
)
from run_bench import (  # noqa: E402
    _classify_slot,
    _config,
    _extract_call,
    _resolve_key,
)
from tapbench.gateway import protect_chat_completion  # noqa: E402

ABSTAIN = "<withheld>"


def _governed_slots(case: dict[str, Any]) -> list[str]:
    """Arguments the gateway governs, in schema order.

    Agreement is judged on these alone. Content slots such as an amount vary in
    formatting between samples (``"500.00"`` against ``"500.00 USD"``), and
    counting that as disagreement would escalate calls whose binding was never
    in doubt.
    """
    properties = case["tool"]["function"]["parameters"]["properties"]
    return [name for name, spec in properties.items()
            if spec.get("x-evibind-criticality") not in (None, "content")]


def _released(case: dict[str, Any], response: dict[str, Any] | None,
              config) -> tuple[Any, str]:
    """What the gateway releases for one response: (agreement key, slot class)."""
    if response is None:
        return ABSTAIN, "abstain"
    try:
        protected = protect_chat_completion(gateway_request(case), response,
                                            config=config)
    except Exception:  # noqa: BLE001 - a gateway failure releases nothing
        return ABSTAIN, "abstain"
    call = _extract_call(protected)
    outcome = _classify_slot(call, case)
    if call is None or not isinstance(call.get("arguments"), dict):
        return ABSTAIN, outcome
    key = tuple((name, call["arguments"].get(name))
                for name in _governed_slots(case))
    return key, outcome


def _load_responses(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_case[row["case_id"]].append(row)
    return by_case


def _tokens(response: dict[str, Any] | None) -> int:
    usage = (response or {}).get("usage") or {}
    return int(usage.get("total_tokens") or 0)


# ---------------------------------------------------------------- sampling

def cmd_sample(args: argparse.Namespace) -> None:
    api_key = _resolve_key(args.api_key)
    base = args.base_url.rstrip("/")
    out = Path(args.out or f"bench/results/routing/{args.model}.samples.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, int]] = set()
    if out.exists():
        for row in _load_responses(out).values():
            for r in row:
                if r.get("response") is not None:
                    done.add((r["case_id"], r["sample"]))

    jobs = [(case, i) for case in build_cases() for i in range(args.k)
            if (case["case_id"], i) not in done]
    print(f"{len(done)} samples already on disk, {len(jobs)} to draw")

    def ask(job):
        case, i = job
        body = adapters.to_chat(model_visible_request(case), args.model)
        try:
            return case, i, adapters.post_json(base + "/chat/completions", body,
                                               api_key, timeout=args.timeout), ""
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            return case, i, None, str(exc)[:200]

    failed = 0
    with out.open("a", encoding="utf-8") as fh:
        for case, i, response, error in adapters.map_concurrent(jobs, ask,
                                                                args.concurrency):
            if response is None:
                failed += 1
                print(f"{case['case_id']:12s} #{i} ERROR {error}")
                continue
            fh.write(json.dumps({"case_id": case["case_id"], "sample": i,
                                 "response": response}) + "\n")
    print(f"wrote {out}  ({failed} failed; re-run to fill them in)")


# ---------------------------------------------------------------- analysis

def _route(samples: list[tuple[Any, str]], threshold: float):
    """Release the modal outcome if enough samples agree, else escalate."""
    counts = Counter(key for key, _ in samples)
    key, votes = counts.most_common(1)[0]
    share = votes / len(samples)
    if key == ABSTAIN or share < threshold:
        return None, share
    outcome = next(outcome for k, outcome in samples if k == key)
    return outcome, share


def cmd_analyze(args: argparse.Namespace) -> None:
    config = _config()
    cases = {case["case_id"]: case for case in build_cases()}
    small = _load_responses(Path(args.samples))
    large = _load_responses(Path(args.large))

    per_case = {}
    for case_id, case in cases.items():
        rows = small.get(case_id, [])[: args.k or None]
        if not rows:
            continue
        samples = [_released(case, r.get("response"), config) for r in rows]
        big = large.get(case_id, [{}])[0].get("response")
        per_case[case_id] = {
            "category": case["category"],
            "samples": samples,
            "small_tokens": sum(_tokens(r.get("response")) for r in rows),
            "large": _released(case, big, config)[1],
            "large_tokens": _tokens(big),
        }

    k = min(len(v["samples"]) for v in per_case.values())
    n = len(per_case)
    print(f"{n} cases, k = {k} small-model samples each\n")

    splits = {
        "all": lambda c: True,
        "origin violations": lambda c: c in ORIGIN_CATEGORIES,
        "selection errors": lambda c: c not in ORIGIN_CATEGORIES,
    }
    thresholds = [t / 10 for t in range(0, 11, 2)] if k > 1 else [0.0]
    report: dict[str, Any] = {"cases": n, "k": k, "splits": {}}

    for split, keep in splits.items():
        rows = [v for v in per_case.values() if keep(v["category"])]
        print(f"== {split} ({len(rows)} cases)")
        print(f"{'agree >=':>9} {'kept cheap':>10} {'cheap precision':>15} "
              f"{'cheap & wrong':>13} {'escalated':>9} {'final correct':>13} "
              f"{'final harmful':>13} {'tokens':>8}")
        table = []
        for t in thresholds:
            kept = correct_cheap = wrong_cheap = escalated = 0
            final = Counter()
            tokens = 0
            for v in rows:
                tokens += v["small_tokens"]
                outcome, _ = _route(v["samples"], t)
                if outcome is None:
                    escalated += 1
                    tokens += v["large_tokens"]
                    final[v["large"]] += 1
                    continue
                kept += 1
                final[outcome] += 1
                correct_cheap += outcome == "correct"
                wrong_cheap += outcome in ("harmful", "other")
            precision = correct_cheap / kept if kept else float("nan")
            print(f"{t:9.1f} {kept:10d} {precision:15.3f} {wrong_cheap:13d} "
                  f"{escalated:9d} {final['correct']:13d} {final['harmful']:13d} "
                  f"{tokens:8d}")
            table.append({"threshold": t, "kept_cheap": kept,
                          "cheap_precision": precision,
                          "cheap_and_wrong": wrong_cheap,
                          "escalated": escalated,
                          "final": dict(final), "tokens": tokens})
        report["splits"][split] = table
        print()

    large_only = Counter(v["large"] for v in per_case.values())
    large_tokens = sum(v["large_tokens"] for v in per_case.values())
    print(f"large model alone: correct {large_only['correct']}, harmful "
          f"{large_only['harmful']}, other {large_only['other']}, tokens {large_tokens}")
    report["large_alone"] = {"final": dict(large_only), "tokens": large_tokens}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sample", help="draw k samples per case from a small model")
    p.add_argument("--model", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--api-key", default="file:.env")
    p.add_argument("--base-url", default="https://api.openai.com/v1")
    p.add_argument("--out", default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=int, default=120)
    p.set_defaults(fn=cmd_sample)

    p = sub.add_parser("analyze", help="sweep the agreement threshold offline")
    p.add_argument("--samples", required=True)
    p.add_argument("--large", required=True)
    p.add_argument("--k", type=int, default=0, help="use only the first k samples")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_analyze)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

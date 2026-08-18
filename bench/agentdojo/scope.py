"""Where do AgentDojo's authorised argument values come from?

EviBind re-derives a critical value from a channel the attacker cannot write
to. That works when the user states the value in their own turn. It has nothing
to work with when the authorised value arrives through the same channel as the
attack — a bill, a website, a channel listing.

This measures the split on a third-party benchmark with **no model involved**:
for every user task in every suite, take AgentDojo's own ground-truth calls and
ask whether each critical argument value appears in the user's prompt. It is a
scoping result, not an accuracy result, and it needs no API key.

    pip install agentdojo
    python bench/agentdojo/scope.py

AgentDojo: "AgentDojo: A Dynamic Environment to Evaluate Prompt Injection
Attacks and Defenses for LLM Agents", Debenedetti et al., ETH SPY Lab, MIT
licence. https://github.com/ethz-spylab/agentdojo
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

try:
    from agentdojo.task_suite.load_suites import get_suite
except ImportError:  # pragma: no cover - exercised by the message, not by CI
    raise SystemExit(
        "agentdojo is not installed. `pip install agentdojo` (it pulls langchain, "
        "cohere and google-genai, so a venv is advisable)."
    )

SUITES = ("banking", "slack", "travel", "workspace")

# Argument names that carry action authority rather than free text. Kept broad
# on purpose: a narrower list would flatter the result by excluding the
# document-sourced identifiers that are the whole point.
CRITICAL = re.compile(
    r"(recipient|iban|account|address|email|user|channel|id$|_id|to$|from$|"
    r"destination|hotel|restaurant|car|flight|company|name)",
    re.I,
)


def measure() -> dict:
    per_suite: dict[str, Counter] = {}
    examples: list[dict] = []
    for suite_name in SUITES:
        suite = get_suite("v1", suite_name)
        env = suite.load_and_inject_default_environment({})
        counts: Counter = Counter()
        for task_id, task in suite.user_tasks.items():
            prompt = task.PROMPT
            try:
                calls = task.ground_truth(env)
            except Exception:  # a few tasks need a mutated environment
                counts["ground_truth_unavailable"] += 1
                continue
            for call in calls:
                for arg, value in (call.args or {}).items():
                    if not isinstance(value, str) or not value.strip():
                        continue
                    if not CRITICAL.search(arg):
                        continue
                    counts["critical_string_args"] += 1
                    if value.lower() in prompt.lower():
                        counts["in_user_turn"] += 1
                    else:
                        counts["not_in_user_turn"] += 1
                        if len(examples) < 10:
                            examples.append({
                                "suite": suite_name, "task": task_id,
                                "function": call.function, "argument": arg,
                                "value": value[:48], "prompt": prompt[:110],
                            })
        per_suite[suite_name] = counts

    totals: Counter = Counter()
    for counts in per_suite.values():
        totals.update(counts)
    return {
        "benchmark": "AgentDojo",
        "source": "https://github.com/ethz-spylab/agentdojo (MIT)",
        "per_suite": {name: dict(counts) for name, counts in per_suite.items()},
        "totals": dict(totals),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = measure()
    per_suite, totals = report["per_suite"], report["totals"]

    print(f"{'suite':12s} {'critical args':>14s} {'in user turn':>13s} {'not in turn':>12s}")
    for name, counts in per_suite.items():
        total = counts.get("critical_string_args", 0)
        share = counts.get("in_user_turn", 0) / total if total else 0
        print(f"{name:12s} {total:>14d} {counts.get('in_user_turn', 0):>13d} "
              f"{counts.get('not_in_user_turn', 0):>12d}   ({share:.0%} re-derivable)")

    total = totals.get("critical_string_args", 0)
    share = totals.get("in_user_turn", 0) / total if total else 0
    print(f"\n{'ALL':12s} {total:>14d} {totals.get('in_user_turn', 0):>13d} "
          f"{totals.get('not_in_user_turn', 0):>12d}   "
          f"({share:.0%} re-derivable from the user's turn)")

    print("\nAuthorised values the user never wrote, so nothing to re-derive from:")
    for row in report["examples"][:6]:
        print(f"  [{row['suite']}/{row['task']}] "
              f"{row['function']}({row['argument']}={row['value']!r})")
        print(f"      user said: {row['prompt']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

"""Cross-suite table for the AgentDojo runs, beside the re-derivability scope.

Both halves have to be read together. Attack success says whether the
*attacker's* value reached a tool; task completion says whether the *user's* did.
A defence that withholds everything wins the first column outright and empties
the second, so neither number means anything alone.

The scope column is the predictor: it is the share of that suite's critical
argument values the user actually wrote, measured by `scope.py` with no model
involved. Where it is high the boundary is close to free; where it is low the
boundary can only withhold.

    python bench/agentdojo/summarize.py --markdown
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

SUITES = ("banking", "workspace", "travel", "slack")

# from scope.py: share of critical argument values present in the user's turn
SCOPE = {"banking": 75, "workspace": 50, "travel": 50, "slack": 27}


def _load(suite: str, model: str, clean: bool) -> dict | None:
    name = f"agentdojo-{suite}{'-clean' if clean else ''}-{model}.json"
    path = RESULTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _row(suite: str, model: str) -> dict | None:
    injected = _load(suite, model, clean=False)
    if injected is None:
        return None
    arms = injected["arms"]
    if "evibind" not in arms:
        return None
    base, guard = arms["baseline"], arms["evibind"]
    clean = _load(suite, model, clean=True)
    clean_arms = (clean or {}).get("arms", {})
    stats = guard.get("guard_stats") or {}
    return {
        "suite": suite,
        "scope": SCOPE.get(suite),
        "cases": base["cases"],
        "attack_base": base.get("attack_succeeded"),
        "attack_guard": guard.get("attack_succeeded"),
        "util_base": base["utility_passed"],
        "util_guard": guard["utility_passed"],
        "clean_base": clean_arms.get("baseline", {}).get("utility_passed"),
        "clean_guard": clean_arms.get("evibind", {}).get("utility_passed"),
        "clean_cases": clean_arms.get("baseline", {}).get("cases"),
        "withheld": stats.get("withheld"),
        "seen": stats.get("calls_seen"),
        "usd": (injected.get("usage") or {}).get("usd"),
        # set by hand in a result file that predates the adapter fixes in
        # FINDINGS section 21. A stale row is marked in the table rather than
        # dropped: leaving it out would make the suite look narrower than it
        # is, and leaving it unmarked would mix two different guards in one
        # table.
        "stale": bool(injected.get("pre_annotation_fix")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    rows = [row for row in (_row(s, args.model) for s in SUITES) if row]
    if not rows:
        raise SystemExit(f"no results for {args.model} in {RESULTS}")

    header = ["suite", "re-derivable", "attack base -> guarded",
              "completed base -> guarded", "clean base -> guarded"]

    def cells(row: dict) -> list[str]:
        clean = ("-" if row["clean_base"] is None else
                 f"{row['clean_base']} -> {row['clean_guard']}"
                 f" of {row['clean_cases']}")
        return [row["suite"] + (" *" if row["stale"] else ""),
                f"{row['scope']}%",
                f"{row['attack_base']} -> {row['attack_guard']}"
                f" of {row['cases']}",
                f"{row['util_base']} -> {row['util_guard']}",
                clean]

    if args.markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            print("| " + " | ".join(cells(row)) + " |")
    else:
        table = [cells(row) for row in rows]
        widths = [max(len(str(x)) for x in [header[i]] + [r[i] for r in table])
                  for i in range(len(header))]
        line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
        print(line)
        print("-" * len(line))
        for row in table:
            print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))

    if any(r["stale"] for r in rows):
        print("\n\\* pre-annotation-fix run (FINDINGS section 21): the guard still "
              "governed read-only\n  tools and matched parameters by description, "
              "so this row understates utility.")

    withheld = sum(r["withheld"] or 0 for r in rows)
    seen = sum(r["seen"] or 0 for r in rows)
    spend = sum(r["usd"] or 0 for r in rows)
    print(f"\ncalls withheld across the injected runs: {withheld} of {seen}")
    print(f"measured spend on the injected runs: ${spend:.2f}")


if __name__ == "__main__":
    main()

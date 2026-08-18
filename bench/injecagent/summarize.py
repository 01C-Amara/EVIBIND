"""Cross-model table for the InjecAgent runs, beside the InjectBench numbers.

The point of the pairing is that both columns describe the *same* model with no
gateway in the way. One column is a tool-selection attack, the other an
argument-substitution attack, and they do not agree.

    python bench/injecagent/summarize.py --split dh_enhanced --markdown
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

ORDER = [
    ("gpt-5.6-sol", "GPT-5.6 Sol"),
    ("gpt-5.6-luna", "GPT-5.6 Luna"),
    ("gpt-5.6-terra", "GPT-5.6 Terra"),
    ("grok-4.6", "Grok 4.6"),
    ("grok-4.5", "Grok 4.5"),
    ("claude-haiku", "Claude Haiku"),
    ("gpt-5.4-mini", "GPT-5.4 mini"),
    ("gpt-5.4-nano", "GPT-5.4 nano"),
    ("gpt-4.1-mini", "GPT-4.1 mini"),
]


def _injecagent(model: str, split: str) -> dict | None:
    path = RESULTS / f"injecagent-{split}-{model}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _injectbench(model: str) -> tuple[int, int] | None:
    path = RESULTS / f"{model}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    origin = data.get("origin_violation", {})
    if "native_slot" not in origin:
        return None
    return origin["native_slot"]["harmful"], origin["n"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dh_enhanced")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    rows = []
    for model, pretty in ORDER:
        external = _injecagent(model, args.split)
        internal = _injectbench(model)
        if external is None and internal is None:
            continue
        if external:
            block = external["security"]
            selection = f"{block['native_attack']}/{block['scored']}"
            guarded = f"{block['guarded_attack']}/{block['scored']}"
            transport = external.get("transport", "native tool calling")
        else:
            selection = guarded = "not run"
            transport = "-"
        substitution = f"{internal[0]}/{internal[1]}" if internal else "not run"
        rows.append([pretty, selection, guarded, substitution,
                     "CLI, emulated" if "CLI" in transport else "native"])

    header = ["model", f"InjecAgent {args.split} native", "released guarded",
              "InjectBench origin native", "tool calling"]
    if args.markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            print("| " + " | ".join(row) + " |")
        return

    widths = [max(len(str(x)) for x in [header[i]] + [r[i] for r in rows])
              for i in range(len(header))]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


if __name__ == "__main__":
    main()

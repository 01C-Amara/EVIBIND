"""Cross-model table over every result JSON in ``bench/results``.

Reports the critical-slot metric: how often the slot the confinement claim is
about was bound to the attacker's value (``harmful``) or to the value the user
actually authorised (``correct``), natively and behind the gateway.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

ORDER = [
    "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6-sol",
    "gpt-5.4-mini", "gpt-5.4-nano", "gpt-4.1-mini",
    "grok-4.6", "grok-4.5", "claude-haiku",
    "mock-last-mention", "scripted-compromised", "transcription-drift",
]


def load() -> list[dict]:
    out = []
    for path in sorted(RESULTS.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "native_slot" not in data:
            continue  # scored before the slot metric existed
        out.append(data)
    key = {name: i for i, name in enumerate(ORDER)}
    out.sort(key=lambda d: (key.get(d["label"], 99), d["label"]))
    return out


def rows(data: dict) -> dict[str, int]:
    origin, sel = data["origin_violation"], data["selection_error"]
    return {
        "n": data["cases"],
        "o_harm_n": origin["native_slot"]["harmful"],
        "o_harm_g": origin["guarded_slot"]["harmful"],
        "o_ok_n": origin["native_slot"]["correct"],
        "o_ok_g": origin["guarded_slot"]["correct"],
        "s_harm_n": sel["native_slot"]["harmful"],
        "s_harm_g": sel["guarded_slot"]["harmful"],
        "s_ok_n": sel["native_slot"]["correct"],
        "s_ok_g": sel["guarded_slot"]["correct"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    data = load()
    header = ["model", "origin: harmful native", "origin: harmful guarded",
              "origin: correct native", "origin: correct guarded",
              "selection: harmful native", "selection: harmful guarded",
              "selection: correct native", "selection: correct guarded"]
    table = []
    for entry in data:
        r = rows(entry)
        table.append([entry["label"],
                      f"{r['o_harm_n']}/60", f"{r['o_harm_g']}/60",
                      f"{r['o_ok_n']}/60", f"{r['o_ok_g']}/60",
                      f"{r['s_harm_n']}/90", f"{r['s_harm_g']}/90",
                      f"{r['s_ok_n']}/90", f"{r['s_ok_g']}/90"])

    if args.markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] * len(header)) + "|")
        for row in table:
            print("| " + " | ".join(row) + " |")
        return

    widths = [max(len(str(x)) for x in [header[i]] + [r[i] for r in table])
              for i in range(len(header))]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for row in table:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


if __name__ == "__main__":
    main()

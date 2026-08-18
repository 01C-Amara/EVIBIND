"""Score a subagent-driven InjecAgent run from its batch answer files.

Pairs with ``prepare_batches.py``. Each answer line is turned into the
chat-completion shape the ordinary scorer expects, so the security arm is
computed by exactly the same code as the API runs.

    python bench/injecagent/score_batches.py --answers DIR --label claude-haiku-agent
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from adapt import load_split, scope_of  # noqa: E402
from run_injecagent import _config, security_row  # noqa: E402


def to_chat_completion(answer: dict[str, Any]) -> dict[str, Any]:
    if not answer.get("call_tool") or not answer.get("tool_name"):
        return {"choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": "answered without a tool"}}]}
    arguments = answer.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    return {"choices": [{"index": 0, "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": None,
                                     "tool_calls": [{
                                         "id": "agent-0", "type": "function",
                                         "function": {"name": answer["tool_name"],
                                                      "arguments": json.dumps(arguments)}}]}}]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--split", default="dh_enhanced")
    parser.add_argument("--label", default="claude-haiku-agent")
    parser.add_argument("--transport", default="Claude Code subagent (agent harness)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cases = {c["case_id"]: c for c in load_split(args.split)
             if not c.get("self_referential")}
    answers: dict[str, dict[str, Any]] = {}
    malformed = 0
    for path in sorted(Path(args.answers).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            case_id = record.get("case_id")
            if case_id in cases:
                answers[case_id] = record
            else:
                malformed += 1

    config = _config()
    rows = [security_row(cases[cid], to_chat_completion(answer), config)
            for cid, answer in answers.items()]
    scoped = [r for r in rows if r["in_scope"]]
    summary = {
        "benchmark": "InjecAgent",
        "source": "https://github.com/uiuc-kang-lab/InjecAgent (MIT)",
        "split": args.split,
        "model": args.label,
        "transport": args.transport,
        "answered": len(answers),
        "expected": len(cases),
        "malformed_lines": malformed,
        "scope": scope_of(list(cases.values())),
        "utility": {},
        "security": {
            "scored": len(rows),
            "native_attack": sum(r["native_attack"] for r in rows),
            "guarded_attack": sum(r["guarded_attack"] for r in rows),
            "in_scope": {"n": len(scoped),
                         "native_attack": sum(r["native_attack"] for r in scoped),
                         "guarded_attack": sum(r["guarded_attack"] for r in scoped)},
            "out_of_scope": {"n": len(rows) - len(scoped),
                             "native_attack": sum(r["native_attack"] for r in rows
                                                  if not r["in_scope"]),
                             "guarded_attack": sum(r["guarded_attack"] for r in rows
                                                   if not r["in_scope"])},
        },
        "rows": rows,
    }
    out = Path(args.out or
               f"bench/results/injecagent-{args.split}-{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    block = summary["security"]
    print(f"InjecAgent {args.split}  {args.label}  ({args.transport})")
    print(f"  answered  : {len(answers)}/{len(cases)}"
          f"{f', {malformed} malformed line(s)' if malformed else ''}")
    print(f"  attacker tool called natively : {block['native_attack']}/{block['scored']}")
    print(f"  attacker tool released guarded: {block['guarded_attack']}/{block['scored']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

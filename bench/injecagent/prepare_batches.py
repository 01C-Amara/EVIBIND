"""Write InjecAgent cases out as batches for a subagent-driven run.

Some models are reachable only as an agent, not as a raw completions endpoint —
Claude Haiku here, via Claude Code subagents. This writes the model-visible
cases to batch files so each subagent reads one file and answers the cases in
it, rather than inlining a very large prompt.

The transport caveat matters and is recorded with the results: a subagent has
its own system prompt and its own tools, so this is an *agent* answering, not a
bare model handed a tool schema. That is arguably closer to InjecAgent's
original ReAct harness than native function calling is, but it is not the same
thing as the API rows and is labelled separately.

    python bench/injecagent/prepare_batches.py --split dh_enhanced --size 30 \\
        --out-dir /tmp/haiku-batches
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from adapt import load_split, model_visible_request  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dh_enhanced")
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cases = [c for c in load_split(args.split) if not c.get("self_referential")]
    if args.limit:
        cases = cases[:args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = 0
    for start in range(0, len(cases), args.size):
        chunk = cases[start:start + args.size]
        payload = []
        for case in chunk:
            visible = model_visible_request(case)
            payload.append({
                "case_id": case["case_id"],
                "tools": visible["tools"],
                "messages": visible["messages"],
            })
        path = out_dir / f"batch-{batches:03d}.json"
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        batches += 1

    print(f"wrote {batches} batches of up to {args.size} cases "
          f"({len(cases)} total) into {out_dir}")


if __name__ == "__main__":
    main()

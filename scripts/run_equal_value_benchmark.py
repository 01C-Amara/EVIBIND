from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tapbench.equal_value_benchmark import (  # noqa: E402
    build_equal_value_pairs,
    evaluate_equal_value_pairs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-pattern", type=int, default=50)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pairs = build_equal_value_pairs(args.per_pattern)
    with (output / "equal_value_pairs.jsonl").open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(
                json.dumps(pair, sort_keys=True, separators=(",", ":")) + "\n"
            )
    report = evaluate_equal_value_pairs(pairs)
    (output / "equal_value_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

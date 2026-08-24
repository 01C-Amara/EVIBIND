from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tapbench.adversarial_boundary import (  # noqa: E402
    build_effect_scenarios,
    run_executable_effects,
    run_separation_suite,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-kind", type=int, default=10)
    parser.add_argument("--separation-repetitions", type=int, default=20)
    args = parser.parse_args()
    report = {
        "executable_effects": run_executable_effects(
            build_effect_scenarios(args.per_kind)
        ),
        "separation": run_separation_suite(args.separation_repetitions),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

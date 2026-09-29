"""Archive AgentDojo per-case traces next to the results they back.

AgentDojo writes traces under the system temp directory, which is not durable:
on this project it was cleared once, taking a working environment with it.
The paired statistics and the residual classification are computed from these
traces, so each post-fix arm is archived as a zip whose member paths match the
frozen replication's archive (``<pipeline>/<suite>/<user_task>/...``).

    python bench/agentdojo/archive_traces.py --pipeline gpt-4o-2024-05-13__actual-gpt-5.4-nano-evibind \\
        --suites banking --out bench/results/agentdojo-banking-gpt-5.4-nano-literal-fix-rerun-traces.zip
"""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default=str(Path(tempfile.gettempdir()) / "agentdojo-runs"))
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--suites", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.runs)
    count = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as archive:
        for suite in args.suites:
            for path in sorted((root / args.pipeline / suite).rglob("*.json")):
                archive.write(path, path.relative_to(root).as_posix())
                count += 1
    size = Path(args.out).stat().st_size
    print(f"archived {count} traces into {args.out} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

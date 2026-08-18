"""Fetch the InjecAgent dataset.

The data is not vendored into this repo. It belongs to its authors, it is 2.9 MB
of JSON, and pinning a copy here would go stale silently. This pulls it into
``bench/injecagent/data/``, which is gitignored.

    python bench/injecagent/fetch.py

InjecAgent: "InjecAgent: Benchmarking Indirect Prompt Injections in Tool-
Integrated Large Language Model Agents", Zhan et al., MIT licence.
https://github.com/uiuc-kang-lab/InjecAgent
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data"
DATA = Path(__file__).resolve().parent / "data"
FILES = (
    "tools.json",
    "test_cases_dh_base.json",
    "test_cases_dh_enhanced.json",
    "test_cases_ds_base.json",
    "test_cases_ds_enhanced.json",
)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = DATA / name
        if target.exists():
            print(f"have  {name} ({target.stat().st_size:,} bytes)")
            continue
        with urllib.request.urlopen(f"{BASE}/{name}", timeout=120) as response:
            payload = response.read()
        target.write_bytes(payload)
        print(f"got   {name} ({len(payload):,} bytes)")
    print(f"\ninto {DATA}")


if __name__ == "__main__":
    main()

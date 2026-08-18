"""Skip test modules that depend on paper-artifact code this repo does not ship.

The product repo ships the gateway and its engine. The paper's analysis and
figure-generation scripts (the ``scripts`` package) are not part of the
distributable, but sixteen test modules import them at module scope. Without
this hook they fail during collection, so ``pytest tests -q`` — the command CI
runs — reports errors that say nothing about the shipped code.

If ``scripts`` is importable (a full research checkout), nothing is skipped.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

collect_ignore: list[str] = []

if importlib.util.find_spec("scripts") is None:
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        source = path.read_text(encoding="utf-8", errors="replace")
        if "from scripts" in source or "import scripts" in source:
            collect_ignore.append(path.name)

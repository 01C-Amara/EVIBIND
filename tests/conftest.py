"""Skip only tests whose separately released research scripts are absent.

The source repository includes deterministic mechanism scripts. Some historical
analysis tests target additional paper-bundle modules that are intentionally not
part of the product checkout. Discover those imports per test module so adding
one public script cannot accidentally unskip every unavailable paper module.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

def _script_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "scripts":
                modules.update(f"scripts.{alias.name}" for alias in node.names)
            elif node.module.startswith("scripts."):
                modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(
                alias.name for alias in node.names if alias.name.startswith("scripts.")
            )
    return modules


collect_ignore: list[str] = []
for path in sorted(Path(__file__).parent.glob("test_*.py")):
    required = _script_imports(path)
    if any(importlib.util.find_spec(module) is None for module in required):
        collect_ignore.append(path.name)

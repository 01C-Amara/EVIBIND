"""Splice the current cross-suite table into `README.md`.

The table in that README is generated, not typed, because it has been wrong
twice: once when the banking numbers were quoted after the annotation fix
changed them, and once when a banking-only result was written up as though it
described the benchmark. Anything between the marker comments is replaced from
whatever is in `bench/results/` right now.

    python bench/agentdojo/refresh_readme.py
"""

from __future__ import annotations

import io
import contextlib
from pathlib import Path

import summarize  # noqa: E402  (same directory)

START = "<!-- generated: summarize.py -->"
END = "<!-- /generated -->"

README = Path(__file__).resolve().parent / "README.md"


def _table() -> str:
    buffer = io.StringIO()
    import sys
    argv = sys.argv
    sys.argv = ["summarize.py", "--markdown"]
    try:
        with contextlib.redirect_stdout(buffer):
            summarize.main()
    finally:
        sys.argv = argv
    return buffer.getvalue().strip()


def main() -> None:
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"marker comments missing from {README}")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    body = _table()
    README.write_text(f"{head}{START}\n\n{body}\n\n{END}{tail}",
                      encoding="utf-8", newline="\n")
    print(f"refreshed the table in {README}")


if __name__ == "__main__":
    main()

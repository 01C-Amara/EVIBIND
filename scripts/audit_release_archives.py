#!/usr/bin/env python3
"""Reject generated, credential-like, or machine-local release content."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path


FORBIDDEN_CONTENT = tuple(
    b"/" + segment + b"/" for segment in (b"home", b"Users", b"projects")
)
GENERATED_PAPER_SUFFIXES = (
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".pdf",
)
FORBIDDEN_MEMBER_MARKERS = (
    "/.env",
    "/private/",
    "/participants.jsonl",
    "human_study_handoff",
)
SECRET_PATTERNS = (
    ("private_key", re.compile(br"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("provider_key", re.compile(br"\bsk-[A-Za-z0-9_-]{20,}\b")),
    (
        "assigned_secret",
        re.compile(
            br"\b(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*"
            br"[\"']?[A-Za-z0-9_./+=-]{24,}",
            re.IGNORECASE,
        ),
    ),
)
PROXY_HUMAN_MARKER = b"handoff_human_" + b"proxy: true"


def _members(path: Path) -> Iterator[tuple[str, bytes]]:
    if path.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    handle = archive.extractfile(member)
                    if handle is not None:
                        yield member.name, handle.read()
        return
    if path.suffix in {".whl", ".zip"}:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith("/"):
                    yield name, archive.read(name)
        return
    raise ValueError(f"unsupported release archive: {path}")


def audit_archives(paths: list[Path]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in paths:
        for name, content in _members(path):
            normalized_name = f"/{name.replace(chr(92), '/')}"
            generated_paper = "/paper/" in normalized_name and name.endswith(
                GENERATED_PAPER_SUFFIXES
            )
            if (
                "/work/" in normalized_name
                or "__pycache__" in name
                or name.endswith(".pyc")
                or generated_paper
            ):
                findings.append(
                    {"archive": str(path), "member": name, "finding": "generated_path"}
                )
            for marker in FORBIDDEN_MEMBER_MARKERS:
                if (
                    marker == "/.env"
                    and normalized_name.lower().endswith("/.env.example")
                ):
                    continue
                if marker.lower() in normalized_name.lower():
                    findings.append(
                        {
                            "archive": str(path),
                            "member": name,
                            "finding": f"forbidden_member:{marker}",
                        }
                    )
            for marker in FORBIDDEN_CONTENT:
                if marker in content:
                    findings.append(
                        {
                            "archive": str(path),
                            "member": name,
                            "finding": f"machine_local_path:{marker.decode()}",
                        }
                    )
            if PROXY_HUMAN_MARKER in content:
                findings.append(
                    {
                        "archive": str(path),
                        "member": name,
                        "finding": "proxy_human_data",
                    }
                )
            for label, pattern in SECRET_PATTERNS:
                if pattern.search(content):
                    findings.append(
                        {
                            "archive": str(path),
                            "member": name,
                            "finding": f"secret_material:{label}",
                        }
                    )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    findings = audit_archives(args.archives)
    print(json.dumps({"archives": len(args.archives), "findings": findings}, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())

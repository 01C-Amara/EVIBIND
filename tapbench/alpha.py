from __future__ import annotations

from collections import defaultdict
from typing import Any


TOKENIZER_FAMILIES = ("qwen", "liquid", "gemma")


def proxy_tokenize_name(name: str, tokenizer_family: str) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    for char in name:
        boundary = char in {"_", "-", "."} or char.isdigit()
        if boundary:
            if current:
                pieces.append("".join(current))
                current = []
            if char.isdigit():
                pieces.append(char)
        else:
            current.append(char)
    if current:
        pieces.append("".join(current))
    if tokenizer_family == "qwen":
        return [piece for chunk in pieces for piece in (chunk[:4], chunk[4:]) if piece]
    if tokenizer_family == "liquid":
        return [piece for chunk in pieces for piece in (chunk[:3], chunk[3:]) if piece]
    return pieces


def schema_names(case: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for tool in case.get("tools", []):
        names.append(str(tool.get("name", "")))
        stack = [tool.get("parameters", {})]
        while stack:
            current = stack.pop()
            if not isinstance(current, dict):
                continue
            properties = current.get("properties", {})
            if isinstance(properties, dict):
                names.extend(str(name) for name in properties)
                stack.extend(value for value in properties.values() if isinstance(value, dict))
    return [name for name in names if name]


def fragmentation_stats(cases: list[dict[str, Any]], tokenizer_families: tuple[str, ...] = TOKENIZER_FAMILIES) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for case in cases:
        alpha = str(case.get("factors", {}).get("alpha", "unknown"))
        names = schema_names(case)
        for family in tokenizer_families:
            counts = [len(proxy_tokenize_name(name, family)) for name in names]
            mean_fragments = sum(counts) / len(counts) if counts else 0.0
            grouped[(alpha, family)].append(mean_fragments)
    for (alpha, family), values in sorted(grouped.items()):
        rows.append(
            {
                "alpha": alpha,
                "tokenizer_family": family,
                "case_count": len(values),
                "mean_name_fragments": sum(values) / len(values),
            }
        )
    by_alpha: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_alpha[row["alpha"]].append(row["mean_name_fragments"])
    return {
        "schema_version": "tapbench.alpha_proxy.v1",
        "primary_proxy": "tokenizer_fragmentation_stats",
        "rows": rows,
        "mean_by_alpha": {alpha: sum(values) / len(values) for alpha, values in sorted(by_alpha.items())},
    }

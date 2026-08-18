from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    compact = "".join(ch for ch in text.lower() if ch.isalnum())
    if len(compact) <= n:
        return [compact] if compact else []
    return [compact[i : i + n] for i in range(len(compact) - n + 1)]


def _vector(text: str, *, arm: str) -> Counter[str]:
    if arm == "cheap_embedding":
        return Counter(_tokens(text) + _char_ngrams(text, 4))
    return Counter(_tokens(text) + _char_ngrams(text, 3))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def rank_tools(case: dict[str, Any], *, arm: str = "tfidf_char") -> list[dict[str, Any]]:
    if arm == "none":
        return list(case.get("tools", []))
    query = " ".join(message.get("content", "") for message in case.get("messages", []) if message.get("role") == "user")
    query_vec = _vector(query, arm=arm)
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for index, tool in enumerate(case.get("tools", [])):
        text = f"{tool.get('name', '')} {tool.get('description', '')}"
        ranked.append((_cosine(query_vec, _vector(text, arm=arm)), -index, tool))
    ranked.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return [tool for _, _, tool in ranked]


def recall_at_k(case: dict[str, Any], *, k: int = 8, arm: str = "tfidf_char") -> bool:
    if case.get("gold_action", {}).get("mode") != "call":
        return True
    target = case.get("gold_action", {}).get("tool")
    top = rank_tools(case, arm=arm)[:k]
    return any(tool.get("canonical_name") == target for tool in top)


def evaluate_retrieval(cases: list[dict[str, Any]], *, k: int = 8, arm: str = "tfidf_char") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        start = time.perf_counter()
        recalled = recall_at_k(case, k=k, arm=arm)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        rows.append(
            {
                "case_id": case["case_id"],
                "hypothesis_grid_id": case["hypothesis_grid_id"],
                "family": case["family"],
                "sigma": case.get("factors", {}).get("sigma"),
                "alpha": case.get("factors", {}).get("alpha"),
                "N": case.get("factors", {}).get("N"),
                "retriever": arm,
                "k": k,
                "recall_at_k": bool(recalled),
                "retriever_latency_ms": elapsed_ms,
            }
        )
    return rows


def recall_summary(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["recall_at_k"]) / len(rows)

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from typing import Any

from .eflrx import _tool_name
from .extractive_candidates import user_request_text
from .multilingual_retriever import (
    RETRIEVER_MODEL_ID,
    RETRIEVER_REVISION,
    forbidden_paths,
)
from .retrieve_pointer import validate_ranking_row
from .source_span_projection import slot_catalog, source_span_catalog


SPAN_SHORTLIST_VERSION = "tapbench.multilingual_e5_span_shortlist.v1"
SPAN_SHORTLIST_SCHEMA_VERSION = "tapbench.span_shortlist.v1"
SPAN_SHORTLIST_ARMS = (
    "slot_request_top16_v1",
    "slot_request_top32_v1",
    "tool_slot_request_top16_v1",
    "tool_slot_request_top32_v1",
)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _query_text(
    *,
    mode: str,
    request: str,
    tool: dict[str, Any],
    slot: dict[str, Any],
) -> str:
    slot_text = (
        f"Argument {slot['name']}: {slot['description']}. "
        f"Request: {request}"
    )
    if mode == "slot_request":
        return "query: " + slot_text
    if mode == "tool_slot_request":
        return (
            "query: Function "
            + _tool_name(tool)
            + ": "
            + str(tool.get("description", ""))
            + ". "
            + slot_text
        )
    raise ValueError(f"unknown span-shortlist query mode: {mode}")


def _passage_text(source_text: str) -> str:
    return "passage: Candidate value: " + source_text


def _batched_encode(
    encoder: Any,
    texts: list[str],
    *,
    batch_size: int,
) -> tuple[Any, list[float], float]:
    import numpy as np

    rows: list[Any] = []
    amortized: list[float] = []
    total = 0.0
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        began = time.perf_counter()
        encoded = encoder.encode(
            batch,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - began
        if len(encoded) != len(batch):
            raise ValueError("span retriever embedding count mismatch")
        rows.extend(encoded)
        amortized.extend([elapsed / len(batch)] * len(batch))
        total += elapsed
    values = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if len(values) and np.any(norms == 0):
        raise ValueError("span retriever produced a zero-norm embedding")
    return values / norms if len(values) else values, amortized, total


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def rank_span_shortlists(
    cases: list[dict[str, Any]],
    tool_ranking_rows: list[dict[str, Any]],
    encoder: Any,
    *,
    query_mode: str,
    top_ks: tuple[int, ...] = (16, 32),
    batch_size: int = 64,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    if not cases:
        raise ValueError("at least one runtime case is required")
    if query_mode not in {"slot_request", "tool_slot_request"}:
        raise ValueError(f"unknown query mode: {query_mode}")
    if not top_ks or any(value <= 0 for value in top_ks):
        raise ValueError("top_ks must contain positive values")
    rankings = {str(row.get("case_id")): row for row in tool_ranking_rows}
    if len(rankings) != len(tool_ranking_rows):
        raise ValueError("tool ranking artifact contains duplicate case IDs")

    case_data: list[dict[str, Any]] = []
    flat_passages: list[str] = []
    for case in cases:
        leaks = forbidden_paths(case)
        if leaks:
            raise ValueError(
                f"runtime case {case.get('case_id')} contains forbidden fields: {leaks}"
            )
        case_id = str(case["case_id"])
        text = user_request_text(case.get("messages", []))
        tools = list(case.get("tools", []))
        ranking_row = rankings.get(case_id)
        if ranking_row is None:
            raise ValueError(f"missing fixed tool ranking for {case_id}")
        fixed_ranking = validate_ranking_row(
            ranking_row,
            case_id=case_id,
            request_text=text,
            tools=tools,
        )
        language = str(case.get("metadata", {}).get("language"))
        lattice = source_span_catalog(text, language)
        unique_by_text: dict[str, dict[str, Any]] = {}
        for span in lattice["spans"]:
            unique_by_text.setdefault(str(span["source_text"]), span)
        unique_spans = sorted(
            unique_by_text.values(), key=lambda row: str(row["span_id"])
        )
        passage_start = len(flat_passages)
        flat_passages.extend(
            _passage_text(str(span["source_text"])) for span in unique_spans
        )
        by_name = {_tool_name(tool): tool for tool in tools}
        retrieved = [
            by_name[str(row["tool"])]
            for row in fixed_ranking
            if str(row["tool"]) in by_name
        ]
        case_data.append(
            {
                "case": case,
                "case_id": case_id,
                "language": language,
                "request": text,
                "tools": tools,
                "retrieved": retrieved,
                "tool_ranking": fixed_ranking,
                "tool_ranking_sha256": ranking_row["ranking_sha256"],
                "tool_ranking_artifact_sha256": None,
                "lattice": lattice,
                "unique_spans": unique_spans,
                "passage_start": passage_start,
                "passage_end": len(flat_passages),
            }
        )

    passage_embeddings, passage_latencies, passage_seconds = _batched_encode(
        encoder,
        flat_passages,
        batch_size=batch_size,
    )
    query_jobs: list[dict[str, Any]] = []
    query_texts: list[str] = []
    for case_index, data in enumerate(case_data):
        rank_by_name = {
            str(row["tool"]): int(row["rank"])
            for row in data["tool_ranking"]
        }
        for tool in data["retrieved"]:
            slots, _ = slot_catalog(tool)
            for slot in slots:
                query_jobs.append(
                    {
                        "case_index": case_index,
                        "tool": tool,
                        "tool_rank": rank_by_name[_tool_name(tool)],
                        "slot": slot,
                    }
                )
                query_texts.append(
                    _query_text(
                        mode=query_mode,
                        request=data["request"],
                        tool=tool,
                        slot=slot,
                    )
                )
    query_embeddings, query_latencies, query_seconds = _batched_encode(
        encoder,
        query_texts,
        batch_size=batch_size,
    )

    ranked_by_job: list[list[dict[str, Any]]] = []
    for job, query_embedding in zip(query_jobs, query_embeddings):
        data = case_data[int(job["case_index"])]
        start = int(data["passage_start"])
        end = int(data["passage_end"])
        scores = passage_embeddings[start:end] @ query_embedding
        spans = data["unique_spans"]
        order = sorted(
            range(len(spans)),
            key=lambda index: (
                -float(scores[index]),
                int(
                    str(spans[index]["end_unit_id"]).split("_")[-1]
                )
                - int(str(spans[index]["start_unit_id"]).split("_")[-1])
                + 1,
                str(spans[index]["span_id"]),
            ),
        )
        ranked_by_job.append(
            [
                {
                    "rank": rank,
                    "span_id": spans[index]["span_id"],
                    "source_text": spans[index]["source_text"],
                    "cosine_score": float(scores[index]),
                    "unit_length": int(
                        str(spans[index]["end_unit_id"]).split("_")[-1]
                    )
                    - int(
                        str(spans[index]["start_unit_id"]).split("_")[-1]
                    )
                    + 1,
                }
                for rank, index in enumerate(order, start=1)
            ]
        )

    output_by_k: dict[int, list[dict[str, Any]]] = {}
    for top_k in sorted(set(top_ks)):
        arm = f"{query_mode}_top{top_k}_v1"
        if arm not in SPAN_SHORTLIST_ARMS:
            raise ValueError(f"undeclared span shortlist arm: {arm}")
        by_case_tool: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
        for job, candidates in zip(query_jobs, ranked_by_job):
            case_index = int(job["case_index"])
            tool = job["tool"]
            name = _tool_name(tool)
            tool_row = by_case_tool[case_index].setdefault(
                name,
                {
                    "tool": name,
                    "tool_rank": int(job["tool_rank"]),
                    "slots": [],
                },
            )
            slot = job["slot"]
            tool_row["slots"].append(
                {
                    "slot_id": slot["slot_id"],
                    "surface_name": slot["name"],
                    "description": slot["description"],
                    "required": slot["required"],
                    "candidates": candidates[:top_k],
                }
            )
        rows: list[dict[str, Any]] = []
        for case_index, data in enumerate(case_data):
            tools = sorted(
                by_case_tool[case_index].values(),
                key=lambda row: int(row["tool_rank"]),
            )
            for tool in tools:
                tool["slots"].sort(key=lambda row: str(row["slot_id"]))
            row = {
                "schema_version": SPAN_SHORTLIST_SCHEMA_VERSION,
                "case_id": data["case_id"],
                "language": data["language"],
                "span_shortlist_version": SPAN_SHORTLIST_VERSION,
                "retriever_model_id": RETRIEVER_MODEL_ID,
                "retriever_revision": RETRIEVER_REVISION,
                "shortlist_arm": arm,
                "top_k_per_slot": top_k,
                "query_mode": query_mode,
                "request_sha256": data["lattice"]["request_sha256"],
                "span_catalog_sha256": data["lattice"]["catalog_sha256"],
                "tool_ranking_sha256": data["tool_ranking_sha256"],
                "unique_source_span_count": len(data["unique_spans"]),
                "tools": tools,
            }
            row["shortlist_sha256"] = _json_sha256(row)
            rows.append(row)
        output_by_k[top_k] = rows

    telemetry = {
        "schema_version": "tapbench.multilingual_e5_span_telemetry.v1",
        "span_shortlist_version": SPAN_SHORTLIST_VERSION,
        "retriever_model_id": RETRIEVER_MODEL_ID,
        "retriever_revision": RETRIEVER_REVISION,
        "query_mode": query_mode,
        "top_ks": sorted(set(top_ks)),
        "case_count": len(cases),
        "slot_query_count": len(query_jobs),
        "unique_span_passage_count": len(flat_passages),
        "passage_embedding_seconds": passage_seconds,
        "query_embedding_seconds": query_seconds,
        "p50_amortized_passage_seconds": _percentile(passage_latencies, 0.50),
        "p95_amortized_passage_seconds": _percentile(passage_latencies, 0.95),
        "p50_amortized_query_seconds": _percentile(query_latencies, 0.50),
        "p95_amortized_query_seconds": _percentile(query_latencies, 0.95),
    }
    return output_by_k, telemetry

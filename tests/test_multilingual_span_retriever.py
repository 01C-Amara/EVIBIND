from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from scripts.score_massive_span_shortlists import score_shortlists
from tapbench.io import write_jsonl
from tapbench.multilingual_retriever import (
    MULTILINGUAL_RETRIEVER_VERSION,
    RETRIEVAL_RANKING_SCHEMA_VERSION,
    RETRIEVER_MODEL_ID,
    RETRIEVER_REVISION,
    catalog_sha256,
    forbidden_paths,
    ranking_sha256,
)
from tapbench.multilingual_span_retriever import rank_span_shortlists


def _tool() -> dict:
    return {
        "name": "email.send",
        "canonical_name": "email.send",
        "description": "Send an email.",
        "parameters": {
            "type": "object",
            "properties": {
                "person": {"type": "string", "description": "recipient person"}
            },
            "required": [],
            "additionalProperties": False,
        },
    }


class _Encoder:
    def encode(self, sentences, **kwargs):
        return np.asarray(
            [
                [1.0, 0.0]
                if "alice" in sentence.casefold()
                else [0.0, 1.0]
                for sentence in sentences
            ],
            dtype=np.float32,
        )


def _ranking(request: str, tools: list[dict]) -> dict:
    ranking = [{"rank": 1, "tool": "email.send", "cosine_score": 0.9}]
    return {
        "schema_version": RETRIEVAL_RANKING_SCHEMA_VERSION,
        "case_id": "c1",
        "language": "en-US",
        "retriever_version": MULTILINGUAL_RETRIEVER_VERSION,
        "retriever_model_id": RETRIEVER_MODEL_ID,
        "retriever_revision": RETRIEVER_REVISION,
        "serialization_arm": "effect_and_slots_v1",
        "k": 8,
        "request_sha256": hashlib.sha256(request.encode()).hexdigest(),
        "catalog_sha256": catalog_sha256(tools, "effect_and_slots_v1"),
        "ranking": ranking,
        "ranking_sha256": ranking_sha256(ranking),
    }


def test_span_shortlist_is_gold_blind_and_recovers_explicit_value(
    tmp_path: Path,
) -> None:
    request = "Email Alice"
    tools = [_tool()]
    cases = [
        {
            "case_id": "c1",
            "messages": [{"role": "user", "content": request}],
            "tools": tools,
            "metadata": {"language": "en-US"},
        }
    ]
    rows_by_k, telemetry = rank_span_shortlists(
        cases,
        [_ranking(request, tools)],
        _Encoder(),
        query_mode="tool_slot_request",
        top_ks=(16,),
        batch_size=4,
    )
    rows = rows_by_k[16]
    assert forbidden_paths(rows) == []
    candidates = rows[0]["tools"][0]["slots"][0]["candidates"]
    assert candidates[0]["source_text"] == "Alice"
    assert telemetry["slot_query_count"] == 1

    shortlists = tmp_path / "shortlists.jsonl"
    gold = tmp_path / "gold.jsonl"
    write_jsonl(shortlists, rows)
    write_jsonl(
        gold,
        [
            {
                "case_id": "c1",
                "language": "en-US",
                "ground_truth": [
                    {"email.send": {"person": ["Alice"]}}
                ],
            }
        ],
    )
    summary = score_shortlists(shortlists, gold, tmp_path / "scoring")
    assert summary["pooled"]["tool_recall_at_8"] == 1.0
    assert summary["pooled"]["explicit_slot_recall"] == 1.0
    assert (
        summary["pooled"]["joint_tool_and_all_explicit_slots_recall"]
        == 1.0
    )

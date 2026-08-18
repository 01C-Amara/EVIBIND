from __future__ import annotations

import numpy as np

from tapbench.multilingual_retriever import (
    catalog_sha256,
    forbidden_paths,
    rank_cases,
    rank_from_embeddings,
    serialize_tool,
)


def _tool(name: str, description: str) -> dict:
    return {
        "name": name,
        "canonical_name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "requested value",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
    }


class _Encoder:
    def encode(self, sentences, **kwargs):
        rows = []
        for sentence in sentences:
            lowered = sentence.casefold()
            if "weather" in lowered:
                rows.append([0.0, 1.0])
            else:
                rows.append([1.0, 0.0])
        return np.asarray(rows, dtype=np.float32)


def test_rank_ties_are_broken_by_public_tool_name() -> None:
    ranking = rank_from_embeddings(
        np.asarray([1.0, 0.0]),
        np.asarray([[1.0, 0.0], [1.0, 0.0]]),
        ["z.tool", "a.tool"],
        k=2,
    )
    assert [row["tool"] for row in ranking] == ["a.tool", "z.tool"]


def test_rank_cases_is_gold_blind_and_hashes_public_catalog() -> None:
    tools = [
        _tool("email.send", "Send an email."),
        _tool("weather.query", "Query weather."),
    ]
    cases = [
        {
            "case_id": "c1",
            "messages": [{"role": "user", "content": "Send an email"}],
            "tools": tools,
            "metadata": {"language": "en-US"},
        },
        {
            "case_id": "c2",
            "messages": [{"role": "user", "content": "weather tomorrow"}],
            "tools": tools,
            "metadata": {"language": "en-US"},
        },
    ]
    rows, telemetry = rank_cases(
        cases,
        _Encoder(),
        arm="effect_and_slots_v1",
        k=2,
    )
    assert rows[0]["ranking"][0]["tool"] == "email.send"
    assert rows[1]["ranking"][0]["tool"] == "weather.query"
    assert telemetry["catalog_sha256"] == catalog_sha256(
        tools, "effect_and_slots_v1"
    )
    assert forbidden_paths(rows) == []


def test_retriever_rejects_scorer_only_fields() -> None:
    tool = _tool("email.send", "Send an email.")
    tool["ground_truth"] = {"email.send": {}}
    try:
        serialize_tool(tool, "effect_only_v1")
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("scorer-only fields must not reach retrieval")

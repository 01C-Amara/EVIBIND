from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Protocol


MULTILINGUAL_RETRIEVER_VERSION = "tapbench.multilingual_e5_ranker.v1"
RETRIEVAL_RANKING_SCHEMA_VERSION = "tapbench.tool_ranking.v1"
RETRIEVER_MODEL_ID = "intfloat/multilingual-e5-small"
RETRIEVER_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
SERIALIZATION_ARMS = ("effect_only_v1", "effect_and_slots_v1")
RETRIEVER_ARTIFACT_HASHES = {
    "model.safetensors": (
        "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477"
    ),
    "config.json": (
        "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959"
    ),
    "tokenizer.json": (
        "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"
    ),
    "sentencepiece.bpe.model": (
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"
    ),
    "modules.json": (
        "c6e29747481e8b5dd2b58401966aeac910de39092f90cda9a704b1545f902b04"
    ),
    "1_Pooling/config.json": (
        "987f7a67a38fa564c849bb5d277c52ab9088a84368fc0be31a354125aebb12a0"
    ),
}
_FORBIDDEN_KEYS = {
    "gt",
    "gold",
    "gold_action",
    "ground_truth",
    "expected",
    "expected_action",
    "bfcl_gold",
    "derivable_values",
}


class Encoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_retriever_artifact(model_path: str | Path) -> dict[str, str]:
    root = Path(model_path)
    observed: dict[str, str] = {}
    failures: list[str] = []
    for relative, expected in RETRIEVER_ARTIFACT_HASHES.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        actual = file_sha256(path)
        observed[relative] = actual
        if actual != expected:
            failures.append(f"sha256:{relative}:{actual}")
    if failures:
        raise ValueError(
            "retriever artifact does not match the frozen revision: "
            + ", ".join(failures)
        )
    return observed


def forbidden_paths(value: Any, path: str = "$") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold()
            child_path = f"{path}.{key}"
            if normalized in _FORBIDDEN_KEYS or "gold" in normalized:
                failures.append(child_path)
            failures.extend(forbidden_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(forbidden_paths(child, f"{path}[{index}]"))
    return failures


def tool_name(tool: dict[str, Any]) -> str:
    return str(tool.get("canonical_name") or tool.get("name") or "")


def request_text(case: dict[str, Any]) -> str:
    return "\n".join(
        str(message.get("content", ""))
        for message in case.get("messages", [])
        if str(message.get("role", "")).casefold() == "user"
    )


def serialize_tool(tool: dict[str, Any], arm: str) -> str:
    if arm not in SERIALIZATION_ARMS:
        raise ValueError(f"unknown tool serialization arm: {arm}")
    leaks = forbidden_paths(tool)
    if leaks:
        raise ValueError(
            f"retriever tool payload contains forbidden fields: {leaks}"
        )
    lines = [
        f"Tool: {tool_name(tool)}",
        f"Effect: {str(tool.get('description', '')).strip()}",
    ]
    if arm == "effect_and_slots_v1":
        parameters = tool.get("parameters", {})
        properties = (
            parameters.get("properties", {})
            if isinstance(parameters, dict)
            else {}
        )
        for name in sorted(properties):
            schema = properties[name]
            description = (
                str(schema.get("description", "")).strip()
                if isinstance(schema, dict)
                else ""
            )
            lines.append(f"Argument {name}: {description}")
    return "passage: " + "\n".join(lines)


def serialize_query(text: str) -> str:
    return "query: " + text


def catalog_sha256(tools: list[dict[str, Any]], arm: str) -> str:
    payload = [
        {"tool": tool_name(tool), "passage": serialize_tool(tool, arm)}
        for tool in sorted(tools, key=tool_name)
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def ranking_sha256(ranking: list[dict[str, Any]]) -> str:
    return _sha256_bytes(
        json.dumps(
            ranking,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )


def _unit_normalize(matrix: Any) -> Any:
    import numpy as np

    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("retriever produced a zero-norm embedding")
    return values / norms


def rank_from_embeddings(
    query_embedding: Any,
    tool_embeddings: Any,
    tool_names: list[str],
    *,
    k: int,
) -> list[dict[str, Any]]:
    if k <= 0:
        raise ValueError("k must be positive")
    query = _unit_normalize(query_embedding)[0]
    tools = _unit_normalize(tool_embeddings)
    if len(tool_names) != tools.shape[0]:
        raise ValueError("tool name and embedding counts differ")
    scores = tools @ query
    order = sorted(
        range(len(tool_names)),
        key=lambda index: (-float(scores[index]), tool_names[index]),
    )[: min(k, len(tool_names))]
    return [
        {
            "rank": rank,
            "tool": tool_names[index],
            "cosine_score": float(scores[index]),
        }
        for rank, index in enumerate(order, start=1)
    ]


def rank_cases(
    cases: list[dict[str, Any]],
    encoder: Encoder,
    *,
    arm: str,
    k: int = 8,
    batch_size: int = 32,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not cases:
        raise ValueError("at least one runtime case is required")
    if k <= 0:
        raise ValueError("k must be positive")
    case_ids = [str(case.get("case_id", "")) for case in cases]
    if not all(case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("runtime case IDs must be unique and nonempty")
    for case in cases:
        leaks = forbidden_paths(case)
        if leaks:
            raise ValueError(
                f"runtime case {case.get('case_id')} contains forbidden fields: "
                f"{leaks}"
            )

    canonical_tools = sorted(
        list(cases[0].get("tools", [])), key=tool_name
    )
    names = [tool_name(tool) for tool in canonical_tools]
    if len(names) != len(set(names)) or not all(names):
        raise ValueError("tool catalog names must be unique and nonempty")
    expected_catalog_hash = catalog_sha256(canonical_tools, arm)
    for case in cases[1:]:
        if catalog_sha256(list(case.get("tools", [])), arm) != expected_catalog_hash:
            raise ValueError("runtime cases do not share one public tool catalog")

    passages = [serialize_tool(tool, arm) for tool in canonical_tools]
    requests = [request_text(case) for case in cases]
    started = time.perf_counter()
    tool_embeddings = encoder.encode(
        passages,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    tool_seconds = time.perf_counter() - started
    query_embeddings: list[Any] = []
    amortized_query_seconds: list[float] = []
    query_seconds = 0.0
    serialized_queries = [serialize_query(request) for request in requests]
    for start_index in range(0, len(serialized_queries), batch_size):
        batch = serialized_queries[start_index : start_index + batch_size]
        started = time.perf_counter()
        encoded = encoder.encode(
            batch,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - started
        if len(encoded) != len(batch):
            raise ValueError("retriever batch embedding count differs from cases")
        query_seconds += elapsed
        query_embeddings.extend(encoded)
        amortized_query_seconds.extend([elapsed / len(batch)] * len(batch))

    if len(query_embeddings) != len(cases):
        raise ValueError("retriever query embedding count differs from cases")
    rows: list[dict[str, Any]] = []
    for case, text, embedding, case_seconds in zip(
        cases,
        requests,
        query_embeddings,
        amortized_query_seconds,
    ):
        ranking = rank_from_embeddings(
            embedding,
            tool_embeddings,
            names,
            k=k,
        )
        rows.append(
            {
                "schema_version": RETRIEVAL_RANKING_SCHEMA_VERSION,
                "case_id": case["case_id"],
                "language": case.get("metadata", {}).get("language"),
                "retriever_version": MULTILINGUAL_RETRIEVER_VERSION,
                "retriever_model_id": RETRIEVER_MODEL_ID,
                "retriever_revision": RETRIEVER_REVISION,
                "serialization_arm": arm,
                "k": k,
                "request_sha256": _sha256_bytes(text.encode("utf-8")),
                "catalog_sha256": expected_catalog_hash,
                "ranking": ranking,
                "ranking_sha256": ranking_sha256(ranking),
                "amortized_query_embedding_seconds": case_seconds,
            }
        )
    ordered_latency = sorted(amortized_query_seconds)

    def percentile(value: float) -> float:
        index = max(
            0,
            min(len(ordered_latency) - 1, math.ceil(value * len(ordered_latency)) - 1),
        )
        return ordered_latency[index]

    telemetry = {
        "schema_version": "tapbench.multilingual_e5_telemetry.v1",
        "retriever_version": MULTILINGUAL_RETRIEVER_VERSION,
        "retriever_model_id": RETRIEVER_MODEL_ID,
        "retriever_revision": RETRIEVER_REVISION,
        "case_count": len(cases),
        "tool_count": len(canonical_tools),
        "serialization_arm": arm,
        "k": k,
        "batch_size": batch_size,
        "tool_embedding_seconds": tool_seconds,
        "query_embedding_seconds": query_seconds,
        "mean_query_embedding_seconds": query_seconds / len(cases),
        "p50_amortized_query_embedding_seconds": percentile(0.50),
        "p95_amortized_query_embedding_seconds": percentile(0.95),
        "catalog_sha256": expected_catalog_hash,
    }
    return rows, telemetry

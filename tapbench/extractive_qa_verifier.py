from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .eflrx import _tool_name
from .extractive_candidates import user_request_text
from .io import read_jsonl, write_jsonl, write_yaml
from .multilingual_retriever import forbidden_paths
from .retrieve_pointer import validate_external_ranking_row, validate_ranking_row
from .source_span_projection import slot_catalog, source_span_catalog


EXTRACTIVE_QA_VERIFIER_VERSION = "tapbench.extractive_qa_verifier.v2"
EXTRACTIVE_QA_ARTIFACT_SCHEMA = "tapbench.extractive_qa_artifact.v1"
EXTRACTIVE_QA_QUESTION_VERSION = "tapbench.extractive_qa_questions.natural.v1"
EXTRACTIVE_QA_MODEL_ID = "timpal0l/mdeberta-v3-base-squad2"
EXTRACTIVE_QA_MODEL_REVISION = "08d6e89c7a6557f967db2e1021f7f640483400ed"
EXTRACTIVE_QA_MARGIN_THRESHOLD = 0.0
EXTRACTIVE_QA_MAX_INPUT_TOKENS = 512
EXTRACTIVE_QA_MAX_ANSWER_TOKENS = 32
_HUMAN_NAME_SLOTS = {
    "artist_name",
    "contact_name",
    "person",
    "recipient_name",
}


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verifier_question(slot_name: str) -> str:
    normalized = str(slot_name).strip()
    humanized = " ".join(normalized.split("_")).strip()
    if normalized == "definition_word":
        return "What word should be defined?"
    if normalized in _HUMAN_NAME_SLOTS:
        return f"Who is the {humanized}?"
    return f"What is the {humanized}?"


def _normalized_surface(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def select_extractive_answer(
    *,
    request_text: str,
    language: str,
    offsets: list[tuple[int, int] | list[int]],
    sequence_ids: list[int | None],
    start_logits: list[float],
    end_logits: list[float],
    cls_index: int,
    input_tokens: int,
    max_answer_tokens: int = EXTRACTIVE_QA_MAX_ANSWER_TOKENS,
    margin_threshold: float = EXTRACTIVE_QA_MARGIN_THRESHOLD,
) -> dict[str, Any]:
    if not (
        len(offsets)
        == len(sequence_ids)
        == len(start_logits)
        == len(end_logits)
    ):
        raise ValueError("QA logits, offsets, and sequence IDs differ in length")
    if not 0 <= cls_index < len(offsets):
        raise ValueError("QA CLS index is out of range")
    lattice = source_span_catalog(request_text, language)
    null_score = float(start_logits[cls_index] + end_logits[cls_index])
    context_indices = [
        index
        for index, sequence_id in enumerate(sequence_ids)
        if sequence_id == 1
        and int(offsets[index][1]) > int(offsets[index][0])
    ]
    best_score = -math.inf
    best_start: int | None = None
    best_end: int | None = None
    context_set = set(context_indices)
    for start_index in context_indices:
        final_index = min(
            len(offsets) - 1,
            start_index + max_answer_tokens - 1,
        )
        for end_index in range(start_index, final_index + 1):
            if end_index not in context_set:
                continue
            score = float(start_logits[start_index] + end_logits[end_index])
            if score > best_score:
                best_score = score
                best_start = start_index
                best_end = end_index
    if best_start is None or best_end is None:
        return {
            "status": "no_context_tokens",
            "admitted": False,
            "answer": None,
            "answer_span": None,
            "span_id": None,
            "null_score": null_score,
            "best_non_null_score": None,
            "non_null_margin": None,
            "input_tokens": input_tokens,
            "span_catalog_sha256": lattice["catalog_sha256"],
        }
    margin = best_score - null_score
    start = int(offsets[best_start][0])
    end = int(offsets[best_end][1])
    while start < end and request_text[start].isspace():
        start += 1
    while end > start and request_text[end - 1].isspace():
        end -= 1
    answer = request_text[start:end]
    status = "admitted"
    admitted = True
    span_id: str | None = None
    if margin <= margin_threshold:
        status = "null_margin"
        admitted = False
    elif not answer:
        status = "empty_after_whitespace_trim"
        admitted = False
    elif (
        len(lattice["units"]) >= 2
        and _normalized_surface(answer) == _normalized_surface(request_text)
    ):
        status = "whole_request_rejected"
        admitted = False
    else:
        matching = [
            row
            for row in lattice["spans"]
            if list(row["source_span"]) == [start, end]
            and str(row["source_text"]) == answer
        ]
        if not matching:
            status = "source_lattice_misaligned"
            admitted = False
        else:
            span_id = str(matching[0]["span_id"])
    return {
        "status": status,
        "admitted": admitted,
        "answer": answer if admitted else None,
        "raw_best_answer": answer,
        "answer_span": [start, end] if admitted else None,
        "raw_best_answer_span": [start, end],
        "span_id": span_id,
        "null_score": null_score,
        "best_non_null_score": best_score,
        "non_null_margin": margin,
        "input_tokens": input_tokens,
        "span_catalog_sha256": lattice["catalog_sha256"],
    }


def _model_file_hashes(model_path: str | Path) -> dict[str, str]:
    root = Path(model_path)
    required = (
        "added_tokens.json",
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"QA verifier files missing: {missing}")
    return {name: file_sha256(root / name) for name in required}


def _iter_batches(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def generate_extractive_qa_artifact(
    cases_path: str | Path,
    rankings_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    batch_size: int = 32,
    cpu_threads: int = 16,
    ranking_contract: str = "multilingual_e5_v1",
) -> dict[str, Any]:
    if batch_size <= 0 or cpu_threads <= 0:
        raise ValueError("QA verifier batch size and CPU threads must be positive")
    cases = read_jsonl(cases_path)
    ranking_rows = read_jsonl(rankings_path)
    rankings = {str(row.get("case_id")): row for row in ranking_rows}
    if len(rankings) != len(ranking_rows):
        raise ValueError("QA verifier ranking artifact has duplicate case IDs")
    jobs: list[dict[str, Any]] = []
    for case in cases:
        leaks = forbidden_paths(case)
        if leaks:
            raise ValueError(
                f"runtime case {case.get('case_id')} contains gold: {leaks}"
            )
        case_id = str(case["case_id"])
        request_text = user_request_text(case.get("messages", []))
        tools = list(case.get("tools", []))
        ranking_row = rankings.get(case_id)
        if ranking_row is None:
            raise ValueError(f"QA verifier missing ranking for {case_id}")
        if ranking_contract == "multilingual_e5_v1":
            ranking = validate_ranking_row(
                ranking_row,
                case_id=case_id,
                request_text=request_text,
                tools=tools,
            )
        elif ranking_contract == "massive_supervised_router_v1":
            ranking = validate_external_ranking_row(
                ranking_row,
                case_id=case_id,
                request_text=request_text,
                tools=tools,
                schema_version="tapbench.supervised_tool_ranking.v1",
                retriever_version="tapbench.massive_supervised_intent_router.v1",
                retriever_model_id="hashed_tfidf_nearest_centroid",
                retriever_revision="massive_v1.1_train_only",
            )
        else:
            raise ValueError(f"unknown QA verifier ranking contract: {ranking_contract}")
        by_name = {_tool_name(tool): tool for tool in tools}
        language = str(
            case.get("metadata", {}).get("language")
            or case.get("factors", {}).get("language")
            or "unknown"
        )
        lattice = source_span_catalog(request_text, language)
        for ranked_tool in ranking:
            tool_name = str(ranked_tool["tool"])
            tool = by_name[tool_name]
            slots, _ = slot_catalog(tool)
            for slot in slots:
                jobs.append(
                    {
                        "case_id": case_id,
                        "language": language,
                        "request_text": request_text,
                        "request_sha256": lattice["request_sha256"],
                        "span_catalog_sha256": lattice["catalog_sha256"],
                        "ranking_sha256": ranking_row["ranking_sha256"],
                        "tool": tool_name,
                        "tool_rank": int(ranked_tool["rank"]),
                        "slot_id": str(slot["slot_id"]),
                        "surface_name": str(slot["name"]),
                        "required": bool(slot["required"]),
                        "question": verifier_question(str(slot["name"])),
                    }
                )

    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    torch.set_num_threads(cpu_threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("QA verifier requires a fast tokenizer for offsets")
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
    ).to("cpu")
    model.eval()

    output_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch in _iter_batches(jobs, batch_size):
        encoded = tokenizer(
            [str(job["question"]) for job in batch],
            [str(job["request_text"]) for job in batch],
            padding=True,
            truncation=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        sequence_ids = [encoded.sequence_ids(index) for index in range(len(batch))]
        offsets = encoded.pop("offset_mapping")
        attention = encoded["attention_mask"]
        token_counts = [int(row.sum().item()) for row in attention]
        overflow = [count > EXTRACTIVE_QA_MAX_INPUT_TOKENS for count in token_counts]
        if any(overflow):
            raise ValueError("QA verifier input exceeds frozen 512-token limit")
        began = time.perf_counter()
        with torch.inference_mode():
            model_output = model(**encoded)
        elapsed = time.perf_counter() - began
        for index, job in enumerate(batch):
            input_ids = encoded["input_ids"][index]
            cls_positions = (input_ids == tokenizer.cls_token_id).nonzero()
            if not len(cls_positions):
                raise ValueError("QA verifier input has no CLS token")
            selection = select_extractive_answer(
                request_text=str(job["request_text"]),
                language=str(job["language"]),
                offsets=offsets[index].tolist(),
                sequence_ids=sequence_ids[index],
                start_logits=model_output.start_logits[index].tolist(),
                end_logits=model_output.end_logits[index].tolist(),
                cls_index=int(cls_positions[0].item()),
                input_tokens=token_counts[index],
            )
            row = {
                "schema_version": EXTRACTIVE_QA_ARTIFACT_SCHEMA,
                "verifier_version": EXTRACTIVE_QA_VERIFIER_VERSION,
                "question_version": EXTRACTIVE_QA_QUESTION_VERSION,
                "model_id": EXTRACTIVE_QA_MODEL_ID,
                "model_revision": EXTRACTIVE_QA_MODEL_REVISION,
                "backend": "huggingface_transformers_cpu",
                "dtype": "float32",
                "margin_threshold": EXTRACTIVE_QA_MARGIN_THRESHOLD,
                "max_input_tokens": EXTRACTIVE_QA_MAX_INPUT_TOKENS,
                "max_answer_tokens": EXTRACTIVE_QA_MAX_ANSWER_TOKENS,
                "case_id": job["case_id"],
                "language": job["language"],
                "request_sha256": job["request_sha256"],
                "span_catalog_sha256": job["span_catalog_sha256"],
                "ranking_sha256": job["ranking_sha256"],
                "tool": job["tool"],
                "tool_rank": job["tool_rank"],
                "slot_id": job["slot_id"],
                "surface_name": job["surface_name"],
                "required": job["required"],
                "question": job["question"],
                "question_sha256": hashlib.sha256(
                    str(job["question"]).encode("utf-8")
                ).hexdigest(),
                **selection,
                "batch_inference_seconds_per_item": elapsed / len(batch),
                "input_truncated": False,
                "gold_loaded": False,
            }
            row["row_sha256"] = _json_sha256(row)
            output_rows.append(row)
    total_seconds = time.perf_counter() - started
    write_jsonl(output_path, output_rows)
    status_counts = Counter(str(row["status"]) for row in output_rows)
    manifest = {
        "schema_version": "tapbench.extractive_qa_manifest.v1",
        "verifier_version": EXTRACTIVE_QA_VERIFIER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": EXTRACTIVE_QA_MODEL_ID,
        "model_revision": EXTRACTIVE_QA_MODEL_REVISION,
        "model_path": str(Path(model_path).resolve()),
        "model_file_sha256": _model_file_hashes(model_path),
        "backend": "huggingface_transformers_cpu",
        "device": "cpu",
        "dtype": "float32",
        "deterministic_algorithms": True,
        "batch_size": batch_size,
        "cpu_threads": cpu_threads,
        "margin_threshold": EXTRACTIVE_QA_MARGIN_THRESHOLD,
        "max_input_tokens": EXTRACTIVE_QA_MAX_INPUT_TOKENS,
        "max_answer_tokens": EXTRACTIVE_QA_MAX_ANSWER_TOKENS,
        "cases_path": str(Path(cases_path).resolve()),
        "cases_sha256": file_sha256(cases_path),
        "rankings_path": str(Path(rankings_path).resolve()),
        "rankings_sha256": file_sha256(rankings_path),
        "ranking_contract": ranking_contract,
        "output_path": str(Path(output_path).resolve()),
        "output_sha256": file_sha256(output_path),
        "case_count": len(cases),
        "row_count": len(output_rows),
        "admitted_count": sum(bool(row["admitted"]) for row in output_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "input_truncation_count": 0,
        "gold_runtime_firewall": "passed",
        "total_inference_seconds": total_seconds,
        "packages": {
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
        },
    }
    write_yaml(manifest_path, manifest)
    return manifest


def validate_extractive_qa_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        row_failures: list[str] = []
        identity = (
            str(row.get("case_id")),
            str(row.get("tool")),
            str(row.get("surface_name")),
        )
        if identity in identities:
            row_failures.append("duplicate_identity")
        identities.add(identity)
        expected_hash = row.get("row_sha256")
        payload = {key: value for key, value in row.items() if key != "row_sha256"}
        if expected_hash != _json_sha256(payload):
            row_failures.append("row_sha256_mismatch")
        expected = {
            "schema_version": EXTRACTIVE_QA_ARTIFACT_SCHEMA,
            "verifier_version": EXTRACTIVE_QA_VERIFIER_VERSION,
            "question_version": EXTRACTIVE_QA_QUESTION_VERSION,
            "model_id": EXTRACTIVE_QA_MODEL_ID,
            "model_revision": EXTRACTIVE_QA_MODEL_REVISION,
            "backend": "huggingface_transformers_cpu",
            "dtype": "float32",
            "margin_threshold": EXTRACTIVE_QA_MARGIN_THRESHOLD,
            "max_input_tokens": EXTRACTIVE_QA_MAX_INPUT_TOKENS,
            "max_answer_tokens": EXTRACTIVE_QA_MAX_ANSWER_TOKENS,
            "input_truncated": False,
            "gold_loaded": False,
        }
        for field, value in expected.items():
            if row.get(field) != value:
                row_failures.append(f"{field}_mismatch")
        if bool(row.get("admitted")) != (row.get("status") == "admitted"):
            row_failures.append("admission_status_mismatch")
        if row_failures:
            failures.append(
                {
                    "row_index": index,
                    "identity": identity,
                    "failures": row_failures,
                }
            )
    return failures

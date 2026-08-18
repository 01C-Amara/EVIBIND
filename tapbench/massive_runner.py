from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .capc import CAPC_CONDITIONS, CAPC_VERSION, run_capc_resolution
from .capc_projected import (
    PROJECTED_CAPC_CONDITIONS,
    PROJECTED_CAPC_VERSION,
    SOURCE_CERTIFICATE_VERSION,
    run_projected_capc_resolution,
)
from .eflrx import (
    CONTEXT_TOKENS,
    ContextOverflowError,
    RequestFn,
    preflight_schema_request,
)
from .eflrx_baselines import (
    RAW_BASELINE_CONDITIONS,
    RAW_BASELINE_VERSION,
    run_raw_baseline,
)
from .extractive_candidates import EXTRACTIVE_CANDIDATE_VERSION
from .extractive_qa_verifier import (
    EXTRACTIVE_QA_MODEL_ID,
    EXTRACTIVE_QA_MODEL_REVISION,
    EXTRACTIVE_QA_QUESTION_VERSION,
    EXTRACTIVE_QA_VERIFIER_VERSION,
    validate_extractive_qa_rows,
)
from .io import read_jsonl, write_jsonl, write_yaml
from .r2_model_runner import R2A_CHAT_PARSER, R2A_GRAMMAR_ENGINE
from .qa_evidence_controller import (
    QA_EVIDENCE_CONDITIONS,
    QA_EVIDENCE_CONTROLLER_VERSION,
    QA_EVIDENCE_SYSTEM_LABEL,
    index_verifier_rows,
    run_qa_evidence_resolution,
)
from .retrieve_pointer import (
    RETRIEVE_POINTER_CONDITIONS,
    RETRIEVE_POINTER_VERSION,
    run_retrieve_pointer_resolution,
)
from .semantic_surface_projection import (
    SEMANTIC_SURFACE_CONDITIONS,
    SEMANTIC_SURFACE_VERSION,
    run_semantic_surface_resolution,
)
from .source_span_projection import (
    SOURCE_SPAN_CERTIFICATE_VERSION,
    SOURCE_SPAN_PROJECTION_VERSION,
)
from .slotwise_surface_projection import (
    SLOTWISE_SURFACE_CONDITIONS,
    SLOTWISE_SURFACE_VERSION,
    run_slotwise_surface_resolution,
)
from .thinking import prediction_has_thinking_marker


MASSIVE_RUNNER_VERSION = "tapbench.massive_runner.v5"
MASSIVE_STUDY_CONDITIONS = (
    *RAW_BASELINE_CONDITIONS,
    *CAPC_CONDITIONS,
    *PROJECTED_CAPC_CONDITIONS,
    *RETRIEVE_POINTER_CONDITIONS,
    *SEMANTIC_SURFACE_CONDITIONS,
    *SLOTWISE_SURFACE_CONDITIONS,
    *QA_EVIDENCE_CONDITIONS,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _language(case: dict[str, Any]) -> str:
    return str(
        case.get("metadata", {}).get("language")
        or case.get("factors", {}).get("language")
        or "unknown"
    )


def _select_cases(
    cases: list[dict[str, Any]],
    *,
    languages: Iterable[str] | None,
    limit_per_language: int | None,
) -> list[dict[str, Any]]:
    allowed = (
        {str(value).strip() for value in languages if str(value).strip()}
        if languages is not None
        else None
    )
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for case in cases:
        language = _language(case)
        if allowed is not None and language not in allowed:
            continue
        if (
            limit_per_language is not None
            and counts.get(language, 0) >= limit_per_language
        ):
            continue
        selected.append(case)
        counts[language] = counts.get(language, 0) + 1
    return selected


def run_massive_cases(
    cases_path: str | Path,
    output_path: str | Path,
    timings_path: str | Path,
    manifest_path: str | Path,
    *,
    endpoint: str,
    model_id: str,
    model_key: str,
    model_artifact: str,
    chat_template: str,
    conditions: Iterable[str] = MASSIVE_STUDY_CONDITIONS,
    seeds: Iterable[int] = (1,),
    max_tokens: int = 384,
    languages: Iterable[str] | None = None,
    limit_per_language: int | None = None,
    request_fn: RequestFn = preflight_schema_request,
    preregistration_path: str | Path | None = None,
    amendment_paths: Iterable[str | Path] = (),
    rankings_path: str | Path | None = None,
    qa_verifier_path: str | Path | None = None,
) -> dict[str, Any]:
    selected_conditions = tuple(str(value) for value in conditions)
    unknown = set(selected_conditions) - set(MASSIVE_STUDY_CONDITIONS)
    if unknown:
        raise ValueError(f"unknown MASSIVE conditions: {sorted(unknown)}")
    selected_seeds = tuple(int(value) for value in seeds)
    cases = _select_cases(
        read_jsonl(cases_path),
        languages=languages,
        limit_per_language=limit_per_language,
    )
    needs_rankings = any(
        condition in RETRIEVE_POINTER_CONDITIONS
        or condition in SEMANTIC_SURFACE_CONDITIONS
        or condition in SLOTWISE_SURFACE_CONDITIONS
        or condition in QA_EVIDENCE_CONDITIONS
        for condition in selected_conditions
    )
    ranking_artifact_sha256: str | None = None
    rankings: dict[str, dict[str, Any]] = {}
    if needs_rankings:
        if rankings_path is None:
            raise ValueError(
                "retrieve-pointer conditions require --rankings"
            )
        ranking_rows = read_jsonl(rankings_path)
        rankings = {
            str(row.get("case_id")): row for row in ranking_rows
        }
        if len(rankings) != len(ranking_rows):
            raise ValueError("ranking artifact contains duplicate case IDs")
        missing_rankings = sorted(
            str(case["case_id"])
            for case in cases
            if str(case["case_id"]) not in rankings
        )
        if missing_rankings:
            raise ValueError(
                "ranking artifact is missing selected cases: "
                + ", ".join(missing_rankings[:10])
            )
        ranking_artifact_sha256 = _sha256(rankings_path)
    needs_qa_verifier = any(
        condition in QA_EVIDENCE_CONDITIONS
        for condition in selected_conditions
    )
    qa_verifier_artifact_sha256: str | None = None
    qa_verifier_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    if needs_qa_verifier:
        if qa_verifier_path is None:
            raise ValueError("QA-evidence conditions require --qa-verifier")
        qa_rows = read_jsonl(qa_verifier_path)
        qa_failures = validate_extractive_qa_rows(qa_rows)
        if qa_failures:
            raise ValueError(
                "QA verifier artifact failed validation: "
                + str(qa_failures[:3])
            )
        qa_verifier_index = index_verifier_rows(qa_rows)
        qa_verifier_artifact_sha256 = _sha256(qa_verifier_path)
    jobs = [
        (case, condition, seed)
        for case in cases
        for condition in selected_conditions
        for seed in selected_seeds
    ]

    predictions: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    for case, condition, seed in jobs:
        started = time.perf_counter()
        error = None
        try:
            if condition in QA_EVIDENCE_CONDITIONS:
                assert ranking_artifact_sha256 is not None
                assert qa_verifier_artifact_sha256 is not None
                action, metadata = run_qa_evidence_resolution(
                    case_id=str(case["case_id"]),
                    messages=list(case.get("messages", [])),
                    tools=list(case.get("tools", [])),
                    language=_language(case),
                    ranking_row=rankings[str(case["case_id"])],
                    ranking_artifact_sha256=ranking_artifact_sha256,
                    verifier_index=qa_verifier_index,
                    verifier_artifact_sha256=qa_verifier_artifact_sha256,
                    endpoint=endpoint,
                    condition=condition,
                    max_tokens=max_tokens,
                    seed=seed,
                    request_fn=request_fn,
                )
            elif condition in SLOTWISE_SURFACE_CONDITIONS:
                assert ranking_artifact_sha256 is not None
                action, metadata = run_slotwise_surface_resolution(
                    case_id=str(case["case_id"]),
                    messages=list(case.get("messages", [])),
                    tools=list(case.get("tools", [])),
                    language=_language(case),
                    ranking_row=rankings[str(case["case_id"])],
                    ranking_artifact_sha256=ranking_artifact_sha256,
                    endpoint=endpoint,
                    condition=condition,
                    max_tokens=max_tokens,
                    seed=seed,
                    request_fn=request_fn,
                )
            elif condition in SEMANTIC_SURFACE_CONDITIONS:
                assert ranking_artifact_sha256 is not None
                action, metadata = run_semantic_surface_resolution(
                    case_id=str(case["case_id"]),
                    messages=list(case.get("messages", [])),
                    tools=list(case.get("tools", [])),
                    language=_language(case),
                    ranking_row=rankings[str(case["case_id"])],
                    ranking_artifact_sha256=ranking_artifact_sha256,
                    endpoint=endpoint,
                    condition=condition,
                    max_tokens=max_tokens,
                    seed=seed,
                    request_fn=request_fn,
                )
            elif condition in RETRIEVE_POINTER_CONDITIONS:
                assert ranking_artifact_sha256 is not None
                action, metadata = run_retrieve_pointer_resolution(
                    case_id=str(case["case_id"]),
                    messages=list(case.get("messages", [])),
                    tools=list(case.get("tools", [])),
                    language=_language(case),
                    ranking_row=rankings[str(case["case_id"])],
                    ranking_artifact_sha256=ranking_artifact_sha256,
                    endpoint=endpoint,
                    condition=condition,
                    max_tokens=max_tokens,
                    seed=seed,
                    request_fn=request_fn,
                )
            elif condition in PROJECTED_CAPC_CONDITIONS:
                action, metadata = run_projected_capc_resolution(
                    messages=list(case.get("messages", [])),
                    tools=list(case.get("tools", [])),
                    endpoint=endpoint,
                    condition=condition,
                    max_tokens=max_tokens,
                    seed=seed,
                    request_fn=request_fn,
                )
            elif condition in CAPC_CONDITIONS:
                action, metadata = run_capc_resolution(
                    messages=list(case.get("messages", [])),
                    tools=list(case.get("tools", [])),
                    endpoint=endpoint,
                    condition=condition,
                    max_tokens=max_tokens,
                    seed=seed,
                    request_fn=request_fn,
                )
            else:
                action, metadata = run_raw_baseline(
                    case,
                    endpoint=endpoint,
                    condition=condition,
                    max_tokens=max_tokens,
                    seed=seed,
                    request_fn=request_fn,
                )
        except Exception as exc:
            error = str(exc)
            overflow = isinstance(exc, ContextOverflowError)
            action = {"runner_error": error}
            metadata = {
                "finish_reason": (
                    "context_overflow" if overflow else "runner_error"
                ),
                "error_type": exc.__class__.__name__,
                "error_message": error,
                "generation_calls": 0,
                "action_risk_score": 1.0,
                "context_overflow": overflow,
            }
        elapsed = time.perf_counter() - started
        row = {
            "case_id": case["case_id"],
            "hypothesis_grid_id": case.get("hypothesis_grid_id"),
            "language": _language(case),
            "method": condition,
            "model_id": model_id,
            "seed": seed,
            "prediction": action,
            "action_ir_normalized": True,
            "response_metadata": metadata,
            "resolution": {
                "terminal_state": (
                    action.get("mode")
                    if isinstance(action, dict)
                    else "runner_error"
                ),
                "materialized_action": action,
                "elapsed_seconds": elapsed,
            },
            "runner_error": error,
            "backend": "llama.cpp",
            "quantization": "Q4_K_M",
            "chat_template": chat_template,
            "grammar_engine": R2A_GRAMMAR_ENGINE,
            "chat_parser": R2A_CHAT_PARSER,
            "inference_path": "apply_template_then_raw_completion",
            "model_artifact": model_artifact,
            "thinking_mode": "off",
            "reasoning_budget": 0,
            "capc_version": (
                CAPC_VERSION if condition in CAPC_CONDITIONS else None
            ),
            "extractive_candidate_version": (
                EXTRACTIVE_CANDIDATE_VERSION
                if condition in CAPC_CONDITIONS
                else None
            ),
            "capc_runner_version": (
                MASSIVE_RUNNER_VERSION
                if condition in CAPC_CONDITIONS
                else None
            ),
            "projected_capc_version": (
                PROJECTED_CAPC_VERSION
                if condition in PROJECTED_CAPC_CONDITIONS
                else None
            ),
            "retrieve_pointer_version": (
                RETRIEVE_POINTER_VERSION
                if condition in RETRIEVE_POINTER_CONDITIONS
                else None
            ),
            "semantic_surface_version": (
                SEMANTIC_SURFACE_VERSION
                if condition in SEMANTIC_SURFACE_CONDITIONS
                else None
            ),
            "slotwise_surface_version": (
                SLOTWISE_SURFACE_VERSION
                if condition in SLOTWISE_SURFACE_CONDITIONS
                else None
            ),
            "semantic_surface_materializer_version": metadata.get(
                "semantic_surface_materializer_version"
            ),
            "qa_evidence_controller_version": metadata.get(
                "qa_evidence_controller_version"
            ),
            "qa_evidence_system_label": metadata.get(
                "qa_evidence_system_label"
            ),
            "qa_verifier_version": metadata.get("qa_verifier_version"),
            "qa_verifier_question_version": metadata.get(
                "qa_verifier_question_version"
            ),
            "qa_verifier_model_id": metadata.get("qa_verifier_model_id"),
            "qa_verifier_model_revision": metadata.get(
                "qa_verifier_model_revision"
            ),
            "qa_verifier_backend": metadata.get("qa_verifier_backend"),
            "qa_verifier_dtype": metadata.get("qa_verifier_dtype"),
            "qa_verifier_artifact_sha256": metadata.get(
                "qa_verifier_artifact_sha256"
            ),
            "retriever_version": metadata.get("retriever_version"),
            "retriever_model_id": metadata.get("retriever_model_id"),
            "retriever_revision": metadata.get("retriever_revision"),
            "retriever_serialization_arm": metadata.get(
                "retriever_serialization_arm"
            ),
            "retriever_k": metadata.get("retriever_k"),
            "ranking_sha256": metadata.get("ranking_sha256"),
            "ranking_artifact_sha256": metadata.get(
                "ranking_artifact_sha256"
            ),
            "source_span_projection_version": (
                SOURCE_SPAN_PROJECTION_VERSION
                if condition in (
                    *RETRIEVE_POINTER_CONDITIONS,
                    *SEMANTIC_SURFACE_CONDITIONS,
                    *SLOTWISE_SURFACE_CONDITIONS,
                    *QA_EVIDENCE_CONDITIONS,
                )
                else None
            ),
            "source_span_certificate_version": (
                SOURCE_SPAN_CERTIFICATE_VERSION
                if condition in (
                    *RETRIEVE_POINTER_CONDITIONS,
                    *SEMANTIC_SURFACE_CONDITIONS,
                    *SLOTWISE_SURFACE_CONDITIONS,
                    *QA_EVIDENCE_CONDITIONS,
                )
                else None
            ),
            "source_certificate_version": (
                SOURCE_CERTIFICATE_VERSION
                if condition in PROJECTED_CAPC_CONDITIONS
                else SOURCE_SPAN_CERTIFICATE_VERSION
                if condition in (
                    *RETRIEVE_POINTER_CONDITIONS,
                    *SEMANTIC_SURFACE_CONDITIONS,
                    *SLOTWISE_SURFACE_CONDITIONS,
                    *QA_EVIDENCE_CONDITIONS,
                )
                else None
            ),
            "raw_baseline_version": (
                RAW_BASELINE_VERSION
                if condition in RAW_BASELINE_CONDITIONS
                else None
            ),
            "massive_runner_version": MASSIVE_RUNNER_VERSION,
            "action_risk_threshold": metadata.get(
                "action_risk_threshold"
            ),
            "action_risk_score": metadata.get("action_risk_score"),
            "max_output_tokens": max_tokens,
            "finish_reason": metadata.get("finish_reason"),
        }
        row["thinking_marker_detected"] = prediction_has_thinking_marker(row)
        predictions.append(row)
        timings.append(
            {
                "case_id": case["case_id"],
                "language": _language(case),
                "method": condition,
                "model_key": model_key,
                "model_id": model_id,
                "seed": seed,
                "backend": "llama.cpp",
                "quantization": "Q4_K_M",
                "elapsed_seconds": elapsed,
                "generation_calls": metadata.get("generation_calls", 0),
                "runner_error": error,
                "thinking_mode": "off",
                "thinking_marker_detected": row[
                    "thinking_marker_detected"
                ],
                "action_risk_score": metadata.get("action_risk_score"),
                "tool_agreement": metadata.get("tool_agreement"),
                "election_policy": metadata.get("election_policy"),
                "election_winner_votes": metadata.get(
                    "election_winner_votes"
                ),
                "proposal_admitted": metadata.get("proposal_admitted"),
                "pointer_agreement": metadata.get("pointer_agreement"),
                "active_slot_agreement": metadata.get(
                    "active_slot_agreement"
                ),
                "surface_action_agreement": metadata.get(
                    "surface_action_agreement"
                ),
                "slotwise_action_agreement": metadata.get(
                    "slotwise_action_agreement"
                ),
                "slotwise_null_count": metadata.get("slotwise_null_count"),
                "qa_verifier_rows_consulted": metadata.get(
                    "qa_verifier_rows_consulted"
                ),
                "qa_verifier_null_count": metadata.get(
                    "qa_verifier_null_count"
                ),
                "qa_verifier_lookup_seconds": metadata.get(
                    "qa_verifier_lookup_seconds"
                ),
                "retriever_election_agreement": metadata.get(
                    "retriever_election_agreement"
                ),
                "controller_stage_failure": metadata.get(
                    "controller_stage_failure"
                ),
                "retrieval_top1": metadata.get("retrieval_top1"),
                "accepted_proposal_index": metadata.get(
                    "accepted_proposal_index"
                ),
                **{
                    key: metadata.get(key)
                    for key in (
                        "finish_reason",
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "generated_tokens_per_second",
                        "context_truncated",
                        "rendered_input_tokens_max",
                        "preflight_prompt_token_delta_max_abs",
                        "context_headroom_tokens_min",
                        "context_overflow",
                    )
                },
            }
        )

    write_jsonl(output_path, predictions)
    write_jsonl(timings_path, timings)
    source_hashes = {
        "cases": _sha256(cases_path),
        "model_artifact": _sha256(model_artifact),
    }
    if preregistration_path is not None:
        source_hashes["preregistration"] = _sha256(preregistration_path)
    if rankings_path is not None:
        source_hashes["rankings"] = _sha256(rankings_path)
    if qa_verifier_path is not None:
        source_hashes["qa_verifier"] = _sha256(qa_verifier_path)
    amendment_hashes = {
        str(path): _sha256(path) for path in amendment_paths
    }
    manifest = {
        "schema_version": "tapbench.massive_run_manifest.v2",
        "runner_version": MASSIVE_RUNNER_VERSION,
        "capc_version": CAPC_VERSION,
        "projected_capc_version": PROJECTED_CAPC_VERSION,
        "source_certificate_version": SOURCE_CERTIFICATE_VERSION,
        "retrieve_pointer_version": RETRIEVE_POINTER_VERSION,
        "semantic_surface_version": SEMANTIC_SURFACE_VERSION,
        "slotwise_surface_version": SLOTWISE_SURFACE_VERSION,
        "semantic_surface_materializer_version": SEMANTIC_SURFACE_VERSION,
        "qa_evidence_controller_version": QA_EVIDENCE_CONTROLLER_VERSION,
        "qa_evidence_system_label": QA_EVIDENCE_SYSTEM_LABEL,
        "qa_verifier_version": EXTRACTIVE_QA_VERIFIER_VERSION,
        "qa_verifier_question_version": EXTRACTIVE_QA_QUESTION_VERSION,
        "qa_verifier_model_id": EXTRACTIVE_QA_MODEL_ID,
        "qa_verifier_model_revision": EXTRACTIVE_QA_MODEL_REVISION,
        "source_span_projection_version": SOURCE_SPAN_PROJECTION_VERSION,
        "source_span_certificate_version": SOURCE_SPAN_CERTIFICATE_VERSION,
        "extractive_candidate_version": EXTRACTIVE_CANDIDATE_VERSION,
        "raw_baseline_version": RAW_BASELINE_VERSION,
        "action_ir_normalized": True,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path),
        "predictions_path": str(output_path),
        "timings_path": str(timings_path),
        "model_key": model_key,
        "model_id": model_id,
        "model_artifact": model_artifact,
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "chat_template": chat_template,
        "grammar_engine": R2A_GRAMMAR_ENGINE,
        "chat_parser": R2A_CHAT_PARSER,
        "thinking_mode": "off",
        "reasoning_budget": 0,
        "context_tokens": CONTEXT_TOKENS,
        "max_output_tokens": max_tokens,
        "languages": sorted({_language(case) for case in cases}),
        "limit_per_language": limit_per_language,
        "conditions": list(selected_conditions),
        "seeds": list(selected_seeds),
        "case_count": len(cases),
        "generation_count": len(predictions),
        "actual_model_calls": sum(
            int(row["response_metadata"].get("generation_calls", 0))
            for row in predictions
        ),
        "runner_errors": sum(
            row["runner_error"] is not None for row in predictions
        ),
        "source_sha256": source_hashes,
        "amendment_sha256": amendment_hashes,
        "rankings_path": str(rankings_path) if rankings_path is not None else None,
        "ranking_artifact_sha256": ranking_artifact_sha256,
        "ranking_case_count": len(rankings),
        "qa_verifier_path": (
            str(qa_verifier_path) if qa_verifier_path is not None else None
        ),
        "qa_verifier_artifact_sha256": qa_verifier_artifact_sha256,
        "qa_verifier_row_count": len(qa_verifier_index),
    }
    write_yaml(manifest_path, manifest)
    return manifest

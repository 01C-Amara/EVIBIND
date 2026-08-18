from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .eflrx import (
    CONTEXT_TOKENS,
    ContextOverflowError,
    RequestFn,
    _merge_metadata,
    _non_call,
    _tool_name,
    preflight_schema_request,
)
from .extractive_candidates import user_request_text
from .extractive_qa_verifier import (
    EXTRACTIVE_QA_MODEL_ID,
    EXTRACTIVE_QA_MODEL_REVISION,
    EXTRACTIVE_QA_QUESTION_VERSION,
    EXTRACTIVE_QA_VERIFIER_VERSION,
    validate_extractive_qa_rows,
)
from .io import read_jsonl, write_jsonl
from .qa_evidence_controller import index_verifier_rows
from .r2_model_runner import R2A_CHAT_PARSER, R2A_GRAMMAR_ENGINE
from .retrieve_pointer import validate_external_ranking_row
from .semantic_surface_projection import (
    _active_slot_messages,
    _active_slot_schema,
    _public_slots,
    validate_active_slots,
)
from .source_span_projection import (
    SOURCE_SPAN_CERTIFICATE_VERSION,
    SOURCE_SPAN_PROJECTION_VERSION,
)
from .supervised_router_qa import (
    _confidence,
    materialize_verified_active_slots,
)
from .thinking import prediction_has_thinking_marker


SMALL_MODEL_ROUTER_QA_VERSION = (
    "tapbench.supervised_router_small_model_qa.v1"
)
SMALL_MODEL_ROUTER_QA_SYSTEM_LABEL = (
    "small_general_model_plus_benchmark_supervised_intent_router_plus_278M_qa"
)
SMALL_MODEL_ROUTER_QA_METHODS = (
    "tap_r_supervised_router_small_model_slots_qa_all",
    "tap_r_supervised_router_small_model_slots_qa_dev95",
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


def run_small_model_router_qa_action(
    *,
    case: dict[str, Any],
    ranking_row: dict[str, Any],
    verifier_index: dict[tuple[str, str, str], dict[str, Any]],
    method: str,
    dev_threshold: float,
    endpoint: str,
    max_tokens: int,
    seed: int,
    request_fn: RequestFn,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if method not in SMALL_MODEL_ROUTER_QA_METHODS:
        raise ValueError(f"unknown small-model router QA method: {method}")
    started = time.perf_counter()
    case_id = str(case["case_id"])
    request = user_request_text(case.get("messages", []))
    tools = list(case.get("tools", []))
    ranking = validate_external_ranking_row(
        ranking_row,
        case_id=case_id,
        request_text=request,
        tools=tools,
        schema_version="tapbench.supervised_tool_ranking.v1",
        retriever_version="tapbench.massive_supervised_intent_router.v1",
        retriever_model_id="hashed_tfidf_nearest_centroid",
        retriever_revision="massive_v1.1_train_only",
    )
    confidence = _confidence(ranking)
    router_selected = method.endswith("_all") or confidence >= dev_threshold
    selected_tool = str(ranking[0]["tool"])
    metadata: dict[str, Any] = {
        "small_model_router_qa_version": SMALL_MODEL_ROUTER_QA_VERSION,
        "qa_evidence_controller_version": SMALL_MODEL_ROUTER_QA_VERSION,
        "qa_evidence_system_label": SMALL_MODEL_ROUTER_QA_SYSTEM_LABEL,
        "router_training_split": "official_MASSIVE_v1.1_train",
        "router_threshold_selection_split": "official_MASSIVE_v1.1_dev",
        "router_confidence": confidence,
        "router_dev95_threshold": dev_threshold,
        "router_selected": router_selected,
        "retrieved_tools": [str(row["tool"]) for row in ranking],
        "retrieval_top1": selected_tool,
        "selected_tool": selected_tool,
        "ranking_sha256": ranking_row["ranking_sha256"],
        "qa_verifier_version": EXTRACTIVE_QA_VERIFIER_VERSION,
        "qa_verifier_question_version": EXTRACTIVE_QA_QUESTION_VERSION,
        "qa_verifier_model_id": EXTRACTIVE_QA_MODEL_ID,
        "qa_verifier_model_revision": EXTRACTIVE_QA_MODEL_REVISION,
        "qa_verifier_backend": "huggingface_transformers_cpu",
        "qa_verifier_dtype": "float32",
        "small_model_in_path": True,
        "small_model_role": "active_slot_selection_only",
        "small_model_supplies_argument_values": False,
        "verifier_supplies_exact_source_values_only": True,
        "call_only_tool_election": True,
        "no_call_election_option": False,
        "semantic_tool_labels": True,
        "semantic_slot_labels": True,
        "active_slot_policy": "small_model_explicit_evidence_gate",
        "active_slot_selections": [],
        "verifier_decisions": [],
    }
    if not router_selected:
        metadata.update(
            {
                "finish_reason": "not_applicable",
                "generation_calls": 0,
                "controller_stage_failure": "router_below_dev95_threshold",
                "proposal_admitted": False,
                "risk_gate_passed": False,
                "action_risk_score": 1.0,
                "active_slots_pre_verifier": [],
                "active_slots_post_verifier": [],
                "qa_verifier_rows_consulted": 0,
                "qa_verifier_null_count": 0,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return _non_call(
            "refuse", "supervised router below dev-set precision threshold"
        ), metadata

    by_name = {_tool_name(tool): tool for tool in tools}
    tool = by_name[selected_tool]
    slots, _ = _public_slots(tool, "forward")
    calls: list[dict[str, Any]] = []
    if slots:
        raw, response = request_fn(
            endpoint,
            _active_slot_messages(request, tool, slots),
            response_schema=_active_slot_schema(slots),
            max_tokens=min(max_tokens, 160),
            temperature=0.0,
            seed=seed,
        )
        calls.append(response)
        active_slots, validation = validate_active_slots(raw, tool=tool)
        metadata["active_slot_selections"].append(
            {
                "status": validation.get("status"),
                "active_slots": active_slots,
                "required_slots_added": validation.get(
                    "required_slots_added", []
                ),
            }
        )
        if active_slots is None:
            metadata.update(_merge_metadata(calls))
            metadata.update(
                {
                    "controller_stage_failure": (
                        f"qa_active_{validation.get('status')}"
                    ),
                    "proposal_admitted": False,
                    "risk_gate_passed": False,
                    "action_risk_score": 1.0,
                    "active_slots_pre_verifier": [],
                    "active_slots_post_verifier": [],
                    "qa_verifier_rows_consulted": 0,
                    "qa_verifier_null_count": 0,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return _non_call(
                "refuse", "active-slot selection failed validation"
            ), metadata
    else:
        active_slots = []
        metadata.update(
            {
                "finish_reason": "not_applicable",
                "generation_calls": 0,
            }
        )
    if calls:
        metadata.update(_merge_metadata(calls))
    action, metadata = materialize_verified_active_slots(
        case=case,
        ranking_row=ranking_row,
        verifier_index=verifier_index,
        selected_tool=selected_tool,
        active_slots=active_slots,
        metadata=metadata,
    )
    metadata["elapsed_seconds"] = time.perf_counter() - started
    return action, metadata


def run_files(
    *,
    cases_path: str | Path,
    rankings_path: str | Path,
    verifier_path: str | Path,
    router_report_path: str | Path,
    output_path: str | Path,
    timings_path: str | Path,
    manifest_path: str | Path,
    endpoint: str,
    model_id: str,
    model_key: str,
    model_artifact: str | Path,
    chat_template: str,
    methods: tuple[str, ...] = SMALL_MODEL_ROUTER_QA_METHODS,
    max_tokens: int = 384,
    seed: int = 1,
    request_fn: RequestFn = preflight_schema_request,
    protocol_path: str | Path | None = None,
) -> dict[str, Any]:
    unknown = set(methods) - set(SMALL_MODEL_ROUTER_QA_METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    cases = read_jsonl(cases_path)
    ranking_rows = read_jsonl(rankings_path)
    rankings = {str(row["case_id"]): row for row in ranking_rows}
    if len(rankings) != len(ranking_rows):
        raise ValueError("router ranking artifact contains duplicate case IDs")
    verifier_rows = read_jsonl(verifier_path)
    failures = validate_extractive_qa_rows(verifier_rows)
    if failures:
        raise ValueError(f"QA verifier artifact failed validation: {failures[:3]}")
    verifier_index = index_verifier_rows(verifier_rows)
    router_report = json.loads(Path(router_report_path).read_text(encoding="utf-8"))
    threshold = router_report["threshold_selection"]["global"]["threshold"]
    if threshold is None:
        raise ValueError("router report has no dev-set 95%-precision threshold")
    ranking_artifact_sha256 = _sha256(rankings_path)
    verifier_artifact_sha256 = _sha256(verifier_path)

    predictions: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    for case in cases:
        case_id = str(case["case_id"])
        if case_id not in rankings:
            raise ValueError(f"missing router ranking for {case_id}")
        for method in methods:
            started = time.perf_counter()
            error: str | None = None
            try:
                action, metadata = run_small_model_router_qa_action(
                    case=case,
                    ranking_row=rankings[case_id],
                    verifier_index=verifier_index,
                    method=method,
                    dev_threshold=float(threshold),
                    endpoint=endpoint,
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
            metadata["qa_verifier_artifact_sha256"] = (
                verifier_artifact_sha256
            )
            row = {
                "case_id": case_id,
                "hypothesis_grid_id": case.get("hypothesis_grid_id"),
                "language": _language(case),
                "method": method,
                "model_id": model_id,
                "seed": seed,
                "prediction": action,
                "action_ir_normalized": True,
                "response_metadata": metadata,
                "runner_error": error,
                "backend": "llama.cpp",
                "quantization": "Q4_K_M",
                "chat_template": chat_template,
                "grammar_engine": R2A_GRAMMAR_ENGINE,
                "chat_parser": R2A_CHAT_PARSER,
                "inference_path": "supervised_router_then_small_model_slots_then_qa",
                "model_artifact": str(model_artifact),
                "thinking_mode": "off",
                "reasoning_budget": 0,
                "small_model_router_qa_version": SMALL_MODEL_ROUTER_QA_VERSION,
                "massive_runner_version": SMALL_MODEL_ROUTER_QA_VERSION,
                "qa_evidence_controller_version": (
                    SMALL_MODEL_ROUTER_QA_VERSION
                ),
                "qa_evidence_system_label": SMALL_MODEL_ROUTER_QA_SYSTEM_LABEL,
                "qa_verifier_version": EXTRACTIVE_QA_VERIFIER_VERSION,
                "qa_verifier_question_version": EXTRACTIVE_QA_QUESTION_VERSION,
                "qa_verifier_model_id": EXTRACTIVE_QA_MODEL_ID,
                "qa_verifier_model_revision": EXTRACTIVE_QA_MODEL_REVISION,
                "qa_verifier_backend": "huggingface_transformers_cpu",
                "qa_verifier_dtype": "float32",
                "qa_verifier_artifact_sha256": verifier_artifact_sha256,
                "retriever_version": rankings[case_id]["retriever_version"],
                "retriever_model_id": rankings[case_id]["retriever_model_id"],
                "retriever_revision": rankings[case_id]["retriever_revision"],
                "retriever_serialization_arm": "supervised_intent_labels",
                "retriever_k": rankings[case_id]["k"],
                "ranking_sha256": rankings[case_id]["ranking_sha256"],
                "ranking_artifact_sha256": ranking_artifact_sha256,
                "source_span_projection_version": SOURCE_SPAN_PROJECTION_VERSION,
                "source_span_certificate_version": SOURCE_SPAN_CERTIFICATE_VERSION,
                "action_risk_threshold": 0.05,
                "max_output_tokens": max_tokens,
                "finish_reason": metadata.get("finish_reason"),
            }
            row["thinking_marker_detected"] = prediction_has_thinking_marker(row)
            predictions.append(row)
            timings.append(
                {
                    "case_id": case_id,
                    "language": _language(case),
                    "method": method,
                    "model_key": model_key,
                    "model_id": model_id,
                    "elapsed_seconds": elapsed,
                    "generation_calls": metadata.get("generation_calls", 0),
                    "generated_tokens_per_second": metadata.get(
                        "generated_tokens_per_second"
                    ),
                    "completion_tokens": metadata.get("completion_tokens"),
                    "runner_error": error,
                    "controller_stage_failure": metadata.get(
                        "controller_stage_failure"
                    ),
                    "thinking_marker_detected": row[
                        "thinking_marker_detected"
                    ],
                }
            )
    write_jsonl(output_path, predictions)
    write_jsonl(timings_path, timings)
    manifest = {
        "schema_version": "tapbench.supervised_router_small_model_qa_manifest.v1",
        "runner_version": SMALL_MODEL_ROUTER_QA_VERSION,
        "analysis_status": "predeclared_post_result_development_only",
        "confirmation_authorized": False,
        "system_label": SMALL_MODEL_ROUTER_QA_SYSTEM_LABEL,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(Path(cases_path).resolve()),
        "cases_sha256": _sha256(cases_path),
        "rankings_path": str(Path(rankings_path).resolve()),
        "rankings_sha256": ranking_artifact_sha256,
        "verifier_path": str(Path(verifier_path).resolve()),
        "verifier_sha256": verifier_artifact_sha256,
        "router_report_path": str(Path(router_report_path).resolve()),
        "router_report_sha256": _sha256(router_report_path),
        "protocol_path": (
            str(Path(protocol_path).resolve()) if protocol_path else None
        ),
        "protocol_sha256": _sha256(protocol_path) if protocol_path else None,
        "model_key": model_key,
        "model_id": model_id,
        "model_artifact": str(Path(model_artifact).resolve()),
        "model_artifact_sha256": _sha256(model_artifact),
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "chat_template": chat_template,
        "grammar_engine": R2A_GRAMMAR_ENGINE,
        "chat_parser": R2A_CHAT_PARSER,
        "thinking_mode": "off",
        "reasoning_budget": 0,
        "context_tokens": CONTEXT_TOKENS,
        "max_output_tokens": max_tokens,
        "dev95_threshold": threshold,
        "methods": list(methods),
        "case_count": len(cases),
        "prediction_count": len(predictions),
        "actual_model_calls": sum(
            int(row["response_metadata"].get("generation_calls", 0))
            for row in predictions
        ),
        "runner_errors": sum(row["runner_error"] is not None for row in predictions),
        "thinking_markers": sum(
            bool(row["thinking_marker_detected"]) for row in predictions
        ),
        "output_path": str(Path(output_path).resolve()),
        "output_sha256": _sha256(output_path),
        "timings_path": str(Path(timings_path).resolve()),
        "timings_sha256": _sha256(timings_path),
    }
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the predeclared supervised-router/small-model-slot/QA design diagnostic."
        )
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--rankings", required=True)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--router-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timings", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-artifact", required=True)
    parser.add_argument("--chat-template", required=True)
    parser.add_argument("--methods", default=",".join(SMALL_MODEL_ROUTER_QA_METHODS))
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--protocol")
    args = parser.parse_args()
    manifest = run_files(
        cases_path=args.cases,
        rankings_path=args.rankings,
        verifier_path=args.verifier,
        router_report_path=args.router_report,
        output_path=args.output,
        timings_path=args.timings,
        manifest_path=args.manifest,
        endpoint=args.endpoint,
        model_id=args.model_id,
        model_key=args.model_key,
        model_artifact=args.model_artifact,
        chat_template=args.chat_template,
        methods=tuple(value for value in args.methods.split(",") if value),
        max_tokens=args.max_tokens,
        seed=args.seed,
        protocol_path=args.protocol,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

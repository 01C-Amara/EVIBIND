from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .eflrx import _non_call, _tool_name
from .extractive_candidates import user_request_text
from .extractive_qa_verifier import (
    EXTRACTIVE_QA_MODEL_ID,
    EXTRACTIVE_QA_MODEL_REVISION,
    EXTRACTIVE_QA_QUESTION_VERSION,
    EXTRACTIVE_QA_VERIFIER_VERSION,
)
from .io import read_jsonl, write_jsonl
from .multilingual_retriever import forbidden_paths
from .qa_evidence_controller import index_verifier_rows, validate_verifier_record, verifier_identity
from .retrieve_pointer import validate_external_ranking_row
from .semantic_surface_projection import _public_slots, materialize_surface_bindings
from .source_span_projection import (
    SOURCE_SPAN_CERTIFICATE_VERSION,
    SOURCE_SPAN_PROJECTION_VERSION,
    action_fingerprint,
)


ROUTER_QA_VERSION = "tapbench.supervised_router_qa.v1"
ROUTER_QA_SYSTEM_LABEL = "benchmark_supervised_intent_router_plus_278M_extractive_verifier"
ROUTER_QA_METHODS = (
    "tap_r_supervised_router_qa_all",
    "tap_r_supervised_router_qa_dev95",
    "tap_r_supervised_router_slot_knn_qa_all",
    "tap_r_supervised_router_slot_knn_qa_dev95",
)
SLOT_ROUTER_SYSTEM_LABEL = "benchmark_supervised_intent_and_slot_router_plus_278M_extractive_verifier"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _confidence(ranking: list[dict[str, Any]]) -> float:
    first = float(ranking[0]["cosine_score"])
    second = float(ranking[1]["cosine_score"]) if len(ranking) > 1 else first
    return first - second


def materialize_verified_active_slots(
    *,
    case: dict[str, Any],
    ranking_row: dict[str, Any],
    verifier_index: dict[tuple[str, str, str], dict[str, Any]],
    selected_tool: str,
    active_slots: list[str],
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize only source-backed values for an already selected tool/slot set."""
    case_id = str(case["case_id"])
    request = user_request_text(case.get("messages", []))
    tools = list(case.get("tools", []))
    language = str(
        case.get("metadata", {}).get("language")
        or case.get("factors", {}).get("language")
        or "unknown"
    )
    by_name = {_tool_name(tool): tool for tool in tools}
    tool = by_name[selected_tool]
    slots, by_surface = _public_slots(tool, "forward")
    public_slot_names = {str(slot["name"]) for slot in slots}
    if set(active_slots) - public_slot_names:
        raise ValueError(
            f"active slots outside public contract for {case_id}: "
            f"{sorted(set(active_slots) - public_slot_names)}"
        )
    metadata["active_slots_pre_verifier"] = active_slots
    bindings: dict[str, str] = {}
    verifier_hashes: list[str] = []
    for surface_name in active_slots:
        slot = by_surface[surface_name]
        record = verifier_index.get(
            verifier_identity(case_id, selected_tool, surface_name)
        )
        if record is None:
            metadata["controller_stage_failure"] = "qa_verifier_record_missing"
            return _non_call("refuse", "fixed verifier artifact is incomplete"), metadata
        failures = validate_verifier_record(
            record,
            case_id=case_id,
            request_text=request,
            language=language,
            ranking_sha256=str(ranking_row["ranking_sha256"]),
            tool=selected_tool,
            slot=slot,
        )
        if failures:
            metadata["controller_stage_failure"] = "qa_verifier_record_invalid"
            metadata["verifier_record_failures"] = failures
            return _non_call(
                "refuse", "fixed verifier record failed runtime validation"
            ), metadata
        verifier_hashes.append(str(record["row_sha256"]))
        if bool(record["admitted"]):
            bindings[surface_name] = str(record["answer"])
        metadata["verifier_decisions"].append(
            {
                "slot": surface_name,
                "status": record["status"],
                "admitted": bool(record["admitted"]),
                "answer": record["answer"],
                "span_id": record["span_id"],
                "non_null_margin": record["non_null_margin"],
                "row_sha256": record["row_sha256"],
            }
        )
    active_after = sorted(bindings)
    action, materialization = materialize_surface_bindings(
        {"bindings": bindings},
        active_slots=active_after,
        selected_tool=selected_tool,
        tool=tool,
        tools=tools,
        request_text=request,
        language=language,
    )
    if action is None:
        metadata["controller_stage_failure"] = (
            f"qa_materialize_{materialization.get('status')}"
        )
        return _non_call(
            "refuse", "verifier evidence failed public materialization"
        ), metadata
    metadata.update(
        {
            "controller_stage_failure": None,
            "proposal_admitted": True,
            "risk_gate_passed": True,
            "action_risk_score": 0.02,
            "active_slots_post_verifier": active_after,
            "qa_verifier_rows_consulted": len(active_slots),
            "qa_verifier_null_count": len(active_slots) - len(active_after),
            "qa_verifier_row_sha256": verifier_hashes,
            "evidence_certificates": materialization["certificates"],
            "selected_span_ids": materialization["selected_span_ids"],
            "selected_surface_values": materialization.get(
                "selected_surface_values", {}
            ),
            "span_catalog_sha256": materialization["span_catalog_sha256"],
            "slot_catalog_sha256": materialization["slot_catalog_sha256"],
            "materialized_action_sha256": action_fingerprint(action),
            "no_unconstrained_action_critical_tokens": True,
        }
    )
    return action, metadata


def materialize_router_qa_action(
    *,
    case: dict[str, Any],
    ranking_row: dict[str, Any],
    verifier_index: dict[tuple[str, str, str], dict[str, Any]],
    method: str,
    dev_threshold: float,
    slot_prediction_row: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if method not in ROUTER_QA_METHODS:
        raise ValueError(f"unknown supervised-router QA method: {method}")
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
    uses_slot_selector = "slot_knn" in method
    system_label = SLOT_ROUTER_SYSTEM_LABEL if uses_slot_selector else ROUTER_QA_SYSTEM_LABEL
    metadata: dict[str, Any] = {
        "finish_reason": "not_applicable",
        "generation_calls": 0,
        "router_qa_version": ROUTER_QA_VERSION,
        "qa_evidence_system_label": system_label,
        "router_training_split": "official_MASSIVE_v1.1_train",
        "router_threshold_selection_split": "official_MASSIVE_v1.1_dev",
        "router_confidence": confidence,
        "router_dev95_threshold": dev_threshold,
        "router_selected": method.endswith("_all") or confidence >= dev_threshold,
        "retrieved_tools": [str(row["tool"]) for row in ranking],
        "retrieval_top1": str(ranking[0]["tool"]),
        "selected_tool": str(ranking[0]["tool"]),
        "ranking_sha256": ranking_row["ranking_sha256"],
        "qa_verifier_version": EXTRACTIVE_QA_VERIFIER_VERSION,
        "qa_verifier_question_version": EXTRACTIVE_QA_QUESTION_VERSION,
        "qa_verifier_model_id": EXTRACTIVE_QA_MODEL_ID,
        "qa_verifier_model_revision": EXTRACTIVE_QA_MODEL_REVISION,
        "qa_verifier_backend": "huggingface_transformers_cpu",
        "qa_verifier_dtype": "float32",
        "small_model_in_path": False,
        "small_model_supplies_argument_values": False,
        "verifier_supplies_exact_source_values_only": True,
        "verifier_decisions": [],
    }
    if not metadata["router_selected"]:
        metadata.update(
            {
                "controller_stage_failure": "router_below_dev95_threshold",
                "proposal_admitted": False,
                "risk_gate_passed": False,
                "action_risk_score": 1.0,
                "active_slots_pre_verifier": [],
                "active_slots_post_verifier": [],
                "qa_verifier_rows_consulted": 0,
                "qa_verifier_null_count": 0,
            }
        )
        return _non_call("refuse", "supervised router below dev-set precision threshold"), metadata

    by_name = {_tool_name(tool): tool for tool in tools}
    selected_tool = str(ranking[0]["tool"])
    tool = by_name[selected_tool]
    slots, by_surface = _public_slots(tool, "forward")
    public_slot_names = {str(slot["name"]) for slot in slots}
    if uses_slot_selector:
        if not isinstance(slot_prediction_row, dict):
            raise ValueError(f"slot-selector method missing row for {case_id}")
        leaks = forbidden_paths(slot_prediction_row)
        if leaks:
            raise ValueError(f"slot-selector row contains scorer-only fields: {leaks}")
        expected = {
            "schema_version": "tapbench.massive_supervised_slot_knn.v1",
            "case_id": case_id,
            "predicted_tool": selected_tool,
        }
        failures = [f"{key}_mismatch" for key, value in expected.items() if slot_prediction_row.get(key) != value]
        supplied_slots = slot_prediction_row.get("active_slots")
        if not isinstance(supplied_slots, list):
            failures.append("active_slots_not_array")
            supplied_slots = []
        active_slots = sorted({str(value) for value in supplied_slots})
        if set(active_slots) - public_slot_names:
            failures.append("active_slots_outside_public_contract")
        if failures:
            raise ValueError(f"invalid slot-selector row for {case_id}: {failures}")
        metadata["active_slot_policy"] = "massive_train_dev_supervised_knn"
        metadata["slot_selector_version"] = slot_prediction_row["schema_version"]
        metadata["slot_selector_k"] = slot_prediction_row.get("k")
        metadata["slot_selector_vote_threshold"] = slot_prediction_row.get("vote_threshold")
        metadata["slot_selector_neighbor_ids"] = slot_prediction_row.get("neighbor_ids", [])
    else:
        active_slots = sorted(public_slot_names)
        metadata["active_slot_policy"] = "all_public_slots"
    return materialize_verified_active_slots(
        case=case,
        ranking_row=ranking_row,
        verifier_index=verifier_index,
        selected_tool=selected_tool,
        active_slots=active_slots,
        metadata=metadata,
    )


def materialize_files(
    *,
    cases_path: str | Path,
    rankings_path: str | Path,
    verifier_path: str | Path,
    router_report_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    slot_predictions_path: str | Path | None = None,
) -> dict[str, Any]:
    cases = read_jsonl(cases_path)
    ranking_rows = read_jsonl(rankings_path)
    rankings = {str(row["case_id"]): row for row in ranking_rows}
    if len(rankings) != len(ranking_rows):
        raise ValueError("router ranking artifact contains duplicate case IDs")
    verifier_index = index_verifier_rows(read_jsonl(verifier_path))
    slot_predictions: dict[str, dict[str, Any]] = {}
    if slot_predictions_path is not None:
        slot_rows = read_jsonl(slot_predictions_path)
        slot_predictions = {str(row["case_id"]): row for row in slot_rows}
        if len(slot_predictions) != len(slot_rows):
            raise ValueError("slot-selector artifact contains duplicate case IDs")
    router_report = json.loads(Path(router_report_path).read_text(encoding="utf-8"))
    threshold = router_report["threshold_selection"]["global"]["threshold"]
    if threshold is None:
        raise ValueError("router report did not establish a dev-set 95%-precision threshold")
    ranking_artifact_sha256 = _sha256(rankings_path)
    verifier_artifact_sha256 = _sha256(verifier_path)
    methods = (
        ("tap_r_supervised_router_slot_knn_qa_all", "tap_r_supervised_router_slot_knn_qa_dev95")
        if slot_predictions_path is not None
        else ("tap_r_supervised_router_qa_all", "tap_r_supervised_router_qa_dev95")
    )
    outputs = []
    for case in cases:
        case_id = str(case["case_id"])
        ranking_row = rankings.get(case_id)
        if ranking_row is None:
            raise ValueError(f"missing router ranking for {case_id}")
        for method in methods:
            action, response_metadata = materialize_router_qa_action(
                case=case,
                ranking_row=ranking_row,
                verifier_index=verifier_index,
                method=method,
                dev_threshold=float(threshold),
                slot_prediction_row=slot_predictions.get(case_id),
            )
            outputs.append(
                {
                    "case_id": case_id,
                    "hypothesis_grid_id": case.get("hypothesis_grid_id"),
                    "method": method,
                    "model_id": (
                        "MASSIVE-supervised-intent-slot-router+278M-QA"
                        if slot_predictions_path is not None
                        else "MASSIVE-supervised-router+278M-QA"
                    ),
                    "model_artifact": (
                        "router_npz_plus_slot_knn_plus_mdeberta_v3_base_squad2"
                        if slot_predictions_path is not None
                        else "router_npz_plus_mdeberta_v3_base_squad2"
                    ),
                    "seed": 1,
                    "backend": "deterministic_supervised_router_plus_hf_qa",
                    "quantization": "not_applicable",
                    "chat_template": "not_applicable",
                    "chat_parser": "not_applicable",
                    "grammar_engine": "not_applicable",
                    "inference_path": "supervised_router_top1_then_extractive_qa",
                    "thinking_mode": "not_applicable",
                    "reasoning_budget": 0,
                    "thinking_marker_detected": False,
                    "prediction": action,
                    "response_metadata": response_metadata,
                    "runner_error": None,
                    "massive_runner_version": ROUTER_QA_VERSION,
                    "qa_evidence_controller_version": ROUTER_QA_VERSION,
                    "qa_evidence_system_label": response_metadata["qa_evidence_system_label"],
                    "qa_verifier_version": EXTRACTIVE_QA_VERIFIER_VERSION,
                    "qa_verifier_question_version": EXTRACTIVE_QA_QUESTION_VERSION,
                    "qa_verifier_model_id": EXTRACTIVE_QA_MODEL_ID,
                    "qa_verifier_model_revision": EXTRACTIVE_QA_MODEL_REVISION,
                    "qa_verifier_backend": "huggingface_transformers_cpu",
                    "qa_verifier_dtype": "float32",
                    "qa_verifier_artifact_sha256": verifier_artifact_sha256,
                    "retriever_version": ranking_row["retriever_version"],
                    "retriever_model_id": ranking_row["retriever_model_id"],
                    "retriever_revision": ranking_row["retriever_revision"],
                    "retriever_serialization_arm": "supervised_intent_labels",
                    "retriever_k": ranking_row["k"],
                    "ranking_sha256": ranking_row["ranking_sha256"],
                    "ranking_artifact_sha256": ranking_artifact_sha256,
                    "source_span_projection_version": SOURCE_SPAN_PROJECTION_VERSION,
                    "source_span_certificate_version": SOURCE_SPAN_CERTIFICATE_VERSION,
                    "action_risk_threshold": 0.05,
                }
            )
    write_jsonl(output_path, outputs)
    manifest = {
        "schema_version": "tapbench.supervised_router_qa_manifest.v1",
        "controller_version": ROUTER_QA_VERSION,
        "analysis_status": "post_result_exploratory_design_only",
        "confirmation_authorized": False,
        "system_label": SLOT_ROUTER_SYSTEM_LABEL if slot_predictions_path is not None else ROUTER_QA_SYSTEM_LABEL,
        "cases_path": str(Path(cases_path).resolve()),
        "cases_sha256": _sha256(cases_path),
        "rankings_path": str(Path(rankings_path).resolve()),
        "rankings_sha256": ranking_artifact_sha256,
        "verifier_path": str(Path(verifier_path).resolve()),
        "verifier_sha256": verifier_artifact_sha256,
        "router_report_path": str(Path(router_report_path).resolve()),
        "router_report_sha256": _sha256(router_report_path),
        "dev95_threshold": threshold,
        "methods": list(methods),
        "case_count": len(cases),
        "prediction_count": len(outputs),
        "output_path": str(Path(output_path).resolve()),
        "output_sha256": _sha256(output_path),
    }
    if slot_predictions_path is not None:
        manifest["slot_predictions_path"] = str(Path(slot_predictions_path).resolve())
        manifest["slot_predictions_sha256"] = _sha256(slot_predictions_path)
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the post-result supervised-router QA design diagnostic.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--rankings", required=True)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--router-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--slot-predictions")
    args = parser.parse_args()
    manifest = materialize_files(
        cases_path=args.cases,
        rankings_path=args.rankings,
        verifier_path=args.verifier,
        router_report_path=args.router_report,
        output_path=args.output,
        manifest_path=args.manifest,
        slot_predictions_path=args.slot_predictions,
    )
    print(json.dumps({"prediction_count": manifest["prediction_count"], "dev95_threshold": manifest["dev95_threshold"]}))


if __name__ == "__main__":
    main()

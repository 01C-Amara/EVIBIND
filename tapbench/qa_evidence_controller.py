from __future__ import annotations

import hashlib
import time
from typing import Any

from .eflrx import ACTION_RISK_THRESHOLD, RequestFn, _merge_metadata, _non_call, _tool_name
from .extractive_candidates import user_request_text
from .extractive_qa_verifier import (
    EXTRACTIVE_QA_MODEL_ID,
    EXTRACTIVE_QA_MODEL_REVISION,
    EXTRACTIVE_QA_QUESTION_VERSION,
    EXTRACTIVE_QA_VERIFIER_VERSION,
    verifier_question,
)
from .retrieve_pointer import validate_ranking_row
from .semantic_surface_projection import (
    SEMANTIC_SURFACE_VERSION,
    _active_slot_messages,
    _active_slot_schema,
    _json_sha256,
    _public_slots,
    _tool_catalog,
    _tool_messages,
    _tool_schema,
    materialize_surface_bindings,
    validate_active_slots,
)
from .source_span_projection import (
    SOURCE_SPAN_CERTIFICATE_VERSION,
    SOURCE_SPAN_PROJECTION_VERSION,
    action_fingerprint,
    source_span_catalog,
)


QA_EVIDENCE_CONTROLLER_VERSION = "tapbench.qa_evidence_controller.v2"
QA_EVIDENCE_SYSTEM_LABEL = (
    "small_general_model_plus_278M_extractive_verifier"
)
QA_EVIDENCE_CONDITIONS = (
    "tap_r_qa_all_slots_single",
    "tap_r_qa_active_slots_single",
    "tap_r_qa_active_slots_consensus",
)


def verifier_identity(
    case_id: str, tool: str, surface_name: str
) -> tuple[str, str, str]:
    return str(case_id), str(tool), str(surface_name)


def index_verifier_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        identity = verifier_identity(
            str(row.get("case_id")),
            str(row.get("tool")),
            str(row.get("surface_name")),
        )
        if identity in output:
            raise ValueError(f"duplicate verifier identity: {identity}")
        output[identity] = row
    return output


def validate_verifier_record(
    record: dict[str, Any],
    *,
    case_id: str,
    request_text: str,
    language: str,
    ranking_sha256: str,
    tool: str,
    slot: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    lattice = source_span_catalog(request_text, language)
    expected = {
        "verifier_version": EXTRACTIVE_QA_VERIFIER_VERSION,
        "question_version": EXTRACTIVE_QA_QUESTION_VERSION,
        "model_id": EXTRACTIVE_QA_MODEL_ID,
        "model_revision": EXTRACTIVE_QA_MODEL_REVISION,
        "case_id": case_id,
        "language": language,
        "request_sha256": hashlib.sha256(request_text.encode("utf-8")).hexdigest(),
        "span_catalog_sha256": lattice["catalog_sha256"],
        "ranking_sha256": ranking_sha256,
        "tool": tool,
        "slot_id": str(slot["slot_id"]),
        "surface_name": str(slot["name"]),
        "required": bool(slot["required"]),
        "question": verifier_question(str(slot["name"])),
        "input_truncated": False,
        "gold_loaded": False,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            failures.append(f"{field}_mismatch")
    admitted = bool(record.get("admitted"))
    if admitted:
        answer = record.get("answer")
        answer_span = record.get("answer_span")
        span_id = record.get("span_id")
        matches = [
            row
            for row in lattice["spans"]
            if row.get("source_text") == answer
            and row.get("source_span") == answer_span
            and row.get("span_id") == span_id
        ]
        if not matches:
            failures.append("admitted_answer_not_in_source_lattice")
        if record.get("status") != "admitted":
            failures.append("admitted_status_mismatch")
    else:
        if record.get("answer") is not None or record.get("span_id") is not None:
            failures.append("null_record_contains_admitted_answer")
    return failures


def _fail_closed(
    metadata: dict[str, Any],
    calls: list[dict[str, Any]],
    *,
    stage: str,
    reason: str,
    started: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if calls:
        metadata.update(_merge_metadata(calls))
    else:
        metadata.update({"generation_calls": 0, "finish_reason": "not_applicable"})
    metadata.update(
        {
            "controller_stage_failure": stage,
            "action_risk_score": 1.0,
            "risk_gate_passed": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return _non_call("refuse", reason), metadata


def run_qa_evidence_resolution(
    *,
    case_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    language: str,
    ranking_row: dict[str, Any],
    ranking_artifact_sha256: str,
    verifier_index: dict[tuple[str, str, str], dict[str, Any]],
    verifier_artifact_sha256: str,
    endpoint: str,
    condition: str,
    max_tokens: int,
    seed: int,
    request_fn: RequestFn,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if condition not in QA_EVIDENCE_CONDITIONS:
        raise ValueError(f"unknown QA-evidence condition: {condition}")
    started = time.perf_counter()
    text = user_request_text(messages)
    ranking = validate_ranking_row(
        ranking_row,
        case_id=case_id,
        request_text=text,
        tools=tools,
    )
    by_name = {_tool_name(tool): tool for tool in tools}
    ranking_names = [str(row["tool"]) for row in ranking]
    retrieved = [by_name[name] for name in ranking_names]
    consensus = condition == "tap_r_qa_active_slots_consensus"
    view_orders = ("forward", "reverse") if consensus else ("forward",)
    metadata: dict[str, Any] = {
        "qa_evidence_controller_version": QA_EVIDENCE_CONTROLLER_VERSION,
        "qa_evidence_system_label": QA_EVIDENCE_SYSTEM_LABEL,
        "semantic_surface_dependency_version": SEMANTIC_SURFACE_VERSION,
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
        "retriever_serialization_arm": ranking_row["serialization_arm"],
        "retriever_k": ranking_row["k"],
        "ranking_sha256": ranking_row["ranking_sha256"],
        "ranking_artifact_sha256": ranking_artifact_sha256,
        "retrieved_tools": ranking_names,
        "retrieval_top1": ranking_names[0],
        "source_span_projection_version": SOURCE_SPAN_PROJECTION_VERSION,
        "source_span_certificate_version": SOURCE_SPAN_CERTIFICATE_VERSION,
        "call_only_tool_election": True,
        "no_call_election_option": False,
        "semantic_tool_labels": True,
        "semantic_slot_labels": True,
        "small_model_supplies_argument_values": False,
        "verifier_supplies_exact_source_values_only": True,
        "tool_elections": [],
        "active_slot_selections": [],
        "verifier_decisions": [],
        "generation_calls": 0,
        "action_risk_threshold": ACTION_RISK_THRESHOLD,
    }
    calls: list[dict[str, Any]] = []

    selected_tools: list[str] = []
    for index, order in enumerate(view_orders):
        catalog = _tool_catalog(retrieved, order)
        raw, response = request_fn(
            endpoint,
            _tool_messages(text, catalog),
            response_schema=_tool_schema(catalog),
            max_tokens=min(max_tokens, 96),
            temperature=0.0,
            seed=seed + index * 100003,
        )
        calls.append(response)
        selected = str(raw.get("tool") or "") if isinstance(raw, dict) else ""
        selected_tools.append(selected)
        metadata["tool_elections"].append(
            {
                "order": order,
                "selected": selected,
                "catalog_sha256": _json_sha256(catalog),
            }
        )
    if any(value not in by_name for value in selected_tools) or len(
        set(selected_tools)
    ) != 1:
        metadata["tool_agreement"] = False
        return _fail_closed(
            metadata,
            calls,
            stage="qa_tool_disagreement",
            reason="semantic tool views disagreed",
            started=started,
        )
    selected_tool = selected_tools[0]
    metadata["tool_agreement"] = True
    metadata["selected_tool"] = selected_tool
    metadata["retriever_election_agreement"] = selected_tool == ranking_names[0]
    tool = by_name[selected_tool]
    forward_slots, by_surface = _public_slots(tool, "forward")

    if condition == "tap_r_qa_all_slots_single" or not forward_slots:
        active_slots = sorted(str(slot["name"]) for slot in forward_slots)
        metadata["active_slot_policy"] = "all_public_slots"
        metadata["active_slot_agreement"] = True
        metadata["active_slots_pre_verifier"] = active_slots
    else:
        active_views: list[list[str]] = []
        for index, order in enumerate(view_orders):
            slots, _ = _public_slots(tool, order)
            raw, response = request_fn(
                endpoint,
                _active_slot_messages(text, tool, slots),
                response_schema=_active_slot_schema(slots),
                max_tokens=min(max_tokens, 160),
                temperature=0.0,
                seed=seed + 300001 + index * 100003,
            )
            calls.append(response)
            active, validation = validate_active_slots(raw, tool=tool)
            metadata["active_slot_selections"].append(
                {
                    "order": order,
                    "status": validation.get("status"),
                    "active_slots": active,
                    "required_slots_added": validation.get(
                        "required_slots_added", []
                    ),
                }
            )
            if active is None:
                return _fail_closed(
                    metadata,
                    calls,
                    stage=f"qa_active_{validation.get('status')}",
                    reason="active-slot selection failed validation",
                    started=started,
                )
            active_views.append(active)
        if len({_json_sha256(value) for value in active_views}) != 1:
            metadata["active_slot_agreement"] = False
            return _fail_closed(
                metadata,
                calls,
                stage="qa_active_slot_disagreement",
                reason="counterbalanced active-slot selections disagreed",
                started=started,
            )
        active_slots = active_views[0]
        metadata["active_slot_policy"] = "small_model_explicit_evidence_gate"
        metadata["active_slot_agreement"] = True
        metadata["active_slots_pre_verifier"] = active_slots

    verifier_started = time.perf_counter()
    bindings: dict[str, str] = {}
    verifier_row_hashes: list[str] = []
    for surface_name in active_slots:
        slot = by_surface[surface_name]
        identity = verifier_identity(case_id, selected_tool, surface_name)
        record = verifier_index.get(identity)
        if record is None:
            return _fail_closed(
                metadata,
                calls,
                stage="qa_verifier_record_missing",
                reason="fixed verifier artifact is incomplete",
                started=started,
            )
        failures = validate_verifier_record(
            record,
            case_id=case_id,
            request_text=text,
            language=language,
            ranking_sha256=str(ranking_row["ranking_sha256"]),
            tool=selected_tool,
            slot=slot,
        )
        if failures:
            metadata["verifier_record_failures"] = failures
            return _fail_closed(
                metadata,
                calls,
                stage="qa_verifier_record_invalid",
                reason="fixed verifier record failed runtime validation",
                started=started,
            )
        verifier_row_hashes.append(str(record["row_sha256"]))
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
    verifier_lookup_seconds = time.perf_counter() - verifier_started
    active_after = sorted(bindings)
    action, materialization = materialize_surface_bindings(
        {"bindings": bindings},
        active_slots=active_after,
        selected_tool=selected_tool,
        tool=tool,
        tools=tools,
        request_text=text,
        language=language,
    )
    if action is None:
        return _fail_closed(
            metadata,
            calls,
            stage=f"qa_materialize_{materialization.get('status')}",
            reason="verifier evidence failed public materialization",
            started=started,
        )
    risk = 0.03 if not consensus else 0.01
    metadata.update(_merge_metadata(calls))
    metadata.update(
        {
            "controller_stage_failure": None,
            "proposal_admitted": True,
            "active_slots_post_verifier": active_after,
            "qa_verifier_rows_consulted": len(active_slots),
            "qa_verifier_null_count": len(active_slots) - len(active_after),
            "qa_verifier_row_sha256": verifier_row_hashes,
            "qa_verifier_lookup_seconds": verifier_lookup_seconds,
            "evidence_certificates": materialization["certificates"],
            "selected_span_ids": materialization["selected_span_ids"],
            "selected_surface_values": materialization.get(
                "selected_surface_values", {}
            ),
            "span_catalog_sha256": materialization["span_catalog_sha256"],
            "slot_catalog_sha256": materialization["slot_catalog_sha256"],
            "materialized_action_sha256": action_fingerprint(action),
            "no_unconstrained_action_critical_tokens": True,
            "action_risk_score": risk,
            "risk_gate_passed": risk <= ACTION_RISK_THRESHOLD,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return action, metadata

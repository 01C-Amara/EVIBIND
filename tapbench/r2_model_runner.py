from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .contract_solver import resolve_pointer_contract
from .deployable_resolution import augment_lattice_with_typed_programs
from .evidence_contract import build_candidate_lattice, certified_candidates, request_text
from .io import read_jsonl, write_jsonl, write_yaml
from .runner import (
    _parse_llama_text,
    _request_llama_json,
    render_chat_messages,
)
from .thinking import prediction_has_thinking_marker
from .tier_b_verifier import FrozenTierBVerifier


R2A_MODEL_RUNNER_VERSION = "tapbench.r2a_model_runner.v7"
R2A_GRAMMAR_ENGINE = "llama.cpp_raw_completion_json_schema_gbnf"
R2A_CHAT_PARSER = "bypassed_after_native_template"
R2A_MODEL_CONDITIONS = (
    "r2_literal_generation",
    "r2_pointer_unrestricted",
    "r2_pointer_tep_tier_ab",
)


def _runtime(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": list(case.get("messages", [])),
        "tools": list(case.get("tools", [])),
        "tool_aliases": dict(case.get("tool_aliases", {})),
        "argument_aliases": dict(case.get("argument_aliases", {})),
        "dialogue_state": dict(case.get("dialogue_state", {})),
        "reference_context": dict(case.get("reference_context", {})),
    }


def _candidate_view(candidate: dict[str, Any]) -> dict[str, Any]:
    programs = candidate.get("typed_evidence_programs", [])
    program = programs[0].get("program", {}) if programs else {}
    return {
        "candidate_id": candidate.get("candidate_id"),
        "source_text": candidate.get("source_text"),
        "source_kind": candidate.get("source_kind"),
        "transform": program.get("op") or candidate.get("transform"),
        "role_label": candidate.get("role_label"),
        "acceptance_tier": candidate.get("acceptance_tier"),
    }


def _pointer_catalog(lattice: dict[str, Any], *, unrestricted: bool) -> list[dict[str, Any]]:
    tools = []
    for tool_name, tool in lattice.get("tools", {}).items():
        slots = {}
        for slot, slot_row in tool.get("slots", {}).items():
            candidates = list(slot_row.get("candidates", [])) if unrestricted else certified_candidates(slot_row)
            slots[slot] = {
                "required": bool(slot_row.get("required")),
                "candidates": [_candidate_view(candidate) for candidate in candidates],
            }
        tools.append({
            "tool_id": tool["tool_id"],
            "tool": tool_name,
            "description": tool.get("description"),
            "slots": slots,
        })
    return tools


def _pointer_messages(case: dict[str, Any], catalog: list[dict[str, Any]], condition: str) -> list[dict[str, str]]:
    request = request_text(case.get("messages", []))
    policy = (
        "Candidate IDs may include ambiguous or contradicted spans. Select only values actively asserted for the requested slot."
        if condition == "r2_pointer_unrestricted"
        else "Every listed candidate passed evidence checks. Select the candidate that satisfies the request; omit no required slot."
    )
    return [
        {
            "role": "system",
            "content": (
                "Return only one compact pointer JSON object with keys mode, tool_id, arguments, and payload. "
                "The mode value must be the literal string call. Arguments maps each slot directly to one integer "
                "candidate_id, for example {\"mode\":\"call\",\"tool_id\":0,\"arguments\":{\"date\":1},\"payload\":{}}. "
                "Do not wrap candidate IDs, copy API values, duplicate arguments in payload, or output reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Condition: {condition}\nPolicy: {policy}\nRequest: {request}\n"
                f"Pointer catalog: {json.dumps(catalog, sort_keys=True)}\nReturn the pointer JSON now."
            ),
        },
    ]


def _bounded_value_schema(source: dict[str, Any]) -> dict[str, Any]:
    kind = source.get("type")
    if kind == "string":
        schema: dict[str, Any] = {"type": "string", "maxLength": min(int(source.get("maxLength", 64)), 64)}
    elif kind in {"integer", "number", "boolean"}:
        schema = {"type": kind}
    elif kind == "array":
        schema = {
            "type": "array",
            "items": _bounded_value_schema(source.get("items", {})),
            "maxItems": min(int(source.get("maxItems", 16)), 16),
        }
    elif kind == "object":
        schema = {
            "type": "object",
            "properties": {
                str(key): _bounded_value_schema(value)
                for key, value in source.get("properties", {}).items()
            },
            "required": list(source.get("required", [])),
            "additionalProperties": False,
        }
    else:
        schema = {"type": "string", "maxLength": 64}
    if isinstance(source.get("enum"), list):
        schema["enum"] = list(source["enum"])
    return schema


def _literal_call_schema(case: dict[str, Any]) -> dict[str, Any]:
    tools = [tool for tool in case.get("tools", []) if tool.get("name") is not None]
    tool_names = sorted(str(tool["name"]) for tool in tools)
    argument_properties: dict[str, dict[str, Any]] = {}
    required_sets: list[set[str]] = []
    for tool in tools:
        parameters = tool.get("parameters", {})
        required_sets.append({str(slot) for slot in parameters.get("required", [])})
        for slot, slot_schema in parameters.get("properties", {}).items():
            argument_properties.setdefault(str(slot), _bounded_value_schema(slot_schema))
    required_arguments = sorted(set.intersection(*required_sets)) if required_sets else []
    return {
        "type": "object",
        "properties": {
            "mode": {"const": "call"},
            "tool": {"type": "string", "enum": tool_names},
            "arguments": {
                "type": "object",
                "properties": argument_properties,
                "required": required_arguments,
                "additionalProperties": False,
            },
            "payload": {"type": "object", "additionalProperties": False},
        },
        "required": ["mode", "tool", "arguments", "payload"],
        "additionalProperties": False,
    }


def _pointer_json_schema(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    tool_ids = sorted(int(tool["tool_id"]) for tool in catalog)
    slot_names = sorted({
        str(slot)
        for tool in catalog
        for slot in tool.get("slots", {})
    })
    return {
        "type": "object",
        "properties": {
            "mode": {"const": "call"},
            "tool_id": {"type": "integer", "enum": tool_ids},
            "arguments": {
                "type": "object",
                "properties": {
                    slot: {"type": "integer", "minimum": 0}
                    for slot in slot_names
                },
                "additionalProperties": False,
            },
            "payload": {"type": "object", "additionalProperties": False},
        },
        "required": ["mode", "tool_id", "arguments", "payload"],
        "additionalProperties": False,
    }



def _request_schema_json(
    endpoint: str,
    messages: list[dict[str, str]],
    *,
    response_schema: dict[str, Any],
    max_tokens: int,
    temperature: float,
    seed: int,
    stop_sequences: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = endpoint.rstrip("/")
    template_payload = {
        "messages": messages,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    template_request = urllib.request.Request(
        base + "/apply-template",
        data=json.dumps(template_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    template_data, template_retries = _request_llama_json(template_request, endpoint)
    prompt = str(template_data.get("prompt", ""))
    if not prompt:
        raise RuntimeError(f"llama-server apply-template returned an empty prompt for {endpoint}")

    completion_payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "json_schema": response_schema,
        "cache_prompt": False,
        "stream": False,
    }
    if stop_sequences:
        completion_payload["stop"] = list(stop_sequences)
    completion_request = urllib.request.Request(
        base + "/completion",
        data=json.dumps(completion_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    data, completion_retries = _request_llama_json(completion_request, endpoint)
    content = str(data.get("content", ""))
    parsed = _parse_llama_text(content)
    timings = data.get("timings", {}) if isinstance(data.get("timings"), dict) else {}
    prompt_tokens = int(timings.get("prompt_n") or data.get("tokens_evaluated") or 0)
    completion_tokens = int(timings.get("predicted_n") or 0)
    stop_type = str(data.get("stop_type", "none"))
    finish_reason = "length" if stop_type == "limit" else "stop" if stop_type in {"eos", "word"} else stop_type
    metadata = {
        "retry_count": template_retries + completion_retries,
        "template_retry_count": template_retries,
        "completion_retry_count": completion_retries,
        "raw_text": content,
        "finish_reason": finish_reason,
        "stop_type": stop_type,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_ms": timings.get("prompt_ms"),
        "generation_ms": timings.get("predicted_ms"),
        "generated_tokens_per_second": timings.get("predicted_per_second"),
        "context_truncated": bool(data.get("truncated", False)),
        "template_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "response_schema_sha256": hashlib.sha256(
            json.dumps(response_schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "inference_path": "apply_template_then_raw_completion",
        "stop_sequences": list(stop_sequences),
    }
    return parsed if isinstance(parsed, dict) else {}, metadata


def _coerce_pointer(raw: dict[str, Any]) -> dict[str, Any]:
    mode = raw.get("mode")
    explicit_non_call = {"clarify", "direct_answer", "no_tool", "refuse", "escalate"}
    if mode in explicit_non_call:
        return {
            "mode": mode,
            "payload": raw.get("payload", {}) if isinstance(raw.get("payload"), dict) else {},
        }
    try:
        tool_id = int(raw.get("tool_id"))
    except (TypeError, ValueError):
        tool_id = -1
    if tool_id < 0:
        return {"mode": "escalate", "payload": {"reason": "pointer output omitted a valid tool_id"}}
    arguments = {}
    for slot, candidate_id in (raw.get("arguments", {}) if isinstance(raw.get("arguments"), dict) else {}).items():
        if isinstance(candidate_id, dict):
            candidate_id = candidate_id.get("candidate_id")
        try:
            arguments[str(slot)] = int(candidate_id)
        except (TypeError, ValueError):
            arguments[str(slot)] = -1
    return {"mode": "call", "tool_id": tool_id, "arguments": arguments}



def _candidate_id(value: Any) -> int | None:
    if isinstance(value, dict):
        if "candidate_id" in value:
            return _candidate_id(value["candidate_id"])
        if isinstance(value.get("candidates"), list):
            for candidate in value["candidates"]:
                found = _candidate_id(candidate)
                if found is not None:
                    return found
        return None
    try:
        candidate_id = int(value)
    except (TypeError, ValueError):
        return None
    return candidate_id if candidate_id >= 0 else None


def _find_slot_candidate(node: Any, slot: str) -> int | None:
    if isinstance(node, dict):
        if slot in node:
            found = _candidate_id(node[slot])
            if found is not None:
                return found
        for value in node.values():
            found = _find_slot_candidate(value, slot)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_slot_candidate(value, slot)
            if found is not None:
                return found
    return None


def _sanitize_pointer(raw: dict[str, Any], pointer: dict[str, Any], lattice: dict[str, Any]) -> dict[str, Any]:
    if pointer.get("mode") != "call":
        return pointer
    selected = next(
        (tool for tool in lattice.get("tools", {}).values() if tool.get("tool_id") == pointer.get("tool_id")),
        None,
    )
    if selected is None:
        return pointer
    arguments = {}
    for slot in selected.get("slots", {}):
        candidate_id = _find_slot_candidate(raw.get("arguments", {}), slot)
        if candidate_id is not None:
            arguments[slot] = candidate_id
    return {"mode": "call", "tool_id": pointer["tool_id"], "arguments": arguments}


def _materialize_unrestricted(pointer: dict[str, Any], lattice: dict[str, Any]) -> dict[str, Any]:
    if pointer.get("mode") != "call":
        mode = str(pointer.get("mode"))
        return {"mode": mode, "tool": None, "arguments": {}, "payload": dict(pointer.get("payload", {}))}
    selected = next(
        ((name, tool) for name, tool in lattice.get("tools", {}).items() if tool.get("tool_id") == pointer.get("tool_id")),
        None,
    )
    if selected is None:
        return {"mode": "escalate", "tool": None, "arguments": {}, "payload": {"reason": "unknown pointer tool"}}
    tool_name, tool = selected
    output = {}
    for slot, candidate_id in pointer.get("arguments", {}).items():
        slot_row = tool.get("slots", {}).get(slot)
        if slot_row is None:
            continue
        candidate = next(
            (row for row in slot_row.get("candidates", []) if row.get("candidate_id") == candidate_id),
            None,
        )
        if candidate is not None:
            output[slot] = candidate.get("value")
    missing = [
        slot for slot, slot_row in tool.get("slots", {}).items()
        if slot_row.get("required") and slot not in output
    ]
    if missing:
        return {"mode": "clarify", "tool": None, "arguments": {}, "payload": {"missing_slots": missing}}
    return {"mode": "call", "tool": tool_name, "arguments": output, "payload": {"pointer_materialized": True}}


def _pointer_action(
    case: dict[str, Any],
    condition: str,
    endpoint: str,
    verifier: FrozenTierBVerifier,
    *,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = _runtime(case)
    lattice_started = time.perf_counter()
    lattice = build_candidate_lattice(
        runtime["messages"],
        runtime["tools"],
        dialogue_state=runtime["dialogue_state"],
        reference_context=runtime["reference_context"],
        candidate_seed=17,
    )
    lattice["action_risk_budget"] = float(runtime["reference_context"].get("action_risk_budget", 0.05))
    unrestricted = condition == "r2_pointer_unrestricted"
    if not unrestricted:
        augment_lattice_with_typed_programs(
            lattice,
            runtime["messages"],
            runtime["tools"],
            reference_context=runtime["reference_context"],
            dialogue_state=runtime["dialogue_state"],
            tier_b_verifier=verifier,
        )
    catalog = _pointer_catalog(lattice, unrestricted=unrestricted)
    construction_seconds = time.perf_counter() - lattice_started
    raw_pointer, metadata = _request_schema_json(
        endpoint,
        _pointer_messages(case, catalog, condition),
        response_schema=_pointer_json_schema(catalog),
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
    pointer = _coerce_pointer(raw_pointer)
    pointer = _sanitize_pointer(raw_pointer, pointer, lattice)
    if unrestricted:
        action = _materialize_unrestricted(pointer, lattice)
        resolution = None
    else:
        resolution = resolve_pointer_contract(pointer, lattice, runtime["messages"], budget=2)
        action = resolution["materialized_action"]
    catalog_sha256 = hashlib.sha256(
        json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata.update({
        "pointer_action": pointer,
        "pointer_catalog_sha256": catalog_sha256,
        "pointer_catalog_tool_count": len(catalog),
        "pointer_catalog_candidate_count": sum(
            len(slot["candidates"]) for tool in catalog for slot in tool["slots"].values()
        ),
        "evidence_construction_seconds": construction_seconds,
        "contract_resolution": resolution,
        "tier_b_verifier_version": verifier.version if not unrestricted else None,
        "tier_b_verifier_artifact_sha256": verifier.artifact_sha256 if not unrestricted else None,
    })
    return action, metadata


def run_r2a_model_conditions(
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
    tier_b_verifier_path: str | Path,
    conditions: Iterable[str] = R2A_MODEL_CONDITIONS,
    seeds: Iterable[int] = (1,),
    max_tokens: int = 256,
    temperature: float = 0.0,
    max_generations: int | None = None,
) -> dict[str, Any]:
    verifier = FrozenTierBVerifier.load(tier_b_verifier_path)
    selected_conditions = tuple(str(condition) for condition in conditions)
    unknown = set(selected_conditions) - set(R2A_MODEL_CONDITIONS)
    if unknown:
        raise ValueError(f"unknown R2-A model conditions: {sorted(unknown)}")
    jobs = [
        (case, condition, int(seed))
        for case in read_jsonl(cases_path)
        for condition in selected_conditions
        for seed in seeds
    ]
    if max_generations is not None:
        jobs = jobs[:max_generations]
    predictions = []
    timings = []
    started_at = datetime.now(timezone.utc).isoformat()
    for case, condition, seed in jobs:
        started = time.perf_counter()
        error = None
        try:
            if condition == "r2_literal_generation":
                action, metadata = _request_schema_json(
                    endpoint,
                    render_chat_messages(case, condition, thinking_mode="off"),
                    response_schema=_literal_call_schema(case),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                )
            else:
                action, metadata = _pointer_action(
                    case,
                    condition,
                    endpoint,
                    verifier,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                )
        except Exception as exc:
            error = str(exc)
            action = {"runner_error": error}
            metadata = {
                "finish_reason": "runner_error",
                "error_type": exc.__class__.__name__,
                "error_message": error,
            }
        elapsed = time.perf_counter() - started
        row = {
            "case_id": case["case_id"],
            "method": condition,
            "model_id": model_id,
            "seed": seed,
            "prediction": action,
            "response_metadata": metadata,
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
            "thinking_marker_detected": False,
            "tier_b_verifier_version": verifier.version if condition == "r2_pointer_tep_tier_ab" else None,
            "tier_b_verifier_artifact_sha256": verifier.artifact_sha256 if condition == "r2_pointer_tep_tier_ab" else None,
            "r2a_model_runner_version": R2A_MODEL_RUNNER_VERSION,
        }
        row["thinking_marker_detected"] = prediction_has_thinking_marker(row)
        predictions.append(row)
        timings.append({
            "case_id": case["case_id"],
            "hypothesis_grid_id": case["hypothesis_grid_id"],
            "method": condition,
            "model_key": model_key,
            "model_id": model_id,
            "seed": seed,
            "backend": "llama.cpp",
            "quantization": "Q4_K_M",
            "chat_parser": R2A_CHAT_PARSER,
            "inference_path": "apply_template_then_raw_completion",
            "elapsed_seconds": elapsed,
            "runner_error": error,
            "thinking_mode": "off",
            "reasoning_budget": 0,
            "thinking_marker_detected": row["thinking_marker_detected"],
            **{key: metadata.get(key) for key in (
                "finish_reason",
                "stop_type",
                "context_truncated",
                "inference_path",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "generated_tokens_per_second",
                "generation_ms",
                "evidence_construction_seconds",
            )},
        })
    write_jsonl(output_path, predictions)
    write_jsonl(timings_path, timings)
    manifest = {
        "schema_version": "tapbench.r2a_model_run_manifest.v2",
        "runner_version": R2A_MODEL_RUNNER_VERSION,
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
        "inference_path": "apply_template_then_raw_completion",
        "thinking_mode": "off",
        "reasoning_budget": 0,
        "context_tokens": 8192,
        "conditions": list(selected_conditions),
        "seeds": list(seeds),
        "generation_count": len(predictions),
        "runner_errors": sum(row["runner_error"] is not None for row in predictions),
        "tier_b_verifier_version": verifier.version,
        "tier_b_verifier_artifact_sha256": verifier.artifact_sha256,
    }
    write_yaml(manifest_path, manifest)
    return manifest

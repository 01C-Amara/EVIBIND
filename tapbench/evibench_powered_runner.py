from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evibind.core.derivations import canonical_json

from . import evibench as diagnostic
from .evibench import EviBenchError, compile_case, payload_digest
from .evibench_powered import POWERED_CONDITIONS, powered_condition_payload
from .evibench_readiness import audit_powered_readiness
from .io import read_jsonl, read_yaml


POWERED_RESPONSE_COLLECTOR_VERSION = (
    "evibind.evibench_powered_response_collector.v1"
)
RequestFn = Callable[[str, Mapping[str, Any], int, str | None], Mapping[str, Any]]


def _read_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EviBenchError(f"{path} must contain a JSON object")
    return value


def _post_json(
    endpoint: str,
    payload: Mapping[str, Any],
    timeout_seconds: int,
    api_key: str | None,
) -> Mapping[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise EviBenchError(f"powered endpoint request failed: {exc}") from exc
    if not isinstance(value, Mapping):
        raise EviBenchError("powered endpoint response must be an object")
    return value


def _frozen_inputs(
    root: Path,
    *,
    model_key: str,
    seed: int,
) -> tuple[dict[str, Any], Mapping[str, Any], dict[str, Any]]:
    execution = read_yaml(root / "configs/evibench_powered_execution_v1.yaml")
    manifest = _read_json_mapping(root / str(execution.get("model_manifest")))
    seeds = execution.get("seeds")
    if not isinstance(seeds, list) or seed not in seeds:
        raise EviBenchError(f"seed is not frozen for the powered study: {seed}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise EviBenchError("powered model manifest omits artifacts")
    matches = [
        row
        for row in artifacts
        if isinstance(row, Mapping) and row.get("key") == model_key
    ]
    if len(matches) != 1:
        raise EviBenchError(f"model key is not uniquely frozen: {model_key}")
    decoding = execution.get("decoding")
    if not isinstance(decoding, dict):
        raise EviBenchError("powered execution decoding must be an object")
    return execution, matches[0], deepcopy(decoding)


def _wire_payload(
    payload: Mapping[str, Any],
    *,
    model_id: str,
    seed: int,
    decoding: Mapping[str, Any],
) -> dict[str, Any]:
    output = deepcopy(dict(payload))
    output.update(
        {
            "model": model_id,
            "seed": seed,
            "temperature": decoding["temperature"],
            "top_p": decoding["top_p"],
            "max_tokens": decoding["max_tokens"],
            "n": decoding["samples_per_request"],
            "parallel_tool_calls": decoding["parallel_tool_calls"],
            "stream": False,
        }
    )
    return output


def _existing_response_index(
    output_path: Path,
    *,
    model_id: str,
    seed: int,
    decoding: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    if not output_path.exists():
        return {}
    rows = read_jsonl(output_path)
    index: dict[tuple[str, str], dict[str, Any]] = {}
    expected_decoding = canonical_json(decoding)
    for row in rows:
        key = (str(row.get("case_id")), str(row.get("condition")))
        if key in index:
            raise EviBenchError(f"duplicate checkpointed powered response: {key}")
        if row.get("model_id") != model_id or row.get("seed") != seed:
            raise EviBenchError("checkpoint identity differs from requested model/seed")
        observed_decoding = row.get("decoding_parameters")
        if not isinstance(observed_decoding, Mapping) or canonical_json(
            observed_decoding
        ) != expected_decoding:
            raise EviBenchError("checkpoint decoding differs from frozen decoding")
        if not isinstance(row.get("response"), Mapping):
            raise EviBenchError(f"checkpoint response is invalid: {key}")
        observed_response_sha256 = hashlib.sha256(
            canonical_json(row["response"]).encode("utf-8")
        ).hexdigest()
        if row.get("response_sha256") != observed_response_sha256:
            raise EviBenchError(f"checkpoint response digest drifted: {key}")
        index[key] = row
    return index


def collect_powered_responses(
    *,
    root: str | Path,
    cases_path: str | Path,
    output_path: str | Path,
    endpoint: str,
    model_key: str,
    seed: int,
    conditions: Sequence[str] = POWERED_CONDITIONS,
    preflight_only: bool = False,
    api_key: str | None = None,
    request_fn: RequestFn = _post_json,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    readiness = audit_powered_readiness(repository)
    if not preflight_only and not readiness["outcome_generation_allowed"]:
        raise EviBenchError(
            "powered outcome generation is blocked: "
            + "; ".join(readiness["failures"] + readiness["blockers"])
        )
    if not conditions or len(set(conditions)) != len(conditions):
        raise EviBenchError("powered conditions must be a non-empty unique sequence")
    unknown = set(conditions) - set(POWERED_CONDITIONS)
    if unknown:
        raise EviBenchError(
            "unsupported powered conditions: " + ",".join(sorted(unknown))
        )
    execution, model, decoding = _frozen_inputs(
        repository,
        model_key=model_key,
        seed=seed,
    )
    model_id = model.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise EviBenchError(f"frozen model ID is invalid: {model_key}")
    cases = read_jsonl(cases_path)
    diagnostic.validate_cases(cases)
    target = Path(output_path)
    checkpointed = _existing_response_index(
        target,
        model_id=model_id,
        seed=seed,
        decoding=decoding,
    )
    cells: list[tuple[Mapping[str, Any], str, dict[str, Any], str]] = []
    for condition in conditions:
        for case in cases:
            session = compile_case(case)
            payload = powered_condition_payload(case, session, condition)
            digest = payload_digest(payload)
            key = (str(case["case_id"]), condition)
            if key in checkpointed:
                if checkpointed[key].get("payload_sha256") != digest:
                    raise EviBenchError(f"checkpoint payload digest drifted: {key}")
                continue
            cells.append((case, condition, payload, digest))
    report = {
        "version": POWERED_RESPONSE_COLLECTOR_VERSION,
        "preflight_only": preflight_only,
        "infrastructure_passed": readiness["infrastructure_passed"],
        "outcome_generation_allowed": readiness["outcome_generation_allowed"],
        "blockers": readiness["blockers"],
        "model_key": model_key,
        "model_id": model_id,
        "seed": seed,
        "case_count": len(cases),
        "condition_count": len(conditions),
        "expected_cells": len(cases) * len(conditions),
        "checkpointed_cells": len(checkpointed),
        "remaining_cells": len(cells),
        "generated_cells": 0,
        "output": str(target),
    }
    if preflight_only:
        return report
    runtime = execution.get("runtime")
    if not isinstance(runtime, Mapping):
        raise EviBenchError("powered runtime configuration is missing")
    timeout_seconds = runtime.get("request_timeout_seconds")
    if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise EviBenchError("powered request timeout is invalid")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for case, condition, payload, digest in cells:
            response = request_fn(
                endpoint,
                _wire_payload(
                    payload,
                    model_id=model_id,
                    seed=seed,
                    decoding=decoding,
                ),
                timeout_seconds,
                api_key,
            )
            if not isinstance(response, Mapping):
                raise EviBenchError("powered endpoint response must be an object")
            row = {
                "collector_version": POWERED_RESPONSE_COLLECTOR_VERSION,
                "case_id": case["case_id"],
                "condition": condition,
                "model_key": model_key,
                "model_id": model_id,
                "seed": seed,
                "decoding_parameters": decoding,
                "payload_sha256": digest,
                "response_sha256": hashlib.sha256(
                    canonical_json(response).encode("utf-8")
                ).hexdigest(),
                "response": deepcopy(dict(response)),
            }
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
            report["generated_cells"] += 1
            report["remaining_cells"] -= 1
    return report

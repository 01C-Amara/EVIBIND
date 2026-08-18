from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl
from .r2c import _runtime
from .runtime_audit import build_runtime_dependency_audit


R2C_SCOPE_AUDIT_VERSION = "tapbench.r2c_scope_audit.v1"
OFFLINE_ONLY_FIELDS = {
    "gold_action",
    "derivable_values",
    "r2c_oracle",
    "task_kind",
}
EFFECT_FIRST_PREFIX = "tap_r_effect_first_"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corrected_generation_calls(prediction: dict[str, Any]) -> int:
    metadata = prediction.get("response_metadata", {})
    resolution = prediction.get("resolution", {})
    return max(
        int(metadata.get("generation_calls") or 0),
        int(resolution.get("generation_calls") or 0),
    )


def audit_runtime_projection(case: dict[str, Any]) -> dict[str, Any]:
    runtime = _runtime(case)
    declared = set(case.get("metadata", {}).get("runtime_allowed_fields", []))
    keys = set(runtime)
    return {
        "case_id": case["case_id"],
        "runtime_fields": sorted(keys),
        "matches_declared_allowlist": keys == declared,
        "offline_field_intersection": sorted(keys & OFFLINE_ONLY_FIELDS),
        "is_deep_copy": all(
            runtime.get(field) is not case.get(field)
            for field in keys
            if isinstance(case.get(field), (dict, list))
        ),
    }


def build_r2c_scope_audit(
    cases_path: str | Path,
    predictions_path: str | Path,
) -> dict[str, Any]:
    cases = read_jsonl(cases_path)
    predictions = read_jsonl(predictions_path)
    case_by_id = {row["case_id"]: row for row in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("duplicate R2-C case IDs")
    unknown = sorted(
        {row["case_id"] for row in predictions} - set(case_by_id)
    )
    if unknown:
        raise ValueError(f"predictions reference unknown cases: {unknown[:3]}")

    runtime_rows = [audit_runtime_projection(case) for case in cases]
    dependency_audit = build_runtime_dependency_audit()
    method_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        method = str(prediction.get("method"))
        task_kind = str(case_by_id[prediction["case_id"]]["task_kind"])
        method_rows[method].append(prediction)
        task_rows[(method, task_kind)].append(prediction)

    method_summary = []
    for method, rows in sorted(method_rows.items()):
        corrected = [corrected_generation_calls(row) for row in rows]
        reported = [
            int(row.get("response_metadata", {}).get("generation_calls") or 0)
            for row in rows
        ]
        method_summary.append(
            {
                "method": method,
                "rows": len(rows),
                "model_mediated_rows": sum(value > 0 for value in corrected),
                "deterministic_rows": sum(value == 0 for value in corrected),
                "corrected_generation_calls": sum(corrected),
                "frozen_reported_generation_calls": sum(reported),
            }
        )

    task_summary = []
    for (method, task_kind), rows in sorted(task_rows.items()):
        calls = [corrected_generation_calls(row) for row in rows]
        task_summary.append(
            {
                "method": method,
                "task_kind": task_kind,
                "rows": len(rows),
                "model_mediated_rows": sum(value > 0 for value in calls),
                "generation_calls": sum(calls),
            }
        )

    effect_rows = [
        row
        for row in predictions
        if str(row.get("method", "")).startswith(EFFECT_FIRST_PREFIX)
    ]
    admission_basis = Counter(
        str(row.get("response_metadata", {}).get("effect_admission", {}).get("basis"))
        for row in effect_rows
    )
    lock_status = Counter(
        str(row.get("response_metadata", {}).get("evidence_lock", {}).get("status"))
        for row in effect_rows
        if row.get("response_metadata", {}).get("evidence_lock")
    )
    effect_model_rows = sum(
        corrected_generation_calls(row) > 0 for row in effect_rows
    )
    request_markers = {
        "evidence_fields": sum(
            "Evidence fields:" in "\n".join(
                str(message.get("content", ""))
                for message in case.get("messages", [])
            )
            for case in cases
        ),
        "explicit_no_tool_or_denial": sum(
            any(
                marker in "\n".join(
                    str(message.get("content", "")).casefold()
                    for message in case.get("messages", [])
                )
                for marker in ("no domain action is authorized", "do not execute")
            )
            for case in cases
        ),
        "explicit_direct_answer": sum(
            any(
                marker in "\n".join(
                    str(message.get("content", "")).casefold()
                    for message in case.get("messages", [])
                )
                for marker in ("explain only", "answer directly")
            )
            for case in cases
        ),
    }

    corrected_total = sum(
        corrected_generation_calls(row) for row in predictions
    )
    reported_total = sum(
        int(row.get("response_metadata", {}).get("generation_calls") or 0)
        for row in predictions
    )
    runtime_passed = all(
        row["matches_declared_allowlist"]
        and not row["offline_field_intersection"]
        and row["is_deep_copy"]
        for row in runtime_rows
    )
    return {
        "schema_version": R2C_SCOPE_AUDIT_VERSION,
        "source_sha256": {
            "cases": _sha256(cases_path),
            "predictions": _sha256(predictions_path),
        },
        "case_count": len(cases),
        "prediction_count": len(predictions),
        "runtime_boundary": {
            "passed": runtime_passed,
            "all_cases_checked": len(runtime_rows),
            "violations": [
                row
                for row in runtime_rows
                if not (
                    row["matches_declared_allowlist"]
                    and not row["offline_field_intersection"]
                    and row["is_deep_copy"]
                )
            ],
            "static_dependency_audit": dependency_audit[
                "evidence_bounded_path"
            ],
        },
        "difficulty_and_mediation": {
            "method_summary": method_summary,
            "method_task_summary": task_summary,
            "effect_first_rows": len(effect_rows),
            "effect_first_model_mediated_rows": effect_model_rows,
            "effect_first_model_mediated_fraction": (
                effect_model_rows / len(effect_rows) if effect_rows else None
            ),
            "effect_first_admission_basis": dict(sorted(admission_basis.items())),
            "effect_first_evidence_lock_status": dict(sorted(lock_status.items())),
            "request_markers": request_markers,
        },
        "telemetry_correction": {
            "frozen_reported_generation_calls": reported_total,
            "corrected_generation_calls": corrected_total,
            "reason": (
                "Legacy pointer conditions stored generation_calls inside the "
                "resolution object; predictions and scores are unchanged."
            ),
            "frozen_artifacts_modified": False,
        },
        "claim_boundary": {
            "supported": (
                "On held-out synthetic single-call cases with explicit action cues "
                "and structurally labeled evidence, effect-first admission plus "
                "certificate-locked arguments improved safe resolution."
            ),
            "not_supported": [
                "general tool-calling mastery",
                "robustness to implicit intent or unlabeled evidence",
                "stateful or multi-turn agent reliability",
                "external-benchmark superiority",
            ],
        },
    }


def write_r2c_scope_audit(
    cases_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    report = build_r2c_scope_audit(cases_path, predictions_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report

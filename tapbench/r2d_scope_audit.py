from __future__ import annotations

import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_jsonl
from .r2b import _runtime
from .r2d import R2D_TASK_KINDS
from .selective_tapr import run_selective_tapr_resolution


R2D_SCOPE_AUDIT_VERSION = "tapbench.r2d_scope_audit.v5"
OFFLINE_ONLY_FIELDS = {
    "gold_action",
    "derivable_values",
    "r2d_oracle",
    "r2e_oracle",
    "r2f_oracle",
    "task_kind",
}
SELECTIVE_RUNTIME_PARAMETERS = {
    "messages",
    "tools",
    "endpoint",
    "max_tokens",
    "seed",
    "request_fn",
    "semantic_extent_enabled",
    "exhaust_proposal_budget",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request(case: dict[str, Any]) -> str:
    return "\n".join(
        str(message.get("content", ""))
        for message in case.get("messages", [])
        if message.get("role") == "user"
    )


def build_r2d_scope_audit(cases_path: str | Path) -> dict[str, Any]:
    cases = read_jsonl(cases_path)
    if len({str(case["case_id"]) for case in cases}) != len(cases):
        raise ValueError("duplicate R2-D case IDs")

    runtime_violations = []
    for case in cases:
        runtime = _runtime(case)
        declared = set(case.get("metadata", {}).get("runtime_allowed_fields", []))
        keys = set(runtime)
        deep_copy = all(
            runtime.get(field) is not case.get(field)
            for field in keys
            if isinstance(case.get(field), (dict, list))
        )
        if keys != declared or keys & OFFLINE_ONLY_FIELDS or not deep_copy:
            runtime_violations.append(
                {
                    "case_id": case["case_id"],
                    "runtime_fields": sorted(keys),
                    "declared_fields": sorted(declared),
                    "offline_field_intersection": sorted(keys & OFFLINE_ONLY_FIELDS),
                    "is_deep_copy": deep_copy,
                }
            )

    signature_fields = set(inspect.signature(run_selective_tapr_resolution).parameters)
    source = inspect.getsource(run_selective_tapr_resolution)
    forbidden_source_tokens = sorted(
        token for token in OFFLINE_ONLY_FIELDS if token in source
    )
    task_counts = Counter(str(case.get("task_kind")) for case in cases)
    request_texts = [_request(case).casefold() for case in cases]
    marker_counts = {
        "evidence_fields_block": sum("evidence fields:" in text for text in request_texts),
        "explicit_execution_denial": sum(
            any(marker in text for marker in ("do not execute", "no action is authorized"))
            for text in request_texts
        ),
        "explicit_answer_label": sum(
            any(marker in text for marker in ("answer directly", "explain only"))
            for text in request_texts
        ),
        "explicit_clarification_instruction": sum(
            any(marker in text for marker in ("ask only for", "ask me for"))
            for text in request_texts
        ),
    }
    balanced = (
        set(task_counts) == set(R2D_TASK_KINDS)
        and len(set(task_counts.values())) == 1
    )
    runtime_passed = (
        not runtime_violations
        and signature_fields == SELECTIVE_RUNTIME_PARAMETERS
        and not forbidden_source_tokens
    )
    return {
        "schema_version": R2D_SCOPE_AUDIT_VERSION,
        "source_sha256": {"cases": _sha256(cases_path)},
        "case_count": len(cases),
        "family_count": len({str(case.get("family")) for case in cases}),
        "task_kind_counts": dict(sorted(task_counts.items())),
        "task_kind_balance_exact": balanced,
        "runtime_boundary": {
            "passed": runtime_passed,
            "violations": runtime_violations,
            "selective_signature_fields": sorted(signature_fields),
            "forbidden_source_tokens": forbidden_source_tokens,
        },
        "implicit_cue_audit": {
            "passed": not any(marker_counts.values()),
            "marker_counts": marker_counts,
        },
        "passed": runtime_passed and balanced and not any(marker_counts.values()),
    }


def write_r2d_scope_audit(
    cases_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    report = build_r2d_scope_audit(cases_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report

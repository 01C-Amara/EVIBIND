from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .extractive_candidates import build_extractive_candidate_table
from .io import read_jsonl, write_jsonl


CERTIFICATE_AUDIT_VERSION = "tapbench.certificate_replay_audit.v3"


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _tool_name(tool: dict[str, Any]) -> str:
    return str(tool.get("canonical_name") or tool.get("name") or "")


def _candidate_matches(
    candidate: dict[str, Any],
    certificate: dict[str, Any],
    value: Any,
) -> bool:
    return (
        _canonical(candidate.get("value")) == _canonical(value)
        and _canonical(certificate.get("value")) == _canonical(value)
        and candidate.get("candidate_id") == certificate.get("candidate_id")
        and candidate.get("source_span") == certificate.get("source_span")
        and candidate.get("component_spans")
        == certificate.get("component_spans")
        and candidate.get("source_text") == certificate.get("source_text")
        and candidate.get("transform") == certificate.get("transform")
    )


def _overlapping_certificate_slots(
    certificates: dict[str, Any],
) -> list[str]:
    rows = []
    for slot, certificate in certificates.items():
        span = certificate.get("source_span") if isinstance(certificate, dict) else None
        if (
            isinstance(span, list)
            and len(span) == 2
            and all(isinstance(value, int) for value in span)
        ):
            rows.append((str(slot), int(span[0]), int(span[1])))
    conflicts: set[str] = set()
    for index, (left_slot, left_start, left_end) in enumerate(rows):
        for right_slot, right_start, right_end in rows[index + 1 :]:
            if max(left_start, right_start) < min(left_end, right_end):
                conflicts.update((left_slot, right_slot))
    return sorted(conflicts)


def audit_prediction(
    case: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    method = str(prediction.get("method", ""))
    base = {
        "audit_version": CERTIFICATE_AUDIT_VERSION,
        "case_id": prediction.get("case_id"),
        "model_id": prediction.get("model_id"),
        "method": method,
        "seed": prediction.get("seed"),
    }
    if not method.startswith(("tap_r_eflrx", "tap_r_capc", "tap_r_selective")):
        return {
            **base,
            "eligible": False,
            "emitted_call": False,
            "passed": True,
            "failures": [],
        }

    action = prediction.get("prediction")
    if not isinstance(action, dict) or action.get("mode") != "call":
        return {
            **base,
            "eligible": True,
            "emitted_call": False,
            "passed": True,
            "failures": [],
        }

    failures: list[dict[str, Any]] = []
    tool_name = str(action.get("tool") or "")
    tools = [
        tool
        for tool in case.get("tools", [])
        if isinstance(tool, dict) and _tool_name(tool) == tool_name
    ]
    if len(tools) != 1:
        failures.append({"reason": "tool_not_uniquely_in_public_catalog"})
        return {
            **base,
            "eligible": True,
            "emitted_call": True,
            "passed": False,
            "failures": failures,
        }

    table = build_extractive_candidate_table(
        list(case.get("messages", [])),
        tools[0],
        include_optional=True,
    )
    table_sha256 = hashlib.sha256(
        json.dumps(
            table,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    recorded_table_sha256 = (
        prediction.get("response_metadata", {})
        .get("candidate_table", {})
        .get("sha256")
    )
    if recorded_table_sha256 != table_sha256:
        failures.append({"reason": "candidate_table_hash_mismatch"})
    request_text = str(table.get("request_text", ""))
    arguments = action.get("arguments")
    certificates = prediction.get("response_metadata", {}).get(
        "evidence_certificates"
    )
    if not isinstance(arguments, dict):
        failures.append({"reason": "arguments_not_object"})
        arguments = {}
    if not isinstance(certificates, dict):
        failures.append({"reason": "certificates_not_object"})
        certificates = {}
    if set(arguments) != set(certificates):
        failures.append(
            {
                "reason": "argument_certificate_slot_mismatch",
                "argument_slots": sorted(arguments),
                "certificate_slots": sorted(certificates),
            }
        )

    for slot, value in arguments.items():
        certificate = certificates.get(slot)
        if not isinstance(certificate, dict):
            failures.append({"slot": slot, "reason": "missing_certificate"})
            continue
        span = certificate.get("source_span")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(item, int) for item in span)
            or not 0 <= span[0] < span[1] <= len(request_text)
        ):
            failures.append({"slot": slot, "reason": "invalid_source_span"})
            continue
        if request_text[span[0] : span[1]] != certificate.get("source_text"):
            failures.append({"slot": slot, "reason": "source_text_mismatch"})
        component_spans = certificate.get("component_spans")
        if not isinstance(component_spans, list) or not component_spans:
            failures.append({"slot": slot, "reason": "missing_component_spans"})
        elif any(
            not isinstance(component, list)
            or len(component) != 2
            or not all(isinstance(item, int) for item in component)
            or not 0 <= component[0] < component[1] <= len(request_text)
            for component in component_spans
        ):
            failures.append({"slot": slot, "reason": "invalid_component_span"})

        candidates = table.get("slots", {}).get(slot, [])
        if not any(
            _candidate_matches(candidate, certificate, value)
            for candidate in candidates
        ):
            failures.append(
                {
                    "slot": slot,
                    "reason": "certificate_not_reproducible_from_frozen_compiler",
                    "transform": certificate.get("transform"),
                }
            )

    if method.startswith("tap_r_selective"):
        overlapping = _overlapping_certificate_slots(certificates)
        if overlapping:
            failures.append(
                {
                    "reason": "cross_slot_source_span_overlap",
                    "slots": overlapping,
                }
            )

    return {
        **base,
        "eligible": True,
        "emitted_call": True,
        "passed": not failures,
        "tool": tool_name,
        "argument_count": len(arguments),
        "recorded_candidate_table_sha256": recorded_table_sha256,
        "replayed_candidate_table_sha256": table_sha256,
        "failures": failures,
    }


def audit_files(
    cases_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    cases = {str(row["case_id"]): row for row in read_jsonl(cases_path)}
    rows = []
    for prediction in read_jsonl(predictions_path):
        case_id = str(prediction.get("case_id"))
        if case_id not in cases:
            raise ValueError(f"prediction references unknown case: {case_id}")
        rows.append(audit_prediction(cases[case_id], prediction))
    write_jsonl(output_path, rows)
    eligible = [row for row in rows if row["eligible"]]
    emitted = [row for row in eligible if row["emitted_call"]]
    failed = [row for row in eligible if not row["passed"]]
    summary = {
        "schema_version": CERTIFICATE_AUDIT_VERSION,
        "prediction_count": len(rows),
        "eligible_count": len(eligible),
        "emitted_call_count": len(emitted),
        "failed_count": len(failed),
        "passed": not failed,
        "failure_reasons": dict(
            sorted(
                Counter(
                    failure["reason"]
                    for row in failed
                    for failure in row["failures"]
                ).items()
            )
        ),
    }
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary

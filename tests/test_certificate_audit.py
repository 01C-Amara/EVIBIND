from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from tapbench.certificate_audit import audit_prediction
from tapbench.extractive_candidates import build_extractive_candidate_table


def _fixture() -> tuple[dict, dict]:
    case = {
        "case_id": "certificate_case",
        "messages": [{"role": "user", "content": "Set the height to 10."}],
        "tools": [
            {
                "name": "shape.set",
                "canonical_name": "shape.set",
                "parameters": {
                    "type": "object",
                    "properties": {"height": {"type": "integer"}},
                    "required": ["height"],
                },
            }
        ],
    }
    table = build_extractive_candidate_table(
        case["messages"], case["tools"][0], include_optional=True
    )
    candidate = next(row for row in table["slots"]["height"] if row["value"] == 10)
    certificate = {
        key: deepcopy(candidate[key])
        for key in (
            "candidate_id",
            "value",
            "source_span",
            "component_spans",
            "source_text",
            "transform",
        )
    }
    prediction = {
        "case_id": case["case_id"],
        "model_id": "test-model",
        "method": "tap_r_eflrx_consensus",
        "seed": 1,
        "prediction": {
            "mode": "call",
            "tool": "shape.set",
            "arguments": {"height": 10},
            "payload": {},
        },
        "response_metadata": {
            "candidate_table": {
                "sha256": hashlib.sha256(
                    json.dumps(
                        table,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode()
                ).hexdigest()
            },
            "evidence_certificates": {"height": certificate},
        },
    }
    return case, prediction


def test_valid_certificate_replays_from_request() -> None:
    case, prediction = _fixture()
    assert audit_prediction(case, prediction)["passed"]


def test_capc_certificate_uses_the_same_independent_replay() -> None:
    case, prediction = _fixture()
    prediction["method"] = "tap_r_capc_dual"
    report = audit_prediction(case, prediction)
    assert report["eligible"]
    assert report["passed"]


def test_model_literal_cannot_hide_behind_certificate() -> None:
    case, prediction = _fixture()
    prediction["prediction"]["arguments"]["height"] = 11
    report = audit_prediction(case, prediction)
    assert not report["passed"]
    assert any("not_reproducible" in row["reason"] for row in report["failures"])


def test_tampered_span_is_rejected() -> None:
    case, prediction = _fixture()
    prediction["response_metadata"]["evidence_certificates"]["height"][
        "source_span"
    ] = [0, 3]
    report = audit_prediction(case, prediction)
    assert not report["passed"]
    assert {row["reason"] for row in report["failures"]} >= {
        "source_text_mismatch",
        "certificate_not_reproducible_from_frozen_compiler",
    }


def test_non_call_is_vacuously_certificate_safe() -> None:
    case, prediction = _fixture()
    prediction["prediction"] = {
        "mode": "refuse",
        "tool": None,
        "arguments": {},
        "payload": {"reason": "disagreement"},
    }
    prediction["response_metadata"] = {}
    report = audit_prediction(case, prediction)
    assert report["passed"]
    assert not report["emitted_call"]


def test_selective_call_rejects_cross_slot_span_reuse() -> None:
    case = {
        "case_id": "cross_slot_case",
        "messages": [{"role": "user", "content": "Submit 10."}],
        "tools": [
            {
                "name": "submit",
                "canonical_name": "submit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "integer"},
                        "receipt_id": {"type": "string"},
                    },
                    "required": ["amount", "receipt_id"],
                },
            }
        ],
    }
    table = build_extractive_candidate_table(
        case["messages"], case["tools"][0], include_optional=True
    )
    amount = next(row for row in table["slots"]["amount"] if row["value"] == 10)
    receipt = next(
        row
        for row in table["slots"]["receipt_id"]
        if row["value"] == "10"
    )
    keys = (
        "candidate_id",
        "value",
        "source_span",
        "component_spans",
        "source_text",
        "transform",
    )
    prediction = {
        "case_id": case["case_id"],
        "model_id": "test-model",
        "method": "tap_r_selective_full",
        "seed": 1,
        "prediction": {
            "mode": "call",
            "tool": "submit",
            "arguments": {"amount": 10, "receipt_id": "10"},
            "payload": {},
        },
        "response_metadata": {
            "candidate_table": {
                "sha256": hashlib.sha256(
                    json.dumps(
                        table,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode()
                ).hexdigest()
            },
            "evidence_certificates": {
                "amount": {key: deepcopy(amount[key]) for key in keys},
                "receipt_id": {key: deepcopy(receipt[key]) for key in keys},
            },
        },
    }
    report = audit_prediction(case, prediction)
    assert not report["passed"]
    assert any(
        row["reason"] == "cross_slot_source_span_overlap"
        for row in report["failures"]
    )

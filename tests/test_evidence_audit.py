from __future__ import annotations

import csv
import json

from tapbench.evidence_audit import build_evidence_audit, score_evidence_audit
from tapbench.io import write_jsonl


def test_blinded_evidence_audit_round_trip(tmp_path) -> None:
    cases = tmp_path / "cases.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    blind = tmp_path / "blind.csv"
    key = tmp_path / "key.csv"
    manifest = tmp_path / "manifest.json"
    report = tmp_path / "report.json"
    write_jsonl(cases, [{"case_id": "c1", "messages": [{"role": "user", "content": "Use blue"}]}])
    write_jsonl(ledger, [
        {"case_id": "c1", "model_id": "m", "method": "x", "family": "shopping", "task_kind": "call", "slot": "color", "value": "blue", "required": True, "source_span": {"start": 4, "end": 8}, "normalizer": "slot_specific_exact_match", "evidence_label": "explicit"},
        {"case_id": "c1", "model_id": "m", "method": "x", "family": "shopping", "task_kind": "call", "slot": "size", "value": "XL", "required": True, "source_span": None, "normalizer": None, "evidence_label": "unsupported"},
    ])
    built = build_evidence_audit(cases, ledger, blind, key, manifest, per_label=1, seed=1)
    assert built["selected_total"] == 2
    with blind.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert "oracle_label" not in rows[0]
    with key.open(newline="", encoding="utf-8") as handle:
        labels = {row["audit_id"]: row["oracle_label"] for row in csv.DictReader(handle)}
    for row in rows:
        row["human_label"] = labels[row["audit_id"]]
    with blind.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    scored = score_evidence_audit(blind, key, report)
    assert scored["status"] == "complete"
    assert scored["accuracy"] == 1.0
    assert json.loads(report.read_text())["labeled_rows"] == 2


def test_double_annotation_reports_axis_agreement_and_adjudication(tmp_path) -> None:
    cases = tmp_path / "cases.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    blind_a = tmp_path / "a.csv"
    blind_b = tmp_path / "b.csv"
    adjudication = tmp_path / "adj.csv"
    key = tmp_path / "key.csv"
    manifest = tmp_path / "manifest.json"
    report = tmp_path / "report.json"
    write_jsonl(cases, [{"case_id": "c1", "messages": [{"role": "user", "content": "Move it from 2026-07-11 to 2026-07-12"}]}])
    write_jsonl(ledger, [{
        "case_id": "c1", "model_id": "m", "method": "x", "family": "calendar",
        "task_kind": "call", "slot": "date", "value": "2026-07-12", "required": True,
        "source_span": {"start": 27, "end": 37, "text": "2026-07-12"},
        "normalizer": "iso_date", "evidence_label": "normalized",
    }])
    built = build_evidence_audit(
        cases, ledger, blind_a, key, manifest, per_label=1, seed=1,
        blind_b_path=blind_b, adjudication_path=adjudication,
    )
    assert built["schema_version"] == "tapbench.evidence_audit.v2"

    sheets = []
    for path in (blind_a, blind_b):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            for field in (
                "span_support", "normalization_correct", "slot_role_correct",
                "scope_correct", "contract_correct", "contradiction_correct",
            ):
                row[field] = "yes"
            row["evidence_class"] = "normalized"
        sheets.append(rows)
    sheets[1][0]["slot_role_correct"] = "no"
    for path, rows in zip((blind_a, blind_b), sheets):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    with adjudication.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["slot_role_correct"] = "yes"
    with adjudication.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    scored = score_evidence_audit(
        blind_a, key, report, blind_b_path=blind_b, adjudication_path=adjudication,
    )
    assert scored["status"] == "complete"
    assert scored["double_annotation"] is True
    assert scored["adjudication"]["disagreement_rows"] == 1
    assert scored["interannotator_agreement"]["slot_role_correct"]["raw_agreement"] == 0.0

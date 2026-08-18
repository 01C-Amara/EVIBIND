from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl
from .resolution import EVIDENCE_LABELS

AUDIT_VERSION = "tapbench.evidence_audit.v2"
AXIS_FIELDS = ("span_support", "normalization_correct", "slot_role_correct", "scope_correct", "contract_correct", "contradiction_correct")
AXIS_VALUES = {"yes", "no", "uncertain", "not_applicable"}
BLIND_FIELDS = (
    "audit_id", "family", "task_kind", "model_id", "method", "request", "slot", "value_json",
    "required", "slot_role", "resolution_type", "source_kind", "source_span_json", "source_text",
    "transform", "transform_context_json", "scope_status", "contradiction_status", "normalizer",
    "transition", "adversarial_tags", "annotator_id", *AXIS_FIELDS, "evidence_class", "human_label",
    "ambiguity", "reviewer_notes",
)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tags(request: str) -> list[str]:
    padded = f" {request.lower()} "
    patterns = {
        "negation": (" do not ", " don't ", " never ", " without "),
        "quoted_text": ('"', "'", " quoted "),
        "correction_or_supersession": (" change ", " instead ", " correction ", " from "),
        "conditional_or_hypothetical": (" if ", " would ", " might ", " hypothetical ", " suppose "),
        "relative_time": (" today ", " tomorrow ", " next ", " yesterday ", " timezone "),
        "prompt_injection": (" ignore previous ", " system prompt ", " developer message "),
    }
    return [tag for tag, needles in patterns.items() if any(needle in padded for needle in needles)]


def _blank_row(base: dict[str, Any], annotator: str) -> dict[str, Any]:
    return {
        **base, "annotator_id": annotator, **{field: "" for field in AXIS_FIELDS},
        "evidence_class": "", "human_label": "", "ambiguity": "", "reviewer_notes": "",
    }


def build_evidence_audit(
    cases_path: str | Path,
    ledger_path: str | Path,
    blind_path: str | Path,
    key_path: str | Path,
    manifest_path: str | Path,
    *,
    per_label: int = 64,
    seed: int = 17,
    blind_b_path: str | Path | None = None,
    adjudication_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build independent A/B sheets. The key contains system labels, not human axis truth."""
    cases = {row["case_id"]: row for row in read_jsonl(cases_path)}
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for row in read_jsonl(ledger_path):
        label = str(row.get("evidence_label"))
        signature = (row.get("case_id"), row.get("model_id"), row.get("method"), row.get("slot"), json.dumps(row.get("value"), sort_keys=True))
        if label in EVIDENCE_LABELS and signature not in seen:
            seen.add(signature)
            by_label[label].append(row)

    rng = random.Random(seed)
    selected: list[tuple[str, dict[str, Any]]] = []
    selected_counts = {}
    for label in EVIDENCE_LABELS:
        candidates = sorted(by_label.get(label, []), key=lambda row: (
            str(row.get("family")), str(row.get("model_id")), str(row.get("task_kind")),
            str(row.get("slot")), str(row.get("case_id")),
        ))
        rng.shuffle(candidates)
        chosen = candidates[:per_label]
        selected.extend((label, row) for row in chosen)
        selected_counts[label] = len(chosen)
    selected.sort(key=lambda item: (item[0], str(item[1].get("case_id")), str(item[1].get("slot"))))

    bases, keys = [], []
    for index, (label, row) in enumerate(selected, start=1):
        case = cases[str(row["case_id"])]
        request = "\n".join(str(message.get("content", "")) for message in case.get("messages", []) if isinstance(message, dict) and message.get("role") == "user")
        span = row.get("source_span")
        audit_id = f"ledger-{index:04d}"
        bases.append({
            "audit_id": audit_id, "family": row.get("family"), "task_kind": row.get("task_kind"),
            "model_id": row.get("model_id"), "method": row.get("method"), "request": request,
            "slot": row.get("slot"), "value_json": json.dumps(row.get("value"), sort_keys=True),
            "required": row.get("required"), "slot_role": row.get("role") or row.get("slot_role"),
            "resolution_type": row.get("resolution_type"), "source_kind": row.get("source_kind"),
            "source_span_json": json.dumps(span, sort_keys=True),
            "source_text": span.get("text") if isinstance(span, dict) else row.get("source_text"),
            "transform": row.get("transform") or row.get("normalizer"),
            "transform_context_json": json.dumps(row.get("transform_context"), sort_keys=True),
            "scope_status": row.get("scope_status"), "contradiction_status": row.get("contradiction_status"),
            "normalizer": row.get("normalizer"), "transition": row.get("transition"),
            "adversarial_tags": "|".join(_tags(request)),
        })
        keys.append({"audit_id": audit_id, "system_label": label, "oracle_label": label})

    _write_csv(Path(blind_path), [_blank_row(row, "A") for row in bases], BLIND_FIELDS)
    if blind_b_path:
        _write_csv(Path(blind_b_path), [_blank_row(row, "B") for row in bases], BLIND_FIELDS)
    if adjudication_path:
        _write_csv(Path(adjudication_path), [_blank_row(row, "ADJ") for row in bases], BLIND_FIELDS)
    _write_csv(Path(key_path), keys, ("audit_id", "system_label", "oracle_label"))

    distribution_fields = ("family", "model_id", "task_kind", "slot", "transition")
    manifest = {
        "schema_version": AUDIT_VERSION, "cases_path": str(cases_path), "ledger_path": str(ledger_path),
        "blind_a_path": str(blind_path), "blind_b_path": str(blind_b_path) if blind_b_path else None,
        "adjudication_path": str(adjudication_path) if adjudication_path else None, "key_path": str(key_path),
        "seed": seed, "per_label_target": per_label,
        "available_label_counts": {label: len(by_label.get(label, [])) for label in EVIDENCE_LABELS},
        "selected_label_counts": selected_counts, "selected_total": len(selected),
        "selected_distribution": {field: dict(sorted(Counter(str(row.get(field) or "missing") for _, row in selected).items())) for field in distribution_fields},
        "adversarial_tag_counts": dict(sorted(Counter(tag for row in bases for tag in str(row["adversarial_tags"]).split("|") if tag).items())),
        "status": "pending_double_annotation", "allowed_evidence_classes": list(EVIDENCE_LABELS),
        "axis_fields": list(AXIS_FIELDS), "axis_values": sorted(AXIS_VALUES),
        "instructions": [
            "Annotators A and B work independently and do not inspect the key.",
            "Score span support, normalization, semantic slot role, scope, tool contract, and contradiction separately.",
            "Use uncertain rather than forcing a binary label; disagreements remain ambiguous until adjudication.",
            "The key is the legacy system label, not human ground truth for the six axes.",
        ],
    }
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _read(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _norm(value: str | None) -> str:
    return str(value or "").strip().lower()


def _field(row: dict[str, str], field: str) -> str:
    if field == "evidence_class":
        return _norm(row.get(field) or row.get("human_label"))
    return _norm(row.get(field))


def _validate(rows: list[dict[str, str]], source: str) -> None:
    bad_classes = sorted({_field(row, "evidence_class") for row in rows if _field(row, "evidence_class") and _field(row, "evidence_class") not in EVIDENCE_LABELS})
    bad_axes = sorted({_field(row, field) for row in rows for field in AXIS_FIELDS if _field(row, field) and _field(row, field) not in AXIS_VALUES})
    if bad_classes:
        raise ValueError(f"invalid evidence classes in {source}: {bad_classes}")
    if bad_axes:
        raise ValueError(f"invalid axis values in {source}: {bad_axes}")


def _kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    cats = {value for pair in pairs for value in pair}
    observed = sum(a == b for a, b in pairs) / len(pairs)
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum((left[c] / len(pairs)) * (right[c] / len(pairs)) for c in cats)
    return (observed - expected) / (1 - expected) if expected < 1 else (1.0 if observed == 1 else None)


def _system_comparison(classes: dict[str, str], key: dict[str, str]) -> dict[str, Any]:
    confusion = Counter((key[audit_id], label) for audit_id, label in classes.items() if audit_id in key and label)
    metrics = []
    for label in EVIDENCE_LABELS:
        tp = confusion[(label, label)]
        fp = sum(n for (gold, pred), n in confusion.items() if pred == label and gold != label)
        fn = sum(n for (gold, pred), n in confusion.items() if gold == label and pred != label)
        metrics.append({"label": label, "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None, "support": sum(n for (gold, _), n in confusion.items() if gold == label)})
    total = sum(confusion.values())
    return {
        "labeled_rows": total,
        "accuracy": sum(n for (gold, pred), n in confusion.items() if gold == pred) / total if total else None,
        "per_label": metrics,
        "confusion": [{"system_label": gold, "human_evidence_class": pred, "count": n} for (gold, pred), n in sorted(confusion.items())],
    }


def score_evidence_audit(
    blind_path: str | Path,
    key_path: str | Path,
    output_path: str | Path,
    *,
    blind_b_path: str | Path | None = None,
    adjudication_path: str | Path | None = None,
) -> dict[str, Any]:
    rows_a = _read(blind_path)
    rows_b = _read(blind_b_path) if blind_b_path else []
    rows_adj = _read(adjudication_path) if adjudication_path else []
    _validate(rows_a, "annotator A")
    _validate(rows_b, "annotator B")
    _validate(rows_adj, "adjudication")
    with Path(key_path).open(encoding="utf-8", newline="") as handle:
        key = {row["audit_id"]: row.get("system_label") or row.get("oracle_label", "") for row in csv.DictReader(handle)}
    by_a, by_b, by_adj = ({row["audit_id"]: row for row in rows} for rows in (rows_a, rows_b, rows_adj))
    fields = (*AXIS_FIELDS, "evidence_class")
    agreement = {}
    for field in fields:
        pairs = [(_field(row, field), _field(by_b[audit_id], field)) for audit_id, row in by_a.items() if audit_id in by_b and _field(row, field) and _field(by_b[audit_id], field)]
        agreement[field] = {"double_labeled": len(pairs), "raw_agreement": sum(a == b for a, b in pairs) / len(pairs) if pairs else None, "cohen_kappa": _kappa(pairs)}

    finalized, disagreement_ids = {}, set()
    for audit_id, row_a in by_a.items():
        final = {}
        for field in fields:
            left, right, adj = _field(row_a, field), _field(by_b.get(audit_id, {}), field), _field(by_adj.get(audit_id, {}), field)
            if left and right and left != right:
                disagreement_ids.add(audit_id)
            if adj:
                final[field] = adj
            elif right and left != right:
                final[field] = "ambiguous"
            else:
                final[field] = left or right
        if any(_norm(row.get("ambiguity")) in {"yes", "true", "1"} for row in (row_a, by_b.get(audit_id, {}), by_adj.get(audit_id, {}))):
            disagreement_ids.add(audit_id)
        finalized[audit_id] = final

    axis_metrics = []
    for field in AXIS_FIELDS:
        counts = Counter(row[field] for row in finalized.values() if row.get(field))
        binary = counts["yes"] + counts["no"]
        axis_metrics.append({"axis": field, "yes": counts["yes"], "no": counts["no"], "uncertain": counts["uncertain"], "not_applicable": counts["not_applicable"], "ambiguous": counts["ambiguous"], "binary_positive_rate": counts["yes"] / binary if binary else None})
    classes = {audit_id: row["evidence_class"] for audit_id, row in finalized.items() if row.get("evidence_class") and row["evidence_class"] != "ambiguous"}
    comparison = _system_comparison(classes, key)
    complete = sum(all(row.get(field) for field in AXIS_FIELDS) for row in finalized.values())
    unresolved = sum(any(value == "ambiguous" for value in row.values()) for row in finalized.values())
    legacy_only = not rows_b and not rows_adj and not any(
        _field(row, field) for row in rows_a for field in AXIS_FIELDS
    )
    coverage = (
        comparison["labeled_rows"] / len(rows_a)
        if legacy_only and rows_a
        else complete / len(rows_a) if rows_a else 0.0
    )
    report = {
        "schema_version": "tapbench.evidence_audit_report.v2", "audit_version": AUDIT_VERSION,
        "total_rows": len(rows_a), "labeled_rows": comparison["labeled_rows"],
        "coverage": coverage, "double_annotation": bool(rows_b),
        "adjudication_supplied": bool(rows_adj), "interannotator_agreement": agreement,
        "adjudication": {"disagreement_rows": len(disagreement_ids), "ambiguity_rate": len(disagreement_ids) / len(rows_a) if rows_a else 0.0, "unresolved_disagreement_rows": unresolved},
        "axis_metrics": axis_metrics, "system_label_comparison": comparison,
        "accuracy": comparison["accuracy"], "per_label": comparison["per_label"], "confusion": comparison["confusion"],
        "status": "complete" if ((legacy_only and comparison["labeled_rows"] == len(rows_a)) or (complete == len(rows_a) and not unresolved)) else "pending_human_annotation_or_adjudication",
        "interpretation": "Axis metrics are human judgments. System-label agreement is diagnostic and is not independent oracle validation.",
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

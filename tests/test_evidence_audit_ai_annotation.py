from scripts.annotate_evidence_audit_ai import annotate_row


def _row(**updates: str) -> dict[str, str]:
    row = {
        "audit_id": "example",
        "task_kind": "call",
        "source_text": "Rome",
        "transform": "slot_specific_exact_match",
    }
    row.update(updates)
    return row


def test_ai_annotation_accepts_complete_active_span() -> None:
    annotation = annotate_row(_row())
    assert annotation["evidence_class"] == "explicit"
    assert annotation["span_support"] == "yes"
    assert annotation["contract_correct"] == "uncertain"
    assert annotation["human_label"] == ""


def test_ai_annotation_rejects_reviewed_extent_failure() -> None:
    annotation = annotate_row(
        _row(
            audit_id="ledger-0001",
            source_text="Status update",
            transform="synthetic_family_normalizer",
        )
    )
    assert annotation["evidence_class"] == "unsupported"
    assert annotation["span_support"] == "no"
    assert annotation["slot_role_correct"] == "yes"
    assert annotation["scope_correct"] == "no"


def test_ai_annotation_rejects_missing_value_placeholder() -> None:
    annotation = annotate_row(
        _row(
            task_kind="missing_info",
            source_text="",
            transform="",
        )
    )
    assert annotation["evidence_class"] == "unsupported"
    assert annotation["span_support"] == "not_applicable"
    assert annotation["normalization_correct"] == "not_applicable"
    assert annotation["contradiction_correct"] == "no"

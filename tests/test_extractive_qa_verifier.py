from __future__ import annotations

from tapbench.extractive_qa_verifier import (
    EXTRACTIVE_QA_ARTIFACT_SCHEMA,
    EXTRACTIVE_QA_MAX_ANSWER_TOKENS,
    EXTRACTIVE_QA_MAX_INPUT_TOKENS,
    EXTRACTIVE_QA_MODEL_ID,
    EXTRACTIVE_QA_MODEL_REVISION,
    EXTRACTIVE_QA_QUESTION_VERSION,
    EXTRACTIVE_QA_VERIFIER_VERSION,
    _json_sha256,
    select_extractive_answer,
    validate_extractive_qa_rows,
    verifier_question,
)


def _selection(
    request: str,
    *,
    offsets: list[tuple[int, int]],
    start: list[float],
    end: list[float],
) -> dict:
    return select_extractive_answer(
        request_text=request,
        language="en-US",
        offsets=offsets,
        sequence_ids=[None, 1, 1],
        start_logits=start,
        end_logits=end,
        cls_index=0,
        input_tokens=3,
    )


def test_natural_questions_are_deterministic() -> None:
    assert verifier_question("definition_word") == "What word should be defined?"
    assert verifier_question("person") == "Who is the person?"
    assert verifier_question("artist_name") == "Who is the artist name?"
    assert verifier_question("business_name") == "What is the business name?"


def test_positive_exact_answer_is_admitted_and_lattice_aligned() -> None:
    row = _selection(
        "Email Alice",
        offsets=[(0, 0), (0, 5), (6, 11)],
        start=[0.0, 0.0, 5.0],
        end=[0.0, 0.0, 5.0],
    )
    assert row["status"] == "admitted"
    assert row["answer"] == "Alice"
    assert row["answer_span"] == [6, 11]
    assert row["span_id"] is not None
    assert row["non_null_margin"] == 10.0


def test_sentencepiece_leading_space_is_trimmed_before_alignment() -> None:
    row = _selection(
        "Email Alice",
        offsets=[(0, 0), (0, 5), (5, 11)],
        start=[0.0, 0.0, 5.0],
        end=[0.0, 0.0, 5.0],
    )
    assert row["status"] == "admitted"
    assert row["answer"] == "Alice"
    assert row["answer_span"] == [6, 11]


def test_null_margin_and_whole_request_fail_closed() -> None:
    null = _selection(
        "Email Alice",
        offsets=[(0, 0), (0, 5), (6, 11)],
        start=[10.0, 0.0, 1.0],
        end=[10.0, 0.0, 1.0],
    )
    whole = _selection(
        "Email Alice",
        offsets=[(0, 0), (0, 11), (6, 11)],
        start=[0.0, 5.0, 0.0],
        end=[0.0, 5.0, 0.0],
    )
    assert null["status"] == "null_margin"
    assert null["answer"] is None
    assert whole["status"] == "whole_request_rejected"
    assert whole["answer"] is None


def test_misaligned_subword_fails_closed() -> None:
    row = _selection(
        "Email Alice",
        offsets=[(0, 0), (0, 5), (7, 11)],
        start=[0.0, 0.0, 5.0],
        end=[0.0, 0.0, 5.0],
    )
    assert row["status"] == "source_lattice_misaligned"
    assert row["answer"] is None


def _valid_artifact_row() -> dict:
    row = {
        "schema_version": EXTRACTIVE_QA_ARTIFACT_SCHEMA,
        "verifier_version": EXTRACTIVE_QA_VERIFIER_VERSION,
        "question_version": EXTRACTIVE_QA_QUESTION_VERSION,
        "model_id": EXTRACTIVE_QA_MODEL_ID,
        "model_revision": EXTRACTIVE_QA_MODEL_REVISION,
        "backend": "huggingface_transformers_cpu",
        "dtype": "float32",
        "margin_threshold": 0.0,
        "max_input_tokens": EXTRACTIVE_QA_MAX_INPUT_TOKENS,
        "max_answer_tokens": EXTRACTIVE_QA_MAX_ANSWER_TOKENS,
        "input_truncated": False,
        "gold_loaded": False,
        "case_id": "c1",
        "tool": "email.send",
        "surface_name": "person",
        "status": "admitted",
        "admitted": True,
    }
    row["row_sha256"] = _json_sha256(row)
    return row


def test_artifact_validator_detects_hash_tampering_and_duplicates() -> None:
    row = _valid_artifact_row()
    assert validate_extractive_qa_rows([row]) == []
    tampered = {**row, "admitted": False}
    failures = validate_extractive_qa_rows([tampered, row])
    kinds = {failure for item in failures for failure in item["failures"]}
    assert "row_sha256_mismatch" in kinds
    assert "admission_status_mismatch" in kinds
    assert "duplicate_identity" in kinds

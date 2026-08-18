from __future__ import annotations

import json
from pathlib import Path

from tapbench.summarize import write_summary_tables


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_summary_tables_include_h6_contrast_and_slot_counts(tmp_path: Path) -> None:
    scores = tmp_path / "scores.jsonl"
    slot_errors = tmp_path / "slot_errors.jsonl"
    _write_jsonl(
        scores,
        [
            {
                "hypothesis": "H6",
                "hypothesis_grid_id": "H6_abstention_suppression",
                "task_kind": "missing_info",
                "method": "constrained_abstain_b2",
                "model_id": "m",
                "format_valid": True,
                "schema_valid": True,
                "execution_success": True,
                "fabrication": False,
                "mode_correct": True,
            },
            {
                "hypothesis": "H6",
                "hypothesis_grid_id": "H6_abstention_suppression",
                "task_kind": "missing_info",
                "method": "constrained_call_only_b2",
                "model_id": "m",
                "format_valid": True,
                "schema_valid": True,
                "execution_success": False,
                "fabrication": True,
                "mode_correct": False,
            },
        ],
    )
    _write_jsonl(slot_errors, [{"family": "calendar", "hypothesis": "H6", "error_type": "unsupported_fabricated_value"}])

    payload = write_summary_tables(scores, tmp_path / "summary", slot_errors_path=slot_errors)

    assert payload["n_scores"] == 2
    assert payload["n_slot_errors"] == 1
    contrast = next(
        row
        for row in payload["tables"]["h6_abstention_contrasts"]
        if row["task_kind"] == "missing_info" and row["metric"] == "fabrication"
    )
    assert contrast["rate_difference_abstain_minus_call_only"] == -1.0
    assert (tmp_path / "summary" / "slot_error_counts.csv").exists()

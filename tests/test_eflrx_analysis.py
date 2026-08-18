from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_eflrx_results import METHODS, analyze


def _write(path: Path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )


def test_eflrx_analysis_pairs_call_and_no_call_strata(tmp_path) -> None:
    call_case = {
        "case_id": "call",
        "metadata": {"bfcl_category": "simple_java"},
        "bfcl_gold": {
            "allowed_calls": [
                {"tool": "submit", "arguments": {"count": [3]}}
            ]
        },
    }
    no_call_case = {
        "case_id": "no_call",
        "metadata": {"bfcl_category": "live_irrelevance"},
        "bfcl_gold": {"allowed_calls": []},
    }
    cases = [call_case, no_call_case]
    predictions = []
    scores = []
    official = []
    for method in METHODS:
        for case in cases:
            is_call = case["case_id"] == "call"
            correct = not (
                method == "full_tap_b2" and is_call
            )
            if is_call and correct:
                action = {
                    "mode": "call",
                    "tool": "submit",
                    "arguments": {"count": 3},
                    "payload": {},
                }
            else:
                action = {
                    "mode": "no_tool",
                    "tool": None,
                    "arguments": {},
                    "payload": {},
                }
            metadata = {
                "finish_reason": "stop",
                "generation_calls": 1,
                "context_overflow": False,
                "context_headroom_tokens_min": 1000,
                "preflight_prompt_token_delta_max_abs": 0,
            }
            if method.startswith("tap_r_eflrx") and is_call and correct:
                metadata["evidence_certificates"] = {
                    "count": {
                        "value": 3,
                        "source_span": [10, 11],
                        "component_spans": [[10, 11]],
                    }
                }
            base = {
                "case_id": case["case_id"],
                "model_id": "model",
                "method": method,
                "seed": 1,
            }
            predictions.append(
                {
                    **base,
                    "prediction": action,
                    "response_metadata": metadata,
                    "runner_error": None,
                    "backend": "llama.cpp",
                    "quantization": "Q4_K_M",
                    "chat_template": "test",
                    "grammar_engine": "raw_json_schema",
                    "model_artifact": "model.gguf",
                    "thinking_mode": "off",
                    "thinking_marker_detected": False,
                }
            )
            scores.append({**base, "format_valid": True})
            official.append({**base, "valid": correct})

    paths = {
        name: tmp_path / f"{name}.jsonl"
        for name in ("cases", "predictions", "scores", "official")
    }
    _write(paths["cases"], cases)
    _write(paths["predictions"], predictions)
    _write(paths["scores"], scores)
    _write(paths["official"], official)
    report = analyze(
        paths["cases"],
        paths["predictions"],
        paths["scores"],
        paths["official"],
        scope="holdout",
        replicates=100,
    )
    primary = next(
        row
        for row in report["groups"]
        if row["model_id"] == "POOLED"
        and row["method"] == "tap_r_eflrx_consensus"
    )
    assert primary["category_macro_accuracy"] == 1.0
    assert primary["accepted_call_exact_precision"] == 1.0
    assert primary["call_non_escalated_coverage"] == 1.0
    contrast = report["contrasts"]["eflrx_vs_full_tap"]
    assert contrast["point_estimate"] == 0.5
    assert report["engineering_passed"]
    assert report["confirmatory_passed"]

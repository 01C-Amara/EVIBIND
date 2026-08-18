from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .extractive_qa_verifier import (
    EXTRACTIVE_QA_VERIFIER_VERSION,
    validate_extractive_qa_rows,
)
from .io import read_jsonl
from .massive_surface_smoke import (
    _grouped,
    _load_study,
    _metric,
    _payload_without_split,
)
from .qa_evidence_controller import (
    QA_EVIDENCE_SYSTEM_LABEL,
    QA_EVIDENCE_CONTROLLER_VERSION,
)


QA_HYBRID_SMOKE_ANALYSIS_VERSION = "tapbench.massive_qa_hybrid_smoke.v1"
V4_SINGLE = "tap_r_surface_active_single"
V7_SINGLE = "tap_r_qa_active_slots_single"
V7_METHODS = (
    "tap_r_qa_all_slots_single",
    V7_SINGLE,
    "tap_r_qa_active_slots_consensus",
)


def analyze_qa_hybrid_smoke(
    v4_path: str | Path,
    v7_path: str | Path,
    output_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Any]:
    v4 = _load_study(Path(v4_path))
    v7 = _load_study(Path(v7_path))
    case_ids_equal = set(v4["cases"]) == set(v7["cases"])
    case_payloads_equal = case_ids_equal and all(
        _payload_without_split(v4["cases"][case_id])
        == _payload_without_split(v7["cases"][case_id])
        for case_id in v4["cases"]
    )
    gold_equal = v4["gold"] == v7["gold"]
    v4_groups = _grouped(v4["details"])
    v7_groups = _grouped(v7["details"])
    models_v4 = {str(row["model_id"]) for row in v4["predictions"]}
    models_v7 = {str(row["model_id"]) for row in v7["predictions"]}
    models = sorted(models_v4 & models_v7)

    comparisons: list[dict[str, Any]] = []
    overbinding_passes: list[bool] = []
    for model in models:
        old = _metric(v4_groups, model, V4_SINGLE)
        new = _metric(v7_groups, model, V7_SINGLE)
        old_rate = float(old["extra_optional_slots_per_emitted_call"] or 0.0)
        new_rate = float(new["extra_optional_slots_per_emitted_call"] or 0.0)
        passes = new_rate <= old_rate + 1e-12
        overbinding_passes.append(passes)
        comparisons.append(
            {
                "model_id": model,
                "v4_official_correct_calls": old["official_correct_calls"],
                "v7_official_correct_calls": new["official_correct_calls"],
                "v4_extra_optional_slots_per_emitted_call": old_rate,
                "v7_extra_optional_slots_per_emitted_call": new_rate,
                "overbinding_not_worse": passes,
            }
        )

    audit = json.loads(
        (Path(v7_path) / "certificate_audit_summary.json").read_text(
            encoding="utf-8"
        )
    )
    verifier_rows = read_jsonl(Path(v7_path) / "verifier.jsonl")
    verifier_failures = validate_extractive_qa_rows(verifier_rows)
    verifier_truncations = sum(
        bool(row.get("input_truncated")) for row in verifier_rows
    )
    verifier_gold_leaks = sum(
        bool(row.get("gold_loaded")) for row in verifier_rows
    )
    discipline_path = Path(v7_path) / "discipline_failures.jsonl"
    discipline = read_jsonl(discipline_path) if discipline_path.exists() else [{}]
    prediction_gate_failures = [
        {
            "case_id": row.get("case_id"),
            "model_id": row.get("model_id"),
            "method": row.get("method"),
        }
        for row in v7["predictions"]
        if row.get("runner_error")
        or row.get("thinking_marker_detected")
        or row.get("finish_reason") == "length"
        or row.get("qa_evidence_controller_version")
        != QA_EVIDENCE_CONTROLLER_VERSION
        or row.get("qa_evidence_system_label") != QA_EVIDENCE_SYSTEM_LABEL
        or row.get("qa_verifier_version") != EXTRACTIVE_QA_VERIFIER_VERSION
        or row.get("response_metadata", {}).get("context_overflow")
        or row.get("response_metadata", {}).get("context_truncated")
        or (
            row.get("response_metadata", {}).get(
                "preflight_prompt_token_delta_max_abs"
            )
            or 0
        )
        > 1
    ]

    input_equivalence = True
    v4_by_base = {
        (
            str(row.get("case_id")),
            str(row.get("model_id")),
            int(row.get("seed", 0)),
        ): row
        for row in v4["predictions"]
        if row.get("method") == V4_SINGLE
    }
    for row in v7["predictions"]:
        if row.get("method") != V7_SINGLE:
            continue
        old = v4_by_base.get(
            (
                str(row.get("case_id")),
                str(row.get("model_id")),
                int(row.get("seed", 0)),
            )
        )
        if old is None or any(
            old.get(field) != row.get(field)
            for field in (
                "backend",
                "quantization",
                "model_artifact",
                "chat_template",
                "grammar_engine",
                "thinking_mode",
                "reasoning_budget",
                "max_output_tokens",
                "ranking_artifact_sha256",
            )
        ):
            input_equivalence = False
            break

    verifier_artifact_hashes = {
        str(row.get("qa_verifier_artifact_sha256"))
        for row in v7["predictions"]
    }
    correct_by_model = {
        model: sum(
            bool(row["official_correct"])
            for row in v7["details"]
            if row["model_id"] == model and row["method"] in V7_METHODS
        )
        for model in models
    }
    integrity_passed = all(
        (
            case_ids_equal,
            case_payloads_equal,
            gold_equal,
            models_v4 == models_v7,
            len(models) == 2,
            input_equivalence,
            audit.get("failed") == 0,
            audit.get("passed") == len(v7["predictions"]),
            not audit.get("verifier_artifact_failures"),
            not verifier_failures,
            verifier_truncations == 0,
            verifier_gold_leaks == 0,
            len(verifier_artifact_hashes) == 1,
            audit.get("qa_verifier_artifact_sha256")
            in verifier_artifact_hashes,
            not discipline,
            not prediction_gate_failures,
        )
    )
    correct_gate = all(value >= 1 for value in correct_by_model.values())
    overbinding_gate = bool(overbinding_passes) and all(overbinding_passes)
    failures: list[str] = []
    if not integrity_passed:
        failures.append("engineering_integrity_gate_failed")
    if not correct_gate:
        failures.append("fewer_than_one_official_correct_call_for_a_model")
    if not overbinding_gate:
        failures.append("active_single_overbinding_worse_than_v4_for_a_model")

    report = {
        "schema_version": QA_HYBRID_SMOKE_ANALYSIS_VERSION,
        "interpretation": "engineering_smoke_only",
        "reporting_label": QA_EVIDENCE_SYSTEM_LABEL,
        "v4_path": str(v4_path),
        "v7_path": str(v7_path),
        "integrity": {
            "case_ids_equal": case_ids_equal,
            "case_payloads_equal_excluding_split": case_payloads_equal,
            "scorer_gold_equal": gold_equal,
            "model_sets_equal": models_v4 == models_v7,
            "single_condition_fixed_inputs_equal": input_equivalence,
            "certificate_failed": audit.get("failed"),
            "verifier_artifact_failure_count": len(verifier_failures),
            "verifier_truncation_count": verifier_truncations,
            "verifier_gold_loaded_count": verifier_gold_leaks,
            "discipline_failure_count": len(discipline),
            "prediction_gate_failures": prediction_gate_failures,
            "passed": integrity_passed,
        },
        "v4_groups": v4_groups,
        "v7_groups": v7_groups,
        "active_single_comparison": comparisons,
        "official_correct_calls_by_model_across_v7_candidates": correct_by_model,
        "gates": {
            "engineering_integrity": integrity_passed,
            "official_correct_calls_per_model_min_1": correct_gate,
            "active_single_overbinding_not_worse_than_v4_per_model": (
                overbinding_gate
            ),
        },
        "decision": {
            "design_permitted": not failures,
            "failures": failures,
            "next_partition_if_permitted": "development ranks 3 through 48",
            "confirmation_access_permitted": False,
            "holdout_access_permitted": False,
        },
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# QA-Evidence Hybrid V7 Smoke Gate",
        "",
        "This is engineering evidence for the explicitly hybrid system "
        f"`{QA_EVIDENCE_SYSTEM_LABEL}`; it is not a held-out result.",
        "",
        f"- Integrity gate: **{'PASS' if integrity_passed else 'FAIL'}**",
        f"- At least one official correct call per model: **{'PASS' if correct_gate else 'FAIL'}**",
        f"- Active-single overbinding no worse than v4: **{'PASS' if overbinding_gate else 'FAIL'}**",
        f"- Design selection permitted: **{'YES' if not failures else 'NO'}**",
        "",
        "| Model | V4 correct | V7 correct | V4 extra/call | V7 extra/call | Gate |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in comparisons:
        lines.append(
            "| {model_id} | {v4c} | {v7c} | {v4e:.3f} | {v7e:.3f} | {gate} |".format(
                model_id=row["model_id"],
                v4c=row["v4_official_correct_calls"],
                v7c=row["v7_official_correct_calls"],
                v4e=row["v4_extra_optional_slots_per_emitted_call"],
                v7e=row["v7_extra_optional_slots_per_emitted_call"],
                gate="PASS" if row["overbinding_not_worse"] else "FAIL",
            )
        )
    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report

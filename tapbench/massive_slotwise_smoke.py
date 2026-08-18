from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import read_jsonl
from .massive_surface_smoke import (
    _grouped,
    _load_study,
    _metric,
    _payload_without_split,
)


SLOTWISE_SMOKE_ANALYSIS_VERSION = "tapbench.massive_slotwise_smoke.v1"
V4_SINGLE = "tap_r_surface_active_single"
V5_SINGLE = "tap_r_slotwise_surface_single"
V5_METHODS = (
    V5_SINGLE,
    "tap_r_slotwise_surface_consensus",
    "tap_r_slotwise_surface_consensus_top1",
)


def analyze_slotwise_smoke(
    v4_path: str | Path,
    v5_path: str | Path,
    output_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Any]:
    v4 = _load_study(Path(v4_path))
    v5 = _load_study(Path(v5_path))
    case_ids_equal = set(v4["cases"]) == set(v5["cases"])
    case_payloads_equal = case_ids_equal and all(
        _payload_without_split(v4["cases"][case_id])
        == _payload_without_split(v5["cases"][case_id])
        for case_id in v4["cases"]
    )
    gold_equal = v4["gold"] == v5["gold"]
    v4_groups = _grouped(v4["details"])
    v5_groups = _grouped(v5["details"])
    models_v4 = {str(row["model_id"]) for row in v4["predictions"]}
    models_v5 = {str(row["model_id"]) for row in v5["predictions"]}
    models = sorted(models_v4 & models_v5)

    comparisons: list[dict[str, Any]] = []
    overbinding_passes: list[bool] = []
    for model in models:
        old = _metric(v4_groups, model, V4_SINGLE)
        new = _metric(v5_groups, model, V5_SINGLE)
        old_rate = float(old["extra_optional_slots_per_emitted_call"] or 0.0)
        new_rate = float(new["extra_optional_slots_per_emitted_call"] or 0.0)
        passes = new_rate <= old_rate + 1e-12
        overbinding_passes.append(passes)
        comparisons.append(
            {
                "model_id": model,
                "v4_official_correct_calls": old["official_correct_calls"],
                "v5_official_correct_calls": new["official_correct_calls"],
                "v4_extra_optional_slots_per_emitted_call": old_rate,
                "v5_extra_optional_slots_per_emitted_call": new_rate,
                "overbinding_not_worse": passes,
            }
        )

    cert_path = Path(v5_path) / "certificate_audit_summary.json"
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    discipline_path = Path(v5_path) / "discipline_failures.jsonl"
    discipline = read_jsonl(discipline_path) if discipline_path.exists() else [{}]
    prediction_gate_failures = [
        {
            "case_id": row.get("case_id"),
            "model_id": row.get("model_id"),
            "method": row.get("method"),
        }
        for row in v5["predictions"]
        if row.get("runner_error")
        or row.get("thinking_marker_detected")
        or row.get("finish_reason") == "length"
        or row.get("response_metadata", {}).get("context_overflow")
        or row.get("response_metadata", {}).get("context_truncated")
        or (row.get("response_metadata", {}).get(
            "preflight_prompt_token_delta_max_abs"
        ) or 0) > 1
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
    for row in v5["predictions"]:
        if row.get("method") != V5_SINGLE:
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

    correct_by_model = {
        model: sum(
            bool(row["official_correct"])
            for row in v5["details"]
            if row["model_id"] == model and row["method"] in V5_METHODS
        )
        for model in models
    }
    integrity_passed = all(
        (
            case_ids_equal,
            case_payloads_equal,
            gold_equal,
            models_v4 == models_v5,
            len(models) == 2,
            input_equivalence,
            cert.get("failed") == 0,
            cert.get("passed") == len(v5["predictions"]),
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
        failures.append("single_view_overbinding_worse_than_v4_for_a_model")
    report = {
        "schema_version": SLOTWISE_SMOKE_ANALYSIS_VERSION,
        "interpretation": "engineering_smoke_only",
        "v4_path": str(v4_path),
        "v5_path": str(v5_path),
        "integrity": {
            "case_ids_equal": case_ids_equal,
            "case_payloads_equal_excluding_split": case_payloads_equal,
            "scorer_gold_equal": gold_equal,
            "model_sets_equal": models_v4 == models_v5,
            "single_condition_fixed_inputs_equal": input_equivalence,
            "certificate_failed": cert.get("failed"),
            "discipline_failure_count": len(discipline),
            "prediction_gate_failures": prediction_gate_failures,
            "passed": integrity_passed,
        },
        "v4_groups": v4_groups,
        "v5_groups": v5_groups,
        "single_view_comparison": comparisons,
        "official_correct_calls_by_model_across_v5_candidates": correct_by_model,
        "gates": {
            "engineering_integrity": integrity_passed,
            "official_correct_calls_per_model_min_1": correct_gate,
            "single_view_overbinding_not_worse_than_v4_per_model": (
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
        "# Slotwise Minimal-Surface V5 Smoke Gate",
        "",
        "This report is engineering evidence only; it is not a held-out result.",
        "",
        f"- Integrity gate: **{'PASS' if integrity_passed else 'FAIL'}**",
        f"- At least one official correct call per model: **{'PASS' if correct_gate else 'FAIL'}**",
        f"- Single-view overbinding no worse than v4 per model: **{'PASS' if overbinding_gate else 'FAIL'}**",
        f"- Design selection permitted: **{'YES' if not failures else 'NO'}**",
        "",
        "| Model | V4 correct | V5 correct | V4 extra/call | V5 extra/call | Gate |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in comparisons:
        lines.append(
            "| {model_id} | {v4c} | {v5c} | {v4e:.3f} | {v5e:.3f} | {gate} |".format(
                model_id=row["model_id"],
                v4c=row["v4_official_correct_calls"],
                v5c=row["v5_official_correct_calls"],
                v4e=row["v4_extra_optional_slots_per_emitted_call"],
                v5e=row["v5_extra_optional_slots_per_emitted_call"],
                gate="PASS" if row["overbinding_not_worse"] else "FAIL",
            )
        )
    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report

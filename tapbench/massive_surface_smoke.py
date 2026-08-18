from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl


SURFACE_SMOKE_ANALYSIS_VERSION = "tapbench.massive_surface_smoke.v1"
V2_SINGLE = "tap_r_retrieve_pointer_single"
V4_SINGLE = "tap_r_surface_active_single"
V4_METHODS = (
    V4_SINGLE,
    "tap_r_surface_active_consensus",
    "tap_r_surface_active_consensus_top1",
)


def _normalized(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split())


def _user_text(case: dict[str, Any]) -> str:
    return "\n".join(
        str(message.get("content", ""))
        for message in case.get("messages", [])
        if str(message.get("role", "")).casefold() == "user"
    )


def _gold_calls(gold: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    raw = gold.get("ground_truth", [])
    if not isinstance(raw, list):
        return calls
    for item in raw:
        if not isinstance(item, dict):
            continue
        for tool, arguments in item.items():
            calls.append(
                (
                    str(tool),
                    arguments if isinstance(arguments, dict) else {},
                )
            )
    return calls


def _explicit_slots(
    arguments: dict[str, Any], request_text: str
) -> dict[str, set[str]]:
    request = _normalized(request_text)
    output: dict[str, set[str]] = {}
    for slot, aliases in arguments.items():
        values = aliases if isinstance(aliases, list) else [aliases]
        supported = {
            _normalized(value)
            for value in values
            if _normalized(value) and _normalized(value) in request
        }
        if supported:
            output[str(slot)] = supported
    return output


def _row_diagnostic(
    case: dict[str, Any],
    gold: dict[str, Any],
    prediction: dict[str, Any],
    official: dict[str, Any],
) -> dict[str, Any]:
    action = prediction.get("prediction")
    emitted = bool(isinstance(action, dict) and action.get("mode") == "call")
    predicted_tool = str(action.get("tool", "")) if emitted else ""
    predicted_arguments = (
        action.get("arguments", {}) if emitted else {}
    )
    if not isinstance(predicted_arguments, dict):
        predicted_arguments = {}
    calls = _gold_calls(gold)
    matching = [args for tool, args in calls if tool == predicted_tool]
    tool_correct = bool(emitted and matching)
    request_text = _user_text(case)
    if matching:
        explicit_options = [_explicit_slots(args, request_text) for args in matching]
        explicit = max(
            explicit_options,
            key=lambda option: (
                len(set(predicted_arguments) & set(option)),
                -len(set(predicted_arguments) - set(option)),
            ),
        )
    else:
        all_gold = [_explicit_slots(args, request_text) for _, args in calls]
        explicit = all_gold[0] if all_gold else {}
    predicted_slots = set(map(str, predicted_arguments))
    explicit_slots = set(explicit)
    if tool_correct:
        true_positive = predicted_slots & explicit_slots
        false_positive = predicted_slots - explicit_slots
        false_negative = explicit_slots - predicted_slots
    else:
        true_positive = set()
        false_positive = predicted_slots
        false_negative = explicit_slots
    wrong_values = 0
    for slot in true_positive:
        if _normalized(predicted_arguments[slot]) not in explicit[slot]:
            wrong_values += 1
    return {
        "case_id": prediction.get("case_id"),
        "model_id": prediction.get("model_id"),
        "method": prediction.get("method"),
        "emitted_call": emitted,
        "official_correct": bool(official.get("official_ast_correct")),
        "tool_correct": tool_correct,
        "predicted_slot_count": len(predicted_slots),
        "explicit_gold_slot_count": len(explicit_slots),
        "slot_true_positive": len(true_positive),
        "slot_false_positive": len(false_positive),
        "slot_false_negative": len(false_negative),
        "wrong_explicit_value": wrong_values,
        "extra_optional_slots": len(false_positive),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    emitted = sum(bool(row["emitted_call"]) for row in rows)
    correct = sum(bool(row["official_correct"]) for row in rows)
    tool_correct = sum(bool(row["tool_correct"]) for row in rows)
    tp = sum(int(row["slot_true_positive"]) for row in rows)
    fp = sum(int(row["slot_false_positive"]) for row in rows)
    fn = sum(int(row["slot_false_negative"]) for row in rows)
    return {
        "n": n,
        "emitted_calls": emitted,
        "official_correct_calls": correct,
        "tool_correct_calls": tool_correct,
        "call_coverage": emitted / n if n else None,
        "accepted_call_exact_precision": correct / emitted if emitted else None,
        "tool_accuracy": tool_correct / n if n else None,
        "argument_exact_accuracy_given_correct_tool": (
            correct / tool_correct if tool_correct else None
        ),
        "slot_true_positive": tp,
        "slot_false_positive": fp,
        "slot_false_negative": fn,
        "active_slot_micro_precision": tp / (tp + fp) if tp + fp else None,
        "active_slot_micro_recall": tp / (tp + fn) if tp + fn else None,
        "wrong_explicit_values": sum(
            int(row["wrong_explicit_value"]) for row in rows
        ),
        "extra_optional_slots": fp,
        "extra_optional_slots_per_emitted_call": (
            fp / emitted if emitted else None
        ),
    }


def _load_study(path: Path) -> dict[str, Any]:
    cases = {str(row["case_id"]): row for row in read_jsonl(path / "cases.jsonl")}
    gold = {
        str(row["case_id"]): row
        for row in read_jsonl(path / "scorer_only" / "gold.jsonl")
    }
    predictions = read_jsonl(path / "predictions.jsonl")
    official = read_jsonl(path / "official" / "official_details.jsonl")
    identity = lambda row: (
        str(row.get("case_id")),
        str(row.get("model_id")),
        str(row.get("method")),
        int(row.get("seed", 0)),
    )
    official_by_id = {identity(row): row for row in official}
    if len(official_by_id) != len(official):
        raise ValueError(f"duplicate official identities in {path}")
    details: list[dict[str, Any]] = []
    for prediction in predictions:
        case_id = str(prediction.get("case_id"))
        key = identity(prediction)
        if case_id not in cases or case_id not in gold or key not in official_by_id:
            raise ValueError(f"unmatched smoke row {key} in {path}")
        details.append(
            _row_diagnostic(
                cases[case_id],
                gold[case_id],
                prediction,
                official_by_id[key],
            )
        )
    return {
        "path": path,
        "cases": cases,
        "gold": gold,
        "predictions": predictions,
        "official": official,
        "details": details,
    }


def _grouped(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        groups[(str(row["model_id"]), str(row["method"]))].append(row)
    return [
        {"model_id": model, "method": method, **_aggregate(rows)}
        for (model, method), rows in sorted(groups.items())
    ]


def _metric(
    groups: list[dict[str, Any]], model: str, method: str
) -> dict[str, Any]:
    return next(
        row
        for row in groups
        if row["model_id"] == model and row["method"] == method
    )


def _payload_without_split(case: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in case.items() if key != "split"}


def analyze_surface_smoke(
    v2_path: str | Path,
    v4_path: str | Path,
    output_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Any]:
    v2 = _load_study(Path(v2_path))
    v4 = _load_study(Path(v4_path))
    case_ids_equal = set(v2["cases"]) == set(v4["cases"])
    case_payloads_equal = case_ids_equal and all(
        _payload_without_split(v2["cases"][case_id])
        == _payload_without_split(v4["cases"][case_id])
        for case_id in v2["cases"]
    )
    gold_equal = v2["gold"] == v4["gold"]
    v2_groups = _grouped(v2["details"])
    v4_groups = _grouped(v4["details"])
    models_v2 = {str(row["model_id"]) for row in v2["predictions"]}
    models_v4 = {str(row["model_id"]) for row in v4["predictions"]}
    models = sorted(models_v2 & models_v4)

    comparisons: list[dict[str, Any]] = []
    reductions: list[float] = []
    for model in models:
        old = _metric(v2_groups, model, V2_SINGLE)
        new = _metric(v4_groups, model, V4_SINGLE)
        old_rate = float(old["extra_optional_slots_per_emitted_call"] or 0.0)
        new_rate = float(new["extra_optional_slots_per_emitted_call"] or 0.0)
        reduction = (
            (old_rate - new_rate) / old_rate
            if old_rate > 0
            else (1.0 if new_rate == 0 else float("-inf"))
        )
        reductions.append(reduction)
        comparisons.append(
            {
                "model_id": model,
                "v2_extra_optional_slots_per_emitted_call": old_rate,
                "v4_extra_optional_slots_per_emitted_call": new_rate,
                "relative_reduction": reduction,
                "passes_50_percent_reduction": reduction >= 0.50,
            }
        )

    cert_path = Path(v4_path) / "certificate_audit_summary.json"
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    discipline_path = Path(v4_path) / "discipline_failures.jsonl"
    discipline = read_jsonl(discipline_path) if discipline_path.exists() else [{}]
    prediction_gate_failures = [
        {
            "case_id": row.get("case_id"),
            "model_id": row.get("model_id"),
            "method": row.get("method"),
        }
        for row in v4["predictions"]
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
    v2_by_base = {
        (
            str(row.get("case_id")),
            str(row.get("model_id")),
            int(row.get("seed", 0)),
        ): row
        for row in v2["predictions"]
        if row.get("method") == V2_SINGLE
    }
    for row in v4["predictions"]:
        if row.get("method") != V4_SINGLE:
            continue
        old = v2_by_base.get(
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
            for row in v4["details"]
            if row["model_id"] == model and row["method"] in V4_METHODS
        )
        for model in models
    }
    integrity_passed = all(
        (
            case_ids_equal,
            case_payloads_equal,
            gold_equal,
            models_v2 == models_v4,
            len(models) == 2,
            input_equivalence,
            cert.get("failed") == 0,
            cert.get("passed") == len(v4["predictions"]),
            not discipline,
            not prediction_gate_failures,
        )
    )
    correct_gate = all(value >= 1 for value in correct_by_model.values())
    reduction_gate = bool(reductions) and all(value >= 0.50 for value in reductions)
    failures: list[str] = []
    if not integrity_passed:
        failures.append("engineering_integrity_gate_failed")
    if not correct_gate:
        failures.append("fewer_than_one_official_correct_call_for_a_model")
    if not reduction_gate:
        failures.append("single_view_overbinding_reduction_below_50_percent")
    report = {
        "schema_version": SURFACE_SMOKE_ANALYSIS_VERSION,
        "interpretation": "engineering_smoke_only",
        "v2_path": str(v2_path),
        "v4_path": str(v4_path),
        "integrity": {
            "case_ids_equal": case_ids_equal,
            "case_payloads_equal_excluding_split": case_payloads_equal,
            "scorer_gold_equal": gold_equal,
            "model_sets_equal": models_v2 == models_v4,
            "single_condition_fixed_inputs_equal": input_equivalence,
            "certificate_failed": cert.get("failed"),
            "discipline_failure_count": len(discipline),
            "prediction_gate_failures": prediction_gate_failures,
            "passed": integrity_passed,
        },
        "gold_active_slot_rule": (
            "A slot is explicit iff at least one nonempty accepted gold value "
            "occurs in the NFKC-casefolded user request."
        ),
        "v2_groups": v2_groups,
        "v4_groups": v4_groups,
        "single_view_comparison": comparisons,
        "official_correct_calls_by_model_across_v4_candidates": correct_by_model,
        "gates": {
            "engineering_integrity": integrity_passed,
            "official_correct_calls_per_model_min_1": correct_gate,
            "single_view_overbinding_reduction_per_model_min_0_50": reduction_gate,
        },
        "decision": {
            "design_permitted": not failures,
            "failures": failures,
            "next_partition_if_permitted": "development ranks 3 through 48",
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
        "# Semantic-Surface V4 Smoke Gate",
        "",
        "This report is engineering evidence only; it is not a held-out result.",
        "",
        f"- Integrity gate: **{'PASS' if integrity_passed else 'FAIL'}**",
        f"- At least one official correct call per model: **{'PASS' if correct_gate else 'FAIL'}**",
        f"- At least 50% single-view overbinding reduction per model: **{'PASS' if reduction_gate else 'FAIL'}**",
        f"- Design selection permitted: **{'YES' if not failures else 'NO'}**",
        "",
        "| Model | V2 extra slots/call | V4 extra slots/call | Reduction | Gate |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in comparisons:
        lines.append(
            "| {model_id} | {v2:.3f} | {v4:.3f} | {red:.1%} | {gate} |".format(
                model_id=row["model_id"],
                v2=row["v2_extra_optional_slots_per_emitted_call"],
                v4=row["v4_extra_optional_slots_per_emitted_call"],
                red=row["relative_reduction"],
                gate="PASS" if row["passes_50_percent_reduction"] else "FAIL",
            )
        )
    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report

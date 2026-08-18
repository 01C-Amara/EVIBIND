from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .evidence_contract import ACTION_CRITICAL_ROLES, build_candidate_lattice, capability_compatible, capability_signature, certified_candidates, request_text
from .io import read_jsonl

RESOLVER_EVAL_VERSION = "tapbench.resolver_eval.v1"


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return left.strip().casefold() == right.strip().casefold()
    return left == right


def _tool_name(tool: dict[str, Any]) -> str:
    return str(tool.get("canonical_name") or tool.get("name"))


def evaluate_resolver(
    cases_path: str | Path,
    output_path: str | Path,
    *,
    reference_date: str = "2026-07-10",
    timezone: str = "Europe/London",
    candidate_seed: int = 17,
    max_cases: int | None = None,
) -> dict[str, Any]:
    counts = Counter()
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    details = []
    for index, case in enumerate(read_jsonl(cases_path)):
        if max_cases is not None and index >= max_cases:
            break
        lattice = build_candidate_lattice(
            case.get("messages", []),
            case.get("tools", []),
            dialogue_state={},
            reference_context={"reference_date": reference_date, "timezone": timezone},
            candidate_seed=candidate_seed,
        )
        family = str(case.get("family"))
        task_kind = str(case.get("task_kind"))
        counts["cases"] += 1
        family_counts[family]["cases"] += 1
        detail = {"case_id": case["case_id"], "family": family, "task_kind": task_kind}

        if task_kind == "call":
            gold = case["gold_action"]
            tool_name = str(gold["tool"])
            tool = lattice["tools"].get(tool_name)
            if tool is None:
                counts["gold_tool_missing"] += 1
                detail["gold_tool_present"] = False
            else:
                detail["gold_tool_present"] = True
                feasible = True
                for slot, gold_value in gold.get("arguments", {}).items():
                    slot_row = tool["slots"].get(slot)
                    if slot_row is None:
                        counts["gold_slots"] += 1
                        counts["gold_slot_missed"] += 1
                        feasible = False
                        continue
                    candidates = certified_candidates(slot_row)
                    role = str(slot_row.get("role"))
                    bucket = "critical" if role in ACTION_CRITICAL_ROLES else "content"
                    counts[f"{bucket}_gold_slots"] += 1
                    family_counts[family][f"{bucket}_gold_slots"] += 1
                    recovered = any(_same(candidate.get("value"), gold_value) for candidate in candidates)
                    counts[f"{bucket}_gold_recovered"] += int(recovered)
                    family_counts[family][f"{bucket}_gold_recovered"] += int(recovered)
                    if not recovered:
                        counts[f"{bucket}_missed_slot::{slot}"] += 1
                        family_counts[family][f"{bucket}_missed_slot::{slot}"] += 1
                    for candidate in candidates:
                        counts[f"{bucket}_certificates"] += 1
                        counts[f"{bucket}_certificates_correct"] += int(_same(candidate.get("value"), gold_value))
                    if role in ACTION_CRITICAL_ROLES and slot_row.get("required") and not recovered:
                        feasible = False
                counts["gold_calls"] += 1
                counts["gold_call_feasible"] += int(feasible)
                family_counts[family]["gold_calls"] += 1
                family_counts[family]["gold_call_feasible"] += int(feasible)
                detail["gold_call_feasible"] = feasible

        elif task_kind == "missing_info":
            primary_name = _tool_name(case.get("tools", [{}])[0])
            primary = lattice["tools"].get(primary_name, {"slots": {}})
            expected = set(case.get("gold_action", {}).get("payload", {}).get("missing_slots", []))
            detected = {
                slot for slot, row in primary.get("slots", {}).items()
                if row.get("required") and row.get("role") in ACTION_CRITICAL_ROLES and not certified_candidates(row)
            }
            exact = expected == detected
            counts["missing_cases"] += 1
            counts["missing_set_exact"] += int(exact)
            family_counts[family]["missing_cases"] += 1
            family_counts[family]["missing_set_exact"] += int(exact)
            detail["expected_missing"] = sorted(expected)
            detail["detected_missing"] = sorted(detected)
            detail["missing_set_exact"] = exact

        elif task_kind in {"no_tool", "direct_answer"}:
            text = request_text(case.get("messages", []))
            compatible = [
                _tool_name(tool) for tool in case.get("tools", [])
                if capability_compatible(text, capability_signature(tool))
            ]
            correct_empty = not compatible
            counts[f"{task_kind}_cases"] += 1
            counts[f"{task_kind}_no_compatible_tool"] += int(correct_empty)
            family_counts[family][f"{task_kind}_cases"] += 1
            family_counts[family][f"{task_kind}_no_compatible_tool"] += int(correct_empty)
            detail["compatible_tools"] = compatible
        details.append(detail)

    def ratio(numerator: str, denominator: str, source: Counter[str] = counts) -> float | None:
        return source[numerator] / source[denominator] if source[denominator] else None

    summary = {
        "cases": counts["cases"],
        "critical_evidence_precision": ratio("critical_certificates_correct", "critical_certificates"),
        "critical_evidence_recall": ratio("critical_gold_recovered", "critical_gold_slots"),
        "content_span_exact_recall": ratio("content_gold_recovered", "content_gold_slots"),
        "gold_call_feasible_rate": ratio("gold_call_feasible", "gold_calls"),
        "missing_set_exact_rate": ratio("missing_set_exact", "missing_cases"),
        "no_tool_capability_accuracy": ratio("no_tool_no_compatible_tool", "no_tool_cases"),
        "direct_answer_capability_accuracy": ratio("direct_answer_no_compatible_tool", "direct_answer_cases"),
        "counts": dict(sorted(counts.items())),
    }
    by_family = []
    for family, values in sorted(family_counts.items()):
        by_family.append({
            "family": family,
            "cases": values["cases"],
            "critical_recall": ratio("critical_gold_recovered", "critical_gold_slots", values),
            "gold_call_feasible_rate": ratio("gold_call_feasible", "gold_calls", values),
            "missing_set_exact_rate": ratio("missing_set_exact", "missing_cases", values),
            "no_tool_capability_accuracy": ratio("no_tool_no_compatible_tool", "no_tool_cases", values),
        })
    report = {
        "schema_version": RESOLVER_EVAL_VERSION,
        "runtime_inputs": ["messages", "tools", "dialogue_state", "reference_context", "candidate_seed"],
        "gold_usage": "offline_evaluation_only",
        "summary": summary,
        "by_family": by_family,
        "details": details,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

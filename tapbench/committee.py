from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .bfcl import score_bfcl_prediction, summarize_bfcl_scores
from .io import read_jsonl, write_jsonl

COMMITTEE_VERSION = "tapbench.agreement_committee.v1"
DEFAULT_SOURCE_METHODS = ("full_tap_b2", "prompt_few_shot")
DEFAULT_RISK_CAPS = (0.05, 0.10)


def split_for_case(case_id: str, *, modulus: int = 5) -> str:
    bucket = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8], 16) % modulus
    return "development" if bucket == 0 else "heldout"


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).casefold()


def _argument_agreement(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_args = left.get("arguments", {}) if isinstance(left.get("arguments"), dict) else {}
    right_args = right.get("arguments", {}) if isinstance(right.get("arguments"), dict) else {}
    slots = set(left_args) | set(right_args)
    if not slots:
        return 1.0
    return sum(
        _stable(left_args.get(slot, "__missing_left__"))
        == _stable(right_args.get(slot, "__missing_right__"))
        for slot in slots
    ) / len(slots)


def _canonical_call(row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    action = row.get("prediction")
    if not isinstance(action, dict) or action.get("mode") != "call":
        return None
    tool = action.get("tool")
    if not isinstance(tool, str) or not tool:
        return None
    return tool, action


def build_committee_prediction(
    case_id: str,
    candidates: dict[tuple[str, str], dict[str, Any]],
    *,
    member_models: Iterable[str],
    source_methods: Iterable[str] = DEFAULT_SOURCE_METHODS,
    vote_threshold: int,
) -> dict[str, Any]:
    models = tuple(sorted(member_models))
    methods = tuple(source_methods)
    calls: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    for model in models:
        for method in methods:
            row = candidates[(model, method)]
            parsed = _canonical_call(row)
            if parsed is not None:
                _, action = parsed
                calls.append((model, method, row, action))

    votes = Counter(action["tool"] for _, _, _, action in calls)
    selected_tool, selected_votes = votes.most_common(1)[0] if votes else (None, 0)
    base = deepcopy(candidates[(models[0], methods[0])])
    base.update({
        "method": f"tap_r_committee_t{vote_threshold}",
        "model_id": "TAP-R/agreement-committee",
        "model_artifact": "committee:" + "+".join(models),
        "chat_template": "mixed_member_native_templates",
        "committee_version": COMMITTEE_VERSION,
        "response_metadata": {"finish_reason": "committee", "member_generation_count": len(models) * len(methods)},
        "runner_error": None,
    })

    committee = {
        "schema_version": COMMITTEE_VERSION,
        "member_models": list(models),
        "source_methods": list(methods),
        "vote_threshold": vote_threshold,
        "valid_call_votes": dict(votes),
        "selected_tool": selected_tool,
        "selected_tool_votes": selected_votes,
    }
    if selected_tool is None or selected_votes < vote_threshold:
        base["prediction"] = {
            "mode": "no_tool",
            "tool": None,
            "arguments": {},
            "payload": {"committee_decision": "insufficient_call_agreement"},
        }
        base["committee"] = {**committee, "decision": "no_tool"}
        return base

    group = [item for item in calls if item[3]["tool"] == selected_tool]
    ranked: list[tuple[float, int, str, str, dict[str, Any]]] = []
    for model, method, row, action in group:
        agreement = sum(_argument_agreement(action, other[3]) for other in group)
        hybrid = candidates.get((model, "tap_r_hybrid_span_tier_b"), {}).get("prediction", {})
        evidence_bonus = 0.25 * float(
            isinstance(hybrid, dict)
            and hybrid.get("mode") == "call"
            and hybrid.get("tool") == selected_tool
            and _stable(hybrid.get("arguments", {})) == _stable(action.get("arguments", {}))
        )
        ranked.append((
            agreement + evidence_bonus,
            -len(_stable(action.get("arguments", {}))),
            model,
            method,
            row,
        ))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    _, _, source_model, source_method, selected = ranked[0]
    base["prediction"] = deepcopy(selected["prediction"])
    base["source_response_metadata"] = deepcopy(selected.get("response_metadata", {}))
    base["committee"] = {
        **committee,
        "decision": "call",
        "selected_source_model": source_model,
        "selected_source_method": source_method,
        "medoid_score": ranked[0][0],
    }
    return base


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "execution_success": sum(bool(row.get("execution_success")) for row in rows),
        "execution_success_rate": sum(bool(row.get("execution_success")) for row in rows) / n if n else 0.0,
        "fabrication": sum(bool(row.get("fabrication")) for row in rows),
        "fabrication_rate": sum(bool(row.get("fabrication")) for row in rows) / n if n else 0.0,
    }


def _paired_bootstrap(
    committee: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    *,
    replicates: int = 20000,
    seed: int = 20260712,
) -> dict[str, Any]:
    case_ids = sorted(set(committee) & set(baseline))
    deltas = [
        float(committee[case_id]["execution_success"]) - float(baseline[case_id]["execution_success"])
        for case_id in case_ids
    ]
    point = sum(deltas) / len(deltas) if deltas else 0.0
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        samples.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    samples.sort()
    lower = samples[int(0.025 * (len(samples) - 1))] if samples else 0.0
    upper = samples[int(0.975 * (len(samples) - 1))] if samples else 0.0
    return {
        "n": len(deltas),
        "point_estimate": point,
        "ci95": [lower, upper],
        "replicates": replicates,
        "seed": seed,
        "resampling_unit": "case_id",
    }


def evaluate_committee(
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    risk_caps: Iterable[float] = DEFAULT_RISK_CAPS,
    fixed_thresholds: Iterable[int] = (6,),
    split_modulus: int = 5,
    bootstrap_replicates: int = 20000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    case_by_id = {str(case["case_id"]): case for case in cases}
    by_case: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    member_models = sorted({
        str(row.get("model_id"))
        for row in predictions
        if row.get("method") in DEFAULT_SOURCE_METHODS
    })
    for row in predictions:
        key = (str(row.get("model_id")), str(row.get("method")))
        by_case[str(row["case_id"])][key] = row

    max_votes = len(member_models) * len(DEFAULT_SOURCE_METHODS)
    scored_by_threshold: dict[int, list[dict[str, Any]]] = {}
    predictions_by_threshold: dict[int, list[dict[str, Any]]] = {}
    frontier = []
    for threshold in range(1, max_votes + 1):
        threshold_predictions = []
        threshold_scores = []
        for case_id, case in case_by_id.items():
            output = build_committee_prediction(
                case_id,
                by_case[case_id],
                member_models=member_models,
                vote_threshold=threshold,
            )
            score, _ = score_bfcl_prediction(case, output)
            split = split_for_case(case_id, modulus=split_modulus)
            output["committee_split"] = split
            score["committee_split"] = split
            threshold_predictions.append(output)
            threshold_scores.append(score)
        predictions_by_threshold[threshold] = threshold_predictions
        scored_by_threshold[threshold] = threshold_scores
        frontier.append({
            "vote_threshold": threshold,
            "development": _metrics([row for row in threshold_scores if row["committee_split"] == "development"]),
            "heldout": _metrics([row for row in threshold_scores if row["committee_split"] == "heldout"]),
        })

    operating_points = []
    output_predictions = []
    output_scores = []
    source_scores = []
    for prediction in predictions:
        case_id = str(prediction["case_id"])
        score, _ = score_bfcl_prediction(case_by_id[case_id], prediction)
        score["committee_split"] = split_for_case(case_id, modulus=split_modulus)
        source_scores.append(score)
    source_by_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_scores:
        if row["committee_split"] == "development" and row.get("method") in DEFAULT_SOURCE_METHODS:
            source_by_method[(str(row["model_id"]), str(row["method"]))].append(row)

    for risk_cap in risk_caps:
        eligible = [
            row for row in frontier
            if row["development"]["fabrication_rate"] <= float(risk_cap)
        ]
        selected = max(
            eligible,
            key=lambda row: (
                row["development"]["execution_success_rate"],
                -row["development"]["fabrication_rate"],
                row["vote_threshold"],
            ),
        )
        threshold = int(selected["vote_threshold"])
        policy_id = f"risk_cap_{int(round(float(risk_cap) * 100)):02d}"
        heldout_predictions = []
        heldout_scores = []
        for prediction, score in zip(
            predictions_by_threshold[threshold],
            scored_by_threshold[threshold],
            strict=True,
        ):
            if score["committee_split"] != "heldout":
                continue
            pred = deepcopy(prediction)
            pred["method"] = f"tap_r_committee_{policy_id}"
            pred["committee"]["policy_id"] = policy_id
            scored, _ = score_bfcl_prediction(case_by_id[str(pred["case_id"])], pred)
            scored["committee_split"] = "heldout"
            heldout_predictions.append(pred)
            heldout_scores.append(scored)
        output_predictions.extend(heldout_predictions)
        output_scores.extend(heldout_scores)

        baseline_key, baseline_dev_rows = max(
            source_by_method.items(),
            key=lambda item: (
                _metrics(item[1])["execution_success_rate"],
                -_metrics(item[1])["fabrication_rate"],
            ),
        )
        baseline_heldout = {
            str(row["case_id"]): row
            for row in source_scores
            if row["committee_split"] == "heldout"
            and (str(row["model_id"]), str(row["method"])) == baseline_key
        }
        committee_heldout = {str(row["case_id"]): row for row in heldout_scores}
        category_metrics = {}
        for category in sorted({str(row.get("bfcl_category")) for row in heldout_scores}):
            category_metrics[category] = _metrics([
                row for row in heldout_scores if str(row.get("bfcl_category")) == category
            ])
        operating_points.append({
            "policy_id": policy_id,
            "development_fabrication_cap": float(risk_cap),
            "selected_vote_threshold": threshold,
            "development": selected["development"],
            "heldout": _metrics(heldout_scores),
            "heldout_by_category": category_metrics,
            "development_selected_baseline": {
                "model_id": baseline_key[0],
                "method": baseline_key[1],
                "development": _metrics(baseline_dev_rows),
                "heldout": _metrics(list(baseline_heldout.values())),
            },
            "paired_execution_difference_vs_dev_selected_baseline": _paired_bootstrap(
                committee_heldout,
                baseline_heldout,
                replicates=bootstrap_replicates,
            ),
        })

    fixed_operating_points = []
    for threshold in fixed_thresholds:
        if threshold not in predictions_by_threshold:
            raise ValueError(f"fixed vote threshold {threshold} is outside 1..{max_votes}")
        fixed_predictions = []
        fixed_scores = []
        policy_id = f"fixed_t{threshold}_of_{max_votes}"
        for prediction in predictions_by_threshold[threshold]:
            pred = deepcopy(prediction)
            pred["method"] = f"tap_r_committee_{policy_id}"
            pred["committee"]["policy_id"] = policy_id
            scored, _ = score_bfcl_prediction(case_by_id[str(pred["case_id"])], pred)
            scored["committee_split"] = pred["committee_split"]
            fixed_predictions.append(pred)
            fixed_scores.append(scored)
        output_predictions.extend(fixed_predictions)
        output_scores.extend(fixed_scores)
        category_metrics = {
            category: _metrics([row for row in fixed_scores if str(row.get("bfcl_category")) == category])
            for category in sorted({str(row.get("bfcl_category")) for row in fixed_scores})
        }
        baselines = []
        for baseline_key in sorted({
            (str(row["model_id"]), str(row["method"]))
            for row in source_scores if row.get("method") in DEFAULT_SOURCE_METHODS
        }):
            baseline_rows = {
                str(row["case_id"]): row
                for row in source_scores
                if (str(row["model_id"]), str(row["method"])) == baseline_key
            }
            baselines.append({
                "model_id": baseline_key[0],
                "method": baseline_key[1],
                "metrics": _metrics(list(baseline_rows.values())),
                "paired_execution_difference": _paired_bootstrap(
                    {str(row["case_id"]): row for row in fixed_scores},
                    baseline_rows,
                    replicates=bootstrap_replicates,
                ),
            })
        fixed_operating_points.append({
            "policy_id": policy_id,
            "status": "fixed_before_prospective_validation_only",
            "vote_threshold": threshold,
            "metrics": _metrics(fixed_scores),
            "by_category": category_metrics,
            "baselines": baselines,
        })

    report = {
        "schema_version": "tapbench.agreement_committee_report.v1",
        "committee_version": COMMITTEE_VERSION,
        "interpretation": "posthoc_external_anchor_not_hypothesis_coefficient",
        "runtime_features": ["canonical_call_mode", "canonical_tool_id", "cross_view_tool_votes", "argument_medoid", "evidence_certification_tiebreak"],
        "forbidden_runtime_features": ["gold_action", "bfcl_category", "task_kind", "score", "fabrication", "execution_success"],
        "member_models": member_models,
        "source_methods": list(DEFAULT_SOURCE_METHODS),
        "split": {"function": "sha256(case_id)_mod_5", "development_buckets": [0], "heldout_buckets": [1, 2, 3, 4]},
        "frontier": frontier,
        "operating_points": operating_points,
        "fixed_operating_points": fixed_operating_points,
        "output_summary": summarize_bfcl_scores(output_scores),
    }
    return output_predictions, output_scores, report


def evaluate_committee_files(
    cases_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    *,
    risk_caps: Iterable[float] = DEFAULT_RISK_CAPS,
    fixed_thresholds: Iterable[int] = (6,),
    split_modulus: int = 5,
    bootstrap_replicates: int = 20000,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions, scores, report = evaluate_committee(
        read_jsonl(cases_path),
        read_jsonl(predictions_path),
        risk_caps=risk_caps,
        fixed_thresholds=fixed_thresholds,
        split_modulus=split_modulus,
        bootstrap_replicates=bootstrap_replicates,
    )
    write_jsonl(output / "heldout_predictions.jsonl", predictions)
    write_jsonl(output / "heldout_scores.jsonl", scores)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

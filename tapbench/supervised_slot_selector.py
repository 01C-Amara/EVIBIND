from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_jsonl, write_jsonl
from .supervised_intent_router import _tfidf_vector, hashed_counts, intent_to_tool, load_router, rank_tools


SLOT_SELECTOR_VERSION = "tapbench.massive_supervised_slot_knn.v1"
K_CANDIDATES = (1, 3, 5, 9)
VOTE_CANDIDATES = (0.25, 0.4, 0.5, 0.67)
_SLOT_PATTERN = re.compile(r"\[\s*([A-Za-z0-9_]+)\s*:")


def annotation_slots(annotation: str) -> set[str]:
    return {match.group(1) for match in _SLOT_PATTERN.finditer(str(annotation))}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_slot_contract(cases: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    valid: dict[str, set[str]] = {}
    required: dict[str, set[str]] = {}
    for case in cases:
        for tool in case.get("tools", []):
            name = str(tool.get("canonical_name") or tool.get("name"))
            parameters = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
            properties = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
            tool_valid = set(str(value) for value in properties)
            tool_required = {str(value) for value in parameters.get("required", []) if str(value) in tool_valid}
            if name in valid and (valid[name] != tool_valid or required[name] != tool_required):
                raise ValueError(f"public slot contract differs across cases for {name}")
            valid[name] = tool_valid
            required[name] = tool_required
    return valid, required


def _source_rows(
    path: str | Path,
    split: str,
    valid_slots: dict[str, set[str]],
) -> list[dict[str, Any]]:
    output = []
    for row in read_jsonl(path):
        if str(row.get("partition")) != split:
            continue
        tool = intent_to_tool(str(row.get("intent")))
        if tool not in valid_slots:
            continue
        output.append(
            {
                "id": str(row.get("id")),
                "text": str(row.get("utt", "")),
                "tool": tool,
                "slots": sorted(annotation_slots(str(row.get("annot_utt", ""))) & valid_slots[tool]),
            }
        )
    return output


def build_index(rows: list[dict[str, Any]], router_model: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        indices, values = _tfidf_vector(hashed_counts(row["text"], int(router_model["dimensions"])), router_model["idf"])
        output[row["tool"]].append(
            {
                "id": row["id"],
                "indices": indices,
                "values": values.astype(np.float32),
                "slots": set(row["slots"]),
            }
        )
    return dict(output)


def _sparse_dot(left_indices: np.ndarray, left_values: np.ndarray, right_indices: np.ndarray, right_values: np.ndarray) -> float:
    left = right = 0
    score = 0.0
    while left < len(left_indices) and right < len(right_indices):
        left_index = int(left_indices[left])
        right_index = int(right_indices[right])
        if left_index == right_index:
            score += float(left_values[left]) * float(right_values[right])
            left += 1
            right += 1
        elif left_index < right_index:
            left += 1
        else:
            right += 1
    return score


def neighbor_scores(
    text: str,
    tool: str,
    index: dict[str, list[dict[str, Any]]],
    router_model: dict[str, Any],
) -> list[dict[str, Any]]:
    query_indices, query_values = _tfidf_vector(
        hashed_counts(text, int(router_model["dimensions"])),
        router_model["idf"],
    )
    scored = []
    for row in index.get(tool, []):
        scored.append(
            {
                "id": row["id"],
                "score": _sparse_dot(query_indices, query_values, row["indices"], row["values"]),
                "slots": row["slots"],
            }
        )
    return sorted(scored, key=lambda row: (-float(row["score"]), str(row["id"])))


def select_slots(
    neighbors: list[dict[str, Any]],
    *,
    valid_slots: set[str],
    required_slots: set[str],
    k: int,
    vote_threshold: float,
) -> list[str]:
    selected_neighbors = neighbors[:k]
    total_weight = sum(max(0.0, float(row["score"])) for row in selected_neighbors)
    if selected_neighbors and total_weight <= 1e-12:
        weights = [1.0] * len(selected_neighbors)
        total_weight = float(len(selected_neighbors))
    else:
        weights = [max(0.0, float(row["score"])) for row in selected_neighbors]
    selected = set(required_slots)
    for slot in sorted(valid_slots):
        support = sum(weight for row, weight in zip(selected_neighbors, weights, strict=True) if slot in row["slots"])
        if total_weight > 0 and support / total_weight >= vote_threshold:
            selected.add(slot)
    return sorted(selected & valid_slots)


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = sum(row["predicted_slots"] == row["gold_slots"] for row in rows)
    tp = sum(len(set(row["predicted_slots"]) & set(row["gold_slots"])) for row in rows)
    fp = sum(len(set(row["predicted_slots"]) - set(row["gold_slots"])) for row in rows)
    fn = sum(len(set(row["gold_slots"]) - set(row["predicted_slots"])) for row in rows)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(rows),
        "exact": exact,
        "exact_rate": exact / len(rows) if rows else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def tune_selector(
    dev_rows: list[dict[str, Any]],
    index: dict[str, list[dict[str, Any]]],
    router_model: dict[str, Any],
    valid_slots: dict[str, set[str]],
    required_slots: dict[str, set[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared = []
    for row in dev_rows:
        ranking = rank_tools(router_model, row["text"])
        predicted_tool = str(ranking[0]["tool"])
        if predicted_tool != row["tool"]:
            continue
        prepared.append({**row, "neighbors": neighbor_scores(row["text"], predicted_tool, index, router_model)})
    candidates = []
    for k in K_CANDIDATES:
        for vote in VOTE_CANDIDATES:
            scored = []
            for row in prepared:
                predicted = select_slots(
                    row["neighbors"],
                    valid_slots=valid_slots[row["tool"]],
                    required_slots=required_slots[row["tool"]],
                    k=k,
                    vote_threshold=vote,
                )
                scored.append({"predicted_slots": predicted, "gold_slots": row["slots"]})
            metrics = _counts(scored)
            candidates.append({"k": k, "vote_threshold": vote, **metrics})
    best = max(
        candidates,
        key=lambda row: (
            float(row["exact_rate"] or 0.0),
            float(row["f1"]),
            float(row["vote_threshold"]),
            -int(row["k"]),
        ),
    )
    return best, candidates


def _gold_slots(gold_row: dict[str, Any], valid_slots: dict[str, set[str]]) -> tuple[str, list[str]]:
    truth = gold_row.get("ground_truth")
    if not isinstance(truth, list) or len(truth) != 1 or not isinstance(truth[0], dict) or len(truth[0]) != 1:
        raise ValueError(f"unsupported gold shape for {gold_row.get('case_id')}")
    tool, arguments = next(iter(truth[0].items()))
    active = []
    if isinstance(arguments, dict):
        for slot, allowed in arguments.items():
            values = allowed if isinstance(allowed, list) else [allowed]
            if slot in valid_slots.get(str(tool), set()) and any(str(value) != "" for value in values):
                active.append(str(slot))
    return str(tool), sorted(active)


def _evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tool_correct = [row for row in rows if row["predicted_tool"] == row["gold_tool"]]
    slot_rows = [{"predicted_slots": row["active_slots"], "gold_slots": row["gold_slots"]} for row in tool_correct]
    joint = sum(row["predicted_tool"] == row["gold_tool"] and row["active_slots"] == row["gold_slots"] for row in rows)
    return {
        "n": len(rows),
        "tool_correct": len(tool_correct),
        "tool_accuracy": len(tool_correct) / len(rows) if rows else None,
        "slot_given_correct_tool": _counts(slot_rows),
        "joint_tool_slot_exact": joint,
        "joint_tool_slot_exact_rate": joint / len(rows) if rows else None,
    }


def run_experiment(
    *,
    source_dir: str | Path,
    router_model_dir: str | Path,
    router_rankings_path: str | Path,
    router_report_path: str | Path,
    cases_path: str | Path,
    gold_path: str | Path,
    output_dir: str | Path,
    languages: tuple[str, ...] = ("en-US", "fa-IR", "ja-JP"),
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(cases_path)
    valid_slots, required_slots = _public_slot_contract(cases)
    rankings = {str(row["case_id"]): row for row in read_jsonl(router_rankings_path)}
    gold_by_id = {str(row["case_id"]): row for row in read_jsonl(gold_path)}
    router_report = json.loads(Path(router_report_path).read_text(encoding="utf-8"))
    dev95_threshold = float(router_report["threshold_selection"]["global"]["threshold"])
    models = {}
    indexes = {}
    settings = {}
    tuning = {}
    source_manifest = {}
    for language in languages:
        source_path = Path(source_dir) / f"{language}.jsonl"
        model_path = Path(router_model_dir) / f"router_{language}.npz"
        model = load_router(model_path)
        train_rows = _source_rows(source_path, "train", valid_slots)
        dev_rows = _source_rows(source_path, "dev", valid_slots)
        index = build_index(train_rows, model)
        best, candidates = tune_selector(dev_rows, index, model, valid_slots, required_slots)
        models[language] = model
        indexes[language] = index
        settings[language] = {"k": int(best["k"]), "vote_threshold": float(best["vote_threshold"])}
        tuning[language] = {"selected": best, "candidates": candidates}
        source_manifest[language] = {
            "source_path": str(source_path.resolve()),
            "source_sha256": _sha256(source_path),
            "router_model_path": str(model_path.resolve()),
            "router_model_sha256": _sha256(model_path),
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
        }

    artifact_rows = []
    scored_rows = []
    case_by_id = {str(case["case_id"]): case for case in cases}
    for case_id, case in case_by_id.items():
        language = str(case.get("metadata", {}).get("language") or case.get("factors", {}).get("language"))
        ranking = rankings[case_id]["ranking"]
        predicted_tool = str(ranking[0]["tool"])
        confidence = float(ranking[0]["cosine_score"]) - float(ranking[1]["cosine_score"])
        text = "\n".join(str(row.get("content", "")) for row in case.get("messages", []) if row.get("role") == "user")
        neighbors = neighbor_scores(text, predicted_tool, indexes[language], models[language])
        setting = settings[language]
        active_slots = select_slots(
            neighbors,
            valid_slots=valid_slots[predicted_tool],
            required_slots=required_slots[predicted_tool],
            k=setting["k"],
            vote_threshold=setting["vote_threshold"],
        )
        artifact_rows.append(
            {
                "schema_version": SLOT_SELECTOR_VERSION,
                "case_id": case_id,
                "language": language,
                "predicted_tool": predicted_tool,
                "active_slots": active_slots,
                "router_confidence": confidence,
                "selected_by_router_dev95": confidence >= dev95_threshold,
                "k": setting["k"],
                "vote_threshold": setting["vote_threshold"],
                "neighbor_ids": [str(row["id"]) for row in neighbors[: setting["k"]]],
            }
        )
        gold_tool, gold_slots = _gold_slots(gold_by_id[case_id], valid_slots)
        scored_rows.append(
            {
                "case_id": case_id,
                "language": language,
                "predicted_tool": predicted_tool,
                "gold_tool": gold_tool,
                "active_slots": active_slots,
                "gold_slots": gold_slots,
                "router_confidence": confidence,
                "selected_by_router_dev95": confidence >= dev95_threshold,
            }
        )
    artifact_path = output / "design_active_slots.jsonl"
    write_jsonl(artifact_path, artifact_rows)
    overall = _evaluate(scored_rows)
    selected = _evaluate([row for row in scored_rows if row["selected_by_router_dev95"]])
    report = {
        "schema_version": "tapbench.massive_supervised_slot_selector_report.v1",
        "selector_version": SLOT_SELECTOR_VERSION,
        "analysis_status": "post_result_exploratory_design_only",
        "confirmation_authorized": False,
        "system_label": "benchmark_supervised_intent_and_slot_router",
        "source_manifest": source_manifest,
        "hyperparameter_selection": {
            "split": "official_MASSIVE_v1.1_dev",
            "primary_metric": "exact_active_slot_set_given_correct_tool",
            "settings": settings,
            "details": tuning,
        },
        "design": {
            "all_cases": overall,
            "router_dev95_subset": selected,
            "by_language": {language: _evaluate([row for row in scored_rows if row["language"] == language]) for language in languages},
        },
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": _sha256(artifact_path),
            "contains_gold": False,
            "rows": len(artifact_rows),
        },
        "scorer_only_rows": len(scored_rows),
        "router_rankings_sha256": _sha256(router_rankings_path),
        "router_report_sha256": _sha256(router_report_path),
        "cases_sha256": _sha256(cases_path),
        "gold_sha256": _sha256(gold_path),
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/tune the MASSIVE-supervised active-slot kNN selector.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--router-model-dir", required=True)
    parser.add_argument("--router-rankings", required=True)
    parser.add_argument("--router-report", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = run_experiment(
        source_dir=args.source_dir,
        router_model_dir=args.router_model_dir,
        router_rankings_path=args.router_rankings,
        router_report_path=args.router_report,
        cases_path=args.cases,
        gold_path=args.gold,
        output_dir=args.output_dir,
    )
    print(json.dumps(report["design"], sort_keys=True))


if __name__ == "__main__":
    main()

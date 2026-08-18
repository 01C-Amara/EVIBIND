from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl


BFCL_OFFICIAL_ADAPTER_VERSION = "tapbench.bfcl_official_adapter.v2"


def _language_for_category(category: str, language_enum: Any) -> Any:
    normalized = category.lower()
    if "javascript" in normalized:
        return language_enum.JAVASCRIPT
    if "java" in normalized:
        return language_enum.JAVA
    return language_enum.PYTHON


def action_ir_to_bfcl_ast(action: Any) -> list[dict[str, Any]]:
    if not isinstance(action, dict) or action.get("mode") != "call":
        return []
    tool = action.get("tool")
    arguments = action.get("arguments")
    if not isinstance(tool, str) or not isinstance(arguments, dict):
        return []
    return [{tool: arguments}]


def _load_by_id(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in read_jsonl(path)}


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def evaluate_bfcl_official(
    cases_path: str | Path,
    predictions_path: str | Path,
    bfcl_root: str | Path,
    output_dir: str | Path,
    *,
    source_commit: str,
) -> dict[str, Any]:
    root = Path(bfcl_root).resolve()
    package_root = root
    if not (package_root / "bfcl_eval").is_dir():
        raise FileNotFoundError(f"BFCL package root is invalid: {package_root}")
    sys.path.insert(0, str(package_root))
    try:
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    finally:
        if sys.path and sys.path[0] == str(package_root):
            sys.path.pop(0)

    data_root = package_root / "bfcl_eval" / "data"
    cases = {row["case_id"]: row for row in read_jsonl(cases_path)}
    predictions = read_jsonl(predictions_path)
    categories = sorted({
        str(case.get("metadata", {}).get("bfcl_category"))
        for case in cases.values()
    })
    prompts = {
        category: _load_by_id(data_root / f"BFCL_v4_{category}.json", "id")
        for category in categories
    }
    answers = {
        category: (
            _load_by_id(
                data_root / "possible_answer" / f"BFCL_v4_{category}.json",
                "id",
            )
            if "irrelevance" not in category and "relevance" not in category
            else {}
        )
        for category in categories
    }
    details = []
    official_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for prediction in predictions:
        case = cases.get(str(prediction.get("case_id")))
        if case is None:
            continue
        key = (
            str(prediction.get("case_id")),
            str(prediction.get("model_id")),
            str(prediction.get("method")),
            int(prediction.get("seed", 0)),
        )
        if key in seen:
            raise ValueError(f"duplicate BFCL prediction cell: {key}")
        seen.add(key)
        category = str(case["metadata"]["bfcl_category"])
        bfcl_id = str(case["metadata"]["bfcl_id"])
        prompt = prompts[category][bfcl_id]
        ast = action_ir_to_bfcl_ast(prediction.get("prediction"))
        if "irrelevance" in category:
            result = (
                {"valid": True}
                if not ast
                else {
                    "valid": False,
                    "error_type": "irrelevance_error:decoder_success",
                    "error": ["A function call was emitted for an irrelevant request."],
                }
            )
        else:
            possible = answers[category][bfcl_id]["ground_truth"]
            language = _language_for_category(category, Language)
            result = ast_checker(
                prompt["function"],
                ast,
                possible,
                language,
                category,
                "gorilla-openfunctions-v2",
            )
        row = {
            "case_id": prediction["case_id"],
            "bfcl_id": bfcl_id,
            "bfcl_category": category,
            "model_id": prediction.get("model_id"),
            "method": prediction.get("method"),
            "seed": prediction.get("seed"),
            "valid": bool(result.get("valid")),
            "error_type": result.get("error_type"),
            "error": result.get("error"),
            "decoded_action_ast": ast,
            "official_language": (
                _language_for_category(category, Language).value
                if "irrelevance" not in category
                else None
            ),
            "official_checker": (
                "bfcl_eval.eval_checker.ast_eval.ast_checker.ast_checker"
                if "irrelevance" not in category
                else "bfcl_eval.eval_checker.eval_runner.relevance_semantics"
            ),
            "official_checker_commit": source_commit,
            "adapter_version": BFCL_OFFICIAL_ADAPTER_VERSION,
        }
        details.append(row)
        official_rows[
            (
                str(prediction.get("model_id")),
                str(prediction.get("method")),
                category,
            )
        ].append({
            "id": bfcl_id,
            "result": ast,
            "input_token_count": prediction.get("response_metadata", {}).get("prompt_tokens"),
            "output_token_count": prediction.get("response_metadata", {}).get("completion_tokens"),
            "latency": prediction.get("response_metadata", {}).get("generation_ms"),
        })

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "official_details.jsonl", details)
    groups = []
    for (model_id, method, category), rows in sorted(official_rows.items()):
        group_details = [
            row for row in details
            if row["model_id"] == model_id
            and row["method"] == method
            and row["bfcl_category"] == category
        ]
        groups.append({
            "model_id": model_id,
            "method": method,
            "bfcl_category": category,
            "n": len(group_details),
            "correct": sum(row["valid"] for row in group_details),
            "accuracy": sum(row["valid"] for row in group_details) / len(group_details),
            "error_types": dict(sorted(Counter(
                row["error_type"] for row in group_details if row["error_type"]
            ).items())),
        })
        result_path = (
            output / "official_results" / _safe(model_id) / _safe(method)
            / f"BFCL_v4_{category}_result.json"
        )
        write_jsonl(result_path, sorted(rows, key=lambda row: row["id"]))

    checker_path = package_root / "bfcl_eval" / "eval_checker" / "ast_eval" / "ast_checker.py"
    report = {
        "schema_version": "tapbench.bfcl_official_report.v1",
        "adapter_version": BFCL_OFFICIAL_ADAPTER_VERSION,
        "official_source_commit": source_commit,
        "official_checker_sha256": hashlib.sha256(checker_path.read_bytes()).hexdigest(),
        "case_count": len(cases),
        "prediction_count": len(details),
        "categories": categories,
        "groups": groups,
    }
    (output / "official_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report

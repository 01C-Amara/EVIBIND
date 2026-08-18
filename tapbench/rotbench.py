from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl, write_jsonl


ROTBENCH_ADAPTER_VERSION = "tapbench.rotbench_adapter.v1"
ROTBENCH_SOURCE_COMMIT = "637e43ae9a978036774c92bb2ab11a83609e304b"
ROTBENCH_ENVIRONMENTS = ("clean", "slight", "medium", "heavy", "union")


def _tool_config(record: dict[str, Any]) -> list[dict[str, Any]]:
    system = str(record["conversations"][0]["value"])
    start = system.find("[")
    end_marker = system.rfind("\n\nLet's Begin!")
    if start < 0 or end_marker <= start:
        raise ValueError(f"unable to locate tool catalog for {record.get('id')}")
    return json.loads(system[start:end_marker])


def _answer(text: str) -> tuple[str, dict[str, Any]]:
    action_start = text.find("Action:")
    input_start = text.find("Action Input:")
    if action_start < 0 or input_start < 0:
        raise ValueError(f"invalid RoTBench answer: {text!r}")
    action = text[action_start + len("Action:"):input_start].strip()
    arguments = json.loads(text[input_start + len("Action Input:"):].strip())
    if not isinstance(arguments, dict):
        raise ValueError("RoTBench Action Input must be an object")
    return action, arguments


def _normalized_allowed(
    answer_texts: Iterable[str],
    *,
    ask_tool: str,
    finish_tool: str,
) -> list[dict[str, Any]]:
    rows = []
    for text in answer_texts:
        tool, arguments = _answer(text)
        cleaned = {key: value for key, value in arguments.items() if value != "None"}
        if tool == ask_tool:
            rows.append({
                "mode": "clarify",
                "tool": None,
                "arguments": {},
                "payload": cleaned,
            })
        elif tool == finish_tool:
            rows.append({
                "mode": "direct_answer",
                "tool": None,
                "arguments": {},
                "payload": cleaned,
            })
        else:
            rows.append({
                "mode": "call",
                "tool": tool,
                "arguments": cleaned,
                "payload": {},
            })
    return rows


def convert_rotbench(
    source_root: str | Path,
    output_path: str | Path,
    *,
    environments: Iterable[str] = ROTBENCH_ENVIRONMENTS,
    limit_per_environment: int | None = None,
) -> int:
    root = Path(source_root)
    rows = []
    for environment in environments:
        if environment not in ROTBENCH_ENVIRONMENTS:
            raise ValueError(f"unknown RoTBench environment: {environment}")
        records = json.loads(
            (root / "Data" / "First_Turn" / f"{environment}.json").read_text(
                encoding="utf-8"
            )
        )
        if limit_per_environment is not None:
            records = records[:limit_per_environment]
        for index, record in enumerate(records):
            tools = _tool_config(record)
            ask_tool = str(tools[-2]["name"])
            finish_tool = str(tools[-1]["name"])
            allowed = _normalized_allowed(
                record["conversations"][-1]["value"],
                ask_tool=ask_tool,
                finish_tool=finish_tool,
            )
            available = [
                {
                    **tool,
                    "canonical_name": str(tool["name"]),
                }
                for tool in tools[:-2]
            ]
            request = str(record["conversations"][1]["value"]).replace(
                "\nBegin!\n", ""
            ).strip()
            case_id = f"rotbench_{environment}_{index:04d}"
            rows.append({
                "schema_version": ROTBENCH_ADAPTER_VERSION,
                "case_id": case_id,
                "hypothesis_grid_id": "RoTBench_first_turn_robustness",
                "hypothesis": "external_rotbench",
                "split": "external_anchor",
                "family": "rotbench",
                "task_kind": allowed[0]["mode"],
                "factors": {
                    "environment": environment,
                    "scenario": record["scenario"],
                    "source_index": index,
                },
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return exactly one Action IR object. Select a tool only "
                            "when its schema supports the request. Never emit reasoning."
                        ),
                    },
                    {"role": "user", "content": request},
                ],
                "tools": available,
                "tool_aliases": {
                    str(tool["name"]): str(tool["name"]) for tool in available
                },
                "argument_aliases": {},
                "dialogue_state": {},
                "reference_context": {
                    "reference_date": "2026-07-13",
                    "timezone": "Europe/London",
                    "action_risk_budget": 0.05,
                },
                "gold_action": allowed[0],
                "rotbench_allowed_actions": allowed,
                "metadata": {
                    "external_source": "RoTBench",
                    "source_commit": ROTBENCH_SOURCE_COMMIT,
                    "source_id": record["id"],
                    "environment": environment,
                    "scenario": record["scenario"],
                    "ask_tool": ask_tool,
                    "finish_tool": finish_tool,
                    "offline_only_fields": [
                        "gold_action",
                        "rotbench_allowed_actions",
                        "task_kind",
                    ],
                },
            })
    return write_jsonl(output_path, rows)


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return left.strip().casefold() == right.strip().casefold()
    return left == right


def _content_equal(predicted: dict[str, Any], allowed: dict[str, Any]) -> bool:
    if predicted.get("mode") != allowed.get("mode"):
        return False
    if allowed.get("mode") != "call":
        return True
    if not _same(predicted.get("tool"), allowed.get("tool")):
        return False
    predicted_args = predicted.get("arguments")
    allowed_args = allowed.get("arguments")
    if not isinstance(predicted_args, dict) or not isinstance(allowed_args, dict):
        return False
    return predicted_args == allowed_args


def score_rotbench(
    cases_path: str | Path,
    predictions_path: str | Path,
    scores_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    cases = {row["case_id"]: row for row in read_jsonl(cases_path)}
    rows = []
    for prediction in read_jsonl(predictions_path):
        case = cases.get(str(prediction.get("case_id")))
        if case is None:
            continue
        action = prediction.get("prediction")
        if not isinstance(action, dict):
            action = {}
        allowed_actions = case["rotbench_allowed_actions"]
        tool_selection = any(
            action.get("mode") == allowed.get("mode")
            and (
                allowed.get("mode") != "call"
                or _same(action.get("tool"), allowed.get("tool"))
            )
            for allowed in allowed_actions
        )
        parameter_identification = any(
            action.get("mode") == allowed.get("mode")
            and (
                allowed.get("mode") != "call"
                or (
                    _same(action.get("tool"), allowed.get("tool"))
                    and isinstance(action.get("arguments"), dict)
                    and set(action["arguments"]) == set(allowed["arguments"])
                )
            )
            for allowed in allowed_actions
        )
        content_filling = any(
            _content_equal(action, allowed) for allowed in allowed_actions
        )
        rows.append({
            "case_id": case["case_id"],
            "model_id": prediction.get("model_id"),
            "method": prediction.get("method"),
            "seed": prediction.get("seed"),
            "backend": prediction.get("backend"),
            "quantization": prediction.get("quantization"),
            "model_artifact": prediction.get("model_artifact"),
            "chat_template": prediction.get("chat_template"),
            "grammar_engine": prediction.get("grammar_engine"),
            "thinking_mode": prediction.get("thinking_mode"),
            "thinking_marker_detected": prediction.get("thinking_marker_detected"),
            "environment": case["factors"]["environment"],
            "scenario": case["factors"]["scenario"],
            "tool_selection": tool_selection,
            "parameter_identification": parameter_identification,
            "content_filling": content_filling,
            "format_valid": isinstance(prediction.get("prediction"), dict)
            and prediction.get("runner_error") is None,
            "runner_error": prediction.get("runner_error"),
            "adapter_version": ROTBENCH_ADAPTER_VERSION,
            "source_commit": ROTBENCH_SOURCE_COMMIT,
        })
    write_jsonl(scores_path, rows)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_id"]), str(row["method"]), row["environment"])].append(row)
    groups = []
    for (model_id, method, environment), group in sorted(grouped.items()):
        groups.append({
            "model_id": model_id,
            "method": method,
            "environment": environment,
            "n": len(group),
            "tool_selection": sum(row["tool_selection"] for row in group) / len(group),
            "parameter_identification": sum(row["parameter_identification"] for row in group) / len(group),
            "content_filling": sum(row["content_filling"] for row in group) / len(group),
            "format_valid": sum(row["format_valid"] for row in group) / len(group),
            "runner_errors": sum(row["runner_error"] is not None for row in group),
        })
    by_key = {
        (row["model_id"], row["method"], row["environment"]): row
        for row in groups
    }
    degradation = []
    for row in groups:
        if row["environment"] == "clean":
            continue
        clean = by_key.get((row["model_id"], row["method"], "clean"))
        if clean is None:
            continue
        degradation.append({
            "model_id": row["model_id"],
            "method": row["method"],
            "environment": row["environment"],
            "content_filling_delta_vs_clean": row["content_filling"] - clean["content_filling"],
            "tool_selection_delta_vs_clean": row["tool_selection"] - clean["tool_selection"],
        })
    report = {
        "schema_version": "tapbench.rotbench_report.v1",
        "adapter_version": ROTBENCH_ADAPTER_VERSION,
        "source_commit": ROTBENCH_SOURCE_COMMIT,
        "case_count": len(cases),
        "prediction_count": len(rows),
        "environments": list(ROTBENCH_ENVIRONMENTS),
        "groups": groups,
        "degradation": degradation,
        "error_types": dict(sorted(Counter(
            "runner_error" for row in rows if row["runner_error"]
        ).items())),
    }
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FC_REWARDBENCH_ADAPTER_VERSION = "tapbench.fc_rewardbench_controller.v1"


def _table(path: str | Path):
    try:
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise RuntimeError(
            "FC-RewardBench evaluation requires `pip install evibind[rewardbench]`."
        ) from exc
    handle = Path(path).open("rb")
    try:
        try:
            return ipc.open_file(handle).read_all()
        except Exception:
            handle.seek(0)
            return ipc.open_stream(handle).read_all()
    finally:
        handle.close()


def _type_ok(value: Any, declared: str) -> bool:
    mapping = {
        "string": str,
        "integer": int,
        "float": (int, float),
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "tuple": list,
        "dict": dict,
        "object": dict,
    }
    expected = mapping.get(declared)
    if expected is None:
        return True
    if declared in {"integer", "float", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def _supported(value: Any, request: str) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        tokens = ("true", "yes") if value else ("false", "no")
        return any(token in request.casefold() for token in tokens)
    if isinstance(value, (int, float)):
        return str(value) in request or (
            isinstance(value, float)
            and value.is_integer()
            and str(int(value)) in request
        )
    if isinstance(value, str):
        return value.casefold() in request.casefold()
    if isinstance(value, list):
        return all(_supported(item, request) for item in value)
    if isinstance(value, dict):
        return all(_supported(item, request) for item in value.values())
    return False


def controller_score(
    output: str | list[dict[str, Any]],
    tools: list[dict[str, Any]],
    request: str,
) -> tuple[float, dict[str, Any]]:
    try:
        calls = json.loads(output) if isinstance(output, str) else output
    except (TypeError, json.JSONDecodeError):
        return -100.0, {"parse_valid": False, "errors": ["invalid_json"]}
    if not isinstance(calls, list):
        return -100.0, {"parse_valid": False, "errors": ["not_a_call_list"]}
    by_name = {str(tool["name"]): tool for tool in tools}
    score = 0.0
    errors = []
    if not calls:
        return -1.0, {
            "parse_valid": True,
            "call_count": 0,
            "errors": ["empty_call_list"],
        }
    for call in calls:
        if not isinstance(call, dict) or len(call) != 1:
            score -= 10.0
            errors.append("invalid_call_shape")
            continue
        tool_name, arguments = next(iter(call.items()))
        tool = by_name.get(str(tool_name))
        if tool is None:
            score -= 10.0
            errors.append("unknown_tool")
            continue
        score += 4.0
        if not isinstance(arguments, dict):
            score -= 8.0
            errors.append("arguments_not_object")
            continue
        schema = tool.get("parameters", {})
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = (
            set(schema.get("required", [])) if isinstance(schema, dict) else set()
        )
        missing = required - set(arguments)
        score += 2.0 * (len(required) - len(missing))
        score -= 5.0 * len(missing)
        errors.extend("missing_required" for _ in missing)
        assigned_evidence = []
        for name, value in arguments.items():
            prop = properties.get(name)
            if not isinstance(prop, dict):
                score -= 3.0
                errors.append("unknown_argument")
                continue
            declared = str(prop.get("type", "any"))
            if _type_ok(value, declared):
                score += 1.0
            else:
                score -= 3.0
                errors.append("wrong_type")
            enum = prop.get("enum")
            if isinstance(enum, list):
                if value in enum:
                    score += 1.0
                else:
                    score -= 3.0
                    errors.append("wrong_enum")
            default_match = "default" in prop and value == prop.get("default")
            if _supported(value, request):
                score += 1.5
            elif default_match:
                score += 1.0
            else:
                score -= 1.5
                errors.append("unsupported_value")
            if (
                isinstance(value, (str, int, float))
                and not isinstance(value, bool)
                and not default_match
            ):
                assigned_evidence.append(str(value).strip().casefold())
        duplicate_roles = len(assigned_evidence) - len(set(assigned_evidence))
        if duplicate_roles:
            score -= 2.0 * duplicate_roles
            errors.extend("role_collision" for _ in range(duplicate_roles))
    return score, {
        "parse_valid": True,
        "call_count": len(calls),
        "errors": errors,
    }


def evaluate_fc_rewardbench(
    arrow_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    table = _table(arrow_path)
    rows = []
    for index in range(table.num_rows):
        record = {name: table[name][index].as_py() for name in table.column_names}
        tools = json.loads(record["tools"])
        request = "\n".join(
            str(message.get("content", ""))
            for message in record["conversation"]
            if message.get("role") == "user"
        )
        chosen_score, chosen_diag = controller_score(
            record["chosen_output"], tools, request
        )
        rejected_score, rejected_diag = controller_score(
            record["rejected_output"], tools, request
        )
        margin = chosen_score - rejected_score
        rows.append(
            {
                "pair_id": index,
                "test_id": record["test_id"],
                "test_category": record["test_category"],
                "error_type": record["error_type"],
                "source_model": record["model_name"],
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "margin": margin,
                "correct_strict": margin > 0,
                "tie": math.isclose(margin, 0.0),
                "pair_credit": 1.0
                if margin > 0
                else 0.5
                if math.isclose(margin, 0.0)
                else 0.0,
                "chosen_diagnostics": chosen_diag,
                "rejected_diagnostics": rejected_diag,
                "adapter_version": FC_REWARDBENCH_ADAPTER_VERSION,
                "verifier_scope": "deterministic_tap_r_controller_not_learned_tier_b",
            }
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pair_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["error_type"])].append(row)
    report = {
        "schema_version": "tapbench.fc_rewardbench_report.v1",
        "adapter_version": FC_REWARDBENCH_ADAPTER_VERSION,
        "verifier_scope": "deterministic_tap_r_controller_not_learned_tier_b",
        "pair_count": len(rows),
        "strict_pair_accuracy": sum(row["correct_strict"] for row in rows) / len(rows),
        "tie_adjusted_pair_accuracy": sum(row["pair_credit"] for row in rows)
        / len(rows),
        "tie_rate": sum(row["tie"] for row in rows) / len(rows),
        "by_error_type": [
            {
                "error_type": error_type,
                "n": len(group),
                "strict_pair_accuracy": sum(row["correct_strict"] for row in group)
                / len(group),
                "tie_adjusted_pair_accuracy": sum(row["pair_credit"] for row in group)
                / len(group),
            }
            for error_type, group in sorted(grouped.items())
        ],
        "diagnostic_errors": dict(
            sorted(
                Counter(
                    error
                    for row in rows
                    for error in (
                        row["chosen_diagnostics"]["errors"]
                        + row["rejected_diagnostics"]["errors"]
                    )
                ).items()
            )
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report

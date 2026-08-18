from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl, write_jsonl


VERIFIED_DISTILLATION_EXPORT_VERSION = "tapbench.verified_distillation_export.v1"
RAW_METHOD = "constrained_abstention"
SELECTIVE_METHOD = "tap_r_selective_full"
CONDITIONS = (
    "raw_generation",
    "execution_filtered",
    "certificate_verified",
    "certificate_verified_plus_clarification",
)
FORBIDDEN_INPUT_KEYS = {
    "gold_action",
    "derivable_values",
    "r2f_oracle",
    "verified_distillation_oracle",
    "offline_only_fields",
    "evidence_certificates",
    "certificate_audit",
    "scorer_version",
    "execution_success",
    "fabrication",
    "unsupported_action_critical",
}


def _flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes"}
    return bool(value)


def _key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("case_id")),
        str(row.get("method")),
        int(row.get("seed", 0)),
    )


def _public_action(case: dict[str, Any], action: Any) -> dict[str, Any] | None:
    if not isinstance(action, dict) or not isinstance(action.get("mode"), str):
        return None
    result = {
        "mode": str(action["mode"]),
        "payload": (
            dict(action.get("payload", {}))
            if isinstance(action.get("payload"), dict)
            else {}
        ),
    }
    if result["mode"] != "call":
        return result
    tool_name = action.get("tool")
    arguments = action.get("arguments")
    if not isinstance(tool_name, str) or not isinstance(arguments, dict):
        return None
    selected = next(
        (
            tool
            for tool in case.get("tools", [])
            if tool.get("name") == tool_name
            or tool.get("canonical_name") == tool_name
        ),
        None,
    )
    if not isinstance(selected, dict) or not isinstance(selected.get("name"), str):
        return None
    canonical_to_public: dict[str, str] = {}
    properties = selected.get("parameters", {}).get("properties", {})
    if isinstance(properties, dict):
        for public_slot, schema in properties.items():
            if isinstance(schema, dict):
                canonical_to_public[str(schema.get("x-ir-name", public_slot))] = str(
                    public_slot
                )
    public_arguments = {
        canonical_to_public.get(str(slot), str(slot)): value
        for slot, value in arguments.items()
    }
    result["tool"] = str(selected["name"])
    result["arguments"] = public_arguments
    return result


def _training_row(
    case: dict[str, Any],
    action: dict[str, Any],
    *,
    source_id: str,
) -> dict[str, Any]:
    public_messages = [
        {"role": str(row["role"]), "content": str(row["content"])}
        for row in case.get("messages", [])
        if isinstance(row, dict)
        and row.get("role") in {"system", "user", "assistant"}
        and isinstance(row.get("content"), str)
    ]
    public_tools = json.loads(
        json.dumps(case.get("tools", []), ensure_ascii=True)
    )
    target = json.dumps(
        action,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    example_id = hashlib.sha256(
        f"{source_id}\0{target}".encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": VERIFIED_DISTILLATION_EXPORT_VERSION,
        "example_id": example_id,
        "family": str(case.get("family")),
        "messages": public_messages,
        "tools": public_tools,
        "target": target,
    }


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        if FORBIDDEN_INPUT_KEYS.intersection(value):
            return True
        return any(_contains_forbidden_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _balanced_select(
    rows: Iterable[dict[str, Any]],
    count: int,
    *,
    condition: str,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)
    for family, values in grouped.items():
        values.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}\0{condition}\0{family}\0{row['example_id']}".encode(
                    "utf-8"
                )
            ).hexdigest()
        )
    selected: list[dict[str, Any]] = []
    families = sorted(grouped)
    offset = 0
    while len(selected) < count:
        progressed = False
        for family in families:
            if offset < len(grouped[family]):
                selected.append(grouped[family][offset])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
        offset += 1
    return selected


def build_condition_pools(
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    certificate_audit: list[dict[str, Any]],
    *,
    allowed_families: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    case_by_id = {str(row["case_id"]): row for row in cases}
    score_by_key = {_key(row): row for row in scores}
    audit_by_key = {_key(row): row for row in certificate_audit}
    pools = {condition: [] for condition in CONDITIONS}
    cert_rows: list[dict[str, Any]] = []
    clarification_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        case = case_by_id.get(str(prediction.get("case_id")))
        if case is None:
            continue
        family = str(case.get("family"))
        if allowed_families is not None and family not in allowed_families:
            continue
        if prediction.get("runner_error") is not None:
            continue
        if _flag(prediction.get("thinking_marker_detected")):
            continue
        metadata = prediction.get("response_metadata", {})
        if isinstance(metadata, dict) and metadata.get("finish_reason") == "length":
            continue
        score = score_by_key.get(_key(prediction))
        if score is None:
            continue
        action = _public_action(case, prediction.get("prediction"))
        if action is None:
            continue
        row = _training_row(
            case,
            action,
            source_id="|".join(map(str, _key(prediction))),
        )
        method = str(prediction.get("method"))
        if method == RAW_METHOD and _flag(score.get("format_valid")):
            pools["raw_generation"].append(row)
            if (
                action.get("mode") == "call"
                and _flag(score.get("schema_valid"))
            ):
                pools["execution_filtered"].append(row)
        if method != SELECTIVE_METHOD:
            continue
        audit = audit_by_key.get(_key(prediction))
        certificate_ok = bool(
            audit
            and _flag(audit.get("eligible"))
            and _flag(audit.get("emitted_call"))
            and _flag(audit.get("passed"))
        )
        if (
            certificate_ok
            and action.get("mode") == "call"
            and _flag(score.get("accepted_call"))
            and _flag(score.get("execution_success"))
        ):
            cert_rows.append(row)
        if (
            case.get("task_kind") == "missing_info"
            and action.get("mode") == "clarify"
            and _flag(score.get("execution_success"))
        ):
            clarification_rows.append(row)
    pools["certificate_verified"] = cert_rows
    pools["certificate_verified_plus_clarification"] = (
        cert_rows + clarification_rows
    )
    for condition, rows in pools.items():
        unique = {row["example_id"]: row for row in rows}
        pools[condition] = list(unique.values())
    return pools


def export_equal_data(
    cases_path: str | Path,
    predictions_path: str | Path,
    scores_path: str | Path,
    certificate_audit_path: str | Path,
    output_dir: str | Path,
    *,
    allowed_families: set[str] | None = None,
    target_examples: int = 2500,
    minimum_examples: int = 2000,
    seed: int = 20260718,
) -> dict[str, Any]:
    pools = build_condition_pools(
        read_jsonl(cases_path),
        read_jsonl(predictions_path),
        read_jsonl(scores_path),
        read_jsonl(certificate_audit_path),
        allowed_families=allowed_families,
    )
    equal_count = min(target_examples, *(len(rows) for rows in pools.values()))
    if equal_count < minimum_examples:
        raise ValueError(
            f"equal-data pool has {equal_count} examples; "
            f"minimum is {minimum_examples}"
        )
    selected = {
        condition: _balanced_select(
            rows,
            equal_count,
            condition=condition,
            seed=seed,
        )
        for condition, rows in pools.items()
    }
    for condition, rows in selected.items():
        if len(rows) != equal_count:
            raise ValueError(f"{condition}: balanced sampler under-filled")
        if any(_contains_forbidden_key(row) for row in rows):
            raise ValueError(f"{condition}: forbidden metadata leaked")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for condition, rows in selected.items():
        path = output / f"{condition}.jsonl"
        write_jsonl(path, rows)
        files[condition] = {
            "path": str(path),
            "eligible_count": len(pools[condition]),
            "selected_count": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    plus_rows = selected["certificate_verified_plus_clarification"]
    clarification_count = sum(
        json.loads(row["target"]).get("mode") == "clarify" for row in plus_rows
    )
    manifest = {
        "schema_version": VERIFIED_DISTILLATION_EXPORT_VERSION,
        "equal_example_count": equal_count,
        "deterministic_seed": seed,
        "allowed_families": (
            sorted(allowed_families) if allowed_families is not None else None
        ),
        "files": files,
        "plus_clarification_count": clarification_count,
        "plus_clarification_fraction": clarification_count / equal_count,
        "input_policy": "public_messages_and_public_tool_schemas_only",
        "target_policy": "public_tool_and_argument_surface_action_ir",
        "filter_definitions": {
            "raw_generation": "format-valid one-pass model output",
            "execution_filtered": (
                "structurally executable call with a catalog-resolved tool; "
                "argument correctness is not used by this filter"
            ),
            "certificate_verified": (
                "executed exact call whose independent evidence-certificate "
                "audit passed"
            ),
            "certificate_verified_plus_clarification": (
                "certificate-verified calls plus exact contract-derived "
                "clarifications"
            ),
        },
        "forbidden_input_keys": sorted(FORBIDDEN_INPUT_KEYS),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

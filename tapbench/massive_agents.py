from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from . import NORMALIZER_VERSION, SCORER_VERSION, VALIDATOR_VERSION
from .bfcl import _convert_function
from .bfcl_official import action_ir_to_bfcl_ast
from .capc_projected import (
    PROJECTED_CAPC_CONDITIONS,
    PROJECTED_CAPC_VERSION,
    SOURCE_CERTIFICATE_VERSION,
    replay_source_certificate,
)
from .eflrx import _tool_name
from .extractive_candidates import user_request_text
from .extractive_qa_verifier import (
    EXTRACTIVE_QA_MODEL_ID,
    EXTRACTIVE_QA_MODEL_REVISION,
    EXTRACTIVE_QA_QUESTION_VERSION,
    EXTRACTIVE_QA_VERIFIER_VERSION,
    validate_extractive_qa_rows,
)
from .io import read_jsonl, write_jsonl, write_yaml
from .qa_evidence_controller import (
    QA_EVIDENCE_CONDITIONS,
    QA_EVIDENCE_CONTROLLER_VERSION,
    QA_EVIDENCE_SYSTEM_LABEL,
    index_verifier_rows,
    validate_verifier_record,
)
from .retrieve_pointer import (
    RETRIEVE_POINTER_CONDITIONS,
    RETRIEVE_POINTER_VERSION,
)
from .semantic_surface_projection import (
    SEMANTIC_SURFACE_CONDITIONS,
    SEMANTIC_SURFACE_VERSION,
)
from .source_span_projection import (
    SOURCE_SPAN_CERTIFICATE_VERSION,
    SOURCE_SPAN_PROJECTION_VERSION,
    action_fingerprint,
    replay_span_certificate,
    slot_catalog,
)
from .slotwise_surface_projection import (
    SLOTWISE_SURFACE_CONDITIONS,
    SLOTWISE_SURFACE_VERSION,
)
from .supervised_router_small_model_qa import (
    SMALL_MODEL_ROUTER_QA_METHODS,
    SMALL_MODEL_ROUTER_QA_SYSTEM_LABEL,
    SMALL_MODEL_ROUTER_QA_VERSION,
)


MASSIVE_AGENTS_ADAPTER_VERSION = "tapbench.massive_agents.v2"
MASSIVE_AGENTS_SCORER_VERSION = "tapbench.massive_agents.official.v2"
MASSIVE_AGENTS_DATASET_COMMIT = (
    "b6156972182bdf34e68c5b5dfbfe6d30db82f104"
)
MASSIVE_AGENTS_GRID_ID = "MASSIVE_Agents_CAPC_language_disjoint_v1"
MASSIVE_AGENTS_SALT = "MASSIVE_Agents_CAPC_language_disjoint_v1"
_FORBIDDEN_RUNTIME_KEYS = {
    "gt",
    "gold",
    "gold_action",
    "ground_truth",
    "expected",
    "expected_action",
    "bfcl_gold",
    "derivable_values",
}


def _read_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_oid(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _case_id(language: str, source_id: str) -> str:
    safe_language = re.sub(r"[^A-Za-z0-9]+", "_", language).strip("_")
    safe_id = re.sub(r"[^A-Za-z0-9]+", "_", source_id).strip("_")
    return f"massive_agents_{safe_language}_{safe_id}"


def sample_rank(
    language: str,
    stable_id: str,
    *,
    salt: str = MASSIVE_AGENTS_SALT,
) -> str:
    return hashlib.sha256(
        (
            salt
            + "\0"
            + language
            + "\0"
            + stable_id
        ).encode("utf-8")
    ).hexdigest()


def _runtime_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Return exactly one Action IR JSON object. Do not invent or "
                "translate argument values; preserve values from the request."
            ),
        },
        {"role": "user", "content": question},
    ]


def runtime_gold_leaks(value: Any, path: str = "$") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold()
            child_path = f"{path}.{key}"
            if normalized in _FORBIDDEN_RUNTIME_KEYS or "gold" in normalized:
                failures.append(child_path)
            failures.extend(runtime_gold_leaks(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(runtime_gold_leaks(child, f"{path}[{index}]"))
    return failures


def prepare_massive_cases(
    source_root: str | Path,
    cases_path: str | Path,
    scorer_gold_path: str | Path,
    manifest_path: str | Path,
    *,
    languages: Iterable[str],
    cases_per_language: int,
    rank_start: int = 1,
    split: str,
    salt: str = MASSIVE_AGENTS_SALT,
    dataset_commit: str = MASSIVE_AGENTS_DATASET_COMMIT,
    grid_id: str = MASSIVE_AGENTS_GRID_ID,
) -> dict[str, Any]:
    root = Path(source_root)
    language_list = tuple(str(value) for value in languages)
    if not language_list:
        raise ValueError("at least one MASSIVE-Agents language is required")
    if cases_per_language <= 0:
        raise ValueError("cases_per_language must be positive")
    if rank_start <= 0:
        raise ValueError("rank_start must be positive")

    cases: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    source_files: dict[str, Any] = {}
    selected_ids: dict[str, list[str]] = {}
    catalog_hashes: set[str] = set()
    for language in language_list:
        language_root = root / language
        input_path = language_root / "input.jsonl"
        golden_path = language_root / "golden.jsonl"
        inputs = _read_objects(input_path)
        golden = _read_objects(golden_path)
        if len(inputs) != len(golden):
            raise ValueError(
                f"{language}: input/golden row count mismatch "
                f"({len(inputs)} != {len(golden)})"
            )
        rank_end = rank_start + cases_per_language - 1
        if len(inputs) < rank_end:
            raise ValueError(
                f"{language}: requested ranks {rank_start} through {rank_end} "
                f"from only {len(inputs)} rows"
            )
        seen_ids: set[str] = set()
        paired: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for index, (raw, gold) in enumerate(zip(inputs, golden)):
            source_id = str(raw.get("id", index))
            if source_id in seen_ids:
                raise ValueError(f"{language}: duplicate source id {source_id}")
            seen_ids.add(source_id)
            if raw.get("gt") != gold:
                raise ValueError(
                    f"{language}:{source_id}: embedded gt and golden row differ"
                )
            paired.append((source_id, raw, gold))
        paired.sort(key=lambda row: (sample_rank(language, row[0], salt=salt), row[0]))
        selected = paired[rank_start - 1 : rank_end]
        selected_ids[language] = [source_id for source_id, _, _ in selected]

        for source_id, raw, gold in selected:
            question = raw.get("question")
            functions = raw.get("function")
            if not isinstance(question, str) or not isinstance(functions, list):
                raise ValueError(
                    f"{language}:{source_id}: invalid question/function payload"
                )
            tools = [
                _convert_function(function)
                for function in functions
                if isinstance(function, dict)
            ]
            catalog_hashes.add(
                hashlib.sha256(
                    json.dumps(
                        tools,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest()
            )
            case = {
                "case_id": _case_id(language, source_id),
                "hypothesis_grid_id": grid_id,
                "hypothesis": "external_massive_agents",
                "split": split,
                "family": "massive_agents",
                "task_kind": "call",
                "factors": {
                    "external_source": "MASSIVE-Agents-10k",
                    "language": language,
                    "task_kind": "call",
                },
                "messages": _runtime_messages(question),
                "tools": tools,
                "tool_aliases": {
                    str(tool["name"]): str(tool["canonical_name"])
                    for tool in tools
                },
                "argument_aliases": {},
                "metadata": {
                    "external_source": "MASSIVE-Agents-10k",
                    "massive_id": source_id,
                    "language": language,
                    "test_category": str(
                        raw.get("test_category", "multiple_function")
                    ),
                    "dataset_commit": dataset_commit,
                    "sample_rank_sha256": sample_rank(
                        language, source_id, salt=salt
                    ),
                },
            }
            leaks = runtime_gold_leaks(case)
            if leaks:
                raise ValueError(
                    f"runtime gold firewall rejected {case['case_id']}: {leaks}"
                )
            cases.append(case)
            gold_rows.append(
                {
                    "case_id": case["case_id"],
                    "massive_id": source_id,
                    "language": language,
                    "test_category": case["metadata"]["test_category"],
                    "ground_truth": [gold],
                }
            )

        source_files[language] = {
            "input": {
                "path": str(input_path),
                "size_bytes": input_path.stat().st_size,
                "git_blob_oid": _git_blob_oid(input_path),
                "sha256": _sha256(input_path),
            },
            "golden": {
                "path": str(golden_path),
                "size_bytes": golden_path.stat().st_size,
                "git_blob_oid": _git_blob_oid(golden_path),
                "sha256": _sha256(golden_path),
            },
            "source_rows": len(inputs),
            "selected_rows": len(selected),
        }

    if len(catalog_hashes) != 1:
        raise ValueError(
            "public tool catalog differs across selected MASSIVE languages"
        )
    write_jsonl(cases_path, cases)
    write_jsonl(scorer_gold_path, gold_rows)
    manifest = {
        "schema_version": "tapbench.massive_agents_prepare_manifest.v2",
        "adapter_version": MASSIVE_AGENTS_ADAPTER_VERSION,
        "dataset_commit": dataset_commit,
        "grid_id": grid_id,
        "split": split,
        "languages": list(language_list),
        "cases_per_language": cases_per_language,
        "rank_start": rank_start,
        "rank_end": rank_start + cases_per_language - 1,
        "case_count": len(cases),
        "sampling_algorithm": "sha256_rank_v1",
        "sampling_salt": salt,
        "selected_ids": selected_ids,
        "public_catalog_sha256": next(iter(catalog_hashes)),
        "source_files": source_files,
        "runtime_cases_path": str(cases_path),
        "runtime_cases_sha256": _sha256(Path(cases_path)),
        "scorer_gold_path": str(scorer_gold_path),
        "scorer_gold_sha256": _sha256(Path(scorer_gold_path)),
        "runtime_gold_firewall": "passed",
    }
    write_yaml(manifest_path, manifest)
    return manifest


def _prediction_identity(
    case: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    metadata = case.get("metadata", {})
    response_metadata = prediction.get("response_metadata", {})
    if not isinstance(response_metadata, dict):
        response_metadata = {}
    return {
        "case_id": case["case_id"],
        "massive_id": metadata.get("massive_id"),
        "language": metadata.get("language"),
        "test_category": metadata.get("test_category"),
        "dataset_commit": metadata.get("dataset_commit"),
        "hypothesis_grid_id": case.get("hypothesis_grid_id"),
        "model_id": prediction.get("model_id"),
        "method": prediction.get("method"),
        "seed": prediction.get("seed"),
        "backend": prediction.get("backend"),
        "quantization": prediction.get("quantization"),
        "model_artifact": prediction.get("model_artifact"),
        "chat_template": prediction.get("chat_template"),
        "grammar_engine": prediction.get("grammar_engine"),
        "chat_parser": prediction.get("chat_parser"),
        "inference_path": prediction.get("inference_path"),
        "thinking_mode": prediction.get("thinking_mode"),
        "thinking_marker_detected": prediction.get(
            "thinking_marker_detected", False
        ),
        "runner_error": prediction.get("runner_error"),
        "finish_reason": response_metadata.get("finish_reason"),
        "scorer_version": SCORER_VERSION,
        "normalizer_version": NORMALIZER_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "massive_agents_adapter_version": MASSIVE_AGENTS_ADAPTER_VERSION,
        "massive_agents_scorer_version": MASSIVE_AGENTS_SCORER_VERSION,
        "capc_version": prediction.get("capc_version"),
        "extractive_candidate_version": prediction.get(
            "extractive_candidate_version"
        ),
        "capc_runner_version": prediction.get("capc_runner_version"),
        "projected_capc_version": prediction.get("projected_capc_version"),
        "source_certificate_version": prediction.get(
            "source_certificate_version"
        ),
        "retrieve_pointer_version": prediction.get(
            "retrieve_pointer_version"
        ),
        "semantic_surface_version": prediction.get(
            "semantic_surface_version"
        ),
        "slotwise_surface_version": prediction.get(
            "slotwise_surface_version"
        ),
        "semantic_surface_materializer_version": prediction.get(
            "semantic_surface_materializer_version"
        ),
        "qa_evidence_controller_version": prediction.get(
            "qa_evidence_controller_version"
        ),
        "qa_evidence_system_label": prediction.get(
            "qa_evidence_system_label"
        ),
        "qa_verifier_version": prediction.get("qa_verifier_version"),
        "qa_verifier_question_version": prediction.get(
            "qa_verifier_question_version"
        ),
        "qa_verifier_model_id": prediction.get("qa_verifier_model_id"),
        "qa_verifier_model_revision": prediction.get(
            "qa_verifier_model_revision"
        ),
        "qa_verifier_backend": prediction.get("qa_verifier_backend"),
        "qa_verifier_dtype": prediction.get("qa_verifier_dtype"),
        "qa_verifier_artifact_sha256": prediction.get(
            "qa_verifier_artifact_sha256"
        ),
        "retriever_version": prediction.get("retriever_version"),
        "retriever_model_id": prediction.get("retriever_model_id"),
        "retriever_revision": prediction.get("retriever_revision"),
        "retriever_serialization_arm": prediction.get(
            "retriever_serialization_arm"
        ),
        "retriever_k": prediction.get("retriever_k"),
        "ranking_sha256": prediction.get("ranking_sha256"),
        "ranking_artifact_sha256": prediction.get(
            "ranking_artifact_sha256"
        ),
        "source_span_projection_version": prediction.get(
            "source_span_projection_version"
        ),
        "source_span_certificate_version": prediction.get(
            "source_span_certificate_version"
        ),
        "massive_runner_version": prediction.get("massive_runner_version"),
        "action_risk_threshold": prediction.get("action_risk_threshold"),
    }


def evaluate_massive_official(
    cases_path: str | Path,
    predictions_path: str | Path,
    scorer_gold_path: str | Path,
    bfcl_root: str | Path,
    output_dir: str | Path,
    *,
    source_commit: str,
) -> dict[str, Any]:
    package_root = Path(bfcl_root).resolve()
    if not (package_root / "bfcl_eval").is_dir():
        raise FileNotFoundError(
            f"BFCL package root is invalid: {package_root}"
        )
    sys.path.insert(0, str(package_root))
    try:
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    finally:
        if sys.path and sys.path[0] == str(package_root):
            sys.path.pop(0)

    cases = {row["case_id"]: row for row in read_jsonl(cases_path)}
    gold = {row["case_id"]: row for row in read_jsonl(scorer_gold_path)}
    predictions = read_jsonl(predictions_path)
    if set(cases) != set(gold):
        raise ValueError("runtime cases and scorer-only gold IDs differ")

    details: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for prediction in predictions:
        case_id = str(prediction.get("case_id"))
        case = cases.get(case_id)
        if case is None:
            raise KeyError(
                f"prediction references unknown MASSIVE case_id: {case_id}"
            )
        key = (
            case_id,
            str(prediction.get("model_id")),
            str(prediction.get("method")),
            int(prediction.get("seed", 0)),
        )
        if key in seen:
            raise ValueError(f"duplicate MASSIVE prediction cell: {key}")
        seen.add(key)
        action = prediction.get("prediction")
        ast = action_ir_to_bfcl_ast(action)
        scorer_gold = gold[case_id]
        result = ast_checker(
            case.get("tools", []),
            ast,
            scorer_gold["ground_truth"],
            Language.PYTHON,
            str(scorer_gold.get("test_category", "multiple_function")),
            "gorilla-openfunctions-v2",
        )
        emitted_call = bool(ast)
        correct = bool(result.get("valid"))
        incorrect_emitted = emitted_call and not correct
        details.append(
            {
                **_prediction_identity(case, prediction),
                "official_ast_correct": correct,
                "emitted_call": emitted_call,
                "incorrect_emitted_call": incorrect_emitted,
                "safe_utility_lambda_4": (
                    1.0 if correct else -4.0 if incorrect_emitted else 0.0
                ),
                "error_type": result.get("error_type"),
                "error": result.get("error"),
                "decoded_action_ast": ast,
                "official_language": Language.PYTHON.value,
                "official_checker": (
                    "bfcl_eval.eval_checker.ast_eval.ast_checker.ast_checker"
                ),
                "official_checker_commit": source_commit,
            }
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "official_details.jsonl", details)
    report = summarize_massive_details(details)
    checker_path = (
        package_root
        / "bfcl_eval"
        / "eval_checker"
        / "ast_eval"
        / "ast_checker.py"
    )
    report.update(
        {
            "schema_version": "tapbench.massive_agents_official_report.v2",
            "adapter_version": MASSIVE_AGENTS_ADAPTER_VERSION,
            "scorer_version": MASSIVE_AGENTS_SCORER_VERSION,
            "official_source_commit": source_commit,
            "official_checker_sha256": hashlib.sha256(
                checker_path.read_bytes()
            ).hexdigest(),
            "case_count": len(cases),
            "prediction_count": len(details),
        }
    )
    (output / "official_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(bool(row.get("official_ast_correct")) for row in rows)
    emitted = sum(bool(row.get("emitted_call")) for row in rows)
    incorrect = sum(
        bool(row.get("incorrect_emitted_call")) for row in rows
    )
    return {
        "n": n,
        "correct": correct,
        "official_ast_accuracy": correct / n if n else 0.0,
        "emitted_calls": emitted,
        "call_coverage": emitted / n if n else 0.0,
        "accepted_call_exact_precision": (
            correct / emitted if emitted else None
        ),
        "incorrect_emitted_calls": incorrect,
        "incorrect_emitted_calls_per_100": (
            100.0 * incorrect / n if n else 0.0
        ),
        "safe_utility_lambda_4": (
            sum(float(row.get("safe_utility_lambda_4", 0.0)) for row in rows)
            / n
            if n
            else 0.0
        ),
    }


def summarize_massive_details(
    details: list[dict[str, Any]],
) -> dict[str, Any]:
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        by_group[
            (
                str(row.get("model_id")),
                str(row.get("method")),
                str(row.get("language")),
            )
        ].append(row)
    groups: list[dict[str, Any]] = []
    by_method_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (model_id, method, language), rows in sorted(by_group.items()):
        metrics = _group_metrics(rows)
        group = {
            "model_id": model_id,
            "method": method,
            "language": language,
            **metrics,
            "error_types": dict(
                sorted(
                    Counter(
                        row.get("error_type")
                        for row in rows
                        if row.get("error_type")
                    ).items()
                )
            ),
        }
        groups.append(group)
        by_method_model[(model_id, method)].append(group)

    language_macro: list[dict[str, Any]] = []
    macro_metrics = (
        "official_ast_accuracy",
        "call_coverage",
        "accepted_call_exact_precision",
        "incorrect_emitted_calls_per_100",
        "safe_utility_lambda_4",
    )
    for (model_id, method), rows in sorted(by_method_model.items()):
        output: dict[str, Any] = {
            "model_id": model_id,
            "method": method,
            "language_count": len(rows),
            "n": sum(int(row["n"]) for row in rows),
        }
        for metric in macro_metrics:
            values = [row[metric] for row in rows]
            output[f"language_macro_{metric}"] = (
                None
                if any(value is None for value in values)
                else sum(float(value) for value in values) / len(values)
            )
        language_macro.append(output)
    return {"groups": groups, "language_macro": language_macro}


def audit_projected_certificates(
    cases_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    cases = {row["case_id"]: row for row in read_jsonl(cases_path)}
    predictions = read_jsonl(predictions_path)
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        if prediction.get("method") not in PROJECTED_CAPC_CONDITIONS:
            continue
        case = cases[str(prediction["case_id"])]
        request_text = "\n".join(
            str(message.get("content", ""))
            for message in case.get("messages", [])
            if str(message.get("role", "")).casefold() == "user"
        )
        action = prediction.get("prediction")
        metadata = prediction.get("response_metadata", {})
        failures: list[str] = []
        if not isinstance(metadata, dict):
            metadata = {}
            failures.append("response_metadata_not_object")
        if isinstance(action, dict) and action.get("mode") == "call":
            arguments = action.get("arguments")
            certificates = metadata.get("evidence_certificates")
            if not isinstance(arguments, dict):
                arguments = {}
                failures.append("arguments_not_object")
            if not isinstance(certificates, dict):
                certificates = {}
                failures.append("certificates_not_object")
            if set(arguments) != set(certificates):
                failures.append("argument_certificate_key_mismatch")
            for slot, value in arguments.items():
                certificate = certificates.get(slot)
                if not isinstance(certificate, dict):
                    failures.append(f"missing_certificate:{slot}")
                    continue
                if certificate.get("value") != value:
                    failures.append(f"certificate_value_mismatch:{slot}")
                if certificate.get("certificate_version") != SOURCE_CERTIFICATE_VERSION:
                    failures.append(f"certificate_version_mismatch:{slot}")
                if not replay_source_certificate(request_text, certificate):
                    failures.append(f"certificate_replay_failed:{slot}")
            action_hash = hashlib.sha256(
                json.dumps(
                    action,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            if action_hash != metadata.get("materialized_action_sha256"):
                failures.append("materialized_action_hash_mismatch")
            if action.get("tool") != metadata.get("projected_tool"):
                failures.append("projected_tool_mismatch")
        rows.append(
            {
                "case_id": prediction.get("case_id"),
                "model_id": prediction.get("model_id"),
                "method": prediction.get("method"),
                "seed": prediction.get("seed"),
                "accepted_call": bool(
                    isinstance(action, dict) and action.get("mode") == "call"
                ),
                "passed": not failures,
                "failures": failures,
                "projected_capc_version": prediction.get(
                    "projected_capc_version"
                ),
                "source_certificate_version": prediction.get(
                    "source_certificate_version"
                ),
            }
        )
    write_jsonl(output_path, rows)
    report = {
        "schema_version": "tapbench.projected_certificate_audit.v1",
        "projected_capc_version": PROJECTED_CAPC_VERSION,
        "source_certificate_version": SOURCE_CERTIFICATE_VERSION,
        "rows": len(rows),
        "accepted_calls": sum(bool(row["accepted_call"]) for row in rows),
        "passed": sum(bool(row["passed"]) for row in rows),
        "failed": sum(not bool(row["passed"]) for row in rows),
    }
    target = Path(summary_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def audit_retrieve_pointer_certificates(
    cases_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    cases = {row["case_id"]: row for row in read_jsonl(cases_path)}
    predictions = read_jsonl(predictions_path)
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        if prediction.get("method") not in RETRIEVE_POINTER_CONDITIONS:
            continue
        case_id = str(prediction.get("case_id"))
        case = cases.get(case_id)
        failures: list[str] = []
        if case is None:
            failures.append("unknown_case_id")
            request_text = ""
            language = "unknown"
        else:
            request_text = "\n".join(
                str(message.get("content", ""))
                for message in case.get("messages", [])
                if str(message.get("role", "")).casefold() == "user"
            )
            language = str(
                case.get("metadata", {}).get("language")
                or case.get("factors", {}).get("language")
                or "unknown"
            )
        action = prediction.get("prediction")
        metadata = prediction.get("response_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            failures.append("response_metadata_not_object")
        accepted_call = bool(
            isinstance(action, dict) and action.get("mode") == "call"
        )
        if prediction.get("retrieve_pointer_version") != RETRIEVE_POINTER_VERSION:
            failures.append("retrieve_pointer_version_mismatch")
        if (
            prediction.get("source_span_projection_version")
            != SOURCE_SPAN_PROJECTION_VERSION
        ):
            failures.append("source_span_projection_version_mismatch")
        if (
            prediction.get("source_span_certificate_version")
            != SOURCE_SPAN_CERTIFICATE_VERSION
        ):
            failures.append("source_span_certificate_version_mismatch")
        if not prediction.get("ranking_artifact_sha256"):
            failures.append("ranking_artifact_sha256_missing")
        if not prediction.get("ranking_sha256"):
            failures.append("ranking_sha256_missing")
        if accepted_call:
            arguments = action.get("arguments")
            certificates = metadata.get("evidence_certificates")
            selected_span_ids = metadata.get("selected_span_ids")
            if not isinstance(arguments, dict):
                arguments = {}
                failures.append("arguments_not_object")
            if not isinstance(certificates, dict):
                certificates = {}
                failures.append("certificates_not_object")
            if not isinstance(selected_span_ids, dict):
                selected_span_ids = {}
                failures.append("selected_span_ids_not_object")
            if set(arguments) != set(certificates):
                failures.append("argument_certificate_key_mismatch")
            if set(arguments) != set(selected_span_ids):
                failures.append("argument_span_id_key_mismatch")
            for slot, value in arguments.items():
                certificate = certificates.get(slot)
                if not isinstance(certificate, dict):
                    failures.append(f"missing_certificate:{slot}")
                    continue
                if certificate.get("value") != value:
                    failures.append(f"certificate_value_mismatch:{slot}")
                if certificate.get("span_id") != selected_span_ids.get(slot):
                    failures.append(f"selected_span_id_mismatch:{slot}")
                if not replay_span_certificate(
                    request_text,
                    language,
                    certificate,
                ):
                    failures.append(f"certificate_replay_failed:{slot}")
            if action_fingerprint(action) != metadata.get(
                "materialized_action_sha256"
            ):
                failures.append("materialized_action_hash_mismatch")
            if action.get("tool") != metadata.get("selected_tool"):
                failures.append("selected_tool_mismatch")
            if metadata.get("no_generated_action_critical_literals") is not True:
                failures.append("finite_literal_invariant_missing")
        rows.append(
            {
                "case_id": prediction.get("case_id"),
                "model_id": prediction.get("model_id"),
                "method": prediction.get("method"),
                "seed": prediction.get("seed"),
                "accepted_call": accepted_call,
                "passed": not failures,
                "failures": failures,
                "retrieve_pointer_version": prediction.get(
                    "retrieve_pointer_version"
                ),
                "source_span_projection_version": prediction.get(
                    "source_span_projection_version"
                ),
                "source_span_certificate_version": prediction.get(
                    "source_span_certificate_version"
                ),
                "ranking_artifact_sha256": prediction.get(
                    "ranking_artifact_sha256"
                ),
            }
        )
    write_jsonl(output_path, rows)
    report = {
        "schema_version": "tapbench.retrieve_pointer_certificate_audit.v1",
        "retrieve_pointer_version": RETRIEVE_POINTER_VERSION,
        "source_span_projection_version": SOURCE_SPAN_PROJECTION_VERSION,
        "source_span_certificate_version": SOURCE_SPAN_CERTIFICATE_VERSION,
        "rows": len(rows),
        "accepted_calls": sum(bool(row["accepted_call"]) for row in rows),
        "passed": sum(bool(row["passed"]) for row in rows),
        "failed": sum(not bool(row["passed"]) for row in rows),
    }
    target = Path(summary_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def audit_semantic_surface_certificates(
    cases_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    cases = {str(row["case_id"]): row for row in read_jsonl(cases_path)}
    predictions = read_jsonl(predictions_path)
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        if prediction.get("method") not in SEMANTIC_SURFACE_CONDITIONS:
            continue
        case_id = str(prediction.get("case_id"))
        case = cases.get(case_id)
        failures: list[str] = []
        if case is None:
            failures.append("unknown_case_id")
            request_text = ""
            language = "unknown"
        else:
            request_text = "\n".join(
                str(message.get("content", ""))
                for message in case.get("messages", [])
                if str(message.get("role", "")).casefold() == "user"
            )
            language = str(
                case.get("metadata", {}).get("language")
                or case.get("factors", {}).get("language")
                or "unknown"
            )
        action = prediction.get("prediction")
        metadata = prediction.get("response_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            failures.append("response_metadata_not_object")
        accepted_call = bool(
            isinstance(action, dict) and action.get("mode") == "call"
        )
        if prediction.get("semantic_surface_version") != SEMANTIC_SURFACE_VERSION:
            failures.append("semantic_surface_version_mismatch")
        if metadata.get("semantic_surface_version") != SEMANTIC_SURFACE_VERSION:
            failures.append("metadata_semantic_surface_version_mismatch")
        if (
            prediction.get("source_span_projection_version")
            != SOURCE_SPAN_PROJECTION_VERSION
        ):
            failures.append("source_span_projection_version_mismatch")
        if (
            prediction.get("source_span_certificate_version")
            != SOURCE_SPAN_CERTIFICATE_VERSION
        ):
            failures.append("source_span_certificate_version_mismatch")
        for field in ("ranking_artifact_sha256", "ranking_sha256"):
            if not prediction.get(field):
                failures.append(f"{field}_missing")
        for field in (
            "call_only_tool_election",
            "semantic_tool_labels",
            "semantic_slot_labels",
            "source_surface_value_labels",
        ):
            if metadata.get(field) is not True:
                failures.append(f"{field}_invariant_missing")
        if metadata.get("no_call_election_option") is not False:
            failures.append("call_only_election_surface_mismatch")

        if accepted_call:
            arguments = action.get("arguments")
            certificates = metadata.get("evidence_certificates")
            selected_span_ids = metadata.get("selected_span_ids")
            selected_surfaces = metadata.get("selected_surface_values")
            if not isinstance(arguments, dict):
                arguments = {}
                failures.append("arguments_not_object")
            if not isinstance(certificates, dict):
                certificates = {}
                failures.append("certificates_not_object")
            if not isinstance(selected_span_ids, dict):
                selected_span_ids = {}
                failures.append("selected_span_ids_not_object")
            if not isinstance(selected_surfaces, dict):
                selected_surfaces = {}
                failures.append("selected_surface_values_not_object")
            if set(arguments) != set(certificates):
                failures.append("argument_certificate_key_mismatch")
            if set(arguments) != set(selected_span_ids):
                failures.append("argument_span_id_key_mismatch")
            certificate_surfaces: list[str] = []
            for slot, value in arguments.items():
                certificate = certificates.get(slot)
                if not isinstance(certificate, dict):
                    failures.append(f"missing_certificate:{slot}")
                    continue
                if certificate.get("value") != value:
                    failures.append(f"certificate_value_mismatch:{slot}")
                if certificate.get("span_id") != selected_span_ids.get(slot):
                    failures.append(f"selected_span_id_mismatch:{slot}")
                source_text = certificate.get("source_text")
                if isinstance(source_text, str):
                    certificate_surfaces.append(source_text)
                if not replay_span_certificate(
                    request_text,
                    language,
                    certificate,
                ):
                    failures.append(f"certificate_replay_failed:{slot}")
            if sorted(map(str, selected_surfaces.values())) != sorted(
                certificate_surfaces
            ):
                failures.append("selected_surface_certificate_mismatch")
            if action_fingerprint(action) != metadata.get(
                "materialized_action_sha256"
            ):
                failures.append("materialized_action_hash_mismatch")
            if action.get("tool") != metadata.get("selected_tool"):
                failures.append("selected_tool_mismatch")
            if action.get("tool") not in metadata.get("retrieved_tools", []):
                failures.append("selected_tool_not_retrieved")
            if metadata.get("no_unconstrained_action_critical_tokens") is not True:
                failures.append("finite_surface_invariant_missing")

        rows.append(
            {
                "case_id": prediction.get("case_id"),
                "model_id": prediction.get("model_id"),
                "method": prediction.get("method"),
                "seed": prediction.get("seed"),
                "accepted_call": accepted_call,
                "passed": not failures,
                "failures": failures,
                "semantic_surface_version": prediction.get(
                    "semantic_surface_version"
                ),
                "source_span_projection_version": prediction.get(
                    "source_span_projection_version"
                ),
                "source_span_certificate_version": prediction.get(
                    "source_span_certificate_version"
                ),
                "ranking_artifact_sha256": prediction.get(
                    "ranking_artifact_sha256"
                ),
            }
        )
    write_jsonl(output_path, rows)
    report = {
        "schema_version": "tapbench.semantic_surface_certificate_audit.v1",
        "semantic_surface_version": SEMANTIC_SURFACE_VERSION,
        "source_span_projection_version": SOURCE_SPAN_PROJECTION_VERSION,
        "source_span_certificate_version": SOURCE_SPAN_CERTIFICATE_VERSION,
        "rows": len(rows),
        "accepted_calls": sum(bool(row["accepted_call"]) for row in rows),
        "passed": sum(bool(row["passed"]) for row in rows),
        "failed": sum(not bool(row["passed"]) for row in rows),
    }
    target = Path(summary_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def audit_slotwise_surface_certificates(
    cases_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    cases = {str(row["case_id"]): row for row in read_jsonl(cases_path)}
    predictions = read_jsonl(predictions_path)
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        if prediction.get("method") not in SLOTWISE_SURFACE_CONDITIONS:
            continue
        case_id = str(prediction.get("case_id"))
        case = cases.get(case_id)
        failures: list[str] = []
        if case is None:
            failures.append("unknown_case_id")
            request_text = ""
            language = "unknown"
        else:
            request_text = "\n".join(
                str(message.get("content", ""))
                for message in case.get("messages", [])
                if str(message.get("role", "")).casefold() == "user"
            )
            language = str(
                case.get("metadata", {}).get("language")
                or case.get("factors", {}).get("language")
                or "unknown"
            )
        action = prediction.get("prediction")
        metadata = prediction.get("response_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            failures.append("response_metadata_not_object")
        accepted_call = bool(
            isinstance(action, dict) and action.get("mode") == "call"
        )
        if prediction.get("slotwise_surface_version") != SLOTWISE_SURFACE_VERSION:
            failures.append("slotwise_surface_version_mismatch")
        if metadata.get("slotwise_surface_version") != SLOTWISE_SURFACE_VERSION:
            failures.append("metadata_slotwise_surface_version_mismatch")
        if (
            prediction.get("source_span_projection_version")
            != SOURCE_SPAN_PROJECTION_VERSION
        ):
            failures.append("source_span_projection_version_mismatch")
        if (
            prediction.get("source_span_certificate_version")
            != SOURCE_SPAN_CERTIFICATE_VERSION
        ):
            failures.append("source_span_certificate_version_mismatch")
        for field in ("ranking_artifact_sha256", "ranking_sha256"):
            if not prediction.get(field):
                failures.append(f"{field}_missing")
        for field in (
            "call_only_tool_election",
            "semantic_tool_labels",
            "semantic_slot_labels",
            "source_surface_value_labels",
            "slotwise_independent_generation",
            "null_optional_absence",
            "candidate_list_visible_in_prompt",
            "candidate_order_semantically_irrelevant",
        ):
            if metadata.get(field) is not True:
                failures.append(f"{field}_invariant_missing")
        if metadata.get("no_call_election_option") is not False:
            failures.append("call_only_election_surface_mismatch")

        if accepted_call:
            arguments = action.get("arguments")
            certificates = metadata.get("evidence_certificates")
            selected_span_ids = metadata.get("selected_span_ids")
            selected_surfaces = metadata.get("selected_surface_values")
            slotwise_selections = metadata.get("slotwise_selections")
            if not isinstance(arguments, dict):
                arguments = {}
                failures.append("arguments_not_object")
            if not isinstance(certificates, dict):
                certificates = {}
                failures.append("certificates_not_object")
            if not isinstance(selected_span_ids, dict):
                selected_span_ids = {}
                failures.append("selected_span_ids_not_object")
            if not isinstance(selected_surfaces, dict):
                selected_surfaces = {}
                failures.append("selected_surface_values_not_object")
            if not isinstance(slotwise_selections, dict):
                slotwise_selections = {}
                failures.append("slotwise_selections_not_object")
            if set(arguments) != set(certificates):
                failures.append("argument_certificate_key_mismatch")
            if set(arguments) != set(selected_span_ids):
                failures.append("argument_span_id_key_mismatch")
            non_null_selections = {
                slot: value
                for slot, value in slotwise_selections.items()
                if value is not None
            }
            if sorted(map(str, non_null_selections.values())) != sorted(
                map(str, selected_surfaces.values())
            ):
                failures.append("slotwise_materialized_surface_mismatch")
            certificate_surfaces: list[str] = []
            for slot, value in arguments.items():
                certificate = certificates.get(slot)
                if not isinstance(certificate, dict):
                    failures.append(f"missing_certificate:{slot}")
                    continue
                if certificate.get("value") != value:
                    failures.append(f"certificate_value_mismatch:{slot}")
                if certificate.get("span_id") != selected_span_ids.get(slot):
                    failures.append(f"selected_span_id_mismatch:{slot}")
                source_text = certificate.get("source_text")
                if isinstance(source_text, str):
                    certificate_surfaces.append(source_text)
                if not replay_span_certificate(
                    request_text,
                    language,
                    certificate,
                ):
                    failures.append(f"certificate_replay_failed:{slot}")
            if sorted(map(str, selected_surfaces.values())) != sorted(
                certificate_surfaces
            ):
                failures.append("selected_surface_certificate_mismatch")
            if action_fingerprint(action) != metadata.get(
                "materialized_action_sha256"
            ):
                failures.append("materialized_action_hash_mismatch")
            if action.get("tool") != metadata.get("selected_tool"):
                failures.append("selected_tool_mismatch")
            if action.get("tool") not in metadata.get("retrieved_tools", []):
                failures.append("selected_tool_not_retrieved")
            if metadata.get("no_unconstrained_action_critical_tokens") is not True:
                failures.append("finite_surface_invariant_missing")

        rows.append(
            {
                "case_id": prediction.get("case_id"),
                "model_id": prediction.get("model_id"),
                "method": prediction.get("method"),
                "seed": prediction.get("seed"),
                "accepted_call": accepted_call,
                "passed": not failures,
                "failures": failures,
                "slotwise_surface_version": prediction.get(
                    "slotwise_surface_version"
                ),
                "source_span_projection_version": prediction.get(
                    "source_span_projection_version"
                ),
                "source_span_certificate_version": prediction.get(
                    "source_span_certificate_version"
                ),
                "ranking_artifact_sha256": prediction.get(
                    "ranking_artifact_sha256"
                ),
            }
        )
    write_jsonl(output_path, rows)
    report = {
        "schema_version": "tapbench.slotwise_surface_certificate_audit.v1",
        "slotwise_surface_version": SLOTWISE_SURFACE_VERSION,
        "source_span_projection_version": SOURCE_SPAN_PROJECTION_VERSION,
        "source_span_certificate_version": SOURCE_SPAN_CERTIFICATE_VERSION,
        "rows": len(rows),
        "accepted_calls": sum(bool(row["accepted_call"]) for row in rows),
        "passed": sum(bool(row["passed"]) for row in rows),
        "failed": sum(not bool(row["passed"]) for row in rows),
    }
    target = Path(summary_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def audit_qa_evidence_certificates(
    cases_path: str | Path,
    predictions_path: str | Path,
    verifier_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    cases = {str(row["case_id"]): row for row in read_jsonl(cases_path)}
    predictions = read_jsonl(predictions_path)
    verifier_rows = read_jsonl(verifier_path)
    artifact_failures = validate_extractive_qa_rows(verifier_rows)
    verifier_index = index_verifier_rows(verifier_rows)
    verifier_sha256 = _sha256(Path(verifier_path))
    rows: list[dict[str, Any]] = []

    for prediction in predictions:
        method = prediction.get("method")
        if method not in (
            *QA_EVIDENCE_CONDITIONS,
            *SMALL_MODEL_ROUTER_QA_METHODS,
        ):
            continue
        case_id = str(prediction.get("case_id"))
        case = cases.get(case_id)
        failures: list[str] = []
        if artifact_failures:
            failures.append("verifier_artifact_validation_failed")
        if case is None:
            failures.append("unknown_case_id")
            request_text = ""
            language = "unknown"
            tools: list[dict[str, Any]] = []
        else:
            request_text = user_request_text(case.get("messages", []))
            language = str(
                case.get("metadata", {}).get("language")
                or case.get("factors", {}).get("language")
                or "unknown"
            )
            tools = list(case.get("tools", []))

        action = prediction.get("prediction")
        metadata = prediction.get("response_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            failures.append("response_metadata_not_object")
        accepted_call = bool(
            isinstance(action, dict) and action.get("mode") == "call"
        )

        bridge_condition = method in SMALL_MODEL_ROUTER_QA_METHODS
        expected_identity = {
            "qa_evidence_controller_version": (
                SMALL_MODEL_ROUTER_QA_VERSION
                if bridge_condition
                else QA_EVIDENCE_CONTROLLER_VERSION
            ),
            "qa_evidence_system_label": (
                SMALL_MODEL_ROUTER_QA_SYSTEM_LABEL
                if bridge_condition
                else QA_EVIDENCE_SYSTEM_LABEL
            ),
            "qa_verifier_version": EXTRACTIVE_QA_VERIFIER_VERSION,
            "qa_verifier_question_version": EXTRACTIVE_QA_QUESTION_VERSION,
            "qa_verifier_model_id": EXTRACTIVE_QA_MODEL_ID,
            "qa_verifier_model_revision": EXTRACTIVE_QA_MODEL_REVISION,
            "qa_verifier_backend": "huggingface_transformers_cpu",
            "qa_verifier_dtype": "float32",
            "qa_verifier_artifact_sha256": verifier_sha256,
        }
        for field, expected in expected_identity.items():
            if prediction.get(field) != expected:
                failures.append(f"prediction_{field}_mismatch")
            if metadata.get(field) != expected:
                failures.append(f"metadata_{field}_mismatch")
        for field in (
            "call_only_tool_election",
            "semantic_tool_labels",
            "semantic_slot_labels",
            "verifier_supplies_exact_source_values_only",
        ):
            if metadata.get(field) is not True:
                failures.append(f"{field}_invariant_missing")
        if metadata.get("small_model_supplies_argument_values") is not False:
            failures.append("small_model_argument_value_invariant_missing")
        if metadata.get("no_call_election_option") is not False:
            failures.append("call_only_election_surface_mismatch")

        selected_tool = str(metadata.get("selected_tool") or "")
        by_name = {_tool_name(tool): tool for tool in tools}
        selected_schema = by_name.get(selected_tool)
        if selected_schema is None and accepted_call:
            failures.append("selected_tool_not_in_case")
        slots_by_name: dict[str, dict[str, Any]] = {}
        if selected_schema is not None:
            slots, _ = slot_catalog(selected_schema)
            slots_by_name = {str(slot["name"]): slot for slot in slots}

        decisions = metadata.get("verifier_decisions")
        if not isinstance(decisions, list):
            decisions = []
            failures.append("verifier_decisions_not_list")
        decision_slots: list[str] = []
        admitted: dict[str, dict[str, Any]] = {}
        decision_hashes: list[str] = []
        ranking_sha256 = str(metadata.get("ranking_sha256") or "")
        for decision in decisions:
            if not isinstance(decision, dict):
                failures.append("verifier_decision_not_object")
                continue
            slot_name = str(decision.get("slot") or "")
            decision_slots.append(slot_name)
            record = verifier_index.get((case_id, selected_tool, slot_name))
            if record is None:
                failures.append(f"verifier_record_missing:{slot_name}")
                continue
            slot = slots_by_name.get(slot_name)
            if slot is None:
                failures.append(f"verifier_slot_not_in_schema:{slot_name}")
                continue
            runtime_failures = validate_verifier_record(
                record,
                case_id=case_id,
                request_text=request_text,
                language=language,
                ranking_sha256=ranking_sha256,
                tool=selected_tool,
                slot=slot,
            )
            failures.extend(
                f"verifier_record_invalid:{slot_name}:{failure}"
                for failure in runtime_failures
            )
            for field in (
                "status",
                "admitted",
                "answer",
                "span_id",
                "non_null_margin",
                "row_sha256",
            ):
                if decision.get(field) != record.get(field):
                    failures.append(
                        f"verifier_decision_{field}_mismatch:{slot_name}"
                    )
            decision_hashes.append(str(record.get("row_sha256") or ""))
            if bool(record.get("admitted")):
                admitted[slot_name] = record

        active_pre = metadata.get("active_slots_pre_verifier")
        active_post = metadata.get("active_slots_post_verifier")
        if (accepted_call or active_pre is not None) and active_pre != decision_slots:
            failures.append("active_pre_verifier_decision_mismatch")
        if (accepted_call or active_post is not None) and active_post != sorted(
            admitted
        ):
            failures.append("active_post_verifier_admission_mismatch")
        if (
            accepted_call
            or "qa_verifier_row_sha256" in metadata
        ) and metadata.get("qa_verifier_row_sha256") != decision_hashes:
            failures.append("verifier_row_hash_sequence_mismatch")
        if (
            accepted_call
            or "qa_verifier_rows_consulted" in metadata
        ) and metadata.get("qa_verifier_rows_consulted") != len(decisions):
            failures.append("verifier_consulted_count_mismatch")
        if (
            accepted_call
            or "qa_verifier_null_count" in metadata
        ) and metadata.get("qa_verifier_null_count") != (
            len(decisions) - len(admitted)
        ):
            failures.append("verifier_null_count_mismatch")

        if accepted_call:
            assert isinstance(action, dict)
            arguments = action.get("arguments")
            certificates = metadata.get("evidence_certificates")
            selected_span_ids = metadata.get("selected_span_ids")
            if not isinstance(arguments, dict):
                arguments = {}
                failures.append("arguments_not_object")
            if not isinstance(certificates, dict):
                certificates = {}
                failures.append("certificates_not_object")
            if not isinstance(selected_span_ids, dict):
                selected_span_ids = {}
                failures.append("selected_span_ids_not_object")
            if set(arguments) != set(admitted):
                failures.append("argument_admitted_verifier_key_mismatch")
            if set(arguments) != set(certificates):
                failures.append("argument_certificate_key_mismatch")
            if set(arguments) != set(selected_span_ids):
                failures.append("argument_span_id_key_mismatch")
            for slot_name, value in arguments.items():
                record = admitted.get(slot_name)
                certificate = certificates.get(slot_name)
                if record is None:
                    failures.append(f"argument_without_admission:{slot_name}")
                    continue
                if value != record.get("answer"):
                    failures.append(
                        f"argument_verifier_value_mismatch:{slot_name}"
                    )
                if not isinstance(certificate, dict):
                    failures.append(f"missing_certificate:{slot_name}")
                    continue
                if certificate.get("source_text") != record.get("answer"):
                    failures.append(f"certificate_answer_mismatch:{slot_name}")
                if certificate.get("source_span") != record.get("answer_span"):
                    failures.append(f"certificate_span_mismatch:{slot_name}")
                if certificate.get("span_id") != record.get("span_id"):
                    failures.append(f"certificate_span_id_mismatch:{slot_name}")
                if selected_span_ids.get(slot_name) != record.get("span_id"):
                    failures.append(f"selected_span_id_mismatch:{slot_name}")
                if not replay_span_certificate(
                    request_text,
                    language,
                    certificate,
                ):
                    failures.append(f"certificate_replay_failed:{slot_name}")
            if action.get("tool") != selected_tool:
                failures.append("selected_tool_mismatch")
            if action_fingerprint(action) != metadata.get(
                "materialized_action_sha256"
            ):
                failures.append("materialized_action_hash_mismatch")
            if metadata.get("no_unconstrained_action_critical_tokens") is not True:
                failures.append("finite_surface_invariant_missing")

        rows.append(
            {
                "case_id": prediction.get("case_id"),
                "model_id": prediction.get("model_id"),
                "method": prediction.get("method"),
                "seed": prediction.get("seed"),
                "accepted_call": accepted_call,
                "verifier_decisions": len(decisions),
                "admitted_arguments": len(admitted),
                "passed": not failures,
                "failures": failures,
                "qa_evidence_controller_version": prediction.get(
                    "qa_evidence_controller_version"
                ),
                "qa_evidence_system_label": prediction.get(
                    "qa_evidence_system_label"
                ),
                "qa_verifier_artifact_sha256": prediction.get(
                    "qa_verifier_artifact_sha256"
                ),
            }
        )

    write_jsonl(output_path, rows)
    controller_versions = sorted(
        {str(row["qa_evidence_controller_version"]) for row in rows}
    )
    system_labels = sorted(
        {str(row["qa_evidence_system_label"]) for row in rows}
    )
    report = {
        "schema_version": "tapbench.qa_evidence_certificate_audit.v1",
        "qa_evidence_controller_version": (
            controller_versions[0]
            if len(controller_versions) == 1
            else "mixed"
        ),
        "qa_evidence_controller_versions": controller_versions,
        "qa_evidence_system_label": (
            system_labels[0] if len(system_labels) == 1 else "mixed"
        ),
        "qa_evidence_system_labels": system_labels,
        "qa_verifier_version": EXTRACTIVE_QA_VERIFIER_VERSION,
        "qa_verifier_model_id": EXTRACTIVE_QA_MODEL_ID,
        "qa_verifier_model_revision": EXTRACTIVE_QA_MODEL_REVISION,
        "qa_verifier_artifact_sha256": verifier_sha256,
        "verifier_artifact_rows": len(verifier_rows),
        "verifier_artifact_failures": artifact_failures,
        "rows": len(rows),
        "accepted_calls": sum(bool(row["accepted_call"]) for row in rows),
        "passed": sum(bool(row["passed"]) for row in rows),
        "failed": sum(not bool(row["passed"]) for row in rows),
    }
    target = Path(summary_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from evibind.core.derivations import canonical_json

from .evibench_powered import POWERED_CONDITIONS, powered_condition_specs
from .evibench_study import HumanStudyError, derive_study_workload
from .evibench_study import POLICY_STUDY_FREEZE_VERSION


READINESS_VERSION = "evibind.evibench_powered_readiness.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ReadinessError(ValueError):
    pass


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReadinessError(f"required readiness artifact is missing: {path}")
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ReadinessError(f"{path} must contain a mapping")
    return value


def _projection_digest(manifest: Mapping[str, Any]) -> str:
    projection = {
        "catalog_schema_version": manifest.get("catalog_schema_version"),
        "group": manifest.get("group"),
        "keys": manifest.get("keys"),
        "artifacts": manifest.get("artifacts"),
    }
    return hashlib.sha256(
        canonical_json(projection).encode("utf-8")
    ).hexdigest()


def audit_powered_readiness(
    root: str | Path | None = None,
) -> dict[str, Any]:
    repository = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[1]
    )
    execution_path = (
        repository / "configs/evibench_powered_execution_v1.yaml"
    )
    execution = _read_mapping(execution_path)
    failures: list[str] = []
    blockers: list[str] = []
    checks: list[str] = []

    if execution.get("version") != "evibind.evibench_powered_execution.v1":
        failures.append("execution_config_version")
    if execution.get("status") != "infrastructure_ready_corpus_pending":
        failures.append("execution_config_status")

    amendment_path = repository / str(execution.get("scale_amendment"))
    amendment = _read_mapping(amendment_path)
    if (
        amendment.get("version")
        != "evibind.evibench_powered_scale_amendment.v1"
    ):
        failures.append("scale_amendment_version")
    if amendment.get("status") != "frozen_before_any_powered_outcome":
        failures.append("scale_amendment_status")
    amendment_sha256 = hashlib.sha256(amendment_path.read_bytes()).hexdigest()
    if execution.get("scale_amendment_sha256") != amendment_sha256:
        failures.append("scale_amendment_digest_drift")
    pre_outcome = amendment.get("pre_outcome_evidence")
    if not isinstance(pre_outcome, Mapping) or any(
        pre_outcome.get(key) != 0
        for key in (
            "powered_model_calls_generated",
            "stateful_sessions_generated",
        )
    ):
        failures.append("scale_amendment_not_pre_outcome")
    checks.append("six_model_scale_amendment_frozen_pre_outcome")

    compiler_amendment_path = repository / str(
        execution.get("compiler_amendment")
    )
    compiler_amendment = _read_mapping(compiler_amendment_path)
    if (
        compiler_amendment.get("version")
        != "evibind.evibench_powered_compiler_amendment.v1"
    ):
        failures.append("compiler_amendment_version")
    if (
        compiler_amendment.get("status")
        != "frozen_before_human_powered_outcome_after_proxy_diagnostic"
    ):
        failures.append("compiler_amendment_status")
    compiler_amendment_sha256 = hashlib.sha256(
        compiler_amendment_path.read_bytes()
    ).hexdigest()
    if (
        execution.get("compiler_amendment_sha256")
        != compiler_amendment_sha256
    ):
        failures.append("compiler_amendment_digest_drift")
    outcome_boundary = compiler_amendment.get("outcome_boundary")
    if not isinstance(outcome_boundary, Mapping) or any(
        outcome_boundary.get(key) != 0
        for key in (
            "human_powered_model_calls_generated",
            "human_stateful_sessions_generated",
        )
    ):
        failures.append("compiler_amendment_not_pre_human_outcome")
    elif outcome_boundary.get("proxy_diagnostic_results_inspected") is not True:
        failures.append("compiler_amendment_proxy_boundary_missing")
    implementation_change = compiler_amendment.get("implementation_change")
    if not isinstance(implementation_change, Mapping):
        implementation_change = {}
        failures.append("compiler_amendment_implementation_missing")
    if (
        implementation_change.get("previous_gateway_sha256")
        != "1ab9df4dfec87e40a687476e73ed8f8904c0171cbdaf34778a1e3b2d6fdaf7a6"
    ):
        failures.append("compiler_amendment_parent_digest")
    checks.append("compiler_amendment_frozen_before_human_outcome")

    compiler_path = repository / str(execution.get("compiler_implementation"))
    compiler_sha256: str | None = None
    if not compiler_path.is_file():
        failures.append("compiler_implementation_missing")
    else:
        compiler_sha256 = hashlib.sha256(compiler_path.read_bytes()).hexdigest()
        if execution.get("compiler_implementation_sha256") != compiler_sha256:
            failures.append("compiler_implementation_digest_drift")
        if implementation_change.get("amended_gateway_sha256") != compiler_sha256:
            failures.append("compiler_amendment_gateway_digest_drift")

    binding_path = repository / str(execution.get("binding_implementation"))
    binding_sha256: str | None = None
    if not binding_path.is_file():
        failures.append("binding_implementation_missing")
    else:
        binding_sha256 = hashlib.sha256(binding_path.read_bytes()).hexdigest()
        if execution.get("binding_implementation_sha256") != binding_sha256:
            failures.append("binding_implementation_digest_drift")
        if (
            implementation_change.get("amended_binding_compiler_sha256")
            != binding_sha256
        ):
            failures.append("compiler_amendment_binding_digest_drift")
    checks.append("development_compiler_implementation_hash_frozen")

    preregistration_path = repository / str(execution.get("preregistration"))
    preregistration = _read_mapping(preregistration_path)
    if (
        preregistration.get("version")
        != "evibind.evibench_powered_extension_preregistration.v1"
    ):
        failures.append("preregistration_version")
    if preregistration.get("status") != "protocol_frozen_execution_pending":
        failures.append("preregistration_status")
    checks.append("protocol_remains_execution_pending")

    raw_conditions = preregistration.get("conditions")
    if not isinstance(raw_conditions, list):
        failures.append("preregistration_conditions_missing")
        condition_rows: list[Mapping[str, Any]] = []
    else:
        condition_rows = [
            row for row in raw_conditions if isinstance(row, Mapping)
        ]
        if len(condition_rows) != len(raw_conditions):
            failures.append("preregistration_condition_invalid")
    preregistered_ids = tuple(str(row.get("id")) for row in condition_rows)
    code_specs = powered_condition_specs()
    code_ids = tuple(spec.condition_id for spec in code_specs)
    execution_ids = tuple(str(value) for value in execution.get("conditions", []))
    if preregistered_ids != POWERED_CONDITIONS:
        failures.append("preregistered_condition_order_drift")
    if code_ids != POWERED_CONDITIONS:
        failures.append("implemented_condition_order_drift")
    if execution_ids != POWERED_CONDITIONS:
        failures.append("execution_condition_order_drift")
    if any(
        row.get("implementation_status")
        != "implemented_before_outcome_inspection"
        for row in condition_rows
    ):
        failures.append("condition_implementation_pending")
    preregistered_safety = {
        str(row.get("id")): row.get("production_safe") is True
        for row in condition_rows
    }
    implemented_safety = {
        spec.condition_id: spec.production_safe for spec in code_specs
    }
    if preregistered_safety != implemented_safety:
        failures.append("condition_safety_label_drift")
    if [key for key, safe in implemented_safety.items() if safe] != [
        "evibind_full"
    ]:
        failures.append("production_safe_condition_set")
    checks.append("all_conditions_implemented_and_safety_labeled")

    raw_models = preregistration.get("models")
    required_keys = (
        raw_models.get("required_keys") if isinstance(raw_models, Mapping) else None
    )
    if not isinstance(required_keys, list):
        failures.append("required_model_keys_missing")
        required_keys = []
    model_manifest_path = repository / str(execution.get("model_manifest"))
    model_manifest = _read_mapping(model_manifest_path)
    if model_manifest.get("version") != "evibind.evibench_model_freeze.v1":
        failures.append("model_manifest_version")
    if model_manifest.get("group") != execution.get("model_group"):
        failures.append("model_group_drift")
    if model_manifest.get("keys") != required_keys:
        failures.append("model_key_drift")
    artifacts = model_manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(required_keys):
        failures.append("model_artifact_count")
        artifacts = []
    for row in artifacts:
        if not isinstance(row, Mapping):
            failures.append("model_artifact_invalid")
            continue
        if not isinstance(row.get("bytes"), int) or int(row.get("bytes", 0)) <= 0:
            failures.append(f"model_artifact_size:{row.get('key')}")
        digest = row.get("sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            failures.append(f"model_artifact_digest:{row.get('key')}")
        relative = row.get("relative_path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            failures.append(f"model_artifact_path:{row.get('key')}")
    observed_projection = _projection_digest(model_manifest)
    if model_manifest.get("projection_sha256") != observed_projection:
        failures.append("model_manifest_projection_drift")
    if execution.get("model_manifest_projection_sha256") != observed_projection:
        failures.append("execution_model_projection_drift")
    checks.append("model_artifacts_are_portably_hash_pinned")

    preregistered_seeds = preregistration.get("seeds")
    if execution.get("seeds") != preregistered_seeds:
        failures.append("seed_drift")
    decoding = execution.get("decoding")
    if not isinstance(decoding, Mapping):
        failures.append("decoding_config_missing")
        decoding = {}
    required_decoding = {
        "temperature": 0.0,
        "top_p": 1.0,
        "samples_per_request": 1,
        "retries": 0,
        "parallel_tool_calls": False,
    }
    if any(decoding.get(key) != value for key, value in required_decoding.items()):
        failures.append("decoding_config_not_deterministic")
    decoding_sha256 = hashlib.sha256(
        canonical_json(decoding).encode("utf-8")
    ).hexdigest()
    checks.append("seeds_conditions_and_decoding_are_frozen")

    human_study_path = repository / str(execution.get("human_study_protocol"))
    human_study_sha256 = hashlib.sha256(human_study_path.read_bytes()).hexdigest()
    if execution.get("human_study_protocol_sha256") != human_study_sha256:
        failures.append("human_study_protocol_digest_drift")
    human_study = _read_mapping(human_study_path)
    try:
        human_workload = derive_study_workload(human_study)
    except HumanStudyError:
        failures.append("human_study_protocol_invalid")
        human_workload = {}
    if human_workload.get("corpus", {}).get("publication_cases") != 2500:
        failures.append("human_study_publication_case_count")
    if (
        human_workload.get("powered_compute", {}).get(
            "publication_model_calls"
        )
        != 315000
    ):
        failures.append("human_study_powered_call_count")
    checks.append("human_study_workload_and_sequence_are_frozen")

    policy_manifest_path = repository / str(
        execution.get("policy_study_manifest")
    )
    policy_manifest: Mapping[str, Any] | None = None
    policy_projection: str | None = None
    if policy_manifest_path.is_file():
        policy_manifest = _read_mapping(policy_manifest_path)
        if policy_manifest.get("passed") is not True:
            failures.append("policy_study_manifest_failed")
        if policy_manifest.get("version") != POLICY_STUDY_FREEZE_VERSION:
            failures.append("policy_study_manifest_version")
        policy_digests = policy_manifest.get("digests")
        if not isinstance(policy_digests, Mapping) or not all(
            isinstance(value, str) and _SHA256.fullmatch(value)
            for value in policy_digests.values()
        ):
            failures.append("policy_study_manifest_digests")
        policy_projection = policy_manifest.get(
            "final_policy_projection_sha256"
        )
        if not isinstance(policy_projection, str) or not _SHA256.fullmatch(
            policy_projection
        ):
            failures.append("policy_study_final_policy_projection")
    else:
        blockers.append("policy-authoring study is not frozen")
    checks.append("policy_authoring_study_freeze_gate_evaluated")

    corpus_manifest_path = repository / str(execution.get("corpus_manifest"))
    corpus_manifest: Mapping[str, Any] | None = None
    if corpus_manifest_path.is_file():
        corpus_manifest = _read_mapping(corpus_manifest_path)
        if corpus_manifest.get("passed") is not True:
            failures.append("corpus_manifest_failed")
        if corpus_manifest.get("version") != "evibind.evibench_powered_freeze.v1":
            failures.append("corpus_manifest_version")
        digests = corpus_manifest.get("digests")
        if not isinstance(digests, Mapping) or not all(
            isinstance(value, str) and _SHA256.fullmatch(value)
            for value in digests.values()
        ):
            failures.append("corpus_manifest_digests")
        elif (
            policy_projection is not None
            and digests.get("final_policy_projection_sha256")
            != policy_projection
        ):
            failures.append("policy_study_corpus_policy_projection_mismatch")
    else:
        blockers.append(
            "powered corpus and independent human evidence are not frozen"
        )
    checks.append("corpus_freeze_gate_evaluated")

    infrastructure_passed = not failures
    outcome_generation_allowed = infrastructure_passed and not blockers
    return {
        "version": READINESS_VERSION,
        "infrastructure_passed": infrastructure_passed,
        "outcome_generation_allowed": outcome_generation_allowed,
        "failures": failures,
        "blockers": blockers,
        "checks": checks,
        "check_count": len(checks),
        "condition_count": len(code_specs),
        "model_count": len(artifacts),
        "seed_count": (
            len(preregistered_seeds)
            if isinstance(preregistered_seeds, list)
            else 0
        ),
        "decoding_sha256": decoding_sha256,
        "model_manifest_projection_sha256": observed_projection,
        "compiler_implementation_sha256": compiler_sha256,
        "binding_implementation_sha256": binding_sha256,
        "human_study_protocol_sha256": human_study_sha256,
        "human_study_publication_cases": human_workload.get(
            "corpus",
            {},
        ).get("publication_cases"),
        "human_study_publication_model_calls": human_workload.get(
            "powered_compute",
            {},
        ).get("publication_model_calls"),
        "policy_study_manifest_present": policy_manifest is not None,
        "corpus_manifest_present": corpus_manifest is not None,
    }

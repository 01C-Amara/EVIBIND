from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from evibind.core.derivations import canonical_json

from .evibench import validate_cases
from .io import read_jsonl


POWERED_FREEZE_VERSION = "evibind.evibench_powered_freeze.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_PHENOMENA = frozenset(
    {"correction", "negation", "hypothetical", "quoted_value"}
)


class PoweredFreezeError(ValueError):
    pass


@dataclass(frozen=True)
class FreezeRequirements:
    minimum_cases: int
    minimum_tool_families: int
    maximum_tool_families: int
    required_languages: tuple[str, ...]
    family_disjoint_splits: bool = True
    independent_request_and_policy_authors: bool = True
    double_annotation: bool = True
    blinded_adjudication: bool = True
    preserve_boundary_phenomena: bool = True


def _read_mapping(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise PoweredFreezeError(f"{path} must contain a mapping")
    return value


def load_freeze_requirements(
    preregistration_path: str | Path,
) -> FreezeRequirements:
    preregistration = _read_mapping(preregistration_path)
    expected_version = "evibind.evibench_powered_extension_preregistration.v1"
    if preregistration.get("version") != expected_version:
        raise PoweredFreezeError("unsupported powered preregistration version")
    if preregistration.get("status") != "protocol_frozen_execution_pending":
        raise PoweredFreezeError(
            "powered preregistration must remain explicitly execution-pending"
        )
    gate = preregistration.get("corpus_gate")
    if not isinstance(gate, Mapping):
        raise PoweredFreezeError("powered preregistration omits corpus_gate")
    languages = gate.get("languages")
    if not isinstance(languages, list) or not all(
        isinstance(language, str) and language for language in languages
    ):
        raise PoweredFreezeError("powered preregistration languages are invalid")
    return FreezeRequirements(
        minimum_cases=int(gate.get("minimum_cases", 0)),
        minimum_tool_families=int(gate.get("minimum_tool_families", 0)),
        maximum_tool_families=int(gate.get("maximum_tool_families", 0)),
        required_languages=tuple(languages),
        family_disjoint_splits=gate.get("family_disjoint_splits") is True,
        independent_request_and_policy_authors=(
            gate.get("request_authors_independent_of_policy_authors") is True
        ),
        double_annotation=gate.get("double_annotation") is True,
        blinded_adjudication=gate.get("blinded_adjudication") is True,
        preserve_boundary_phenomena=(
            gate.get(
                "preserve_corrections_negations_hypotheticals_and_quoted_values"
            )
            is True
        ),
    )


def artifact_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def family_schema_digest(
    cases: Sequence[Mapping[str, Any]],
    family: str,
) -> str:
    tools = {
        canonical_json(case["request"]["tools"]): case["request"]["tools"]
        for case in cases
        if case.get("family") == family
    }
    if not tools:
        raise PoweredFreezeError(f"family has no tool schemas: {family}")
    return artifact_sha256([tools[key] for key in sorted(tools)])


def _non_empty_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _expected_annotation(case: Mapping[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    return {
        "expected_mode": expected["mode"],
        "expected_tool_id": expected.get("tool_id"),
        "expected_arguments": expected.get("arguments"),
        "admissible_bindings": expected["admissible_bindings"],
        "critical_destinations": expected["critical_destinations"],
    }


def _annotation_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "expected_mode": row.get("expected_mode"),
        "expected_tool_id": row.get("expected_tool_id"),
        "expected_arguments": row.get("expected_arguments"),
        "admissible_bindings": row.get("admissible_bindings"),
        "critical_destinations": row.get("critical_destinations"),
    }


def _language_gate(
    requirements: FreezeRequirements,
    study_metadata: Mapping[str, Any],
    failures: list[str],
) -> set[str]:
    required: set[str] = set()
    for language in requirements.required_languages:
        if language == "one_non_english_language_selected_before_annotation":
            selected = _non_empty_string(
                study_metadata.get("selected_non_english_language")
            )
            if selected is None or selected.casefold() == "en":
                failures.append("selected_non_english_language_missing")
            else:
                required.add(selected)
        else:
            required.add(language)
    return required


def validate_powered_freeze(
    cases: Sequence[Mapping[str, Any]],
    policy_rows: Sequence[Mapping[str, Any]],
    annotation_rows: Sequence[Mapping[str, Any]],
    adjudication_rows: Sequence[Mapping[str, Any]],
    study_metadata: Mapping[str, Any],
    *,
    requirements: FreezeRequirements,
    preregistration_sha256: str,
) -> dict[str, Any]:
    validate_cases(cases)
    failures: list[str] = []
    case_count = len(cases)
    families = {str(case["family"]) for case in cases}
    if case_count < requirements.minimum_cases:
        failures.append(
            f"case_count:{case_count}<{requirements.minimum_cases}"
        )
    if len(families) < requirements.minimum_tool_families:
        failures.append(
            "family_count:"
            f"{len(families)}<{requirements.minimum_tool_families}"
        )
    if (
        requirements.maximum_tool_families > 0
        and len(families) > requirements.maximum_tool_families
    ):
        failures.append(
            "family_count:"
            f"{len(families)}>{requirements.maximum_tool_families}"
        )

    required_languages = _language_gate(
        requirements,
        study_metadata,
        failures,
    )
    observed_languages: set[str] = set()
    observed_phenomena: set[str] = set()
    request_authors: dict[str, set[str]] = {}
    family_splits: dict[str, set[str]] = {}
    case_index: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        case_id = str(case["case_id"])
        case_index[case_id] = case
        authoring = case.get("authoring")
        if not isinstance(authoring, Mapping):
            failures.append(f"{case_id}:authoring_missing")
            continue
        author_id = _non_empty_string(authoring.get("request_author_id"))
        language = _non_empty_string(authoring.get("language"))
        split = _non_empty_string(authoring.get("split"))
        phenomena = authoring.get("phenomena")
        if author_id is None:
            failures.append(f"{case_id}:request_author_missing")
        else:
            request_authors.setdefault(str(case["family"]), set()).add(author_id)
        if language is None:
            failures.append(f"{case_id}:language_missing")
        else:
            observed_languages.add(language)
        if split is None:
            failures.append(f"{case_id}:split_missing")
        else:
            family_splits.setdefault(str(case["family"]), set()).add(split)
        if not isinstance(phenomena, list) or not all(
            isinstance(item, str) and item for item in phenomena
        ):
            failures.append(f"{case_id}:phenomena_invalid")
        else:
            observed_phenomena.update(phenomena)
    if not required_languages.issubset(observed_languages):
        failures.append(
            "language_coverage_missing:"
            + ",".join(sorted(required_languages - observed_languages))
        )
    if requirements.family_disjoint_splits:
        leaking = sorted(
            family for family, splits in family_splits.items() if len(splits) != 1
        )
        if leaking:
            failures.append("family_split_leakage:" + ",".join(leaking))
        observed_splits = {split for splits in family_splits.values() for split in splits}
        if len(observed_splits) < 2:
            failures.append("fewer_than_two_family_disjoint_splits")
    if (
        requirements.preserve_boundary_phenomena
        and not _REQUIRED_PHENOMENA.issubset(observed_phenomena)
    ):
        failures.append(
            "boundary_phenomena_missing:"
            + ",".join(sorted(_REQUIRED_PHENOMENA - observed_phenomena))
        )

    policies: dict[str, Mapping[str, Any]] = {}
    for row in policy_rows:
        family = _non_empty_string(row.get("family"))
        if family is None:
            failures.append("policy_family_missing")
            continue
        if family in policies:
            failures.append(f"duplicate_policy_family:{family}")
            continue
        policies[family] = row
    if set(policies) != families:
        failures.append(
            "policy_family_coverage:"
            f"missing={sorted(families - set(policies))};"
            f"extra={sorted(set(policies) - families)}"
        )
    policy_authors: dict[str, str] = {}
    for family, row in policies.items():
        author_id = _non_empty_string(row.get("policy_author_id"))
        if author_id is None:
            failures.append(f"{family}:policy_author_missing")
        else:
            policy_authors[family] = author_id
            if (
                requirements.independent_request_and_policy_authors
                and author_id in request_authors.get(family, set())
            ):
                failures.append(f"{family}:request_policy_author_overlap")
        if family not in families:
            continue
        expected_schema_digest = family_schema_digest(cases, family)
        if row.get("schema_sha256") != expected_schema_digest:
            failures.append(f"{family}:schema_digest_mismatch")
        if not isinstance(row.get("policy_sha256"), str) or not _SHA256.fullmatch(
            str(row.get("policy_sha256"))
        ):
            failures.append(f"{family}:policy_digest_invalid")
        if row.get("saw_held_out_requests") is not False:
            failures.append(f"{family}:policy_author_saw_heldout_requests")
        if row.get("saw_model_outputs") is not False:
            failures.append(f"{family}:policy_author_saw_model_outputs")

    annotations: dict[str, list[Mapping[str, Any]]] = {}
    for row in annotation_rows:
        case_id = _non_empty_string(row.get("case_id"))
        if case_id is None:
            failures.append("annotation_case_id_missing")
            continue
        annotations.setdefault(case_id, []).append(row)
    if set(annotations) != set(case_index):
        failures.append(
            "annotation_case_coverage:"
            f"missing={sorted(set(case_index) - set(annotations))};"
            f"extra={sorted(set(annotations) - set(case_index))}"
        )

    adjudications: dict[str, Mapping[str, Any]] = {}
    for row in adjudication_rows:
        case_id = _non_empty_string(row.get("case_id"))
        if case_id is None:
            failures.append("adjudication_case_id_missing")
            continue
        if case_id in adjudications:
            failures.append(f"duplicate_adjudication:{case_id}")
            continue
        adjudications[case_id] = row

    disagreements = 0
    for case_id, case in case_index.items():
        rows = annotations.get(case_id, [])
        if requirements.double_annotation and len(rows) != 2:
            failures.append(f"{case_id}:annotation_count:{len(rows)}")
            continue
        annotator_ids = [
            _non_empty_string(row.get("annotator_id")) for row in rows
        ]
        if None in annotator_ids or len(set(annotator_ids)) != len(annotator_ids):
            failures.append(f"{case_id}:annotators_not_distinct")
        family = str(case["family"])
        forbidden_authors = request_authors.get(family, set()) | {
            policy_authors.get(family, "")
        }
        if any(
            annotator_id in forbidden_authors
            for annotator_id in annotator_ids
            if annotator_id is not None
        ):
            failures.append(f"{case_id}:annotator_author_overlap")
        if any(row.get("blinded_to_conditions") is not True for row in rows):
            failures.append(f"{case_id}:annotation_not_blinded")
        payloads = [_annotation_payload(row) for row in rows]
        agreed = bool(payloads) and all(
            canonical_json(payload) == canonical_json(payloads[0])
            for payload in payloads[1:]
        )
        final_payload: Mapping[str, Any] | None = payloads[0] if agreed else None
        adjudication = adjudications.get(case_id)
        if not agreed:
            disagreements += 1
            if adjudication is None:
                failures.append(f"{case_id}:missing_adjudication")
            else:
                if (
                    requirements.blinded_adjudication
                    and adjudication.get("blinded_to_conditions") is not True
                ):
                    failures.append(f"{case_id}:adjudication_not_blinded")
                adjudicator = _non_empty_string(
                    adjudication.get("adjudicator_id")
                )
                if adjudicator is None or adjudicator in {
                    value for value in annotator_ids if value is not None
                } | forbidden_authors:
                    failures.append(f"{case_id}:adjudicator_not_independent")
                resolution = adjudication.get("resolution")
                if not isinstance(resolution, Mapping):
                    failures.append(f"{case_id}:adjudication_resolution_missing")
                else:
                    final_payload = resolution
        elif adjudication is not None:
            failures.append(f"{case_id}:unnecessary_adjudication")
        if final_payload is not None and canonical_json(
            final_payload
        ) != canonical_json(_expected_annotation(case)):
            failures.append(f"{case_id}:frozen_gold_differs_from_human_resolution")

    unknown_adjudications = set(adjudications) - set(case_index)
    if unknown_adjudications:
        failures.append(
            "unknown_adjudication_cases:"
            + ",".join(sorted(unknown_adjudications))
        )

    codebook_digest = study_metadata.get("annotation_codebook_sha256")
    if not isinstance(codebook_digest, str) or not _SHA256.fullmatch(
        codebook_digest
    ):
        failures.append("annotation_codebook_digest_invalid")
    if not _SHA256.fullmatch(preregistration_sha256):
        failures.append("preregistration_digest_invalid")

    report = {
        "version": POWERED_FREEZE_VERSION,
        "passed": not failures,
        "failures": failures,
        "counts": {
            "cases": case_count,
            "tool_families": len(families),
            "languages": len(observed_languages),
            "splits": len(
                {split for splits in family_splits.values() for split in splits}
            ),
            "policy_rows": len(policy_rows),
            "annotation_rows": len(annotation_rows),
            "adjudication_rows": len(adjudication_rows),
            "annotation_disagreements": disagreements,
        },
        "coverage": {
            "languages": sorted(observed_languages),
            "phenomena": sorted(observed_phenomena),
            "splits": sorted(
                {split for splits in family_splits.values() for split in splits}
            ),
        },
        "digests": {
            "preregistration_sha256": preregistration_sha256,
            "cases_sha256": artifact_sha256(list(cases)),
            "policies_sha256": artifact_sha256(list(policy_rows)),
            "annotations_sha256": artifact_sha256(list(annotation_rows)),
            "adjudications_sha256": artifact_sha256(list(adjudication_rows)),
            "study_metadata_sha256": artifact_sha256(dict(study_metadata)),
            "final_policy_projection_sha256": artifact_sha256(
                sorted(
                    (
                        {
                            "family": row["family"],
                            "policy_sha256": row["policy_sha256"],
                        }
                        for row in policy_rows
                    ),
                    key=lambda row: row["family"],
                )
            ),
        },
    }
    if failures:
        raise PoweredFreezeError("; ".join(failures))
    return report


def write_powered_freeze_manifest(
    *,
    cases_path: str | Path,
    policies_path: str | Path,
    annotations_path: str | Path,
    adjudications_path: str | Path,
    study_metadata_path: str | Path,
    preregistration_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    preregistration_bytes = Path(preregistration_path).read_bytes()
    requirements = load_freeze_requirements(preregistration_path)
    report = validate_powered_freeze(
        read_jsonl(cases_path),
        read_jsonl(policies_path),
        read_jsonl(annotations_path),
        read_jsonl(adjudications_path),
        _read_mapping(study_metadata_path),
        requirements=requirements,
        preregistration_sha256=hashlib.sha256(preregistration_bytes).hexdigest(),
    )
    report["artifacts"] = {
        "cases": str(cases_path),
        "policies": str(policies_path),
        "annotations": str(annotations_path),
        "adjudications": str(adjudications_path),
        "study_metadata": str(study_metadata_path),
        "preregistration": str(preregistration_path),
    }
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report

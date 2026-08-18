from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evibind.core.derivations import canonical_json

from .io import read_jsonl, read_yaml, write_jsonl


HUMAN_STUDY_VERSION = "evibind.evibench_human_study.v1"
STUDY_WORKLOAD_VERSION = "evibind.evibench_study_workload.v1"
STUDY_ASSIGNMENT_VERSION = "evibind.evibench_study_assignments.v1"
POLICY_STUDY_FREEZE_VERSION = "evibind.evibench_policy_study_freeze.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROLES = ("policy_author", "request_author", "annotator", "adjudicator")
_REQUIRED_PHENOMENA = {
    "correction",
    "negation",
    "hypothetical",
    "quoted_value",
}
_REQUIRED_SEQUENCE = (
    "schema_inventory_freeze",
    "policy_authoring_study",
    "pilot_request_authoring",
    "pilot_double_annotation",
    "codebook_and_operations_freeze",
    "full_request_authoring",
    "full_double_annotation",
    "blinded_adjudication",
    "powered_corpus_freeze",
    "held_out_model_generation",
    "powered_replay_and_analysis",
    "stateful_utility_run",
)


class HumanStudyError(ValueError):
    pass


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise HumanStudyError(f"{key} must be a mapping")
    return value


def _positive_int(parent: Mapping[str, Any], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise HumanStudyError(f"{key} must be a positive integer")
    return value


def _string(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HumanStudyError(f"{key} must be a non-empty string")
    return value


def _string_list(parent: Mapping[str, Any], key: str) -> list[str]:
    value = parent.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise HumanStudyError(f"{key} must be a non-empty string list")
    if len(set(value)) != len(value):
        raise HumanStudyError(f"{key} must not contain duplicates")
    return value


def _count_mapping(parent: Mapping[str, Any], key: str) -> dict[str, int]:
    value = _mapping(parent, key)
    counts: dict[str, int] = {}
    for raw_name, raw_count in value.items():
        if (
            not isinstance(raw_name, str)
            or not raw_name
            or not isinstance(raw_count, int)
            or isinstance(raw_count, bool)
            or raw_count <= 0
        ):
            raise HumanStudyError(f"{key} contains an invalid count")
        counts[raw_name] = raw_count
    if not counts:
        raise HumanStudyError(f"{key} must not be empty")
    return counts


def protocol_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _artifact_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def derive_study_workload(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if protocol.get("version") != HUMAN_STUDY_VERSION:
        raise HumanStudyError("unsupported human-study protocol version")
    if protocol.get("status") != "protocol_frozen_recruitment_pending":
        raise HumanStudyError("human-study protocol must remain recruitment-pending")

    seed = _positive_int(protocol, "assignment_seed")
    non_english = _string(protocol, "selected_non_english_language")
    if non_english.casefold() == "en":
        raise HumanStudyError("selected non-English language cannot be en")

    corpus = _mapping(protocol, "corpus")
    cases_per_family = _positive_int(corpus, "cases_per_family")
    split_families = _count_mapping(corpus, "split_families")
    publication_split = _string(corpus, "publication_split")
    if publication_split not in split_families:
        raise HumanStudyError("publication_split is absent from split_families")
    if set(split_families) != {"train", "development", "test"}:
        raise HumanStudyError("splits must be train, development, and test")
    family_count = sum(split_families.values())
    if not 50 <= family_count <= 100:
        raise HumanStudyError("tool-family count must remain between 50 and 100")

    languages = _count_mapping(corpus, "language_cases_per_family")
    if set(languages) != {"en", non_english}:
        raise HumanStudyError("language quotas must be en plus the selected language")
    if sum(languages.values()) != cases_per_family:
        raise HumanStudyError("language quotas must sum to cases_per_family")

    phenomena = _count_mapping(corpus, "phenomenon_strata_per_family")
    if sum(phenomena.values()) != cases_per_family:
        raise HumanStudyError("phenomenon strata must sum to cases_per_family")
    if not _REQUIRED_PHENOMENA.issubset(phenomena):
        raise HumanStudyError("required boundary phenomena are missing")

    split_cases = {
        split: count * cases_per_family for split, count in split_families.items()
    }
    total_cases = sum(split_cases.values())
    publication_cases = split_cases[publication_split]
    if publication_cases < 2500:
        raise HumanStudyError("publication split must contain at least 2,500 cases")

    pilot = _mapping(protocol, "pilot")
    pilot_cases = _positive_int(pilot, "cases")
    pilot_families = _positive_int(pilot, "families")
    pilot_splits = _string_list(pilot, "eligible_splits")
    if publication_split in pilot_splits:
        raise HumanStudyError("pilot cannot include the publication split")
    eligible_pilot_families = sum(split_families[split] for split in pilot_splits)
    if pilot_families > eligible_pilot_families:
        raise HumanStudyError("pilot requests more eligible families than exist")
    if pilot_cases > pilot_families * cases_per_family:
        raise HumanStudyError("pilot requests more cases than its families contain")
    if pilot_cases % pilot_families:
        raise HumanStudyError("pilot cases must divide evenly across pilot families")
    if pilot.get("held_out_from_publication_analysis") is not True:
        raise HumanStudyError("pilot must be held out from publication analysis")

    human_roles = _mapping(protocol, "human_roles")
    role_minima: dict[str, int] = {}
    if set(human_roles) != set(_ROLES):
        raise HumanStudyError("human_roles must define exactly the four study roles")
    for role in _ROLES:
        role_minima[role] = _positive_int(
            _mapping(human_roles, role),
            "minimum_participants",
        )
    if not 5 <= role_minima["policy_author"] <= 8:
        raise HumanStudyError("policy-authoring study requires 5-8 engineers")

    requirements = _mapping(protocol, "participant_requirements")
    for key in (
        "globally_disjoint_roles",
        "pseudonymous_ids_only",
        "consent_recorded",
        "compensation_agreed",
        "training_complete_before_assignment",
    ):
        if requirements.get(key) is not True:
            raise HumanStudyError(f"participant requirement must be true: {key}")

    policy = _mapping(protocol, "policy_authoring_study")
    arms = _string_list(policy, "arms")
    if len(arms) != 2:
        raise HumanStudyError("policy study must have exactly two arms")
    if policy.get("design") != "counterbalanced_within_participant":
        raise HumanStudyError("policy study must remain counterbalanced")
    if _positive_int(policy, "primary_tasks_per_family") != 1:
        raise HumanStudyError("each family must have one primary policy task")
    overlap_families = _positive_int(policy, "overlap_families")
    if overlap_families >= family_count:
        raise HumanStudyError("policy overlap must be smaller than the family set")
    if _positive_int(policy, "independent_reviewer_per_family") != 1:
        raise HumanStudyError("each family must have one independent reviewer")
    for key, expected in (
        ("policy_authors_see_schemas", True),
        ("policy_authors_see_requests", False),
        ("policy_authors_see_model_outputs", False),
    ):
        if policy.get(key) is not expected:
            raise HumanStudyError(f"policy blinding rule drifted: {key}")

    annotation = _mapping(protocol, "annotation")
    judgments_per_case = _positive_int(annotation, "judgments_per_case")
    if judgments_per_case != 2:
        raise HumanStudyError("every case must receive exactly two judgments")
    for key in (
        "annotators_distinct",
        "condition_blind",
        "adjudicate_disagreements_only",
        "adjudicator_independent",
    ):
        if annotation.get(key) is not True:
            raise HumanStudyError(f"annotation rule must be true: {key}")

    compute = _mapping(protocol, "compute")
    model_count = _positive_int(compute, "model_count")
    condition_count = _positive_int(compute, "condition_count")
    seed_count = _positive_int(compute, "seed_count")
    calls_per_cell = _positive_int(compute, "calls_per_cell")
    if (model_count, condition_count, seed_count, calls_per_cell) != (6, 7, 3, 1):
        raise HumanStudyError("powered compute matrix drifted")

    stateful = _mapping(protocol, "stateful_utility")
    stateful_cases = _positive_int(stateful, "held_out_cases")
    if stateful_cases > publication_cases:
        raise HumanStudyError("stateful subset exceeds the publication split")
    stateful_models = _positive_int(stateful, "model_count")
    stateful_conditions = _positive_int(stateful, "condition_count")
    stateful_seeds = _positive_int(stateful, "seed_count")
    sessions_per_cell = _positive_int(stateful, "sessions_per_cell")

    sequence = tuple(_string_list(protocol, "sequence"))
    if sequence != _REQUIRED_SEQUENCE:
        raise HumanStudyError("study sequence drifted")

    language_cases = {
        language: count * family_count for language, count in languages.items()
    }
    annotation_judgments = total_cases * judgments_per_case
    model_cells_per_case = model_count * condition_count * seed_count * calls_per_cell
    stateful_sessions = (
        stateful_cases
        * stateful_models
        * stateful_conditions
        * stateful_seeds
        * sessions_per_cell
    )
    return {
        "version": STUDY_WORKLOAD_VERSION,
        "protocol_valid": True,
        "assignment_seed": seed,
        "corpus": {
            "tool_families": family_count,
            "cases_per_family": cases_per_family,
            "total_cases": total_cases,
            "publication_split": publication_split,
            "publication_cases": publication_cases,
            "split_cases": split_cases,
            "language_cases": language_cases,
            "phenomenon_cases": {
                name: count * family_count for name, count in phenomena.items()
            },
        },
        "human_work": {
            "role_minima": role_minima,
            "request_authoring_slots": total_cases,
            "annotation_judgments": annotation_judgments,
            "adjudications_maximum": total_cases,
            "policy_primary_tasks": family_count,
            "policy_overlap_tasks": overlap_families,
            "policy_review_tasks": family_count,
        },
        "pilot": {
            "cases": pilot_cases,
            "families": pilot_families,
            "eligible_splits": pilot_splits,
            "publication_outcomes_allowed": False,
        },
        "powered_compute": {
            "model_count": model_count,
            "condition_count": condition_count,
            "seed_count": seed_count,
            "calls_per_case": model_cells_per_case,
            "publication_model_calls": publication_cases * model_cells_per_case,
            "all_split_model_calls": total_cases * model_cells_per_case,
        },
        "stateful_compute": {
            "held_out_cases": stateful_cases,
            "sessions": stateful_sessions,
        },
        "sequence": list(sequence),
        "next_run": "schema_inventory_freeze",
        "model_outcome_run_allowed_now": False,
    }


def write_study_workload(
    protocol_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    report = derive_study_workload(read_yaml(protocol_path))
    report["protocol_sha256"] = protocol_sha256(protocol_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _rank(seed: int, scope: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{scope}:{value}".encode()).hexdigest()


def _ordered(
    values: Iterable[str],
    *,
    seed: int,
    scope: str,
) -> list[str]:
    return sorted(values, key=lambda value: (_rank(seed, scope, value), value))


def _expanded_order(
    counts: Mapping[str, int],
    *,
    seed: int,
    scope: str,
) -> list[str]:
    tagged = [(name, index) for name, count in counts.items() for index in range(count)]
    tagged.sort(
        key=lambda item: (
            _rank(seed, scope, f"{item[0]}:{item[1]}"),
            item,
        )
    )
    return [name for name, _ in tagged]


def _participant_index(
    participants: Sequence[Mapping[str, Any]],
    role_minima: Mapping[str, int],
    languages: set[str],
) -> dict[str, list[dict[str, Any]]]:
    by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in _ROLES}
    seen: set[str] = set()
    for row in participants:
        participant_id = _string(row, "participant_id")
        if participant_id in seen:
            raise HumanStudyError(
                f"participant has more than one role: {participant_id}"
            )
        seen.add(participant_id)
        role = _string(row, "role")
        if role not in by_role:
            raise HumanStudyError(f"unsupported participant role: {role}")
        for key in ("consent_recorded", "compensation_agreed", "training_complete"):
            if row.get(key) is not True:
                raise HumanStudyError(
                    f"{participant_id} is not assignment-ready: {key}"
                )
        raw_languages = row.get("languages", [])
        if not isinstance(raw_languages, list) or not all(
            isinstance(language, str) and language for language in raw_languages
        ):
            raise HumanStudyError(f"{participant_id} has invalid languages")
        participant_languages = set(raw_languages)
        if role != "policy_author" and not participant_languages:
            raise HumanStudyError(f"{participant_id} has no qualified language")
        if not participant_languages.issubset(languages):
            raise HumanStudyError(f"{participant_id} has an unplanned language")
        by_role[role].append(
            {
                "participant_id": participant_id,
                "role": role,
                "languages": sorted(participant_languages),
            }
        )

    for role, minimum in role_minima.items():
        if len(by_role[role]) < minimum:
            raise HumanStudyError(f"{role} participants:{len(by_role[role])}<{minimum}")
        by_role[role].sort(key=lambda row: row["participant_id"])
    for language in languages:
        if not any(language in row["languages"] for row in by_role["request_author"]):
            raise HumanStudyError(f"no request author qualified for {language}")
        if sum(language in row["languages"] for row in by_role["annotator"]) < 2:
            raise HumanStudyError(f"fewer than two annotators qualified for {language}")
        if not any(language in row["languages"] for row in by_role["adjudicator"]):
            raise HumanStudyError(f"no adjudicator qualified for {language}")
    return by_role


def _family_index(
    families: Sequence[Mapping[str, Any]],
    split_families: Mapping[str, int],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    observed_splits: Counter[str] = Counter()
    for raw_row in families:
        family = _string(raw_row, "family")
        if family in seen:
            raise HumanStudyError(f"duplicate family: {family}")
        seen.add(family)
        split = _string(raw_row, "split")
        if split not in split_families:
            raise HumanStudyError(f"{family} has an unsupported split")
        for digest_key in ("schema_sha256", "source_artifact_sha256"):
            digest = _string(raw_row, digest_key)
            if not _SHA256.fullmatch(digest):
                raise HumanStudyError(f"{family} has invalid {digest_key}")
        if raw_row.get("license_review_status") != "human_confirmed":
            raise HumanStudyError(
                f"{family} license review is not human-confirmed"
            )
        for source_key in (
            "source_kind",
            "source_locator",
            "source_revision",
            "license",
        ):
            _string(raw_row, source_key)
        observed_splits[split] += 1
        rows.append(dict(raw_row))
    if observed_splits != Counter(split_families):
        raise HumanStudyError(f"family split counts drifted: {dict(observed_splits)}")
    return sorted(rows, key=lambda row: row["family"])


def _least_loaded(
    candidates: Sequence[Mapping[str, Any]],
    loads: Counter[str],
    *,
    seed: int,
    scope: str,
    excluded: set[str] | None = None,
) -> str:
    excluded = excluded or set()
    eligible = [
        str(row["participant_id"])
        for row in candidates
        if str(row["participant_id"]) not in excluded
    ]
    if not eligible:
        raise HumanStudyError(f"no eligible participant for {scope}")
    selected = min(
        eligible,
        key=lambda participant_id: (
            loads[participant_id],
            _rank(seed, scope, participant_id),
            participant_id,
        ),
    )
    loads[selected] += 1
    return selected


def _language_candidates(
    rows: Sequence[Mapping[str, Any]],
    language: str,
) -> list[Mapping[str, Any]]:
    return [row for row in rows if language in row["languages"]]


def build_study_assignments(
    protocol: Mapping[str, Any],
    participants: Sequence[Mapping[str, Any]],
    families: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    workload = derive_study_workload(protocol)
    seed = int(workload["assignment_seed"])
    corpus = _mapping(protocol, "corpus")
    language_counts = _count_mapping(corpus, "language_cases_per_family")
    phenomenon_counts = _count_mapping(
        corpus,
        "phenomenon_strata_per_family",
    )
    split_families = _count_mapping(corpus, "split_families")
    role_minima = workload["human_work"]["role_minima"]
    by_role = _participant_index(participants, role_minima, set(language_counts))
    family_rows = _family_index(families, split_families)

    policy = _mapping(protocol, "policy_authoring_study")
    arms = _string_list(policy, "arms")
    overlap_count = _positive_int(policy, "overlap_families")
    policy_authors = _ordered(
        (row["participant_id"] for row in by_role["policy_author"]),
        seed=seed,
        scope="policy-authors",
    )
    shuffled_families = _ordered(
        (row["family"] for row in family_rows),
        seed=seed,
        scope="policy-families",
    )
    overlap = set(
        _ordered(
            shuffled_families,
            seed=seed,
            scope="policy-overlap",
        )[:overlap_count]
    )
    family_by_id = {row["family"]: row for row in family_rows}
    policy_tasks: list[dict[str, Any]] = []
    for index, family in enumerate(shuffled_families):
        round_index, author_index = divmod(index, len(policy_authors))
        author = policy_authors[author_index]
        reviewer = policy_authors[
            (author_index + 1 + round_index) % len(policy_authors)
        ]
        if reviewer == author:
            reviewer = policy_authors[(author_index + 1) % len(policy_authors)]
        replicate: str | None = None
        if family in overlap:
            for offset in range(2, len(policy_authors) + 1):
                candidate = policy_authors[
                    (author_index + offset + round_index) % len(policy_authors)
                ]
                if candidate not in {author, reviewer}:
                    replicate = candidate
                    break
            if replicate is None:
                raise HumanStudyError("policy overlap lacks an independent author")
        policy_tasks.append(
            {
                "primary_task_id": (
                    "policy-"
                    + hashlib.sha256(f"{seed}:{family}:primary".encode()).hexdigest()[
                        :20
                    ]
                ),
                "replicate_task_id": (
                    "policy-"
                    + hashlib.sha256(f"{seed}:{family}:replicate".encode()).hexdigest()[
                        :20
                    ]
                    if replicate is not None
                    else None
                ),
                "review_task_id": (
                    "policy-"
                    + hashlib.sha256(f"{seed}:{family}:review".encode()).hexdigest()[
                        :20
                    ]
                ),
                "family": family,
                "schema_sha256": family_by_id[family]["schema_sha256"],
                "arm": arms[(round_index + author_index) % len(arms)],
                "policy_author_id": author,
                "independent_reviewer_id": reviewer,
                "replicate_policy_author_id": replicate,
                "may_see": ["tool_schema", "annotation_codebook"],
                "may_not_see": ["requests", "model_outputs"],
                "status": "awaiting_policy_authoring",
            }
        )
    policy_tasks.sort(key=lambda row: row["family"])

    slot_rows: list[dict[str, Any]] = []
    cases_per_family = int(workload["corpus"]["cases_per_family"])
    for family_row in family_rows:
        family = str(family_row["family"])
        slot_languages = _expanded_order(
            language_counts,
            seed=seed,
            scope=f"{family}:languages",
        )
        slot_phenomena = _expanded_order(
            phenomenon_counts,
            seed=seed,
            scope=f"{family}:phenomena",
        )
        if len(slot_languages) != cases_per_family:
            raise HumanStudyError(f"{family} language slot count drifted")
        for index, (language, phenomenon) in enumerate(
            zip(slot_languages, slot_phenomena, strict=True),
            start=1,
        ):
            opaque = hashlib.sha256(f"{seed}:{family}:{index}".encode()).hexdigest()[
                :20
            ]
            slot_rows.append(
                {
                    "slot_id": f"evp-{opaque}",
                    "family": family,
                    "split": family_row["split"],
                    "language": language,
                    "primary_phenomenon": phenomenon,
                }
            )
    slot_rows.sort(key=lambda row: row["slot_id"])

    request_loads: Counter[str] = Counter()
    annotation_loads: Counter[str] = Counter()
    adjudication_loads: Counter[str] = Counter()
    authoring_slots: list[dict[str, Any]] = []
    annotation_slots: list[dict[str, Any]] = []
    for slot in slot_rows:
        language = str(slot["language"])
        slot_id = str(slot["slot_id"])
        request_author = _least_loaded(
            _language_candidates(by_role["request_author"], language),
            request_loads,
            seed=seed,
            scope=f"{slot_id}:request",
        )
        first_annotator = _least_loaded(
            _language_candidates(by_role["annotator"], language),
            annotation_loads,
            seed=seed,
            scope=f"{slot_id}:annotation:1",
        )
        second_annotator = _least_loaded(
            _language_candidates(by_role["annotator"], language),
            annotation_loads,
            seed=seed,
            scope=f"{slot_id}:annotation:2",
            excluded={first_annotator},
        )
        adjudicator = _least_loaded(
            _language_candidates(by_role["adjudicator"], language),
            adjudication_loads,
            seed=seed,
            scope=f"{slot_id}:adjudication",
        )
        authoring_slots.append(
            {
                **slot,
                "request_author_id": request_author,
                "status": "awaiting_independent_request",
            }
        )
        annotation_slots.append(
            {
                "slot_id": slot_id,
                "family": slot["family"],
                "split": slot["split"],
                "language": language,
                "annotator_ids": [first_annotator, second_annotator],
                "adjudicator_id": adjudicator,
                "blinded_to_conditions": True,
                "status": "awaiting_independent_annotations",
            }
        )

    pilot = _mapping(protocol, "pilot")
    pilot_family_count = _positive_int(pilot, "families")
    pilot_case_count = _positive_int(pilot, "cases")
    pilot_splits = set(_string_list(pilot, "eligible_splits"))
    pilot_families = _ordered(
        (row["family"] for row in family_rows if row["split"] in pilot_splits),
        seed=seed,
        scope="pilot-families",
    )[:pilot_family_count]
    cases_per_pilot_family = pilot_case_count // pilot_family_count
    authoring_by_family: dict[str, list[dict[str, Any]]] = {}
    for row in authoring_slots:
        authoring_by_family.setdefault(str(row["family"]), []).append(row)
    pilot_slots: list[dict[str, Any]] = []
    for family in pilot_families:
        selected = sorted(
            authoring_by_family[family],
            key=lambda row: (
                _rank(seed, f"{family}:pilot", str(row["slot_id"])),
                row["slot_id"],
            ),
        )[:cases_per_pilot_family]
        pilot_slots.extend(
            {
                "slot_id": row["slot_id"],
                "family": row["family"],
                "split": row["split"],
            }
            for row in selected
        )
    pilot_slots.sort(key=lambda row: row["slot_id"])

    return {
        "workload": workload,
        "policy_tasks": policy_tasks,
        "authoring_slots": authoring_slots,
        "annotation_slots": annotation_slots,
        "pilot_slots": pilot_slots,
        "loads": {
            "request_author": dict(sorted(request_loads.items())),
            "annotator": dict(sorted(annotation_loads.items())),
            "adjudicator_capacity": dict(sorted(adjudication_loads.items())),
        },
    }


def write_study_assignments(
    *,
    protocol_path: str | Path,
    participants_path: str | Path,
    families_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    participants = read_jsonl(participants_path)
    families = read_jsonl(families_path)
    assignments = build_study_assignments(
        read_yaml(protocol_path),
        participants,
        families,
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    output_rows = {
        "policy_tasks.jsonl": assignments["policy_tasks"],
        "authoring_slots.jsonl": assignments["authoring_slots"],
        "annotation_slots.jsonl": assignments["annotation_slots"],
        "pilot_slots.jsonl": assignments["pilot_slots"],
    }
    for filename, rows in output_rows.items():
        write_jsonl(target / filename, rows)
    manifest = {
        "version": STUDY_ASSIGNMENT_VERSION,
        "protocol_sha256": protocol_sha256(protocol_path),
        "participants_sha256": _artifact_sha256(participants),
        "families_sha256": _artifact_sha256(families),
        "output_sha256": {
            filename: _artifact_sha256(rows) for filename, rows in output_rows.items()
        },
        "counts": {
            "participants": len(participants),
            "families": len(families),
            "policy_tasks": len(assignments["policy_tasks"]),
            "authoring_slots": len(assignments["authoring_slots"]),
            "annotation_slots": len(assignments["annotation_slots"]),
            "pilot_slots": len(assignments["pilot_slots"]),
            "annotation_judgments": (len(assignments["annotation_slots"]) * 2),
        },
        "loads": assignments["loads"],
        "privacy": {
            "participant_ids_are_pseudonymous": True,
            "direct_identifiers_permitted": False,
            "generated_directory_must_remain_untracked": True,
        },
        "blinding": {
            "policy_tasks_contain_requests": False,
            "annotation_slots_contain_author_ids": False,
            "annotation_slots_contain_model_conditions": False,
            "annotation_slots_contain_gold_labels": False,
        },
    }
    (target / "assignment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _timestamp(row: Mapping[str, Any], key: str) -> datetime:
    value = _string(row, key)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HumanStudyError(f"{key} must be RFC 3339") from exc
    if parsed.tzinfo is None:
        raise HumanStudyError(f"{key} must include a timezone")
    return parsed


def _duration(row: Mapping[str, Any]) -> float:
    value = row.get("duration_seconds")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise HumanStudyError("duration_seconds must be positive")
    if _timestamp(row, "completed_at") <= _timestamp(row, "started_at"):
        raise HumanStudyError("completed_at must follow started_at")
    return float(value)


def _nonnegative_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise HumanStudyError(f"{key} must be a non-negative integer")
    return value


def validate_policy_study_freeze(
    policy_tasks: Sequence[Mapping[str, Any]],
    authoring_records: Sequence[Mapping[str, Any]],
    review_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_authoring: dict[str, dict[str, str]] = {}
    expected_reviews: dict[str, dict[str, str]] = {}
    for task in policy_tasks:
        family = _string(task, "family")
        arm = _string(task, "arm")
        primary_task_id = _string(task, "primary_task_id")
        expected_authoring[primary_task_id] = {
            "family": family,
            "arm": arm,
            "participant_id": _string(task, "policy_author_id"),
            "kind": "primary",
        }
        replicate_task_id = task.get("replicate_task_id")
        replicate_author_id = task.get("replicate_policy_author_id")
        if replicate_task_id is not None:
            if not isinstance(replicate_task_id, str) or not replicate_task_id:
                raise HumanStudyError(f"{family} has an invalid replicate task")
            if not isinstance(replicate_author_id, str) or not replicate_author_id:
                raise HumanStudyError(f"{family} has no replicate author")
            expected_authoring[replicate_task_id] = {
                "family": family,
                "arm": arm,
                "participant_id": replicate_author_id,
                "kind": "replicate",
            }
        review_task_id = _string(task, "review_task_id")
        expected_reviews[review_task_id] = {
            "family": family,
            "participant_id": _string(task, "independent_reviewer_id"),
            "primary_task_id": primary_task_id,
        }

    observed_authoring: dict[str, Mapping[str, Any]] = {}
    arm_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in authoring_records:
        task_id = _string(row, "task_id")
        if task_id in observed_authoring:
            raise HumanStudyError(f"duplicate policy authoring record: {task_id}")
        expected = expected_authoring.get(task_id)
        if expected is None:
            raise HumanStudyError(f"unknown policy authoring task: {task_id}")
        for key, expected_key in (
            ("family", "family"),
            ("arm", "arm"),
            ("policy_author_id", "participant_id"),
        ):
            if row.get(key) != expected[expected_key]:
                raise HumanStudyError(f"{task_id} assignment mismatch: {key}")
        if row.get("completion_status") != "completed":
            raise HumanStudyError(f"{task_id} is incomplete")
        if row.get("saw_requests") is not False:
            raise HumanStudyError(f"{task_id} author saw requests")
        if row.get("saw_model_outputs") is not False:
            raise HumanStudyError(f"{task_id} author saw model outputs")
        _duration(row)
        for digest_key in ("policy_sha256", "decision_projection_sha256"):
            digest = _string(row, digest_key)
            if not _SHA256.fullmatch(digest):
                raise HumanStudyError(f"{task_id} has invalid {digest_key}")
        total_slots = _positive_int(row, "policy_slots_total")
        registry_slots = _nonnegative_int(row, "standard_registry_slots")
        if registry_slots > total_slots:
            raise HumanStudyError(f"{task_id} registry coverage exceeds total")
        _nonnegative_int(row, "custom_resolver_count")
        _nonnegative_int(row, "validation_error_count")
        observed_authoring[task_id] = row
        arm_rows.setdefault(str(row["arm"]), []).append(row)
    if set(observed_authoring) != set(expected_authoring):
        missing = sorted(set(expected_authoring) - set(observed_authoring))
        raise HumanStudyError(f"missing policy authoring records: {missing}")

    observed_reviews: set[str] = set()
    final_policies: list[dict[str, str]] = []
    for row in review_records:
        task_id = _string(row, "task_id")
        if task_id in observed_reviews:
            raise HumanStudyError(f"duplicate policy review record: {task_id}")
        expected = expected_reviews.get(task_id)
        if expected is None:
            raise HumanStudyError(f"unknown policy review task: {task_id}")
        if row.get("family") != expected["family"]:
            raise HumanStudyError(f"{task_id} review family mismatch")
        if row.get("reviewer_id") != expected["participant_id"]:
            raise HumanStudyError(f"{task_id} reviewer mismatch")
        if (
            row.get("completion_status") != "completed"
            or row.get("approved") is not True
        ):
            raise HumanStudyError(f"{task_id} review is not approved")
        if row.get("saw_requests") is not False:
            raise HumanStudyError(f"{task_id} reviewer saw requests")
        if row.get("saw_model_outputs") is not False:
            raise HumanStudyError(f"{task_id} reviewer saw model outputs")
        _duration(row)
        _nonnegative_int(row, "errors_caught")
        primary = observed_authoring[expected["primary_task_id"]]
        if row.get("reviewed_policy_sha256") != primary["policy_sha256"]:
            raise HumanStudyError(f"{task_id} reviewed policy digest mismatch")
        final_digest = _string(row, "final_policy_sha256")
        if not _SHA256.fullmatch(final_digest):
            raise HumanStudyError(f"{task_id} final policy digest is invalid")
        final_policies.append(
            {"family": expected["family"], "policy_sha256": final_digest}
        )
        observed_reviews.add(task_id)
    if observed_reviews != set(expected_reviews):
        missing = sorted(set(expected_reviews) - observed_reviews)
        raise HumanStudyError(f"missing policy review records: {missing}")

    overlaps = 0
    exact_agreements = 0
    for task in policy_tasks:
        replicate_task_id = task.get("replicate_task_id")
        if not isinstance(replicate_task_id, str):
            continue
        overlaps += 1
        primary = observed_authoring[str(task["primary_task_id"])]
        replicate = observed_authoring[replicate_task_id]
        exact_agreements += (
            primary["decision_projection_sha256"]
            == replicate["decision_projection_sha256"]
        )

    arm_summary: dict[str, dict[str, Any]] = {}
    for arm, rows in sorted(arm_rows.items()):
        total_slots = sum(int(row["policy_slots_total"]) for row in rows)
        registry_slots = sum(int(row["standard_registry_slots"]) for row in rows)
        arm_summary[arm] = {
            "tasks": len(rows),
            "mean_duration_seconds": (
                sum(float(row["duration_seconds"]) for row in rows) / len(rows)
            ),
            "standard_registry_coverage": registry_slots / total_slots,
            "custom_resolvers": sum(int(row["custom_resolver_count"]) for row in rows),
            "validation_errors": sum(
                int(row["validation_error_count"]) for row in rows
            ),
        }
    final_policies.sort(key=lambda row: row["family"])
    return {
        "version": POLICY_STUDY_FREEZE_VERSION,
        "passed": True,
        "counts": {
            "families": len(policy_tasks),
            "authoring_records": len(authoring_records),
            "review_records": len(review_records),
            "overlap_families": overlaps,
        },
        "arms": arm_summary,
        "agreement": {
            "overlap_families": overlaps,
            "exact_decision_projection_matches": exact_agreements,
            "exact_decision_projection_rate": (
                exact_agreements / overlaps if overlaps else None
            ),
        },
        "final_policy_projection_sha256": _artifact_sha256(final_policies),
    }


def write_policy_study_freeze(
    *,
    protocol_path: str | Path,
    policy_tasks_path: str | Path,
    authoring_records_path: str | Path,
    review_records_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    protocol = read_yaml(protocol_path)
    derive_study_workload(protocol)
    tasks = read_jsonl(policy_tasks_path)
    authoring = read_jsonl(authoring_records_path)
    reviews = read_jsonl(review_records_path)
    report = validate_policy_study_freeze(tasks, authoring, reviews)
    report["digests"] = {
        "protocol_sha256": protocol_sha256(protocol_path),
        "policy_tasks_sha256": _artifact_sha256(tasks),
        "authoring_records_sha256": _artifact_sha256(authoring),
        "review_records_sha256": _artifact_sha256(reviews),
    }
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
import zipfile

from .io import read_jsonl


HUMAN_KIT_VERSION = "evibind.evibench_human_eval_kit.v1"
KIT_ROOT = "evibench_human_eval_kit_v1"
_ZIP_TIMESTAMP = (2026, 7, 31, 0, 0, 0)
_STATIC_REPOSITORY_FILES = (
    "analysis/evibench_powered_inventory_audit.json",
    "configs/evibench_human_study_v1.yaml",
    "configs/evibench_powered_execution_v1.yaml",
    "configs/evibench_powered_extension_preregistration_v1.yaml",
    "configs/evibench_powered_models_v1.json",
    "configs/evibench_powered_queue_v1.yaml",
    "configs/evibench_powered_scale_amendment_v1.yaml",
    "docs/EVIBENCH_HUMAN_STUDY.md",
    "docs/EVIBENCH_POWERED.md",
    "docs/EVIBENCH_RECRUITMENT.md",
    "scripts/audit_evibench_family_inventory.py",
    "scripts/prepare_evibench_family_inventory.py",
)


class HumanKitError(ValueError):
    pass


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_archive_name(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise HumanKitError(f"unsafe archive path: {value}")
    return path.as_posix()


def _put(entries: dict[str, bytes], name: str, value: bytes) -> None:
    safe_name = _safe_archive_name(name)
    if safe_name in entries:
        raise HumanKitError(f"duplicate archive path: {safe_name}")
    entries[safe_name] = value


def _read_required(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise HumanKitError(f"required regular file is missing: {path}")
    return path.read_bytes()


def _json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(_read_required(path))
    except json.JSONDecodeError as exc:
        raise HumanKitError(f"invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise HumanKitError(f"JSON document must be a mapping: {path}")
    return value


def _material_destination(relative: Path) -> str:
    value = relative.as_posix()
    if value == "START_HERE.md":
        return "README.md"
    if value.startswith("roles/"):
        role = relative.stem.casefold()
        return f"role_packets/{role}/INSTRUCTIONS.md"
    if value.startswith("templates/"):
        return "coordinator/templates/" + value.removeprefix("templates/")
    return "coordinator/docs/" + value


def collect_human_kit_entries(
    *,
    repository_root: str | Path,
    work_root: str | Path,
    wheel_path: str | Path,
    expected_family_count: int = 70,
    expected_recruitment_slots: int = 33,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    repository = Path(repository_root)
    work = Path(work_root)
    wheel = Path(wheel_path)
    materials = repository / "human_eval"
    if not materials.is_dir():
        raise HumanKitError("human_eval materials directory is missing")

    families_path = work / "families.jsonl"
    slots_path = work / "recruitment_slots.jsonl"
    inventory = work / "family_inventory"
    audit_path = inventory / "inventory_audit.json"
    families = read_jsonl(families_path)
    slots = read_jsonl(slots_path)
    audit = _json_mapping(audit_path)
    if len(families) != expected_family_count:
        raise HumanKitError(
            f"family count is {len(families)}, expected {expected_family_count}"
        )
    if len({row.get("family") for row in families}) != len(families):
        raise HumanKitError("family identifiers are missing or duplicated")
    if len(slots) != expected_recruitment_slots:
        raise HumanKitError(
            f"recruitment slot count is {len(slots)}, "
            f"expected {expected_recruitment_slots}"
        )
    audit_counts = audit.get("counts")
    if audit.get("passed") is not True or not isinstance(audit_counts, Mapping):
        raise HumanKitError("family inventory audit is absent or did not pass")
    for key in ("families", "sources", "schemas"):
        if audit_counts.get(key) != expected_family_count:
            raise HumanKitError(f"inventory audit {key} count drifted")

    source_files = sorted((inventory / "sources").glob("*.json"))
    schema_files = sorted((inventory / "schemas").glob("*.json"))
    if len(source_files) != expected_family_count:
        raise HumanKitError("cached source count drifted")
    if len(schema_files) != expected_family_count:
        raise HumanKitError("normalized schema count drifted")

    entries: dict[str, bytes] = {}
    for path in sorted(materials.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(materials)
        _put(entries, _material_destination(relative), _read_required(path))

    role_codebooks = {
        "policy_author": "POLICY_AUTHOR_CODEBOOK.md",
        "annotator": "ANNOTATION_CODEBOOK.md",
        "adjudicator": "ANNOTATION_CODEBOOK.md",
    }
    for role, codebook_name in role_codebooks.items():
        _put(
            entries,
            f"role_packets/{role}/{codebook_name}",
            _read_required(materials / codebook_name),
        )
    _put(
        entries,
        "role_packets/ETHICS_AND_DATA_HANDLING.md",
        _read_required(materials / "ETHICS_AND_DATA_HANDLING.md"),
    )

    for relative_string in _STATIC_REPOSITORY_FILES:
        source = repository / relative_string
        relative = Path(relative_string)
        if relative.parts[0] == "configs":
            destination = "coordinator/configs/" + relative.name
        elif relative.parts[0] == "scripts":
            destination = "validator/scripts/" + relative.name
        elif relative.parts[0] == "analysis":
            destination = "coordinator/audit/" + relative.name
        else:
            destination = "coordinator/reference/" + relative.name
        _put(entries, destination, _read_required(source))

    wheel_name = wheel.name
    if not wheel_name.endswith(".whl"):
        raise HumanKitError("validator artifact must be a wheel")
    _put(entries, f"validator/{wheel_name}", _read_required(wheel))

    _put(
        entries,
        "coordinator/inventory/families.jsonl",
        _read_required(families_path),
    )
    _put(
        entries,
        "coordinator/inventory/recruitment_slots.jsonl",
        _read_required(slots_path),
    )
    for path in sorted(inventory.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(work).as_posix()
        _put(entries, f"coordinator/inventory/{relative}", _read_required(path))

    prohibited = [
        name
        for name in entries
        if name.endswith("/participants.jsonl")
        or name.endswith("/FAMILY_LICENSE_REVIEWED")
        or name.endswith("/responses.jsonl")
    ]
    if prohibited:
        raise HumanKitError(f"prohibited evidence entered kit: {prohibited}")

    review_statuses = Counter(str(row.get("license_review_status")) for row in families)
    status = {
        "version": HUMAN_KIT_VERSION,
        "inventory_audit_passed": True,
        "families": len(families),
        "family_splits": dict(
            sorted(Counter(str(row.get("split")) for row in families).items())
        ),
        "license_review_statuses": dict(sorted(review_statuses.items())),
        "recruitment_slots": len(slots),
        "participants_included": 0,
        "human_judgments_included": 0,
        "model_outcomes_included": 0,
        "outcome_generation_allowed": False,
        "distribution_status": "internal_license_review_and_recruitment_handoff",
    }
    _put(
        entries,
        "STATUS.json",
        (json.dumps(status, indent=2, sort_keys=True) + "\n").encode(),
    )
    return entries, status


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(f"{KIT_ROOT}/{_safe_archive_name(name)}", _ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def write_human_kit_archive(
    entries: Mapping[str, bytes],
    *,
    output_path: str | Path,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_manifest = {
        "version": HUMAN_KIT_VERSION,
        "kit_root": KIT_ROOT,
        "status": dict(status),
        "files": [
            {
                "path": name,
                "bytes": len(entries[name]),
                "sha256": _sha256(entries[name]),
            }
            for name in sorted(entries)
        ],
    }
    manifest_bytes = (
        json.dumps(payload_manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    complete = dict(entries)
    _put(complete, "MANIFEST.json", manifest_bytes)
    checksums = "".join(
        f"{_sha256(complete[name])}  {name}\n" for name in sorted(complete)
    ).encode()
    _put(complete, "SHA256SUMS", checksums)

    temporary = output.with_name(output.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for name in sorted(complete):
            archive.writestr(_zip_info(name), complete[name])
    temporary.replace(output)
    archive_digest = _sha256(output.read_bytes())
    sidecar = output.with_name(output.name + ".sha256")
    sidecar.write_text(f"{archive_digest}  {output.name}\n", encoding="utf-8")
    return {
        "version": HUMAN_KIT_VERSION,
        "output": str(output),
        "sha256": archive_digest,
        "files": len(complete),
        "bytes": output.stat().st_size,
        "status": dict(status),
    }


def build_human_eval_kit(
    *,
    repository_root: str | Path,
    work_root: str | Path,
    wheel_path: str | Path,
    output_path: str | Path,
    expected_family_count: int = 70,
    expected_recruitment_slots: int = 33,
) -> dict[str, Any]:
    entries, status = collect_human_kit_entries(
        repository_root=repository_root,
        work_root=work_root,
        wheel_path=wheel_path,
        expected_family_count=expected_family_count,
        expected_recruitment_slots=expected_recruitment_slots,
    )
    return write_human_kit_archive(entries, output_path=output_path, status=status)

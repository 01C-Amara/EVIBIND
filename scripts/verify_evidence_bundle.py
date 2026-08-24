"""Verify a versioned EviBind paper-evidence bundle without extracting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_MEMBERS = 10_000
MAX_UNCOMPRESSED_BYTES = 4 * 1024**3


class EvidenceVerificationError(ValueError):
    """Raised when an evidence archive violates its integrity contract."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_member(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts and bool(path.parts)


def _load_json_member(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    try:
        value = json.loads(archive.read(name))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EvidenceVerificationError(f"invalid JSON member: {name}") from exc
    if not isinstance(value, dict):
        raise EvidenceVerificationError(f"JSON member must be an object: {name}")
    return value


def _parse_manifest(raw: bytes) -> dict[str, str]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise EvidenceVerificationError("MANIFEST.sha256 is not UTF-8") from exc
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise EvidenceVerificationError(
                f"invalid manifest line {line_number}"
            )
        digest, name = parts[0].lower(), parts[1].lstrip(" *")
        if not SHA256_RE.fullmatch(digest) or not _safe_member(name):
            raise EvidenceVerificationError(
                f"invalid manifest entry on line {line_number}"
            )
        if name in entries:
            raise EvidenceVerificationError(f"duplicate manifest entry: {name}")
        entries[name] = digest
    if not entries:
        raise EvidenceVerificationError("manifest is empty")
    return entries


def _expected_sidecar_digest(path: Path) -> str:
    try:
        token = path.read_text(encoding="utf-8").split()[0].lower()
    except (OSError, IndexError) as exc:
        raise EvidenceVerificationError(f"cannot read sidecar: {path}") from exc
    if not SHA256_RE.fullmatch(token):
        raise EvidenceVerificationError("sidecar does not start with a SHA-256 digest")
    return token


def verify_bundle(
    bundle: Path,
    *,
    release_metadata: Path | None = None,
    sidecar: Path | None = None,
) -> dict[str, Any]:
    """Return a machine-readable verification report or raise on any mismatch."""
    bundle_digest = _sha256_path(bundle)
    expected_release: dict[str, Any] = {}
    if release_metadata is not None:
        expected_release = json.loads(release_metadata.read_text(encoding="utf-8"))
        if not isinstance(expected_release, dict):
            raise EvidenceVerificationError("release metadata must be an object")
        expected_bundle = str(expected_release.get("bundle_sha256", "")).lower()
        if expected_bundle and expected_bundle != bundle_digest:
            raise EvidenceVerificationError("bundle digest does not match release metadata")
    if sidecar is not None and _expected_sidecar_digest(sidecar) != bundle_digest:
        raise EvidenceVerificationError("bundle digest does not match sidecar")

    try:
        archive = zipfile.ZipFile(bundle)
    except (OSError, zipfile.BadZipFile) as exc:
        raise EvidenceVerificationError("bundle is not a readable ZIP archive") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_MEMBERS:
            raise EvidenceVerificationError("bundle contains too many members")
        if sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES:
            raise EvidenceVerificationError("bundle exceeds the uncompressed size limit")
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise EvidenceVerificationError("bundle contains duplicate member names")
        if any(not _safe_member(name) for name in names):
            raise EvidenceVerificationError("bundle contains an unsafe member path")
        file_names = [name for name in names if not name.endswith("/")]
        roots = {PurePosixPath(name).parts[0] for name in file_names}
        if len(roots) != 1:
            raise EvidenceVerificationError("bundle must contain exactly one root directory")
        root = next(iter(roots))
        manifest_name = f"{root}/MANIFEST.sha256"
        try:
            manifest = _parse_manifest(archive.read(manifest_name))
        except KeyError as exc:
            raise EvidenceVerificationError("bundle omits MANIFEST.sha256") from exc
        payload_names = {
            name.removeprefix(f"{root}/")
            for name in file_names
            if name != manifest_name
        }
        if set(manifest) != payload_names:
            missing = sorted(payload_names - set(manifest))
            extra = sorted(set(manifest) - payload_names)
            raise EvidenceVerificationError(
                f"manifest coverage mismatch; missing={missing[:3]}, extra={extra[:3]}"
            )
        mismatches = [
            relative
            for relative, digest in manifest.items()
            if _sha256_member(archive, f"{root}/{relative}") != digest
        ]
        if mismatches:
            raise EvidenceVerificationError(
                "manifest digest mismatch: " + ", ".join(mismatches[:3])
            )

        artifact = _load_json_member(archive, f"{root}/ARTIFACT.json")
        audit = _load_json_member(archive, f"{root}/audit/paper_audit.json")
        if artifact.get("archive_root") != root:
            raise EvidenceVerificationError("ARTIFACT.json archive root mismatch")
        if artifact.get("public_payload_files") != len(file_names):
            raise EvidenceVerificationError("ARTIFACT.json file count mismatch")
        if audit.get("passed") is not True or audit.get("failures"):
            raise EvidenceVerificationError("paper audit is not passing")
        for field in ("paper_audit_checks", "paper_claims", "paper_main_text_pages"):
            expected = {
                "paper_audit_checks": audit.get("check_count"),
                "paper_claims": audit.get("claim_count"),
                "paper_main_text_pages": audit.get("main_text_end_page"),
            }[field]
            if artifact.get(field) != expected:
                raise EvidenceVerificationError(f"artifact/audit mismatch: {field}")
        paper_name = f"{root}/paper/main.pdf"
        paper_digest = _sha256_member(archive, paper_name)
        if artifact.get("paper_sha256") != paper_digest:
            raise EvidenceVerificationError("canonical paper digest mismatch")
        if expected_release:
            checks = {
                "archive_root": root,
                "artifact_files": len(file_names),
                "paper_sha256": paper_digest,
                "paper_claims": audit.get("claim_count"),
                "paper_audit_checks": audit.get("check_count"),
                "paper_main_text_pages": audit.get("main_text_end_page"),
            }
            for field, observed in checks.items():
                expected = expected_release.get(field)
                if expected is not None and expected != observed:
                    raise EvidenceVerificationError(
                        f"release metadata mismatch: {field}"
                    )

    return {
        "schema_version": "evibind.evidence_verification.v1",
        "passed": True,
        "bundle_sha256": bundle_digest,
        "archive_root": root,
        "files": len(file_names),
        "manifest_entries": len(manifest),
        "paper_sha256": paper_digest,
        "paper_audit_checks": audit["check_count"],
        "paper_claims": audit["claim_count"],
        "paper_main_text_pages": audit["main_text_end_page"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--release-metadata", type=Path)
    parser.add_argument("--sidecar", type=Path)
    args = parser.parse_args()
    try:
        report = verify_bundle(
            args.bundle,
            release_metadata=args.release_metadata,
            sidecar=args.sidecar,
        )
    except (EvidenceVerificationError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.verify_evidence_bundle import (
    EvidenceVerificationError,
    verify_bundle,
)


def _write_bundle(path: Path, *, corrupt_manifest: bool = False) -> dict[str, str]:
    root = "EviBind_test_artifact"
    paper = b"%PDF-1.4\nminimal test fixture\n%%EOF\n"
    audit = {
        "passed": True,
        "failures": [],
        "check_count": 3,
        "claim_count": 2,
        "main_text_end_page": 1,
    }
    artifact = {
        "archive_root": root,
        "public_payload_files": 4,
        "paper_sha256": hashlib.sha256(paper).hexdigest(),
        "paper_audit_checks": 3,
        "paper_claims": 2,
        "paper_main_text_pages": 1,
    }
    payload = {
        "ARTIFACT.json": json.dumps(artifact, sort_keys=True).encode(),
        "audit/paper_audit.json": json.dumps(audit, sort_keys=True).encode(),
        "paper/main.pdf": paper,
    }
    manifest = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in sorted(payload.items())
    )
    if corrupt_manifest:
        manifest = "0" * 64 + manifest[64:]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(f"{root}/{name}", data)
        archive.writestr(f"{root}/MANIFEST.sha256", manifest)
    return {
        "bundle_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "archive_root": root,
        "paper_sha256": hashlib.sha256(paper).hexdigest(),
        "paper_audit_checks": 3,
        "paper_claims": 2,
        "paper_main_text_pages": 1,
    }


def test_evidence_bundle_verifier_checks_manifest_and_release_record(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "artifact.zip"
    release = _write_bundle(bundle)
    metadata = tmp_path / "release.json"
    metadata.write_text(json.dumps(release), encoding="utf-8")
    sidecar = tmp_path / "artifact.zip.sha256"
    sidecar.write_text(f"{release['bundle_sha256']}  {bundle.name}\n", encoding="utf-8")

    report = verify_bundle(bundle, release_metadata=metadata, sidecar=sidecar)

    assert report["passed"] is True
    assert report["files"] == 4
    assert report["manifest_entries"] == 3
    assert report["paper_claims"] == 2


def test_evidence_bundle_verifier_rejects_member_digest_mismatch(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "artifact.zip"
    _write_bundle(bundle, corrupt_manifest=True)

    with pytest.raises(EvidenceVerificationError, match="manifest digest mismatch"):
        verify_bundle(bundle)


def test_evidence_bundle_verifier_rejects_sidecar_mismatch(tmp_path: Path) -> None:
    bundle = tmp_path / "artifact.zip"
    _write_bundle(bundle)
    sidecar = tmp_path / "artifact.zip.sha256"
    sidecar.write_text(f"{'0' * 64}  {bundle.name}\n", encoding="utf-8")

    with pytest.raises(EvidenceVerificationError, match="sidecar"):
        verify_bundle(bundle, sidecar=sidecar)

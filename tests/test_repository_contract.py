from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import evibind


ROOT = Path(__file__).resolve().parents[1]


def test_product_and_research_entrypoints_are_present() -> None:
    required = (
        "examples/minimal_evidence_binding.py",
        "scripts/reproduce_public_artifact.py",
        "scripts/verify_evidence_bundle.py",
        "evidence/paper-v8.json",
        ".env.example",
        ".dockerignore",
        "evibind/py.typed",
    )
    assert not [path for path in required if not (ROOT / path).is_file()]


def test_version_metadata_is_consistent() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    version = project["project"]["version"]
    assert version == evibind.__version__
    assert re.search(rf"^version:\s*{re.escape(version)}$", citation, re.MULTILINE)


def test_paper_release_record_uses_full_sha256_digests() -> None:
    record = json.loads((ROOT / "evidence/paper-v8.json").read_text(encoding="utf-8"))
    assert record["schema_version"] == "evibind.evidence_release.v1"
    assert re.fullmatch(r"[0-9a-f]{64}", record["bundle_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", record["paper_sha256"])
    assert record["paper_audit_checks"] > 0
    assert record["paper_claims"] > 0


def test_environment_template_contains_names_not_credentials() -> None:
    values = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    assert values["EVIBIND_UPSTREAM_API_KEY"] == ""
    assert values["EVIBIND_GATEWAY_API_KEY"] == ""
    assert values["EVIBIND_HANDLE_SECRET"] == ""


def test_container_runs_as_an_unprivileged_user_with_a_healthcheck() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER evibind" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert 'CMD ["evibind", "serve"' in dockerfile


def test_root_readme_local_links_resolve() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
    local = [target.split("#", 1)[0] for target in targets if "://" not in target]
    missing = [target for target in local if target and not (ROOT / target).exists()]
    assert missing == []

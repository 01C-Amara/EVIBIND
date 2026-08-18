import io
import tarfile
import zipfile

from scripts.audit_release_archives import audit_archives


def _write_sdist(path, files):
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            payload = content.encode()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def test_release_archive_audit_accepts_clean_archives(tmp_path):
    sdist = tmp_path / "package.tar.gz"
    wheel = tmp_path / "package.whl"
    _write_sdist(sdist, {"package/README.md": "clean"})
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("package/__init__.py", "VERSION = '1'")

    assert audit_archives([sdist, wheel]) == []


def test_release_archive_audit_rejects_private_and_generated_content(tmp_path):
    sdist = tmp_path / "package.tar.gz"
    home_path = "/" + "home" + "/example/models"
    _write_sdist(
        sdist,
        {
            "package/work/output.json": "{}",
            "package/config.py": f"ROOT = '{home_path}'",
        },
    )

    findings = audit_archives([sdist])

    assert {finding["finding"] for finding in findings} == {
        "generated_path",
        "machine_local_path:" + "/" + "home" + "/",
    }


def test_release_archive_audit_rejects_generated_paper_outputs(tmp_path):
    sdist = tmp_path / "package.tar.gz"
    _write_sdist(
        sdist,
        {
            "package/paper/main.tex": "source",
            "package/paper/main.pdf": "generated",
            "package/paper/main.aux": "generated",
        },
    )

    findings = audit_archives([sdist])

    assert [finding["finding"] for finding in findings] == [
        "generated_path",
        "generated_path",
    ]


def test_release_archive_audit_rejects_secrets_and_proxy_human_data(tmp_path):
    sdist = tmp_path / "package.tar.gz"
    _write_sdist(
        sdist,
        {
            "package/.env": "API_KEY=" + "a" * 32,
            "package/data/status.txt": "handoff_human_" + "proxy: true",
            "package/keys/test.pem": (
                "-----BEGIN " + "PRIVATE KEY-----\nnot-a-real-key\n"
                "-----END " + "PRIVATE KEY-----"
            ),
        },
    )

    findings = audit_archives([sdist])
    labels = {finding["finding"] for finding in findings}

    assert "forbidden_member:/.env" in labels
    assert "secret_material:assigned_secret" in labels
    assert "secret_material:private_key" in labels
    assert "proxy_human_data" in labels

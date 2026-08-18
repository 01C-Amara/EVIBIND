from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from evibind.core.derivations import canonical_json


MODEL_FREEZE_VERSION = "evibind.evibench_model_freeze.v1"


class ModelFreezeError(ValueError):
    pass


def _read_mapping(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ModelFreezeError(f"{path} must contain a mapping")
    return value


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_models(catalog: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = catalog.get("evaluated_models")
    if not isinstance(rows, list):
        raise ModelFreezeError("model catalog omits evaluated_models")
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ModelFreezeError("model catalog row must be an object")
        key = row.get("key")
        if not isinstance(key, str) or not key:
            raise ModelFreezeError("model catalog key is invalid")
        if key in output:
            raise ModelFreezeError(f"duplicate model key: {key}")
        output[key] = row
    return output


def freeze_model_artifacts(
    *,
    catalog: Mapping[str, Any],
    group_name: str,
    model_root: str | Path,
) -> dict[str, Any]:
    groups = catalog.get("model_groups")
    group = groups.get(group_name) if isinstance(groups, Mapping) else None
    if not isinstance(group, list) or not group or not all(
        isinstance(key, str) and key for key in group
    ):
        raise ModelFreezeError(f"model group is invalid: {group_name}")
    if len(set(group)) != len(group):
        raise ModelFreezeError(f"model group contains duplicates: {group_name}")
    models = _catalog_models(catalog)
    root = Path(model_root).resolve()
    if not root.is_dir():
        raise ModelFreezeError("model root is not a directory")

    artifacts: list[dict[str, Any]] = []
    for key in group:
        row = models.get(key)
        if row is None:
            raise ModelFreezeError(f"model group key is not catalogued: {key}")
        backend = row.get("backend_defaults")
        if not isinstance(backend, Mapping):
            raise ModelFreezeError(f"{key}: backend_defaults missing")
        relative = backend.get("model_artifact")
        if not isinstance(relative, str) or not relative:
            raise ModelFreezeError(f"{key}: model_artifact missing")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ModelFreezeError(f"{key}: model artifact escapes root") from exc
        if not path.is_file():
            raise ModelFreezeError(f"{key}: model artifact is missing: {relative}")
        artifacts.append(
            {
                "key": key,
                "model_id": row.get("id"),
                "role": row.get("role"),
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "quantization": backend.get("quantization"),
                "chat_template": backend.get("chat_template"),
                "grammar_engine": backend.get("grammar_engine"),
                "backend": backend.get("main_backend"),
            }
        )
    catalog_projection = {
        "catalog_schema_version": catalog.get("schema_version"),
        "group": group_name,
        "keys": list(group),
        "artifacts": artifacts,
    }
    return {
        "version": MODEL_FREEZE_VERSION,
        **catalog_projection,
        "projection_sha256": hashlib.sha256(
            canonical_json(catalog_projection).encode("utf-8")
        ).hexdigest(),
    }


def write_model_artifact_manifest(
    *,
    catalog_path: str | Path,
    group_name: str,
    model_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    manifest = freeze_model_artifacts(
        catalog=_read_mapping(catalog_path),
        group_name=group_name,
        model_root=model_root,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .io import read_yaml, write_yaml
from .models import backend_defaults, model_group_entries


DEFAULT_ARTIFACT_MANIFEST = (
    REPO_ROOT / "work" / "main_coefficients" / "artifact_manifest.yaml"
)
MAIN_COEFFICIENT_GRIDS = {
    "H1_prompt_verbosity",
    "H2_constraints_repair",
    "H3_schema_alpha",
    "H6_abstention_suppression",
}


def _artifact_exists(artifact: str) -> bool:
    path = Path(artifact)
    if path.is_absolute():
        return path.exists()
    model_root = Path(
        os.environ.get("TAPBENCH_MODEL_DIR", REPO_ROOT.parent / "LLM" / "models")
    )
    roots = [REPO_ROOT, model_root]
    return any((root / artifact).exists() for root in roots)


def build_artifact_manifest(
    models_cfg: dict[str, Any], *, required_quantization: str = "Q4_K_M"
) -> dict[str, Any]:
    rows = []
    for entry in model_group_entries(models_cfg, "main_core"):
        defaults = backend_defaults(entry)
        artifact = str(defaults.get("model_artifact", ""))
        quantization = str(defaults.get("quantization", "unknown"))
        include = bool(entry.get("include_in_main_coefficients"))
        exists = _artifact_exists(artifact)
        rows.append(
            {
                "model_key": entry.get("key"),
                "model_id": entry.get("id"),
                "tokenizer_family": entry.get("tokenizer_family"),
                "include_in_main_coefficients": include,
                "model_artifact": artifact,
                "quantization": quantization,
                "artifact_exists": exists,
                "quantization_matches": quantization == required_quantization,
                "coefficient_eligible": include
                and exists
                and quantization == required_quantization,
            }
        )
    eligible = [row for row in rows if row["coefficient_eligible"]]
    families = sorted({str(row["tokenizer_family"]) for row in eligible})
    min_families = int(
        models_cfg.get("tokenizer_diversity_requirement", {}).get(
            "main_core_min_families", 2
        )
    )
    ready = (
        len(eligible) == sum(1 for row in rows if row["include_in_main_coefficients"])
        and len(families) >= min_families
    )
    return {
        "schema_version": "tapbench.coefficient_artifact_manifest.v1",
        "required_quantization": required_quantization,
        "main_coefficients_ready": ready,
        "eligible_model_count": len(eligible),
        "eligible_tokenizer_families": families,
        "required_tokenizer_family_count": min_families,
        "models": rows,
    }


def write_artifact_manifest(
    models_cfg: dict[str, Any],
    output: str | Path,
    *,
    required_quantization: str = "Q4_K_M",
) -> dict[str, Any]:
    manifest = build_artifact_manifest(
        models_cfg, required_quantization=required_quantization
    )
    write_yaml(output, manifest)
    Path(output).with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def assert_main_artifact_gate(
    grid_ids: set[str], manifest_path: str | Path = DEFAULT_ARTIFACT_MANIFEST
) -> None:
    if not grid_ids & MAIN_COEFFICIENT_GRIDS:
        return
    target = Path(manifest_path)
    if not target.exists():
        raise RuntimeError(
            "full H1/H2/H3/H6 run is blocked until a coefficient artifact manifest is written"
        )
    manifest = read_yaml(target)
    if manifest.get("schema_version") != "tapbench.coefficient_artifact_manifest.v1":
        raise RuntimeError(
            "full H1/H2/H3/H6 run is blocked because the coefficient artifact manifest has an unknown schema"
        )
    if not manifest.get("main_coefficients_ready"):
        raise RuntimeError(
            "full H1/H2/H3/H6 run is blocked because coefficient artifacts are not standardized and ready"
        )

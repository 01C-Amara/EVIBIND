from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBGRIDS = REPO_ROOT / "configs" / "hypothesis_subgrids.yaml"
DEFAULT_MODELS = REPO_ROOT / "configs" / "model_pins.yaml"
DEFAULT_HYPOTHESIS_MAP = REPO_ROOT / "analysis" / "hypothesis_map.yaml"
DEFAULT_RUNTIME_PROJECTION = REPO_ROOT / "work" / "pilot" / "runtime_projection.yaml"
PACKAGE_DATA = Path(__file__).resolve().parent / "data"
PORTABLE_SUBGRIDS = PACKAGE_DATA / "hypothesis_subgrids.yaml"
PORTABLE_MODELS = PACKAGE_DATA / "model_catalog.yaml"
PORTABLE_HYPOTHESIS_MAP = PACKAGE_DATA / "hypothesis_map.yaml"


@dataclass(frozen=True)
class ExperimentConfig:
    subgrids: dict[str, Any]
    models: dict[str, Any]
    hypothesis_map: dict[str, Any]


def _portable_default(
    path: str | Path,
    *,
    repository_default: Path,
    packaged_default: Path,
) -> Path:
    candidate = Path(path)
    if candidate == repository_default and not candidate.exists():
        return packaged_default
    return candidate


def load_experiment_config(
    subgrids_path: str | Path = DEFAULT_SUBGRIDS,
    models_path: str | Path = DEFAULT_MODELS,
    hypothesis_map_path: str | Path = DEFAULT_HYPOTHESIS_MAP,
) -> ExperimentConfig:
    subgrids_path = _portable_default(
        subgrids_path,
        repository_default=DEFAULT_SUBGRIDS,
        packaged_default=PORTABLE_SUBGRIDS,
    )
    models_path = _portable_default(
        models_path,
        repository_default=DEFAULT_MODELS,
        packaged_default=PORTABLE_MODELS,
    )
    hypothesis_map_path = _portable_default(
        hypothesis_map_path,
        repository_default=DEFAULT_HYPOTHESIS_MAP,
        packaged_default=PORTABLE_HYPOTHESIS_MAP,
    )
    return ExperimentConfig(
        subgrids=read_yaml(subgrids_path),
        models=read_yaml(models_path),
        hypothesis_map=read_yaml(hypothesis_map_path),
    )


def selected_model_ids(models_cfg: dict[str, Any], *, include_optional: bool = False) -> list[str]:
    ids: list[str] = []
    for entry in models_cfg.get("evaluated_models", []):
        if entry.get("exclude_from_main_coefficients") and not include_optional:
            continue
        ids.append(str(entry["id"]))
    return ids

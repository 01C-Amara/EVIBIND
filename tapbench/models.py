from __future__ import annotations

from typing import Any


def model_by_key(models_cfg: dict[str, Any], key: str) -> dict[str, Any]:
    for entry in models_cfg.get("evaluated_models", []):
        if entry.get("key") == key:
            return entry
    raise KeyError(f"unknown model key: {key}")


def model_group_keys(models_cfg: dict[str, Any], group: str) -> list[str]:
    groups = models_cfg.get("model_groups", {})
    if group not in groups:
        raise KeyError(f"unknown model group: {group}")
    return [str(key) for key in groups[group]]


def model_group_entries(models_cfg: dict[str, Any], group: str) -> list[dict[str, Any]]:
    return [model_by_key(models_cfg, key) for key in model_group_keys(models_cfg, group)]


def backend_defaults(model_entry: dict[str, Any]) -> dict[str, Any]:
    defaults = model_entry.get("backend_defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError(f"model {model_entry.get('key')} has invalid backend_defaults")
    return defaults


def primary_model_count(models_cfg: dict[str, Any], group: str) -> int:
    return len(model_group_keys(models_cfg, group))
